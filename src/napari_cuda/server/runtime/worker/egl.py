"""Render-thread orchestration for the server.

`EGLRendererWorker` lives in this module and is still EGL-backed internally, but
the neutral module name reflects its broader responsibilities:

* bootstrap the napari Viewer/VisPy canvas and keep it on the worker thread,
* drain `RenderUpdateMailbox` updates and apply them to the viewer/visuals,
* drive the render loop, including camera animation and policy evaluation,
* capture frames via `CaptureFacade`, hand them to the encoder, and surface
  timing metadata for downstream metrics.

The worker owns the EGL + CUDA context pair, renders into an FBO-backed texture,
maps the texture into CUDA memory, stages the pixels for NVENC, and hands the
encoded packets back to the asyncio side through callbacks.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import numpy as np

try:
    import dask.array as da  # type: ignore
except Exception as _da_err:
    da = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning("dask.array not available; OME-Zarr features disabled: %s", _da_err)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

from vispy import scene  # type: ignore

import napari_cuda.server.data.lod as lod
import napari_cuda.server.data.policy as level_policy
from napari.components.viewer_model import ViewerModel
from napari_cuda.server.app.config import LevelPolicySettings, ServerCtx
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.data.hw_limits import get_hw_limits
from napari_cuda.server.data.level_budget import (
    LevelBudgetError,
)
from napari_cuda.server.data.level_logging import (
    LayerAssignmentLogger,
    LevelSwitchLogger,
)

from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.rendering.capture import FrameTimings, encode_frame
from napari_cuda.server.rendering.debug_tools import DebugConfig, DebugDumper
from napari_cuda.server.runtime.camera import (
    CameraCommandQueue,
    CameraPoseApplied,
)
from napari_cuda.server.runtime.camera.animator import animate_if_enabled
from napari_cuda.server.runtime.camera.controller import (
    process_camera_deltas as _process_camera_deltas,
)
from napari_cuda.server.runtime.ipc import LevelSwitchIntent
from napari_cuda.server.runtime.core.snapshot_build import RenderLedgerSnapshot
from napari_cuda.server.runtime.worker.snapshots import (
    apply_plane_metadata,
    apply_slice_level,
    apply_slice_roi,
    apply_volume_level,
    apply_volume_metadata,
)
from napari_cuda.server.runtime.viewport.roi import viewport_roi_for_level
from napari_cuda.server.runtime.data import (
    SliceROI,
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
    roi_chunk_signature,
)
from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdate, RenderUpdateMailbox
from napari_cuda.server.runtime.core import (
    ensure_scene_source,
    ledger_axis_labels,
    ledger_displayed,
    ledger_level,
    ledger_level_shapes,
    ledger_ndisplay,
    ledger_order,
    ledger_step,
)
from napari_cuda.server.runtime.worker.loop import run_render_tick
from napari_cuda.server.runtime.worker.resources import WorkerResources
from napari_cuda.server.runtime.viewport import (
    PlaneState,
    RenderMode,
    ViewportRunner,
    ViewportState,
    VolumeState,
)
from napari_cuda.server.scene import CameraDeltaCommand
from . import render_updates as _render_updates
from . import viewer as _viewer

logger = logging.getLogger(__name__)


# Back-compat for tests patching the selector directly
select_level = lod.select_level

## Camera ops now live in napari_cuda.server.runtime.camera.ops as free functions.

class _LevelBudgetError(RuntimeError):
    """Raised when a multiscale level exceeds memory/voxel budgets."""


class EGLRendererWorker:
    """Headless VisPy renderer using EGL with CUDA interop and NVENC."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        use_volume: bool = False,
        fps: int = 60,
        volume_depth: int = 64,
        volume_dtype: str = "float32",
        volume_relative_step: Optional[float] = None,
        animate: bool = False,
        animate_dps: float = 30.0,
        zarr_path: Optional[str] = None,
        zarr_level: Optional[str] = None,
        zarr_axes: Optional[str] = None,
        zarr_z: Optional[int] = None,
        camera_pose_cb: Callable[[CameraPoseApplied], None] | None = None,
        level_intent_cb: Callable[[LevelSwitchIntent], None] | None = None,
        policy_name: Optional[str] = None,
        *,
        camera_queue: CameraCommandQueue,
        ctx: ServerCtx,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._budget_error_cls = _LevelBudgetError
        self.fps = int(fps)
        self.volume_depth = int(volume_depth)
        self.volume_dtype = str(volume_dtype)
        self.volume_relative_step = volume_relative_step
        self._ctx: ServerCtx = ctx
        self._debug_policy = ctx.debug_policy
        self._raw_dump_budget = max(0, int(self._debug_policy.dumps.raw_budget))
        self._debug_policy_logged = False
        self._env: Optional[Mapping[str, str]] = dict(env) if env is not None else None

        # Viewport state shared between runner and worker.
        self._viewport_state = ViewportState()
        self._viewport_state.mode = RenderMode.VOLUME if use_volume else RenderMode.PLANE
        self._viewport_runner = ViewportRunner(self._viewport_state.plane)

        self._configure_animation(animate, animate_dps)
        self._init_render_components()
        self._init_scene_state()
        self._camera_queue = camera_queue
        self._init_locks(camera_pose_cb, level_intent_cb)
        self._last_slice_signature: Optional[
            tuple[int, Optional[tuple[int, ...]], Optional[tuple[int, int, int, int]]]
        ] = None
        self._last_snapshot_signature: Optional[tuple] = None
        self._configure_debug_flags()
        self._configure_policy(self._ctx.policy)
        self._configure_roi_settings()
        self._configure_budget_limits()
        self._last_dims_signature: Optional[tuple] = None
        self._applied_versions: dict[tuple[str, str, str], int] = {}
        self._last_plane_pose: Optional[tuple] = None
        self._last_volume_pose: Optional[tuple] = None

        if policy_name:
            try:
                self.set_policy(policy_name)
            except Exception:
                logger.exception("policy init set failed; continuing with default")

        # Zarr/NGFF dataset configuration (optional); prefer explicit args from server
        self._zarr_path = zarr_path
        self._zarr_level = zarr_level
        self._zarr_axes = (zarr_axes or 'zyx')
        self._zarr_init_z = (zarr_z if (zarr_z is not None and int(zarr_z) >= 0) else None)

        self._zarr_shape: Optional[tuple[int, ...]] = None
        self._zarr_dtype: Optional[str] = None
        self._z_index: Optional[int] = None
        self._zarr_clim: Optional[tuple[float, float]] = None
        self._hw_limits = get_hw_limits()

        self._is_ready = False
        self._debug: Optional[DebugDumper] = None
        self._debug_config = DebugConfig.from_policy(self._debug_policy.dumps)
        self._ledger: ServerStateLedger = ServerStateLedger()

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def set_camera_pose_callback(
        self,
        callback: Callable[[CameraPoseApplied], None],
    ) -> None:
        self._camera_pose_callback = callback

    @property
    def viewport_state(self) -> ViewportState:
        return self._viewport_state

    def _current_level_index(self) -> int:
        if self._viewport_state.mode is RenderMode.VOLUME:
            return int(self._viewport_state.volume.level)
        return int(self._viewport_state.plane.applied_level)

    def _set_current_level_index(self, value: int) -> None:
        level = int(value)
        self._viewport_state.plane.applied_level = level
        self._viewport_state.volume.level = level

    @property
    def _volume_scale(self) -> tuple[float, float, float]:
        scale = self._viewport_state.volume.scale
        if scale is None:
            return (1.0, 1.0, 1.0)
        return scale

    @_volume_scale.setter
    def _volume_scale(self, value: tuple[float, float, float]) -> None:
        self._viewport_state.volume.scale = tuple(float(component) for component in value)

    def _set_dims_range_for_level(self, source: ZarrSceneSource, level: int) -> None:
        """Set napari dims.range to match the chosen level shape.

        This avoids Z slider reflecting base-resolution depth when rendering
        a coarser level.
        """
        if self._viewer is None:
            return
        descriptor = source.level_descriptors[level]
        shape = tuple(int(s) for s in descriptor.shape)
        ranges = tuple((0, max(0, s - 1), 1) for s in shape)
        self._viewer.dims.range = ranges

    # Legacy zoom-delta policy path removed; selection uses oversampling + stabilizer

    # Legacy _apply_zoom_based_level_switch removed

    def _init_egl(self) -> None:
        self.cuda_ctx = self._resources.bootstrap(server_ctx=self._ctx, fps_hint=int(self.fps))

    def _init_cuda(self) -> None:
        # CUDA context is established during bootstrap; nothing additional required.
        return

    def _create_scene_source(self) -> Optional[ZarrSceneSource]:
        if self._zarr_path is None:
            return None
        if da is None:
            raise RuntimeError("ZarrSceneSource requires dask.array to be available")
        axes_override = tuple(self._zarr_axes) if isinstance(self._zarr_axes, str) else None
        source = ZarrSceneSource(
            self._zarr_path,
            preferred_level=self._zarr_level,
            axis_override=axes_override,
        )
        return source

    def _ensure_scene_source(self) -> ZarrSceneSource:
        return ensure_scene_source(self)

    def _init_vispy_scene(self) -> None:
        """Adapter-only scene initialization (legacy path removed)."""
        source = self._create_scene_source()
        if source is not None:
            self._scene_source = source
            try:
                self._zarr_shape = source.level_shape(0)
                self._zarr_dtype = str(source.dtype)
                self._set_current_level_index(source.current_level)
                descriptor = source.level_descriptors[source.current_level]
                self._zarr_level = descriptor.path or None
            except Exception:
                logger.debug("scene source metadata bootstrap failed", exc_info=True)

        try:
            self._init_viewer_scene(source)
        except Exception:
            logger.exception("Adapter scene initialization failed (legacy path removed)")
            # Fail fast: adapter is mandatory
            raise

    # --- Multiscale: request switch (thread-safe) ---
    def request_multiscale_level(self, level: int, path: Optional[str] = None) -> None:
        """Emit a manual level intent back to the controller."""

        if getattr(self, "_lock_level", None) is not None:
            if self._log_layer_debug and logger.isEnabledFor(logging.INFO):
                logger.info("request_multiscale_level ignored due to lock_level=%s", str(self._lock_level))
            return

        callback = self._level_intent_callback
        if callback is None:
            logger.debug("manual level switch ignored (no callback)")
            return

        source = self._ensure_scene_source()
        target = int(level)
        if path:
            target = int(source.level_index_for_path(path))

        descriptors = source.level_descriptors
        if descriptors:
            target = max(0, min(target, len(descriptors) - 1))

        downgraded = False
        if self._viewport_state.mode is RenderMode.VOLUME:
            target, downgraded = self._resolve_volume_intent_level(source, target)

        self._viewport_state.volume.downgraded = bool(downgraded)

        decision = lod.LevelDecision(
            desired_level=int(target),
            selected_level=int(target),
            reason="manual",
            timestamp=time.perf_counter(),
            oversampling={},
            downgraded=bool(downgraded),
        )
        ledger = self._ledger
        step_hint = ledger_step(ledger)
        context = lod.build_level_context(
            decision,
            source=source,
            prev_level=int(self._current_level_index()),
            last_step=last_step,
        )
        apply_plane_metadata(self, source, context)

        intent = LevelSwitchIntent(
            desired_level=int(target),
            selected_level=int(context.level),
            reason="manual",
            previous_level=int(self._current_level_index()),
            context=context,
            oversampling={},
            timestamp=decision.timestamp,
            downgraded=bool(downgraded),
            zoom_ratio=None,
            lock_level=self._lock_level,
            mode=self.viewport_state.mode,
            plane_state=deepcopy(self.viewport_state.plane),
            volume_state=deepcopy(self.viewport_state.volume),
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "intent.level_switch: prev=%d target=%d reason=%s downgraded=%s",
                int(self._current_level_index()),
                int(context.level),
                intent.reason,
                intent.downgraded,
            )

        self._viewport_runner.ingest_snapshot(RenderLedgerSnapshot(current_level=int(context.level)))
        self._last_interaction_ts = time.perf_counter()
        callback(intent)
        self._mark_render_tick_needed()

    def _mark_render_tick_needed(self) -> None:
        self._render_tick_required = True

    def _mark_render_tick_complete(self) -> None:
        self._render_tick_required = False

    def _mark_render_loop_started(self) -> None:
        self._render_loop_started = True

    # ---- Init helpers -------------------------------------------------------

    def _configure_animation(self, animate: bool, animate_dps: float) -> None:
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception as exc:
            logger.debug("Invalid animate_dps=%r; using default 30.0: %s", animate_dps, exc)
            self._animate_dps = 30.0
        self._anim_start = time.perf_counter()

    def _init_render_components(self) -> None:
        self._resources = WorkerResources(width=self.width, height=self.height)
        self.cuda_ctx = None
        self.canvas: Optional[scene.SceneCanvas] = None
        self.view = None
        self._plane_visual_handle = None
        self._volume_visual_handle = None

    @property
    def resources(self) -> WorkerResources:
        """Expose render resources owned by the worker."""

        return self._resources

    def _init_scene_state(self) -> None:
        self._viewer: Optional[ViewerModel] = None
        self._napari_layer = None
        self._scene_source: Optional[ZarrSceneSource] = None
        self._set_current_level_index(0)
        self._viewport_state.volume.downgraded = False
        self._data_wh = (int(self.width), int(self.height))
        self._data_d = None
        self._volume_scale = (1.0, 1.0, 1.0)
        self._layer_logger = LayerAssignmentLogger(logger)
        self._switch_logger = LevelSwitchLogger(logger)
        # Monotonic sequence for programmatic camera pose commits (op close)
        self._pose_seq: int = 1
        # Track highest command_seq observed from camera deltas
        self._max_camera_command_seq: int = 0
        self._ledger: Optional[ServerStateLedger] = None
        self._level_policy_suppressed = False


    def _init_locks(
        self,
        camera_pose_cb: Optional[Callable[[CameraPoseApplied], None]],
        level_intent_cb: Optional[Callable[[LevelSwitchIntent], None]] = None,
    ) -> None:
        self._state_lock = threading.RLock()
        self._render_mailbox = RenderUpdateMailbox()
        self._last_ensure_log: Optional[tuple[int, Optional[str]]] = None
        self._last_ensure_log_ts = 0.0
        self._render_tick_required = False
        self._render_loop_started = False
        if camera_pose_cb is None:
            raise ValueError("EGLRendererWorker requires camera_pose callback")
        self._camera_pose_callback = camera_pose_cb
        self._level_intent_callback = level_intent_cb

    def _configure_debug_flags(self) -> None:
        worker_dbg = self._debug_policy.worker
        self._debug_zoom_drift = bool(worker_dbg.debug_zoom_drift)
        self._debug_pan = bool(worker_dbg.debug_pan)
        self._debug_reset = bool(worker_dbg.debug_reset)
        self._debug_orbit = bool(worker_dbg.debug_orbit)
        self._orbit_el_min = float(worker_dbg.orbit_el_min)
        self._orbit_el_max = float(worker_dbg.orbit_el_max)

    def _configure_policy(self, policy_cfg: LevelPolicySettings) -> None:
        self._policy_func = level_policy.resolve_policy('oversampling')
        self._policy_name = 'oversampling'
        self._last_interaction_ts = time.perf_counter()
        policy_logging = self._debug_policy.logging
        worker_dbg = self._debug_policy.worker
        self._log_layer_debug = bool(policy_logging.log_layer_debug)
        self._log_policy_eval = bool(policy_cfg.log_policy_eval)
        self._lock_level = worker_dbg.lock_level
        self._level_threshold_in = float(policy_cfg.threshold_in)
        self._level_threshold_out = float(policy_cfg.threshold_out)
        self._level_hysteresis = float(policy_cfg.hysteresis)
        self._level_fine_threshold = float(policy_cfg.fine_threshold)
        self._sticky_contrast = bool(policy_cfg.sticky_contrast)
        self._level_switch_cooldown_ms = float(policy_cfg.cooldown_ms)
        self._last_level_switch_ts = 0.0
        self._oversampling_thresholds = (
            {int(k): float(v) for k, v in policy_cfg.oversampling_thresholds.items()}
            if getattr(policy_cfg, "oversampling_thresholds", None)
            else None
        )
        self._oversampling_hysteresis = float(getattr(policy_cfg, "oversampling_hysteresis", 0.1))

    def _configure_roi_settings(self) -> None:
        worker_dbg = self._debug_policy.worker
        self._roi_edge_threshold = int(worker_dbg.roi_edge_threshold)
        self._roi_align_chunks = bool(worker_dbg.roi_align_chunks)
        self._roi_ensure_contains_viewport = bool(worker_dbg.roi_ensure_contains_viewport)
        self._roi_pad_chunks = 1
        self._idr_on_z = False

    def _configure_budget_limits(self) -> None:
        cfg = self._ctx.cfg
        self._slice_max_bytes = int(max(0, getattr(cfg, 'max_slice_bytes', 0)))
        self._volume_max_bytes = int(max(0, getattr(cfg, 'max_volume_bytes', 0)))
        self._volume_max_voxels = int(max(0, getattr(cfg, 'max_volume_voxels', 0)))

    def _init_viewer_scene(self, source: Optional[ZarrSceneSource]) -> None:
        _viewer._init_viewer_scene(self, source)

    def _configure_camera_for_mode(self) -> None:
        _viewer._configure_camera_for_mode(self)

    def _frame_volume_camera(self, w: float, h: float, d: float) -> None:
        _viewer._frame_volume_camera(self, w, h, d)

    def _bootstrap_camera_pose(
        self,
        mode: RenderMode,
        source: Optional[ZarrSceneSource],
        *,
        reason: str,
    ) -> None:
        _viewer._bootstrap_camera_pose(self, mode, source, reason=reason)

    def _enter_volume_mode(self) -> None:
        _viewer._enter_volume_mode(self)

    def _exit_volume_mode(self) -> None:
        _viewer._exit_volume_mode(self)

    def _register_plane_visual(self, node: Any) -> None:
        _viewer._register_plane_visual(self, node)

    def _register_volume_visual(self, node: Any) -> None:
        _viewer._register_volume_visual(self, node)

    def _ensure_plane_visual(self) -> Any:
        return _viewer._ensure_plane_visual(self)

    def _ensure_volume_visual(self) -> Any:
        return _viewer._ensure_volume_visual(self)

    def _apply_camera_reset(self, cam) -> None:
        _viewer._apply_camera_reset(self, cam)

    def _emit_current_camera_pose(self, reason: str) -> None:
        _viewer._emit_current_camera_pose(self, reason)

    def _emit_pose_from_camera(self, camera, reason: str) -> None:
        _viewer._emit_pose_from_camera(self, camera, reason)

    def _pose_from_camera(self, camera, target: str, command_seq: int) -> Optional[CameraPoseApplied]:
        return _viewer._pose_from_camera(self, camera, target, command_seq)

    def _snapshot_camera_pose(self, target: str, command_seq: int) -> Optional[CameraPoseApplied]:
        return _viewer._snapshot_camera_pose(self, target, command_seq)

    def _current_panzoom_rect(self) -> Optional[tuple[float, float, float, float]]:
        return _viewer._current_panzoom_rect(self)

    def snapshot_dims_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}

        axes = ledger_axis_labels(ledger)
        if axes:
            meta["axis_labels"] = list(axes)
            meta["axes"] = list(axes)

        order = ledger_order(ledger)
        if order:
            meta["order"] = list(order)

        displayed = ledger_displayed(ledger)
        if displayed:
            meta["displayed"] = list(displayed)

        ndisplay = ledger_ndisplay(ledger)
        if ndisplay is not None:
            meta["ndisplay"] = int(ndisplay)
            meta["mode"] = "volume" if int(ndisplay) >= 3 else "plane"

        current_step = ledger_step(ledger)
        if current_step:
            meta["current_step"] = list(current_step)

        level_shapes = ledger_level_shapes(ledger)
        if level_shapes:
            meta["level_shapes"] = [list(shape) for shape in level_shapes]
        level_idx = ledger_level(ledger)
        if level_shapes and level_idx is not None and 0 <= level_idx < len(level_shapes):
            current_shape = level_shapes[level_idx]
            meta["sizes"] = [int(size) for size in current_shape]

        ndim_candidates: list[int] = []
        if axes:
            ndim_candidates.append(len(axes))
        if order:
            ndim_candidates.append(len(order))
        if current_step:
            ndim_candidates.append(len(current_step))
        if level_shapes:
            ndim_candidates.extend(len(shape) for shape in level_shapes if shape)
        if displayed:
            ndim_candidates.append(len(displayed))

        ndim = max(ndim_candidates) if ndim_candidates else 1
        meta["ndim"] = ndim

        if "sizes" not in meta:
            meta["sizes"] = [1] * ndim

        meta["range"] = [[0, max(0, size - 1)] for size in meta["sizes"]]

        return meta

    def _log_debug_policy_once(self) -> None:
        if self._debug_policy_logged:
            return
        if logger.isEnabledFor(logging.INFO):
            logger.info("Debug policy resolved: %s", self._debug_policy)
        self._debug_policy_logged = True

    # ---- Level helpers ------------------------------------------------------

    def _update_level_metadata(self, descriptor, applied) -> None:
        self._set_current_level_index(applied.level)
        self._z_index = applied.z_index
        self._zarr_level = descriptor.path or None
        self._zarr_shape = descriptor.shape
        self._zarr_axes = applied.axes
        self._zarr_dtype = applied.dtype
        self._zarr_clim = applied.contrast
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "worker level applied: level=%s step=%s z=%s shape=%s",
                applied.level,
                applied.step,
                applied.z_index,
                descriptor.shape,
            )

    def set_policy(self, name: str) -> None:
        new_name = str(name or '').strip().lower()
        # Accept only simplified oversampling policy (and close synonyms)
        allowed = {'oversampling', 'thresholds', 'ratio'}
        if new_name not in allowed:
            raise ValueError(f"Unsupported policy: {new_name}")
        self._policy_name = new_name
        # Map all aliases to the same selector
        self._policy_func = level_policy.resolve_policy(new_name)
        # No history maintenance; snapshot excludes history
        if self._log_layer_debug:
            logger.info("policy set: name=%s", new_name)

    def _physical_scale_for_level(self, source: ZarrSceneSource, level: int) -> tuple[float, float, float]:
        try:
            scale = source.level_scale(level)
        except Exception:
            return (1.0, 1.0, 1.0)
        values = [float(s) for s in scale]
        if not values:
            return (1.0, 1.0, 1.0)
        if len(values) >= 3:
            return (values[-3], values[-2], values[-1])
        if len(values) == 2:
            return (1.0, values[0], values[1])
        # single element; reuse across axes
        return (values[0], values[0], values[0])

    def _oversampling_for_level(self, source: ZarrSceneSource, level: int) -> float:
        try:
            roi = viewport_roi_for_level(self, source, level, quiet=True, for_policy=True)
            if roi.is_empty():
                h, w = plane_wh_for_level(source, level)
            else:
                h, w = roi.height, roi.width
        except Exception:
            h, w = self.height, self.width
        vh = max(1, int(self.height))
        vw = max(1, int(self.width))
        return float(max(h / vh, w / vw))


    def _load_slice(
        self,
        source: ZarrSceneSource,
        level: int,
        z_idx: int,
    ) -> np.ndarray:
        roi = viewport_roi_for_level(self, source, level)
        slab = source.slice(level, z_idx, compute=True, roi=roi)
        if not isinstance(slab, np.ndarray):
            slab = np.asarray(slab, dtype=np.float32)
        return slab

    def _load_volume(
        self,
        source: ZarrSceneSource,
        level: int,
    ) -> np.ndarray:
        volume = source.level_volume(level, compute=True)
        if not isinstance(volume, np.ndarray):
            volume = np.asarray(volume, dtype=np.float32)
        return volume

    def _build_policy_context(
        self,
        source: ZarrSceneSource,
        *,
        requested_level: Optional[int],
    ) -> level_policy.LevelSelectionContext:
        levels = tuple(source.level_descriptors)
        overs_map: dict[int, float] = {}
        for descriptor in levels:
            try:
                lvl = int(descriptor.index)
            except Exception:
                lvl = int(levels.index(descriptor))
            try:
                overs_map[lvl] = float(self._oversampling_for_level(source, lvl))
            except Exception:
                overs_map[lvl] = float('nan')
        return level_policy.LevelSelectionContext(
            levels=levels,
            current_level=int(self._current_level_index()),
            requested_level=int(requested_level) if requested_level is not None else None,
            level_oversampling=overs_map,
            thresholds=self._oversampling_thresholds,
            hysteresis=self._oversampling_hysteresis,
        )

    def _estimate_level_bytes(self, source: ZarrSceneSource, level: int) -> tuple[int, int]:
        descriptor = source.level_descriptors[level]
        voxels = 1
        for dim in descriptor.shape:
            voxels *= max(1, int(dim))
        dtype_size = np.dtype(source.dtype).itemsize
        return int(voxels), int(voxels * dtype_size)

    def _volume_budget_allows(self, source: ZarrSceneSource, level: int) -> None:
        voxels, bytes_est = self._estimate_level_bytes(source, level)
        limit_bytes = self._volume_max_bytes or self._hw_limits.volume_max_bytes
        limit_voxels = self._volume_max_voxels or self._hw_limits.volume_max_voxels
        if limit_voxels and voxels > limit_voxels:
            msg = f"voxels={voxels} exceeds cap={limit_voxels}"
            if self._log_layer_debug:
                logger.info("budget check (volume): level=%d voxels=%d bytes=%d -> REJECT: %s", level, voxels, bytes_est, msg)
            raise _LevelBudgetError(msg)
        if limit_bytes and bytes_est > limit_bytes:
            msg = f"bytes={bytes_est} exceeds cap={limit_bytes}"
            if self._log_layer_debug:
                logger.info("budget check (volume): level=%d voxels=%d bytes=%d -> REJECT: %s", level, voxels, bytes_est, msg)
            raise _LevelBudgetError(msg)
        if self._log_layer_debug:
            logger.info("budget check (volume): level=%d voxels=%d bytes=%d -> OK", level, voxels, bytes_est)

    def _resolve_volume_intent_level(
        self,
        source: ZarrSceneSource,
        requested_level: int,
    ) -> tuple[int, bool]:
        # Volume rendering always uses the coarsest level for now, bypassing budgets/clamps.
        descriptors = source.level_descriptors
        if not descriptors:
            return int(requested_level), False
        coarsest = max(0, len(descriptors) - 1)
        return int(coarsest), bool(coarsest != int(requested_level))

    def _slice_budget_allows(self, source: ZarrSceneSource, level: int) -> None:
        """Enforce optional max-bytes budget for a single 2D slice.

        Uses YX plane dimensions and the source dtype size. Only applies when
        NAPARI_CUDA_MAX_SLICE_BYTES is non-zero.
        """
        limit_bytes = int(self._slice_max_bytes or 0)
        if limit_bytes <= 0:
            return
        h, w = plane_wh_for_level(source, level)
        dtype_size = int(np.dtype(source.dtype).itemsize)
        bytes_est = int(h) * int(w) * dtype_size
        if bytes_est > limit_bytes:
            msg = f"slice bytes={bytes_est} exceeds cap={limit_bytes} at level={level}"
            if self._log_layer_debug:
                logger.info(
                    "budget check (slice): level=%d h=%d w=%d dtype=%s bytes=%d -> REJECT: %s",
                    level, h, w, str(source.dtype), bytes_est, msg,
                )
            raise _LevelBudgetError(msg)
        if self._log_layer_debug:
            logger.info(
                "budget check (slice): level=%d h=%d w=%d dtype=%s bytes=%d -> OK",
                level, h, w, str(source.dtype), bytes_est,
            )

    def _get_level_volume(
        self,
        source: ZarrSceneSource,
        level: int,
    ) -> np.ndarray:
        self._volume_budget_allows(source, level)
        return self._load_volume(source, level)

    def _apply_level(
        self,
        source: ZarrSceneSource,
        level: int,
        *,
        prev_level: Optional[int] = None,
    ) -> lod.LevelContext:
        runner_step: Optional[tuple[int, ...]] = None
        if self._viewport_runner is not None:
            runner_step = self._viewport_runner.state.target_step

        if runner_step is not None:
            step_hint: Optional[tuple[int, ...]] = runner_step
        else:
            recorded = ledger_step(ledger)
            step_hint = (
                tuple(int(v) for v in recorded)
                if recorded is not None
                else None
            )

        decision = lod.LevelDecision(
            desired_level=int(level),
            selected_level=int(level),
            reason="direct",
            timestamp=time.perf_counter(),
            oversampling={},
            downgraded=False,
        )

        context = lod.build_level_context(
            decision,
            source=source,
            prev_level=prev_level,
            last_step=step_hint,
        )
        if self._viewport_state.mode is RenderMode.VOLUME:
            apply_volume_metadata(self, source, context)
        else:
            apply_plane_metadata(self, source, context)
        return context

    def _apply_volume_level(
        self,
        source: ZarrSceneSource,
        applied: lod.LevelContext,
    ) -> None:
        apply_volume_level(
            self,
            source,
            applied,
            downgraded=bool(self._viewport_state.volume.downgraded),
        )

    def _apply_slice_level(
        self,
        source: ZarrSceneSource,
        applied: lod.LevelContext,
    ) -> None:
        apply_slice_level(self, source, applied)

    def _format_level_roi(self, source: ZarrSceneSource, level: int) -> str:
        if self._viewport_state.mode is RenderMode.VOLUME:
            return "volume"
        roi = viewport_roi_for_level(self, source, level)
        if roi.is_empty():
            return "full"
        return f"y={roi.y_start}:{roi.y_stop} x={roi.x_start}:{roi.x_stop}"

    def _aligned_roi_signature(
        self,
        source: ZarrSceneSource,
        level: int,
        roi: SliceROI,
    ) -> tuple[SliceROI, Optional[tuple[int, int]], Optional[tuple[int, int, int, int]]]:
        """Return chunk-aligned ROI and its signature for comparison."""

        chunk_shape = chunk_shape_for_level(source, int(level))
        aligned_roi = roi
        if self._roi_align_chunks and chunk_shape is not None:
            full_h, full_w = plane_wh_for_level(source, int(level))
            aligned_roi = align_roi_to_chunk_grid(
                roi,
                chunk_shape,
                int(self._roi_pad_chunks),
                height=full_h,
                width=full_w,
            )
        signature = roi_chunk_signature(aligned_roi, chunk_shape)
        return aligned_roi, chunk_shape, signature

    def _volume_shape_for_view(self) -> Optional[tuple[int, int, int]]:
        if self._data_d is not None and self._data_wh is not None:
            w, h = self._data_wh
            return int(self._data_d), int(h), int(w)
        if self._zarr_shape is not None and len(self._zarr_shape) >= 3:
            z, y, x = self._zarr_shape[:3]
            return int(z), int(y), int(x)
        return None

    def _volume_world_extents(self) -> Optional[tuple[float, float, float]]:
        shape = self._volume_shape_for_view()
        if shape is None:
            return None
        z, y, x = shape
        sz, sy, sx = self._volume_scale
        width = float(x) * float(sx)
        height = float(y) * float(sy)
        depth = float(z) * float(sz)
        extents = (width, height, depth)
        self._viewport_state.volume.world_extents = extents
        return extents

    def reset_encoder(self) -> None:
        self._resources.reset_encoder(fps_hint=int(self.fps), server_ctx=self._ctx)

    def force_idr(self) -> None:
        logger.debug("Requesting encoder force IDR")
        self._resources.force_idr()

    def _request_encoder_idr(self) -> None:
        self._resources.request_idr()

    def render_frame(self, azimuth_deg: Optional[float] = None) -> None:
        if azimuth_deg is not None:
            assert self.view is not None and self.view.camera is not None
            self.view.camera.azimuth = float(azimuth_deg)
        _render_updates.drain_scene_updates(self)
        t0 = time.perf_counter()
        self.canvas.render()
        self._render_tick_required = False
        self._render_loop_started = True

    def _dims_signature(self, snapshot: RenderLedgerSnapshot) -> tuple:
        return (
            int(snapshot.ndisplay) if snapshot.ndisplay is not None else None,
            tuple(int(v) for v in snapshot.order) if snapshot.order is not None else None,
            tuple(int(v) for v in snapshot.displayed) if snapshot.displayed is not None else None,
            tuple(int(v) for v in snapshot.current_step) if snapshot.current_step is not None else None,
            int(snapshot.current_level) if snapshot.current_level is not None else None,
            tuple(str(v) for v in snapshot.axis_labels) if snapshot.axis_labels is not None else None,
        )

    def _apply_dims_from_snapshot(self, snapshot: RenderLedgerSnapshot, *, signature: tuple) -> None:
        viewer = self._viewer
        if viewer is None:
            return

        self._last_dims_signature = signature

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "dims.apply: ndisplay=%s order=%s displayed=%s current_step=%s",
                str(snapshot.ndisplay), str(snapshot.order), str(snapshot.displayed), str(snapshot.current_step)
            )

        dims = viewer.dims
        ndim = int(getattr(dims, "ndim", 0) or 0)

        axis_labels_src = snapshot.axis_labels
        if axis_labels_src:
            labels = tuple(str(v) for v in axis_labels_src)
            dims.axis_labels = labels
            ndim = max(ndim, len(labels))

        order_src = snapshot.order
        if order_src:
            ndim = max(ndim, len(tuple(int(v) for v in order_src)))

        fallback_order: Optional[tuple[int, ...]] = None
        current_order = getattr(dims, "order", None)
        if current_order is not None:
            fallback_order = tuple(int(v) for v in current_order)

        if snapshot.level_shapes and snapshot.current_level is not None:
            level_shapes = snapshot.level_shapes
            level_idx = int(snapshot.current_level)
            if level_shapes and 0 <= level_idx < len(level_shapes):
                ndim = max(ndim, len(level_shapes[level_idx]))

        if ndim <= 0:
            ndim = max(len(dims.current_step), len(dims.axis_labels)) or 1
        if dims.ndim != ndim:
            dims.ndim = ndim

        if snapshot.current_step is not None:
            step_tuple = tuple(int(v) for v in snapshot.current_step)
            if len(step_tuple) < ndim:
                step_tuple = step_tuple + tuple(0 for _ in range(ndim - len(step_tuple)))
            elif len(step_tuple) > ndim:
                step_tuple = step_tuple[:ndim]
            dims.current_step = step_tuple
        if snapshot.current_level is not None:
            self._set_current_level_index(int(snapshot.current_level))

        if snapshot.order is not None:
            order_values = tuple(int(v) for v in snapshot.order)
        else:
            assert fallback_order is not None, "ledger missing dims order"
            order_values = fallback_order
        assert order_values, "ledger emitted empty dims order"
        dims.order = order_values

        displayed_src = snapshot.displayed
        if displayed_src is not None:
            displayed_tuple = tuple(int(v) for v in displayed_src)
        else:
            current_displayed = getattr(dims, "displayed", None)
            assert current_displayed is not None, "ledger missing dims displayed"
            displayed_tuple = tuple(int(v) for v in current_displayed)
        assert displayed_tuple, "ledger emitted empty dims displayed"
        expected_displayed = tuple(order_values[-len(displayed_tuple):])
        assert displayed_tuple == expected_displayed, "ledger displayed mismatch order/ndisplay"

        if snapshot.ndisplay is not None:
            dims.ndisplay = max(1, int(snapshot.ndisplay))

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "dims.applied: ndim=%d order=%s displayed=%s ndisplay=%d",
                int(dims.ndim), str(tuple(dims.order)), str(tuple(dims.displayed)), int(dims.ndisplay)
            )


    def _update_z_index_from_snapshot(self, snapshot: RenderLedgerSnapshot) -> None:
        if snapshot.axis_labels is None or snapshot.current_step is None:
            return
        labels = [str(label).lower() for label in snapshot.axis_labels]
        if "z" not in labels:
            return
        idx = labels.index("z")
        if idx < len(snapshot.current_step):
            self._z_index = int(snapshot.current_step[idx])

    def enqueue_update(self, delta: RenderUpdate) -> None:
        """Normalize and enqueue a render delta for the worker mailbox."""

        if (
            delta.mode is not None
            or delta.plane_state is not None
            or delta.volume_state is not None
        ):
            self._render_mailbox.set_viewport_state(
                mode=delta.mode,
                plane_state=delta.plane_state,
                volume_state=delta.volume_state,
            )

        scene_state = None
        if delta.scene_state is not None:
            self._last_interaction_ts = time.perf_counter()
            scene_state = _render_updates.normalize_scene_state(delta.scene_state)
        if scene_state is not None:
            self._render_mailbox.set_scene_state(scene_state)
            self._mark_render_tick_needed()
            return

    def process_camera_deltas(self, commands: Sequence[CameraDeltaCommand]) -> None:
        if not commands:
            return

        outcome = _process_camera_deltas(self, commands)
        last_seq = getattr(outcome, "last_command_seq", None)
        if last_seq is not None:
            last_seq_int = int(last_seq)
            self._max_camera_command_seq = max(int(self._max_camera_command_seq), last_seq_int)
            self._pose_seq = max(int(self._pose_seq), int(self._max_camera_command_seq))

        self._viewport_runner.ingest_camera_deltas(commands)
        if (
            self._viewport_runner is not None
            and self._viewport_state.mode is RenderMode.PLANE
        ):
            rect = self._current_panzoom_rect()
            if rect is not None:
                self._viewport_runner.update_camera_rect(rect)
        self._mark_render_tick_needed()
        self._user_interaction_seen = True
        self._last_interaction_ts = time.perf_counter()

        if outcome.camera_changed and self._viewport_state.mode is RenderMode.VOLUME:
            self._emit_current_camera_pose("camera-delta")

        self._run_viewport_tick()

        if (
            bool(getattr(outcome, "policy_triggered", False))
            and self._viewport_state.mode is not RenderMode.VOLUME
        ):
            self._evaluate_level_policy()

    def _record_zoom_hint(self, commands: Sequence[CameraDeltaCommand]) -> None:
        """Legacy hook kept for tests; delegates to render mailbox."""

        for command in reversed(commands):
            if getattr(command, "kind", None) != "zoom":
                continue
            factor = getattr(command, "factor", None)
            if factor is None:
                continue
            factor = float(factor)
            if factor > 0.0:
                self._render_mailbox.record_zoom_hint(factor)
                break

    def _capture_blit_gpu_ns(self) -> Optional[int]:
        return self._resources.capture.pipeline.capture_blit_gpu_ns()


    def capture_and_encode_packet(self) -> tuple[FrameTimings, Optional[bytes], int, int]:
        """Same as capture_and_encode, but also returns the packet and flags.

        Flags bit 0x01 indicates keyframe (IDR/CRA).
        """
        encoded = encode_frame(
            capture=self._resources.capture,
            render_frame=self.render_tick,
            obtain_encoder=lambda: self._resources.encoder,
            encoder_lock=self._resources.enc_lock,
            debug_dumper=self._debug,
        )

        return encoded.timings, encoded.packet, encoded.flags, encoded.sequence

    def render_tick(self) -> float:
        canvas = self.canvas
        assert canvas is not None, "render_tick requires an initialized canvas"

        return run_render_tick(
            animate_camera=lambda: animate_if_enabled(
                enabled=bool(self._animate),
                view=getattr(self, "view", None),
                width=self.width,
                height=self.height,
                animate_dps=float(self._animate_dps),
                anim_start=float(self._anim_start),
            ),
            drain_scene_updates=lambda: self._drain_camera_ops_then_scene_updates(),
            render_canvas=canvas.render,
            evaluate_policy_if_needed=lambda: None,
            mark_tick_complete=self._mark_render_tick_complete,
            mark_loop_started=self._mark_render_loop_started,
        )

    def _drain_camera_ops_then_scene_updates(self) -> None:
        commands = self._camera_queue.pop_all()
        policy_triggered = False

        if commands:
            outcome = _process_camera_deltas(self, commands)
            policy_triggered = bool(getattr(outcome, "policy_triggered", False))
            last_seq = getattr(outcome, "last_command_seq", None)
            if last_seq is not None:
                last_seq_int = int(last_seq)
                self._max_camera_command_seq = max(int(self._max_camera_command_seq), last_seq_int)
                self._pose_seq = max(int(self._pose_seq), int(self._max_camera_command_seq))

            self._viewport_runner.ingest_camera_deltas(commands)
            if self._viewport_state.mode is RenderMode.PLANE:
                rect = self._current_panzoom_rect()
                if rect is not None:
                    self._viewport_runner.update_camera_rect(rect)
            self._mark_render_tick_needed()
            self._user_interaction_seen = True
            self._last_interaction_ts = time.perf_counter()
            self._level_policy_suppressed = False

            if outcome.camera_changed and self._viewport_state.mode is RenderMode.VOLUME:
                self._emit_current_camera_pose("camera-delta")

        _render_updates.drain_scene_updates(self)
        self._run_viewport_tick()

        if (
            policy_triggered
            and self._viewport_state.mode is not RenderMode.VOLUME
            and not self._level_policy_suppressed
        ):
            self._evaluate_level_policy()

    def _run_viewport_tick(self) -> None:
        if self._level_policy_suppressed:
            runner = self._viewport_runner
            if runner is not None:
                runner.state.zoom_hint = None
            return

        if self._viewport_state.mode is RenderMode.VOLUME:
            runner = self._viewport_runner
            if runner is not None:
                runner.state.zoom_hint = None
            return

        source = self._scene_source or self._ensure_scene_source()

        def _resolve(level: int, _rect: tuple[float, float, float, float]) -> SliceROI:
            return viewport_roi_for_level(self, source, int(level), quiet=True)

        plan = self._viewport_runner.plan_tick(source=source, roi_resolver=_resolve)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "viewport.tick plan level_change=%s slice=%s pose_event=%s target_level=%d applied_level=%d awaiting=%s",
                plan.level_change,
                plan.slice_task.signature if plan.slice_task is not None else None,
                plan.pose_event.value if plan.pose_event is not None else None,
                int(self._viewport_runner.state.target_level),
                int(self._viewport_runner.state.applied_level),
                self._viewport_runner.state.awaiting_level_confirm,
            )

        if plan.zoom_hint is not None:
            self._render_mailbox.record_zoom_hint(float(plan.zoom_hint))

        level_applied = False

        if plan.level_change:
            target_level = int(self._viewport_runner.state.request.level)
            prev_level = int(self._current_level_index())
            applied_context = self._apply_level(
                source,
                target_level,
                prev_level=prev_level,
            )

            self._set_current_level_index(int(applied_context.level))
            if self._viewport_state.mode is RenderMode.VOLUME:
                self._apply_volume_level(source, applied_context)
            else:
                self._apply_slice_level(source, applied_context)
            level_applied = True
            self._viewport_runner.mark_level_applied(int(applied_context.level))

        if (
            plan.slice_task is not None
            and self._viewport_state.mode is not RenderMode.VOLUME
            and not level_applied
        ):
            slice_task = plan.slice_task
            level_int = int(slice_task.level)
            step_tuple = tuple(int(v) for v in slice_task.step) if slice_task.step is not None else None
            signature_token = (level_int, step_tuple, slice_task.signature)
            if self._last_slice_signature == signature_token:
                self._viewport_runner.mark_slice_applied(slice_task)
            else:
                apply_slice_roi(
                    self,
                    source,
                    level_int,
                    slice_task.roi,
                    update_contrast=False,
                    step=step_tuple,
                )
                self._viewport_runner.mark_slice_applied(slice_task)

        rect = self._current_panzoom_rect()
        if rect is not None:
            self._viewport_runner.update_camera_rect(rect)
        if plan.pose_event is not None:
            self._emit_current_camera_pose(plan.pose_event.value)


    # ---- C6 selection (napari-anchored) -------------------------------------
    def _evaluate_level_policy(self) -> None:
        """Evaluate multiscale policy inputs and perform a level switch if needed."""
        if not self._zarr_path:
            return
        try:
            source = self._ensure_scene_source()
        except Exception:
            logger.debug("ensure_scene_source failed in selection", exc_info=True)
            return

        current = int(self._current_level_index())
        zoom_hint = self._render_mailbox.consume_zoom_hint(max_age=0.5)
        zoom_ratio = float(zoom_hint.ratio) if zoom_hint is not None else None

        config = lod.LevelPolicyConfig(
            threshold_in=float(self._level_threshold_in),
            threshold_out=float(self._level_threshold_out),
            fine_threshold=float(self._level_fine_threshold),
            hysteresis=float(self._level_hysteresis),
            cooldown_ms=float(self._level_switch_cooldown_ms),
        )

        outcome = lod.evaluate_policy(
            source=source,
            current_level=current,
            oversampling_for_level=self._oversampling_for_level,
            zoom_ratio=zoom_ratio,
            lock_level=self._lock_level,
            last_switch_ts=float(self._last_level_switch_ts),
            config=config,
            log_policy_eval=self._log_policy_eval,
            select_level_fn=select_level,
            logger_ref=logger,
        )

        if logger.isEnabledFor(logging.DEBUG):
            try:
                current_overs = self._oversampling_for_level(source, current)
            except Exception:
                current_overs = float("nan")
            logger.debug(
                "policy.evaluate: level=%d zoom_hint=%s overs_current=%.3f",
                current,
                f"{zoom_ratio:.3f}" if zoom_ratio is not None else None,
                current_overs,
            )

        if outcome is None:
            return

        def _budget_check(scene: ZarrSceneSource, level: int) -> None:
            try:
                if self._viewport_state.mode is RenderMode.VOLUME:
                    self._volume_budget_allows(scene, level)
                else:
                    self._slice_budget_allows(scene, level)
            except _LevelBudgetError as exc:
                raise LevelBudgetError(str(exc)) from exc

        decision = lod.enforce_budgets(
            outcome,
            source=source,
            use_volume=self._viewport_state.mode is RenderMode.VOLUME,
            budget_check=_budget_check,
            log_layer_debug=self._log_layer_debug,
        )

        self._last_level_switch_ts = float(decision.timestamp)

        ledger = self._ledger
        step_hint = ledger_step(ledger)
        context = lod.build_level_context(
            decision,
            source=source,
            prev_level=current,
            last_step=step_hint,
        )

        requested = self._viewport_runner.request_level(int(context.level))
        if not requested:
            logger.debug("level intent suppressed (no change or already pending)")
            return

        if self._viewport_state.mode is RenderMode.VOLUME:
            apply_volume_metadata(self, source, context)
        else:
            apply_plane_metadata(self, source, context)

        intent = LevelSwitchIntent(
            desired_level=int(decision.desired_level),
            selected_level=int(context.level),
            reason=decision.reason,
            previous_level=current,
            context=context,
            oversampling=decision.oversampling,
            timestamp=decision.timestamp,
            downgraded=decision.downgraded,
            zoom_ratio=zoom_ratio,
            lock_level=self._lock_level,
            mode=self.viewport_state.mode,
            plane_state=deepcopy(self.viewport_state.plane),
            volume_state=deepcopy(self.viewport_state.volume),
        )

        logger.info(
            "intent.level_switch: prev=%d target=%d reason=%s downgraded=%s",
            int(current),
            int(context.level),
            decision.reason,
            intent.downgraded,
        )

        callback = self._level_intent_callback
        if callback is None:
            logger.debug("level intent callback missing; skipping emission")
            return

        callback(intent)
        self._mark_render_tick_needed()

    # (packer is now provided by bitstream.py)

    # Removed legacy _torch_from_cupy helper (unused)

    def cleanup(self) -> None:
        self._resources.cleanup()
        self.cuda_ctx = None

        try:
            if self.canvas is not None:
                self.canvas.close()
        except Exception:
            logger.debug("Cleanup: canvas close failed", exc_info=True)
        self.canvas = None
        self._viewer = None
        self._napari_layer = None
        if self._plane_visual_handle is not None:
            self._plane_visual_handle.detach()
        if self._volume_visual_handle is not None:
            self._volume_visual_handle.detach()
        self._plane_visual_handle = None
        self._volume_visual_handle = None

        self._is_ready = False

    def viewer_model(self) -> Optional[ViewerModel]:
        """Expose the napari ``ViewerModel`` when adapter mode is active."""
        return self._viewer

    # Adapter is always used; legacy path removed
    def attach_ledger(self, ledger: ServerStateLedger) -> None:
        self._ledger = ledger
