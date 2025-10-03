"""Render-thread orchestration for the server.

`EGLRendererWorker` lives in this module and is still EGL-backed internally, but
the neutral module name reflects its broader responsibilities:

* bootstrap the napari Viewer/VisPy canvas and keep it on the worker thread,
* drain `RenderMailbox` updates and apply them through `SceneStateApplier`,
* drive the render loop, including camera animation and policy evaluation,
* capture frames via `CaptureFacade`, hand them to the encoder, and surface
  timing metadata for downstream metrics.

The worker owns the EGL + CUDA context pair, renders into an FBO-backed texture,
maps the texture into CUDA memory, stages the pixels for NVENC, and hands the
encoded packets back to the asyncio side through callbacks.
"""

from __future__ import annotations

import os
import time
import logging

from typing import Any, Callable, Deque, Dict, Optional, Mapping, Sequence, Tuple, List
import threading
import math

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

from napari.components.viewer_model import ViewerModel
from napari._vispy.layers.image import VispyImageLayer, _napari_cmap_to_vispy

import pycuda.driver as cuda  # type: ignore
# Bitstream packing happens in the server layer
from .patterns import make_rgba_image
from .debug_tools import DebugConfig, DebugDumper
from .hw_limits import get_hw_limits
from .zarr_source import ZarrSceneSource
from .scene_types import SliceROI
from napari_cuda.server.rendering.viewer_builder import ViewerBuilder
from napari_cuda.server.camera_animator import animate_if_enabled
from . import policy as level_policy
import napari_cuda.server.lod as lod
from napari_cuda.server.lod import apply_level
from napari_cuda.server.level_budget import LevelBudgetError
from napari_cuda.server.config import LevelPolicySettings, ServerCtx
from napari_cuda.server.rendering.egl_context import EglContext
from napari_cuda.server.rendering.encoder import Encoder
from napari_cuda.server.capture import CaptureFacade, FrameTimings, encode_frame
from napari_cuda.server.render_loop import run_render_tick
from napari_cuda.server.roi_applier import (
    SliceUpdatePlanner,
    refresh_slice,
)
from napari_cuda.server.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
)
from napari_cuda.server.scene_state_applier import (
    SceneStateApplier,
    SceneStateApplyContext,
    SceneDrainResult,
)
from napari_cuda.server.camera_controller import process_commands
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.server_scene import ServerSceneCommand
from napari_cuda.server.plane_restore_state import PlaneRestoreState
from napari_cuda.server.render_mailbox import (
    PendingRenderUpdate,
    RenderDelta,
    RenderMailbox,
)
from napari_cuda.server.policy_metrics import PolicyMetrics
from napari_cuda.server.level_logging import LayerAssignmentLogger, LevelSwitchLogger
from napari_cuda.server.worker_runtime import (
    apply_worker_level,
    apply_worker_volume_level,
    apply_worker_slice_level,
    format_worker_level_roi,
    viewport_roi_for_level,
    set_level_with_budget,
    perform_level_switch,
    ensure_scene_source,
    notify_scene_refresh,
    refresh_worker_slice_if_needed,
    reset_worker_camera,
)
from napari_cuda.server.display_mode import apply_ndisplay_switch

logger = logging.getLogger(__name__)

# Back-compat for tests patching the selector directly
select_level = lod.select_level

## Camera ops now live in napari_cuda.server.camera_ops as free functions.

class _LevelBudgetError(RuntimeError):
    """Raised when a multiscale level exceeds memory/voxel budgets."""
    pass


class EGLRendererWorker:
    """Headless VisPy renderer using EGL with CUDA interop and NVENC."""

    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False, fps: int = 60,
                 volume_depth: int = 64, volume_dtype: str = "float32", volume_relative_step: Optional[float] = None,
                 animate: bool = False, animate_dps: float = 30.0,
                 zarr_path: Optional[str] = None, zarr_level: Optional[str] = None,
                 zarr_axes: Optional[str] = None, zarr_z: Optional[int] = None,
                 scene_refresh_cb: Optional[Callable[[], None]] = None,
                 policy_name: Optional[str] = None,
                 *,
                 ctx: ServerCtx,
                 env: Optional[Mapping[str, str]] = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.use_volume = bool(use_volume)
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

        self._configure_animation(animate, animate_dps)
        self._init_render_components()
        self._init_scene_state(scene_refresh_cb)
        self._init_locks()
        self._configure_debug_flags()
        self._configure_policy(self._ctx.policy)
        self._configure_roi_settings()
        self._configure_budget_limits()

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

        self._bootstrapped = False
        self._bootstrapping = False
        self._debug: Optional[DebugDumper] = None
        self._debug_config = DebugConfig.from_policy(self._debug_policy.dumps)

    @property
    def is_bootstrapped(self) -> bool:
        return self._bootstrapped

    @property
    def is_bootstrapping(self) -> bool:
        return self._bootstrapping

    def set_scene_refresh_callback(self, callback: Optional[Callable[[object], None]]) -> None:
        self._scene_refresh_cb = callback

    def bootstrap(self) -> None:
        """Complete GPU/VisPy setup once the lifecycle registers the worker."""

        if self._bootstrapped:
            return
        if self._bootstrapping:
            return

        self._bootstrapping = True
        try:
            # Initialize CUDA first (independent of GL)
            self._init_cuda()
            # Create VisPy SceneCanvas (EGL backend) which makes a GL context current
            self._init_vispy_scene()
            # Adopt the current EGL context created by VisPy for capture
            self._init_egl()
            self._init_capture()
            self._init_cuda_interop()
            self._init_encoder()
        except Exception as exc:
            logger.warning("Bootstrap failed; attempting cleanup: %s", exc, exc_info=True)
            try:
                self.cleanup()
            except Exception as cleanup_exc:
                logger.debug("Cleanup after failed bootstrap also failed: %s", cleanup_exc)
            raise
        finally:
            self._bootstrapping = False

        self._debug = DebugDumper(self._debug_config)
        if self._debug.cfg.enabled:
            self._debug.log_env_once()
            self._debug.ensure_out_dir()
        self._capture.pipeline.set_debug(self._debug)
        self._capture.pipeline.set_raw_dump_budget(self._raw_dump_budget)
        self._capture.cuda.set_force_tight_pitch(self._debug_policy.worker.force_tight_pitch)
        self._log_debug_policy_once()
        self._bootstrapped = True

        logger.info(
            "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s, zarr=%s",
            self.width,
            self.height,
            self._capture.pipeline.enc_input_format,
            self.fps,
            self._animate,
            bool(self._zarr_path),
        )

    def _frame_volume_camera(self, w: float, h: float, d: float) -> None:
        """Choose stable initial center and distance for TurntableCamera.

        ``w``, ``h``, ``d`` are world-space extents (after scale). We center the
        camera on the volume midpoint and set distance so the full height fits.
        """
        cam = self.view.camera
        if not isinstance(cam, scene.cameras.TurntableCamera):
            return
        center = (float(w) * 0.5, float(h) * 0.5, float(d) * 0.5)
        cam.center = center  # type: ignore[attr-defined]
        fov_deg = float(getattr(cam, 'fov', 60.0) or 60.0)
        fov_rad = math.radians(max(1e-3, min(179.0, fov_deg)))
        dist = (0.5 * float(h)) / max(1e-6, math.tan(0.5 * fov_rad))
        cam.distance = float(dist * 1.1)  # type: ignore[attr-defined]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "frame_volume_camera extent=(%.3f, %.3f, %.3f) center=%s dist=%.3f",
                w,
                h,
                d,
                center,
                float(cam.distance),
            )

    def _init_viewer_scene(self, source: Optional[ZarrSceneSource]) -> None:
        builder = ViewerBuilder(self)
        canvas, view, viewer = builder.build(source)
        # Mirror attributes for call sites expecting them
        self.canvas = canvas
        self.view = view
        self._viewer = viewer
        # Ensure the camera frames current data extents on first init
        assert self.view is not None, "adapter must supply a VisPy view"
        cam = self.view.camera
        if cam is not None:
            self._apply_camera_reset(cam)

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
        self._egl.ensure()

    def _init_cuda(self) -> None:
        cuda.init()
        dev = cuda.Device(0)
        ctx = dev.retain_primary_context()
        ctx.push()
        self.cuda_ctx = ctx

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
                self._active_ms_level = source.current_level
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
        """Queue a multiscale level switch; applied on render thread next frame."""
        if getattr(self, '_lock_level', None) is not None:
            if self._log_layer_debug and logger.isEnabledFor(logging.INFO):
                logger.info("request_multiscale_level ignored due to lock_level=%s", str(self._lock_level))
            return
        if self._log_layer_debug and logger.isEnabledFor(logging.INFO):
            logger.info(
                "request multiscale level: level=%d path=%s",
                int(level),
                str(path) if path else "<default>",
            )
        self._render_mailbox.enqueue_multiscale(int(level), str(path) if path else None)
        self._last_interaction_ts = time.perf_counter()
        self._mark_render_tick_needed()

    def _notify_scene_refresh(self) -> None:
        notify_scene_refresh(self)

    # --- Plane restore helpers -------------------------------------------------

    def snapshot_plane_state(self) -> Optional[PlaneRestoreState]:
        """Capture the current plane state for later restoration."""

        if self._last_step is None:
            return None

        step_tuple = tuple(int(value) for value in self._last_step)
        level = int(self._active_ms_level)

        roi_level: Optional[int] = None
        roi_value: Optional[SliceROI] = None
        if self._last_roi is not None:
            roi_level = int(self._last_roi[0])
            roi_value = self._last_roi[1]

        camera_state = None
        if self.view is not None and self.view.camera is not None:
            camera_state = dict(self.view.camera.get_state())

        data_wh = None
        if self._data_wh is not None:
            data_wh = (int(self._data_wh[0]), int(self._data_wh[1]))

        state = PlaneRestoreState(
            step=step_tuple,
            level=level,
            roi_level=roi_level,
            roi=roi_value,
            camera_state=camera_state,
            data_wh=data_wh,
        )
        self._plane_restore_state = state
        return state

    def schedule_plane_restore(self, state: PlaneRestoreState) -> None:
        """Queue a previously captured plane state for restoration."""

        self._plane_restore_state = state
        self._pending_plane_restore = state
        self._level_policy_refresh_needed = True
        self._mark_render_tick_needed()

    def _apply_pending_plane_restore(self) -> None:
        if self._pending_plane_restore is None or self.use_volume:
            return

        state = self._pending_plane_restore
        self._pending_plane_restore = None
        self._plane_restore_state = state

        source = self._ensure_scene_source()
        with self._state_lock:
            source.set_current_slice(tuple(int(v) for v in state.step), int(state.level))

        self._active_ms_level = int(state.level)
        self._last_step = tuple(int(v) for v in state.step)

        if state.roi_level is not None and state.roi is not None:
            self._last_roi = (int(state.roi_level), state.roi)

        if state.data_wh is not None:
            self._data_wh = (int(state.data_wh[0]), int(state.data_wh[1]))

        if self._viewer is not None:
            self._viewer.dims.current_step = tuple(int(v) for v in state.step)

        if state.camera_state is not None and self.view is not None and self.view.camera is not None:
            self.view.camera.set_state(state.camera_state)  # type: ignore[arg-type]

        # Notify clients with the restored step metadata.
        self._notify_scene_refresh()

    def _mark_render_tick_needed(self) -> None:
        self._render_tick_required = True

    def _mark_render_tick_complete(self) -> None:
        self._render_tick_required = False

    def _mark_render_loop_started(self) -> None:
        self._render_loop_started = True

    def _evaluate_policy_if_needed(self) -> None:
        if not self._level_policy_refresh_needed or self.use_volume:
            return
        self._level_policy_refresh_needed = False
        self._evaluate_level_policy()

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
        self.cuda_ctx: Optional[cuda.Context] = None
        self.canvas: Optional[scene.SceneCanvas] = None
        self.view = None
        self._visual = None
        self._egl = EglContext(self.width, self.height)
        self._capture = CaptureFacade(width=self.width, height=self.height)
        self._encoder: Optional[Encoder] = None

    def _init_scene_state(self, scene_refresh_cb: Optional[Callable[[], None]]) -> None:
        self._viewer: Optional[ViewerModel] = None
        self._napari_layer = None
        self._scene_source: Optional[ZarrSceneSource] = None
        self._active_ms_level = 0
        self._level_downgraded = False
        self._scene_refresh_cb = scene_refresh_cb
        self._zoom_accumulator = 0.0
        self._last_zoom_hint_ts = 0.0
        self._zoom_hint_hold_s = 0.35
        self._data_wh = (int(self.width), int(self.height))
        self._data_d = None
        self._volume_scale = (1.0, 1.0, 1.0)
        self._orientation_ready = False
        self._policy_metrics = PolicyMetrics()
        self._layer_logger = LayerAssignmentLogger(logger)
        self._switch_logger = LevelSwitchLogger(logger)
        self._plane_restore_state: Optional[PlaneRestoreState] = None
        self._pending_plane_restore: Optional[PlaneRestoreState] = None

    def _init_locks(self) -> None:
        self._enc_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._render_mailbox = RenderMailbox()
        self._last_ensure_log: Optional[tuple[int, Optional[str]]] = None
        self._last_ensure_log_ts = 0.0
        self._ensure_log_interval_s = 1.0
        self._render_tick_required = False
        self._render_loop_started = False
        self._level_policy_refresh_needed = False

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
        self._policy_metrics.reset()
        policy_logging = self._debug_policy.logging
        worker_dbg = self._debug_policy.worker
        self._log_layer_debug = bool(policy_logging.log_layer_debug)
        self._log_roi_anchor = bool(policy_logging.log_roi_anchor)
        self._log_policy_eval = bool(policy_cfg.log_policy_eval)
        self._lock_level = worker_dbg.lock_level
        self._level_threshold_in = float(policy_cfg.threshold_in)
        self._level_threshold_out = float(policy_cfg.threshold_out)
        self._level_hysteresis = float(policy_cfg.hysteresis)
        self._level_fine_threshold = float(policy_cfg.fine_threshold)
        self._preserve_view_on_switch = bool(policy_cfg.preserve_view_on_switch)
        self._sticky_contrast = bool(policy_cfg.sticky_contrast)
        self._last_step: Optional[tuple[int, ...]] = None
        self._level_switch_cooldown_ms = float(policy_cfg.cooldown_ms)
        self._last_level_switch_ts = 0.0
        self._oversampling_thresholds = (
            {int(k): float(v) for k, v in policy_cfg.oversampling_thresholds.items()}
            if getattr(policy_cfg, "oversampling_thresholds", None)
            else None
        )
        self._oversampling_hysteresis = float(getattr(policy_cfg, "oversampling_hysteresis", 0.1))

    def _configure_roi_settings(self) -> None:
        self._roi_cache: Dict[int, tuple[Optional[tuple[float, ...]], SliceROI]] = {}
        self._roi_log_state: Dict[int, tuple[SliceROI, float]] = {}
        self._roi_log_interval = 0.25
        worker_dbg = self._debug_policy.worker
        self._roi_edge_threshold = int(worker_dbg.roi_edge_threshold)
        self._roi_align_chunks = bool(worker_dbg.roi_align_chunks)
        self._roi_ensure_contains_viewport = bool(worker_dbg.roi_ensure_contains_viewport)
        self._roi_pad_chunks = 1
        self._idr_on_z = False
        self._last_roi: Optional[tuple[int, SliceROI]] = None

    def _configure_budget_limits(self) -> None:
        cfg = self._ctx.cfg
        self._slice_max_bytes = int(max(0, getattr(cfg, 'max_slice_bytes', 0)))
        self._volume_max_bytes = int(max(0, getattr(cfg, 'max_volume_bytes', 0)))
        self._volume_max_voxels = int(max(0, getattr(cfg, 'max_volume_voxels', 0)))

    def snapshot_dims_metadata(self) -> Dict[str, Any]:
        viewer = self._viewer
        if viewer is None:
            return {}
        dims = getattr(viewer, "dims", None)
        if dims is None:
            return {}

        meta: Dict[str, Any] = {}

        source = self._scene_source
        if source is not None and source.axes:
            axes_sequence = tuple(str(axis) for axis in source.axes)
        elif self._zarr_axes:
            axes_sequence = tuple(self._zarr_axes)
        else:
            axes_sequence = ()
        if axes_sequence:
            meta["axes"] = list(axes_sequence)

        if self._zarr_shape:
            level_shape = tuple(int(size) for size in self._zarr_shape)
        elif source is not None and source.level_descriptors:
            idx = int(self._active_ms_level)
            descriptors = source.level_descriptors
            if idx < 0 or idx >= len(descriptors):
                idx = 0
            level_shape = tuple(int(size) for size in descriptors[idx].shape)
        else:
            level_shape = ()

        # napari's ViewerModel may report transient steps during bootstrap/level flips;
        # rely on the worker's tracked step instead so metadata reflects what we applied.
        last_step = self._last_step
        raw_step = tuple(int(value) for value in last_step) if last_step else ()

        ndim_candidates: list[int] = []
        if dims.ndim:
            ndim_candidates.append(int(dims.ndim))
        if axes_sequence:
            ndim_candidates.append(len(axes_sequence))
        if level_shape:
            ndim_candidates.append(len(level_shape))
        if raw_step:
            ndim_candidates.append(len(raw_step))
        dims_nsteps = getattr(dims, "nsteps", None)
        if isinstance(dims_nsteps, (list, tuple)) and dims_nsteps:
            ndim_candidates.append(len(dims_nsteps))

        ndim = max(ndim_candidates) if ndim_candidates else 1
        meta["ndim"] = ndim

        ndisplay = int(dims.ndisplay) if dims.ndisplay else 2
        meta["ndisplay"] = ndisplay
        meta["mode"] = "volume" if ndisplay >= 3 else "plane"

        order_list = [int(value) for value in dims.order]
        if len(order_list) != ndim:
            order_list = list(range(ndim))
        meta["order"] = order_list

        axis_labels = [str(text) for text in dims.axis_labels if str(text)]
        if len(axis_labels) < ndim:
            for idx in range(len(axis_labels), ndim):
                axis_labels.append(str(axes_sequence[idx]) if idx < len(axes_sequence) else f"axis-{idx}")
        else:
            axis_labels = axis_labels[:ndim]
        meta["axis_labels"] = axis_labels

        if dims.displayed:
            meta["displayed"] = [int(value) for value in dims.displayed]

        if level_shape:
            sizes = [level_shape[idx] if idx < len(level_shape) else 1 for idx in range(ndim)]
        else:
            sizes = [max(1, int(value)) for value in dims.nsteps[:ndim]] if dims.nsteps else [1] * ndim
        meta["sizes"] = sizes

        meta["range"] = [[0, max(0, size - 1)] for size in sizes]

        return meta

    def _log_debug_policy_once(self) -> None:
        if self._debug_policy_logged:
            return
        if logger.isEnabledFor(logging.INFO):
            logger.info("Debug policy resolved: %s", self._debug_policy)
        self._debug_policy_logged = True

    # ---- Level helpers ------------------------------------------------------

    def _update_level_metadata(self, descriptor, applied) -> None:
        self._active_ms_level = applied.level
        self._last_step = applied.step
        if not self.use_volume:
            self.snapshot_plane_state()
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
        self._policy_metrics.reset()
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

    def policy_metrics_snapshot(self) -> Dict[str, object]:
        return self._policy_metrics.snapshot(
            policy=self._policy_name,
            active_level=int(self._active_ms_level),
            downgraded=bool(self._level_downgraded),
        )

    def _load_slice(
        self,
        source: ZarrSceneSource,
        level: int,
        z_idx: int,
    ) -> np.ndarray:
        roi = viewport_roi_for_level(self, source, level)
        # Remember ROI used for this level so we can place the slab in world coords
        try:
            self._last_roi = (int(level), roi)
        except Exception:
            self._last_roi = None
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
        overs_map: Dict[int, float] = {}
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
            current_level=int(self._active_ms_level),
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
    ) -> lod.AppliedLevel:
        return apply_worker_level(self, source, level, prev_level=prev_level)

    def _apply_volume_level(
        self,
        source: ZarrSceneSource,
        applied: lod.AppliedLevel,
    ) -> None:
        apply_worker_volume_level(self, source, applied)

    def _apply_slice_level(
        self,
        source: ZarrSceneSource,
        applied: lod.AppliedLevel,
    ) -> None:
        apply_worker_slice_level(self, source, applied)

    def _format_level_roi(self, source: ZarrSceneSource, level: int) -> str:
        return format_worker_level_roi(self, source, level)

    def _apply_multiscale_switch(self, level: int, path: Optional[str]) -> None:
        if not self._zarr_path:
            return
        source = self._ensure_scene_source()
        target = level
        if path:
            target = source.level_index_for_path(path)
        perform_level_switch(
            self,
            target_level=int(target),
            reason="intent",
            requested_level=int(target),
            selected_level=int(target),
            source=source,
            budget_error=self._budget_error_cls,
        )

    def _clear_visual(self) -> None:
        if self._visual is not None:
            self._visual.parent = None  # type: ignore[attr-defined]
        self._visual = None

    def _resolve_visual(self) -> Any:
        if self._visual is not None:
            return self._visual

        view = self.view
        assert view is not None, "VisPy view must exist before resolving the active visual"

        children = getattr(view.scene, "children", None) or ()
        for child in children:
            if not hasattr(child, "set_data"):
                continue
            if not (hasattr(child, "cmap") or hasattr(child, "clim")):
                continue
            self._visual = child
            return child

        assert False, "EGLRendererWorker could not locate an active VisPy visual"

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
        try:
            sz, sy, sx = self._volume_scale
        except Exception:
            sz = sy = sx = 1.0
        width = float(x) * float(sx)
        height = float(y) * float(sy)
        depth = float(z) * float(sz)
        return (width, height, depth)

    def _init_capture(self) -> None:
        self._capture.ensure()

    def _init_cuda_interop(self) -> None:
        """Init CUDA-GL interop and allocate destination via one-time DLPack bridge."""
        self._capture.initialize_cuda_interop()

    def _init_encoder(self) -> None:
        with self._enc_lock:
            if self._encoder is None:
                self._encoder = Encoder(
                    self.width,
                    self.height,
                    fps_hint=int(self.fps),
                )
            encoder = self._encoder
            encoder.set_fps_hint(int(self.fps))
            encoder.setup(self._ctx)
            self._capture.pipeline.set_enc_input_format(encoder.input_format)

    def reset_encoder(self) -> None:
        with self._enc_lock:
            encoder = self._encoder
            if encoder is None:
                return
            encoder.set_fps_hint(int(self.fps))
            encoder.reset(self._ctx)
            self._enc_input_fmt = encoder.input_format

    def force_idr(self) -> None:
        logger.debug("Requesting encoder force IDR")
        with self._enc_lock:
            encoder = self._encoder
            if encoder is None:
                return
            encoder.force_idr()

    def _request_encoder_idr(self) -> None:
        with self._enc_lock:
            encoder = self._encoder
            if encoder is None:
                return
            encoder.request_idr()

    def render_frame(self, azimuth_deg: Optional[float] = None) -> None:
        if azimuth_deg is not None:
            assert self.view is not None and self.view.camera is not None
            self.view.camera.azimuth = float(azimuth_deg)
        self.drain_scene_updates()
        t0 = time.perf_counter()
        self.canvas.render()
        self._render_tick_required = False
        self._render_loop_started = True

    def _normalize_scene_state(self, state: ServerSceneState) -> ServerSceneState:
        """Return the render-thread-friendly copy of *state*."""

        layer_updates = None
        if state.layer_updates:
            updates: Dict[str, Dict[str, object]] = {}
            for layer_id, props in state.layer_updates.items():
                if not props:
                    continue
                updates[str(layer_id)] = {str(key): value for key, value in props.items()}
            if updates:
                layer_updates = updates

        return ServerSceneState(
            center=(tuple(float(c) for c in state.center) if state.center is not None else None),
            zoom=(float(state.zoom) if state.zoom is not None else None),
            angles=(tuple(float(a) for a in state.angles) if state.angles is not None else None),
            current_step=(
                tuple(int(s) for s in state.current_step)
                if state.current_step is not None
                else None
            ),
            volume_mode=(str(state.volume_mode) if state.volume_mode is not None else None),
            volume_colormap=(
                str(state.volume_colormap) if state.volume_colormap is not None else None
            ),
            volume_clim=(
                tuple(float(v) for v in state.volume_clim)
                if state.volume_clim is not None
                else None
            ),
            volume_opacity=(
                float(state.volume_opacity) if state.volume_opacity is not None else None
            ),
            volume_sample_step=(
                float(state.volume_sample_step)
                if state.volume_sample_step is not None
                else None
            ),
            layer_updates=layer_updates,
        )

    def enqueue_update(self, delta: RenderDelta) -> None:
        """Normalize and enqueue a render delta for the worker mailbox."""

        scene_state = None
        if delta.scene_state is not None:
            self._last_interaction_ts = time.perf_counter()
            scene_state = self._normalize_scene_state(delta.scene_state)

        normalized = RenderDelta(
            display_mode=delta.display_mode,
            multiscale=delta.multiscale,
            scene_state=scene_state,
        )

        if (
            normalized.display_mode is None
            and normalized.multiscale is None
            and normalized.scene_state is None
        ):
            return

        self._render_mailbox.enqueue(normalized)
        self._mark_render_tick_needed()

    def apply_state(self, state: ServerSceneState) -> None:
        """Queue a complete scene state snapshot for the next frame."""

        self.enqueue_update(RenderDelta(scene_state=state))

    def process_camera_commands(self, commands: Sequence[ServerSceneCommand]) -> None:
        process_commands(self, commands)

    def _apply_camera_reset(self, cam) -> None:
        reset_worker_camera(self, cam)

    def _build_scene_state_context(self, cam) -> SceneStateApplyContext:
        return SceneStateApplyContext(
            use_volume=bool(self.use_volume),
            viewer=self._viewer,
            camera=cam,
            visual=self._resolve_visual(),
            layer=self._napari_layer,
            scene_source=self._scene_source,
            active_ms_level=int(self._active_ms_level),
            z_index=self._z_index,
            last_roi=self._last_roi,
            preserve_view_on_switch=self._preserve_view_on_switch,
            sticky_contrast=self._sticky_contrast,
            idr_on_z=self._idr_on_z,
            data_wh=self._data_wh,
            volume_scale=getattr(self, '_volume_scale', None),
            state_lock=self._state_lock,
            ensure_scene_source=self._ensure_scene_source,
            plane_scale_for_level=plane_scale_for_level,
            load_slice=self._load_slice,
            mark_render_tick_needed=self._mark_render_tick_needed,
            request_encoder_idr=self._request_encoder_idr,
        )

    def drain_scene_updates(self) -> None:
        updates: PendingRenderUpdate = self._render_mailbox.drain()

        previous_volume = self.use_volume

        if updates.display_mode is not None:
            apply_ndisplay_switch(self, updates.display_mode)
            if not previous_volume and self.use_volume:
                # entering volume: render loop is the only place that can
                # broadcast updated dims metadata.
                self._notify_scene_refresh()

        if not self.use_volume:
            self._apply_pending_plane_restore()

        if updates.multiscale is not None:
            lvl = int(updates.multiscale.level)
            pth = updates.multiscale.path
            self._apply_multiscale_switch(lvl, pth)

        state = updates.scene_state
        if state is None:
            return

        view = self.view
        assert view is not None, "drain_scene_updates requires an active VisPy view"
        cam = view.camera

        ctx = self._build_scene_state_context(cam)
        drain_res: SceneDrainResult = SceneStateApplier.drain_updates(
            ctx,
            state=state,
            mailbox=self._render_mailbox,
        )

        if drain_res.z_index is not None:
            self._z_index = int(drain_res.z_index)
        if drain_res.data_wh is not None:
            self._data_wh = (int(drain_res.data_wh[0]), int(drain_res.data_wh[1]))
        if drain_res.last_step is not None:
            self._last_step = tuple(int(x) for x in drain_res.last_step)
            if not self.use_volume:
                self.snapshot_plane_state()
                self._notify_scene_refresh()

        if drain_res.policy_refresh_needed and not self.use_volume:
            self._evaluate_level_policy()

    # --- Public coalesced toggles ------------------------------------------------
    def request_ndisplay(self, ndisplay: int) -> None:
        """Queue a 2D/3D view switch to apply on the render thread."""
        self._render_mailbox.enqueue_display_mode(3 if int(ndisplay) >= 3 else 2)

    def _capture_blit_gpu_ns(self) -> Optional[int]:
        return self._capture.pipeline.capture_blit_gpu_ns()


    def capture_and_encode_packet(self) -> tuple[FrameTimings, Optional[bytes], int, int]:
        """Same as capture_and_encode, but also returns the packet and flags.

        Flags bit 0x01 indicates keyframe (IDR/CRA).
        """
        encoded = encode_frame(
            capture=self._capture,
            render_frame=self.render_tick,
            obtain_encoder=lambda: self._encoder,
            encoder_lock=self._enc_lock,
            debug_dumper=self._debug,
        )

        self._orientation_ready = encoded.orientation_ready

        return encoded.timings, encoded.packet, encoded.flags, encoded.sequence

    # ---- C5 helpers (pure refactor; no behavior change) ---------------------
    def _refresh_slice_if_needed(self) -> None:
        refresh_worker_slice_if_needed(self)

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
            drain_scene_updates=self.drain_scene_updates,
            refresh_slice=self._refresh_slice_if_needed,
            render_canvas=canvas.render,
            evaluate_policy_if_needed=self._evaluate_policy_if_needed,
            mark_tick_complete=self._mark_render_tick_complete,
            mark_loop_started=self._mark_render_loop_started,
        )

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

        current = int(self._active_ms_level)
        zoom_hint = self._render_mailbox.consume_zoom_hint(max_age=0.5)
        zoom_ratio = float(zoom_hint.ratio) if zoom_hint is not None else None

        config = lod.LevelPolicyConfig(
            threshold_in=float(self._level_threshold_in),
            threshold_out=float(self._level_threshold_out),
            fine_threshold=float(self._level_fine_threshold),
            hysteresis=float(self._level_hysteresis),
            cooldown_ms=float(self._level_switch_cooldown_ms),
        )

        try:
            outcome = lod.run_policy_switch(
                source=source,
                current_level=current,
                oversampling_for_level=self._oversampling_for_level,
                zoom_ratio=zoom_ratio,
                lock_level=self._lock_level,
                last_switch_ts=float(self._last_level_switch_ts),
                config=config,
                log_policy_eval=self._log_policy_eval,
                apply_level=lambda level, reason: set_level_with_budget(
                    self,
                    level,
                    reason=reason,
                    budget_error=self._budget_error_cls,
                ),
                budget_error=self._budget_error_cls,
                select_level_fn=select_level,
                logger_ref=logger,
            )
        except Exception as exc:
            action = getattr(getattr(exc, "policy_decision", None), "action", "unknown")
            logger.exception("ms.switch: level apply failed (reason=%s)", action)
            raise

        if outcome is None:
            return

        prev = current
        self._last_level_switch_ts = float(outcome.timestamp)
        if outcome.decision.action in {"zoom-in", "zoom-out"}:
            self._last_zoom_hint_ts = float(outcome.timestamp)

        logger.info(
            "ms.switch: %d -> %d (reason=%s) overs=%s",
            int(prev),
            int(self._active_ms_level),
            outcome.decision.action,
            lod.format_oversampling(outcome.oversampling),
        )

    # (packer is now provided by bitstream.py)

    # Removed legacy _torch_from_cupy helper (unused)

    def cleanup(self) -> None:
        try:
            self._capture.cleanup()
        except Exception:
            logger.debug("Cleanup: capture cleanup failed", exc_info=True)

        with self._enc_lock:
            encoder = self._encoder
            if encoder is not None:
                encoder.shutdown()

        try:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
        except Exception:
            logger.debug("Cleanup: CUDA context pop/detach failed", exc_info=True)
        self.cuda_ctx = None

        try:
            if self.canvas is not None:
                self.canvas.close()
        except Exception:
            logger.debug("Cleanup: canvas close failed", exc_info=True)
        self.canvas = None
        self._viewer = None
        self._napari_layer = None

        self._egl.cleanup()

    def viewer_model(self) -> Optional[ViewerModel]:
        """Expose the napari ``ViewerModel`` when adapter mode is active."""
        return self._viewer

    # Adapter is always used; legacy path removed
