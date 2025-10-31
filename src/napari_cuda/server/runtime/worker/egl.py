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
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Optional

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")


import napari_cuda.server.data.lod as lod
from napari.components.viewer_model import ViewerModel
from napari_cuda.server.app.config import ServerCtx
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.data.hw_limits import get_hw_limits
from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.rendering.capture import FrameTimings
from napari_cuda.server.rendering.debug_tools import DebugConfig, DebugDumper
from napari_cuda.server.runtime.bootstrap import (
    setup_camera as viewer_camera_ops,
    setup_viewer as viewer_setup,
    setup_visuals as viewer_visuals,
)
from napari_cuda.server.runtime.camera import (
    CameraCommandQueue,
    CameraPoseApplied,
)
from napari_cuda.server.runtime.bootstrap.runtime_driver import (
    cleanup_render_worker,
    init_egl as core_init_egl,
    init_vispy_scene as core_init_vispy_scene,
    setup_worker_runtime,
)
from napari_cuda.server.runtime.bootstrap.scene_setup import ensure_scene_source
from napari_cuda.server.runtime.render_loop.apply.snapshots.build import (
    RenderLedgerSnapshot,
)
from napari_cuda.server.runtime.render_loop.plan.ledger_access import (
    axis_labels as ledger_axis_labels,
    displayed as ledger_displayed,
    level as ledger_level,
    level_shapes as ledger_level_shapes,
    ndisplay as ledger_ndisplay,
    order as ledger_order,
    step as ledger_step,
)
from napari_cuda.server.runtime.ipc import LevelSwitchIntent
from napari_cuda.server.runtime.ipc.mailboxes import (
    RenderUpdate,
)
from napari_cuda.server.runtime.lod import level_policy
from napari_cuda.server.runtime.lod.context import build_level_context
from napari_cuda.server.runtime.render_loop import (
    render_updates as _render_updates,
)
from napari_cuda.server.runtime.render_loop.ticks import (
    capture as capture_tick,
)
from napari_cuda.server.runtime.render_loop.apply_interface import RenderApplyInterface
from napari_cuda.server.runtime.render_loop.apply.snapshots.viewer_metadata import (
    apply_plane_metadata,
    apply_volume_metadata,
)
from napari_cuda.server.runtime.viewport import (
    RenderMode,
    ViewportState,
)
from napari_cuda.server.runtime.worker.resources import WorkerResources

logger = logging.getLogger(__name__)


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
        self._budget_error_cls = level_policy.BudgetGuardError
        self.fps = int(fps)
        self.volume_depth = int(volume_depth)
        self.volume_dtype = str(volume_dtype)
        self.volume_relative_step = volume_relative_step
        self._ctx: ServerCtx = ctx
        self._debug_policy = ctx.debug_policy
        self._raw_dump_budget = max(0, int(self._debug_policy.dumps.raw_budget))
        self._debug_policy_logged = False
        self._env: Optional[Mapping[str, str]] = dict(env) if env is not None else None

        setup_worker_runtime(
            self,
            use_volume=use_volume,
            animate=animate,
            animate_dps=animate_dps,
            camera_queue=camera_queue,
            camera_pose_cb=camera_pose_cb,
            level_intent_cb=level_intent_cb,
            policy_name=policy_name,
        )

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

    @property
    def resources(self) -> WorkerResources:
        """Expose render resources owned by the worker."""

        return self._resources

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
        core_init_egl(self)

    def _init_cuda(self) -> None:
        # CUDA context is established during bootstrap; nothing additional required.
        return

    def _ensure_scene_source(self) -> ZarrSceneSource:
        return ensure_scene_source(self)

    def _init_vispy_scene(self) -> None:
        core_init_vispy_scene(self)

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
            target, downgraded = level_policy.resolve_volume_intent_level(self, source, target)

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
        context = build_level_context(
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


    def _init_viewer_scene(self, source: Optional[ZarrSceneSource]) -> None:
        viewer_setup._init_viewer_scene(self, source)

    def _configure_camera_for_mode(self) -> None:
        viewer_camera_ops._configure_camera_for_mode(self)

    def _frame_volume_camera(self, w: float, h: float, d: float) -> None:
        viewer_camera_ops._frame_volume_camera(self, w, h, d)

    def _bootstrap_camera_pose(
        self,
        mode: RenderMode,
        source: Optional[ZarrSceneSource],
        *,
        reason: str,
    ) -> None:
        viewer_camera_ops._bootstrap_camera_pose(self, mode, source, reason=reason)

    def _enter_volume_mode(self) -> None:
        viewer_camera_ops._enter_volume_mode(self)

    def _exit_volume_mode(self) -> None:
        viewer_camera_ops._exit_volume_mode(self)

    def _register_plane_visual(self, node: Any) -> None:
        viewer_visuals._register_plane_visual(self, node)

    def _register_volume_visual(self, node: Any) -> None:
        viewer_visuals._register_volume_visual(self, node)

    def _ensure_plane_visual(self) -> Any:
        return viewer_visuals._ensure_plane_visual(self)

    def _ensure_volume_visual(self) -> Any:
        return viewer_visuals._ensure_volume_visual(self)

    def _apply_camera_reset(self, cam) -> None:
        viewer_camera_ops._apply_camera_reset(self, cam)

    def _emit_current_camera_pose(self, reason: str) -> None:
        viewer_camera_ops._emit_current_camera_pose(self, reason)

    def _emit_pose_from_camera(self, camera, reason: str) -> None:
        viewer_camera_ops._emit_pose_from_camera(self, camera, reason)

    def _pose_from_camera(self, camera, target: str, command_seq: int) -> Optional[CameraPoseApplied]:
        return viewer_camera_ops._pose_from_camera(self, camera, target, command_seq)

    def _snapshot_camera_pose(self, target: str, command_seq: int) -> Optional[CameraPoseApplied]:
        return viewer_camera_ops._snapshot_camera_pose(self, target, command_seq)

    def _current_panzoom_rect(self) -> Optional[tuple[float, float, float, float]]:
        return viewer_camera_ops._current_panzoom_rect(self)

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
        # Map all aliases to the default selector
        self._policy_func = lod.select_level
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

    def _load_volume(
        self,
        source: ZarrSceneSource,
        level: int,
    ) -> np.ndarray:
        volume = source.level_volume(level, compute=True)
        if not isinstance(volume, np.ndarray):
            volume = np.asarray(volume, dtype=np.float32)
        return volume

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

    def _capture_blit_gpu_ns(self) -> Optional[int]:
        return capture_tick.capture_blit_gpu_ns(self)

    def capture_and_encode_packet(self) -> tuple[FrameTimings, Optional[bytes], int, int]:
        return capture_tick.capture_and_encode_packet(self)

    def render_tick(self) -> float:
        return capture_tick.render_tick(self)


    # ---- C6 selection (napari-anchored) -------------------------------------
    def _evaluate_level_policy(self) -> None:
        """Evaluate multiscale policy inputs and perform a level switch if needed."""

        current_level = int(self._current_level_index())
        evaluation = level_policy.evaluate(self)
        if evaluation is None:
            return

        decision = evaluation.decision
        context = evaluation.context
        source = evaluation.source
        zoom_ratio = evaluation.zoom_ratio

        self._last_level_switch_ts = float(decision.timestamp)

        requested = self._viewport_runner.request_level(int(context.level))
        if not requested:
            logger.debug("level intent suppressed (no change or already pending)")
            return

        apply_iface = RenderApplyInterface(self)
        if self._viewport_state.mode is RenderMode.VOLUME:
            apply_volume_metadata(apply_iface, source, context)
        else:
            apply_plane_metadata(apply_iface, source, context)

        intent = LevelSwitchIntent(
            desired_level=int(decision.desired_level),
            selected_level=int(context.level),
            reason=decision.reason,
            previous_level=current_level,
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
            int(current_level),
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
        cleanup_render_worker(self)

    def viewer_model(self) -> Optional[ViewerModel]:
        """Expose the napari ``ViewerModel`` when adapter mode is active."""
        return self._viewer

    # Adapter is always used; legacy path removed
    def attach_ledger(self, ledger: ServerStateLedger) -> None:
        self._ledger = ledger
