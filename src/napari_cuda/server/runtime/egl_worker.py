"""Render-thread orchestration for the server.

`EGLRendererWorker` lives in this module and is still EGL-backed internally, but
the neutral module name reflects its broader responsibilities:

* bootstrap the napari Viewer/VisPy canvas and keep it on the worker thread,
* drain `RenderUpdateMailbox` updates and apply them through `SceneStateApplier`,
* drive the render loop, including camera animation and policy evaluation,
* capture frames via `CaptureFacade`, hand them to the encoder, and surface
  timing metadata for downstream metrics.

The worker owns the EGL + CUDA context pair, renders into an FBO-backed texture,
maps the texture into CUDA memory, stages the pixels for NVENC, and hands the
encoded packets back to the asyncio side through callbacks.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from copy import deepcopy
from dataclasses import replace
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

import pycuda.driver as cuda  # type: ignore
from vispy import scene  # type: ignore
from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera, TurntableCamera

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

# No direct use of roi_applier here; slice application goes through SceneStateApplier.
from napari_cuda.server.data.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
)
from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.rendering.capture import (
    CaptureFacade,
    FrameTimings,
    encode_frame,
)
from napari_cuda.server.rendering.debug_tools import DebugConfig, DebugDumper
from napari_cuda.server.rendering.egl_context import EglContext
from napari_cuda.server.rendering.encoder import Encoder

# Bitstream packing happens in the server layer
from napari_cuda.server.rendering.viewer_builder import ViewerBuilder
from napari_cuda.server.runtime.camera_animator import animate_if_enabled
from napari_cuda.server.runtime.camera_command_queue import CameraCommandQueue
from napari_cuda.server.runtime.camera_controller import (
    process_camera_deltas as _process_camera_deltas,
)
from napari_cuda.server.runtime.camera_pose import CameraPoseApplied
from napari_cuda.server.runtime.intents import LevelSwitchIntent
from napari_cuda.server.runtime.render_ledger_snapshot import (
    RenderLedgerSnapshot,
)
from napari_cuda.server.runtime.render_snapshot import apply_render_snapshot
from napari_cuda.server.runtime.plane_loader import (
    apply_plane_slice_roi,
    viewport_roi_for_level,
)
from napari_cuda.server.runtime.plane_snapshot import apply_slice_level
from napari_cuda.server.runtime.volume_snapshot import apply_volume_level
from napari_cuda.server.runtime.viewer_stage import (
    apply_plane_metadata,
    apply_volume_metadata,
)
from napari_cuda.server.runtime.render_update_mailbox import (
    RenderUpdate,
    RenderUpdateMailbox,
)
from napari_cuda.server.runtime.runtime_loop import run_render_tick
from napari_cuda.server.runtime.scene_state_applier import (
    SceneDrainResult,
    SceneStateApplier,
    SceneStateApplyContext,
)
from napari_cuda.server.runtime.scene_types import SliceROI
from napari_cuda.server.runtime.state_structs import PlaneState, RenderMode, ViewportState, VolumeState
from napari_cuda.server.runtime.viewport_runner import ViewportRunner
from napari_cuda.server.runtime.worker_runtime import (
    ensure_scene_source,
    reset_worker_camera,
)
from napari_cuda.server.scene import CameraDeltaCommand

logger = logging.getLogger(__name__)


def _rect_to_tuple(rect: Rect) -> tuple[float, float, float, float]:
    """Normalize a VisPy Rect into (left, bottom, width, height)."""

    return (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))


def _coarsest_level_index(source: ZarrSceneSource) -> Optional[int]:
    descriptors = source.level_descriptors
    if not descriptors:
        return None
    return len(descriptors) - 1


# Back-compat for tests patching the selector directly
select_level = lod.select_level

## Camera ops now live in napari_cuda.server.runtime.camera_ops as free functions.

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
        step_hint = self._ledger_step()
        level_hint = self._ledger_level()
        axis_labels = self._ledger_axis_labels()
        order = self._ledger_order()
        ndisplay = self._ledger_ndisplay()
        canvas, view, viewer = builder.build(
            source,
            level=level_hint,
            step=step_hint,
            axis_labels=axis_labels,
            order=order,
            ndisplay=ndisplay,
        )
        # Mirror attributes for call sites expecting them
        self.canvas = canvas
        self.view = view
        self._viewer = viewer
        # Ensure the camera frames current data extents on first init
        assert self.view is not None, "adapter must supply a VisPy view"
        cam = self.view.camera
        if cam is not None:
            self._apply_camera_reset(cam)
            # Persist initial camera pose so plane caches exist deterministically
            callback = self._camera_pose_callback
            if callback is not None:
                pose = self._snapshot_camera_pose("main", 1)
                if pose is not None:
                    callback(pose)

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
        last_step = self._ledger_step()
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

    def _configure_camera_for_mode(self) -> None:
        view = self.view
        if view is None:
            return

        if self._viewport_state.mode is RenderMode.VOLUME:
            view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
            extent = self._volume_world_extents()
            if extent is None:
                width_px, height_px = self._data_wh
                depth_px = self._data_d or 1
                extent = (float(width_px), float(height_px), float(depth_px))
            view.camera.set_range(
                x=(0.0, max(1.0, extent[0])),
                y=(0.0, max(1.0, extent[1])),
                z=(0.0, max(1.0, extent[2])),
            )
            self._frame_volume_camera(extent[0], extent[1], extent[2])
        else:
            cam = scene.cameras.PanZoomCamera(aspect=1.0)
            view.camera = cam
            plane_state = self.viewport_state.plane
            rect = plane_state.camera_rect
            center = plane_state.camera_center
            zoom = plane_state.camera_zoom

            if rect is not None and len(rect) >= 4:
                cam.rect = Rect(
                    float(rect[0]),
                    float(rect[1]),
                    float(rect[2]),
                    float(rect[3]),
                )
            else:
                self._apply_camera_reset(cam)

            if center is not None and len(center) >= 2:
                cam.center = (
                    float(center[0]),
                    float(center[1]),
                )
            if zoom is not None:
                cam.zoom = float(zoom)

    def _enter_volume_mode(self) -> None:
        if self._viewport_state.mode is RenderMode.VOLUME:
            return

        if self._level_intent_callback is None:
            raise RuntimeError("level intent callback required for volume mode")

        source = self._ensure_scene_source()
        requested_level = _coarsest_level_index(source)
        assert requested_level is not None and requested_level >= 0, "Volume mode requires a multiscale level"
        selected_level, downgraded = self._resolve_volume_intent_level(source, int(requested_level))
        self._viewport_state.volume.downgraded = bool(downgraded)

        decision = lod.LevelDecision(
            desired_level=int(requested_level),
            selected_level=int(selected_level),
            reason="ndisplay-3d",
            timestamp=time.perf_counter(),
            oversampling={},
            downgraded=bool(downgraded),
        )
        last_step = self._ledger_step()
        context = lod.build_level_context(
            decision,
            source=source,
            prev_level=int(self._current_level_index()),
            last_step=last_step,
        )
        apply_volume_metadata(self, source, context)

        # Defer level application to the controller transaction.
        self._viewport_state.mode = RenderMode.VOLUME

        intent = LevelSwitchIntent(
            desired_level=int(requested_level),
            selected_level=int(context.level),
            reason="ndisplay-3d",
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
        logger.info(
            "intent.level_switch: prev=%d target=%d reason=%s downgraded=%s",
            int(self._current_level_index()),
            int(context.level),
            intent.reason,
            intent.downgraded,
        )
        requested = self._viewport_runner.request_level(int(context.level))
        if requested and self._level_intent_callback is not None:
            self._level_intent_callback(intent)

        self._configure_camera_for_mode()
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "toggle.enter_3d: requested_level=%d selected_level=%d downgraded=%s",
                int(requested_level),
                int(selected_level),
                bool(downgraded),
            )
        # Apply cached 3D camera pose if present
        cam = self.view.camera if self.view is not None else None
        if cam is not None:
            cen = self._ledger.get("camera_volume", "main", "center")
            ang = self._ledger.get("camera_volume", "main", "angles")
            dist = self._ledger.get("camera_volume", "main", "distance")
            fov = self._ledger.get("camera_volume", "main", "fov")
            if cen is not None and ang is not None and dist is not None and fov is not None:
                cx, cy, cz = tuple(float(v) for v in cen.value)
                az, el, rl = tuple(float(v) for v in ang.value)
                cam.center = (cx, cy, cz)  # type: ignore[attr-defined]
                cam.azimuth = float(az)  # type: ignore[attr-defined]
                cam.elevation = float(el)  # type: ignore[attr-defined]
                cam.roll = float(rl)
                cam.distance = float(dist.value)  # type: ignore[attr-defined]
                cam.fov = float(fov.value)  # type: ignore[attr-defined]
        self._request_encoder_idr()
        self._mark_render_tick_needed()
        self._emit_current_camera_pose(reason="enter-3d")

    def _exit_volume_mode(self) -> None:
        if self._viewport_state.mode is not RenderMode.VOLUME:
            return
        self._viewport_state.mode = RenderMode.PLANE
        # Restore plane target level/step and plane camera rect from caches
        lvl_entry = self._ledger.get("view_cache", "plane", "level")
        step_entry = self._ledger.get("view_cache", "plane", "step")
        assert lvl_entry is not None, "plane restore requires view_cache.plane.level"
        assert step_entry is not None, "plane restore requires view_cache.plane.step"
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "toggle.exit_3d: restore level=%s step=%s", str(lvl_entry.value), str(step_entry.value)
            )
        source = self._ensure_scene_source()
        lvl_idx = int(lvl_entry.value)
        plane_entry = self._ledger.get("viewport", "plane", "state")
        assert plane_entry is not None and isinstance(plane_entry.value, Mapping), "plane camera cache missing viewport state"
        plane_state = PlaneState(**dict(plane_entry.value))  # type: ignore[arg-type]
        self._viewport_state.plane = PlaneState(**dict(plane_entry.value))  # sync worker cache
        assert plane_state.camera_rect is not None, "plane camera cache missing rect"
        rect = tuple(float(v) for v in plane_state.camera_rect)
        step_tuple = tuple(int(v) for v in step_entry.value)

        view = self.view
        if view is not None:
            view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
            cam = view.camera
            sy, sx = plane_scale_for_level(source, lvl_idx)
            h_full, w_full = plane_wh_for_level(source, lvl_idx)
            world_w = float(w_full) * float(max(1e-12, sx))
            world_h = float(h_full) * float(max(1e-12, sy))
            cam.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
            cam.rect = Rect(*rect)
            if plane_state.camera_center is not None:
                cx, cy = plane_state.camera_center
                cam.center = (float(cx), float(cy), 0.0)  # type: ignore[attr-defined]
            if plane_state.camera_zoom is not None:
                cam.zoom = float(plane_state.camera_zoom)

        decision = lod.LevelDecision(
            desired_level=int(lvl_idx),
            selected_level=int(lvl_idx),
            reason="ndisplay-2d",
            timestamp=time.perf_counter(),
            oversampling={},
        )
        context = lod.build_level_context(
            decision,
            source=source,
            prev_level=int(self._current_level_index()),
            last_step=step_tuple,
        )
        apply_plane_metadata(self, source, context)
        intent = LevelSwitchIntent(
            desired_level=int(lvl_idx),
            selected_level=int(context.level),
            reason="ndisplay-2d",
            previous_level=int(self._current_level_index()),
            context=context,
            oversampling={},
            timestamp=decision.timestamp,
            downgraded=False,
            mode=self.viewport_state.mode,
            plane_state=deepcopy(self.viewport_state.plane),
            volume_state=deepcopy(self.viewport_state.volume),
        )

        callback = self._level_intent_callback
        if callback is None:
            logger.debug("plane restore intent dropped (no callback)")
            return

        requested = self._viewport_runner.request_level(int(context.level))
        if requested:
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
        self.cuda_ctx: Optional[cuda.Context] = None
        self.canvas: Optional[scene.SceneCanvas] = None
        self.view = None
        self._visual = None
        self._egl = EglContext(self.width, self.height)
        self._capture = CaptureFacade(width=self.width, height=self.height)
        self._encoder: Optional[Encoder] = None

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
        self._enc_lock = threading.Lock()
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
        self._preserve_view_on_switch = bool(policy_cfg.preserve_view_on_switch)
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
        self._roi_cache: dict[int, tuple[Optional[tuple[float, ...]], SliceROI]] = {}
        self._roi_log_state: dict[int, tuple[SliceROI, float]] = {}
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

    def snapshot_dims_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}

        axes = self._ledger_axis_labels()
        if axes:
            meta["axis_labels"] = list(axes)
            meta["axes"] = list(axes)

        order = self._ledger_order()
        if order:
            meta["order"] = list(order)

        displayed = self._ledger_displayed()
        if displayed:
            meta["displayed"] = list(displayed)

        ndisplay = self._ledger_ndisplay()
        if ndisplay is not None:
            meta["ndisplay"] = int(ndisplay)
            meta["mode"] = "volume" if int(ndisplay) >= 3 else "plane"

        current_step = self._ledger_step()
        if current_step:
            meta["current_step"] = list(current_step)

        level_shapes = self._ledger_level_shapes()
        if level_shapes:
            meta["level_shapes"] = [list(shape) for shape in level_shapes]
        level_idx = self._ledger_level()
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
            recorded = self._ledger_step()
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

        logger.warning("EGLRendererWorker could not locate an active VisPy visual; proceeding without visual")
        return None

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

    def _normalize_scene_state(self, state: RenderLedgerSnapshot) -> RenderLedgerSnapshot:
        """Return the render-thread-friendly copy of *state*."""

        layer_values = None
        if state.layer_values:
            values: dict[str, dict[str, object]] = {}
            for layer_id, props in state.layer_values.items():
                if not props:
                    continue
                values[str(layer_id)] = {str(key): value for key, value in props.items()}
            if values:
                layer_values = values

        layer_versions = None
        if state.layer_versions:
            versions: dict[str, dict[str, int]] = {}
            for layer_id, props in state.layer_versions.items():
                if not props:
                    continue
                mapped: dict[str, int] = {}
                for key, version in props.items():
                    mapped[str(key)] = int(version)
                if mapped:
                    versions[str(layer_id)] = mapped
            if versions:
                layer_versions = versions

        return RenderLedgerSnapshot(
            center=(tuple(float(c) for c in state.center) if state.center is not None else None),
            zoom=(float(state.zoom) if state.zoom is not None else None),
            angles=(tuple(float(a) for a in state.angles) if state.angles is not None else None),
            current_step=(
                tuple(int(s) for s in state.current_step)
                if state.current_step is not None
                else None
            ),
            dims_version=(int(state.dims_version) if state.dims_version is not None else None),
            ndisplay=(int(state.ndisplay) if state.ndisplay is not None else None),
            view_version=(int(state.view_version) if state.view_version is not None else None),
            displayed=(
                tuple(int(idx) for idx in state.displayed)
                if state.displayed is not None
                else None
            ),
            order=(
                tuple(int(idx) for idx in state.order)
                if state.order is not None
                else None
            ),
            axis_labels=(
                tuple(str(label) for label in state.axis_labels)
                if state.axis_labels is not None
                else None
            ),
            level_shapes=(
                tuple(
                    tuple(int(dim) for dim in shape)
                    for shape in state.level_shapes
                )
                if state.level_shapes is not None
                else None
            ),
            current_level=(int(state.current_level) if state.current_level is not None else None),
            multiscale_level_version=(
                int(state.multiscale_level_version)
                if state.multiscale_level_version is not None
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
            layer_values=layer_values,
            layer_versions=layer_versions,
        )

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
            scene_state = self._normalize_scene_state(delta.scene_state)
        if scene_state is not None:
            self._render_mailbox.set_scene_state(scene_state)
            self._mark_render_tick_needed()
            return

    def _record_snapshot_versions(self, state: RenderLedgerSnapshot) -> None:
        if state.dims_version is not None:
            self._applied_versions[("dims", "main", "current_step")] = int(state.dims_version)
        if state.view_version is not None:
            self._applied_versions[("view", "main", "ndisplay")] = int(state.view_version)
        if state.multiscale_level_version is not None:
            self._applied_versions[("multiscale", "main", "level")] = int(state.multiscale_level_version)
        if state.camera_versions:
            for key, version in state.camera_versions.items():
                self._applied_versions[("camera", "main", str(key))] = int(version)

    def _extract_layer_changes(self, state: RenderLedgerSnapshot) -> dict[str, dict[str, Any]]:
        layer_changes: dict[str, dict[str, Any]] = {}
        layer_versions = state.layer_versions or {}
        if not state.layer_values:
            return layer_changes

        for raw_layer_id, props in state.layer_values.items():
            if not props:
                continue
            layer_id = str(raw_layer_id)
            version_map = layer_versions.get(layer_id)
            if version_map is None and raw_layer_id in layer_versions:
                version_map = layer_versions[raw_layer_id]
            for raw_prop, value in props.items():
                prop = str(raw_prop)
                version_value = None
                if version_map is not None and prop in version_map:
                    version_value = int(version_map[prop])
                    key = ("layer", layer_id, prop)
                    previous = self._applied_versions.get(key)
                    if previous is not None and previous == version_value:
                        continue
                    self._applied_versions[key] = version_value
                layer_changes.setdefault(layer_id, {})[prop] = value
        return layer_changes

    def _consume_render_snapshot(
        self,
        state: RenderLedgerSnapshot,
        *,
        apply_camera_pose: bool = True,
    ) -> None:
        """Queue a complete scene state snapshot for the next frame."""

        normalized = self._normalize_scene_state(state)
        self._render_mailbox.set_scene_state(normalized, apply_camera_pose=apply_camera_pose)
        self._mark_render_tick_needed()
        self._last_interaction_ts = time.perf_counter()

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

    def _apply_camera_reset(self, cam) -> None:
        reset_worker_camera(self, cam)

    def _emit_current_camera_pose(self, reason: str) -> None:
        """Snapshot and emit the current camera pose for ledger sync."""

        callback = self._camera_pose_callback
        if callback is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("pose.emit skipped (no callback) reason=%s", reason)
            return

        base_seq = max(int(self._pose_seq), int(self._max_camera_command_seq))
        next_seq = base_seq + 1
        pose = self._snapshot_camera_pose("main", int(next_seq))
        if pose is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("pose.emit skipped (no pose) reason=%s seq=%d", reason, int(self._pose_seq))
            return

        if self._viewport_state.mode is RenderMode.VOLUME:
            volume_key = (
                tuple(float(v) for v in pose.center) if pose.center is not None else None,
                tuple(float(v) for v in pose.angles) if pose.angles is not None else None,
                float(pose.distance) if pose.distance is not None else None,
                float(pose.fov) if pose.fov is not None else None,
            )
            if self._last_volume_pose == volume_key:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("pose.emit skipped (unchanged volume pose) reason=%s", reason)
                return
            self._last_volume_pose = volume_key
        else:
            plane_key = (
                tuple(float(v) for v in pose.center) if pose.center is not None else None,
                float(pose.zoom) if pose.zoom is not None else None,
                tuple(float(v) for v in pose.rect) if pose.rect is not None else None,
            )
            if self._last_plane_pose == plane_key:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("pose.emit skipped (unchanged plane pose) reason=%s", reason)
                return
            self._last_plane_pose = plane_key

        self._pose_seq = int(next_seq)

        if logger.isEnabledFor(logging.INFO):
            mode = "volume" if self._viewport_state.mode is RenderMode.VOLUME else "plane"
            rect = pose.rect if pose.rect is not None else None
            logger.info(
                "pose.emit: seq=%d reason=%s mode=%s rect=%s center=%s zoom=%s",
                int(next_seq),
                reason,
                mode,
                rect,
                pose.center,
                pose.zoom,
            )

        callback(pose)
        self._max_camera_command_seq = max(int(self._max_camera_command_seq), int(next_seq))

    def _snapshot_camera_pose(self, target: str, command_seq: int) -> Optional[CameraPoseApplied]:
        view = self.view
        if view is None:
            return None
        cam = view.camera
        def _center_tuple(camera) -> Optional[tuple[float, ...]]:
            center_value = getattr(camera, "center", None)
            if callable(center_value):
                center_value = center_value()
            if center_value is None:
                return None
            return tuple(float(component) for component in center_value)

        volume_state = self._viewport_state.volume
        plane_state = self._viewport_state.plane

        if isinstance(cam, TurntableCamera):
            center_tuple = _center_tuple(cam)
            distance_val = float(cam.distance)
            fov_val = float(cam.fov)
            azimuth = float(cam.azimuth)
            elevation = float(cam.elevation)
            roll = float(getattr(cam, "roll", 0.0))
            if center_tuple is not None:
                volume_state.pose_center = tuple(float(v) for v in center_tuple)
            volume_state.pose_distance = distance_val
            volume_state.pose_fov = fov_val
            volume_state.pose_angles = (azimuth, elevation, roll)
            return CameraPoseApplied(
                target=str(target or "main"),
                command_seq=int(command_seq),
                center=center_tuple,
                zoom=None,
                angles=(azimuth, elevation, roll),
                distance=distance_val,
                fov=fov_val,
                rect=None,
            )
        if isinstance(cam, PanZoomCamera):
            center_tuple = _center_tuple(cam)
            rect_obj = cam.rect
            rect_tuple: Optional[tuple[float, float, float, float]]
            if rect_obj is None:
                rect_tuple = None
            else:
                rect_tuple = (
                    float(rect_obj.left),
                    float(rect_obj.bottom),
                    float(rect_obj.width),
                    float(rect_obj.height),
                )
            zoom_attr = cam.zoom
            if callable(zoom_attr):
                zoom_attr = cam._zoom if hasattr(cam, "_zoom") else 1.0  # type: ignore[attr-defined]
            zoom_val = float(zoom_attr)
            if rect_tuple is not None:
                plane_state.camera_rect = rect_tuple
            if center_tuple is not None and len(center_tuple) >= 2:
                plane_state.camera_center = (float(center_tuple[0]), float(center_tuple[1]))
            plane_state.camera_zoom = zoom_val
            return CameraPoseApplied(
                target=str(target or "main"),
                command_seq=int(command_seq),
                center=center_tuple,
                zoom=zoom_val,
                angles=None,
                distance=None,
                fov=None,
                rect=rect_tuple,
            )
        return CameraPoseApplied(
            target=str(target or "main"),
            command_seq=int(command_seq),
            center=_center_tuple(cam),
            zoom=None,
            angles=None,
            distance=None,
            fov=None,
            rect=None,
        )

    def _current_panzoom_rect(self) -> Optional[tuple[float, float, float, float]]:
        view = self.view
        if view is None:
            return None
        cam = view.camera
        if not isinstance(cam, PanZoomCamera):
            return None
        rect = cam.rect
        if rect is None:
            return None
        return (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))

    def _build_scene_state_context(self, cam) -> SceneStateApplyContext:
        applied_roi = self._viewport_runner.state.applied_roi
        applied_level = self._viewport_runner.state.applied_level
        last_roi_tuple = None
        if applied_roi is not None:
            last_roi_tuple = (int(applied_level), applied_roi)

        return SceneStateApplyContext(
            use_volume=self._viewport_state.mode is RenderMode.VOLUME,
            viewer=self._viewer,
            camera=cam,
            visual=self._resolve_visual(),
            layer=self._napari_layer,
            scene_source=self._scene_source,
            active_ms_level=int(self._current_level_index()),
            z_index=self._z_index,
            last_roi=last_roi_tuple,
            preserve_view_on_switch=self._preserve_view_on_switch,
            sticky_contrast=self._sticky_contrast,
            idr_on_z=self._idr_on_z,
            data_wh=self._data_wh,
            volume_scale=self._volume_scale,
            state_lock=self._state_lock,
            ensure_scene_source=self._ensure_scene_source,
            plane_scale_for_level=plane_scale_for_level,
            load_slice=self._load_slice,
            mark_render_tick_needed=self._mark_render_tick_needed,
            request_encoder_idr=self._request_encoder_idr,
        )

    def _apply_viewport_state_snapshot(
        self,
        *,
        mode: Optional[RenderMode],
        plane_state: Optional[PlaneState],
        volume_state: Optional[VolumeState],
    ) -> None:
        """Apply a mailbox-only viewport update (no scene snapshot)."""

        runner = self._viewport_runner
        updated = False

        if plane_state is not None:
            self._viewport_state.plane = deepcopy(plane_state)
            if runner is not None:
                runner._plane = self._viewport_state.plane  # type: ignore[attr-defined]
            updated = True

        if volume_state is not None:
            self._viewport_state.volume = deepcopy(volume_state)
            updated = True

        if mode is not None and mode is not self._viewport_state.mode:
            self._viewport_state.mode = mode
            self._configure_camera_for_mode()
            updated = True

        if not updated:
            return

        if runner is not None:
            # Ensure runner sees latest camera rect when switching modes before the next snapshot.
            if self._viewport_state.mode is RenderMode.PLANE:
                rect = self._current_panzoom_rect()
                if rect is not None:
                    runner.update_camera_rect(rect)

    def drain_scene_updates(self) -> None:
        updates: RenderUpdate = self._render_mailbox.drain()
        state = updates.scene_state
        if state is None:
            self._apply_viewport_state_snapshot(
                mode=updates.mode,
                plane_state=updates.plane_state,
                volume_state=updates.volume_state,
            )
            return

        snapshot_op_seq = updates.op_seq
        if snapshot_op_seq is None:
            snapshot_op_seq = int(state.op_seq) if state.op_seq is not None else 0
        if int(snapshot_op_seq) < int(self.viewport_state.op_seq):
            logger.debug(
                "render snapshot skipped: stale op_seq snapshot=%d latest=%d",
                int(snapshot_op_seq),
                int(self.viewport_state.op_seq),
            )
            return
        self.viewport_state.op_seq = int(snapshot_op_seq)

        self._record_snapshot_versions(state)
        layer_changes = self._extract_layer_changes(state)

        if layer_changes:
            layer_version_subset: dict[str, dict[str, int]] = {}
            original_versions = state.layer_versions or {}
            for layer_id, props in layer_changes.items():
                version_map = original_versions.get(layer_id, {})
                subset: dict[str, int] = {}
                for prop in props:
                    if prop in version_map:
                        subset[prop] = int(version_map[prop])
                if subset:
                    layer_version_subset[layer_id] = subset
            state_for_apply = replace(
                state,
                layer_values=layer_changes,
                layer_versions=layer_version_subset or None,
            )
        else:
            state_for_apply = replace(state, layer_values=None, layer_versions=None)

        if not updates.apply_camera_pose:
            state_for_apply = replace(
                state_for_apply,
                center=None,
                zoom=None,
                angles=None,
                distance=None,
                fov=None,
                rect=None,
            )

        if updates.plane_state is not None:
            self._viewport_state.plane = deepcopy(updates.plane_state)
            if self._viewport_runner is not None:
                self._viewport_runner._plane = self._viewport_state.plane  # type: ignore[attr-defined]
        if updates.volume_state is not None:
            self._viewport_state.volume = deepcopy(updates.volume_state)

        previous_mode = self._viewport_state.mode
        signature_changed = self._render_mailbox.update_state_signature(state)
        if signature_changed:
            apply_render_snapshot(self, state_for_apply)

        if updates.mode is not None:
            self._viewport_state.mode = updates.mode
        else:
            self._viewport_state.mode = previous_mode

        if self._viewport_state.mode is RenderMode.VOLUME:
            self._level_policy_suppressed = False
        elif updates.mode is RenderMode.PLANE or previous_mode is RenderMode.VOLUME:
            self._level_policy_suppressed = True

        if signature_changed and self._viewport_runner is not None:
            self._viewport_runner.ingest_snapshot(state)

        view = self.view
        assert view is not None, "drain_scene_updates requires an active VisPy view"
        cam = view.camera

        ctx = self._build_scene_state_context(cam)
        drain_res: SceneDrainResult = SceneStateApplier.drain_updates(
            ctx,
            state=state_for_apply,
            mailbox=self._render_mailbox,
        )

        if drain_res.z_index is not None:
            self._z_index = int(drain_res.z_index)
        if drain_res.data_wh is not None:
            self._data_wh = (int(drain_res.data_wh[0]), int(drain_res.data_wh[1]))

        if (
            self._viewport_runner is not None
            and self._viewport_state.mode is RenderMode.PLANE
        ):
            rect = self._current_panzoom_rect()
            if rect is not None:
                self._viewport_runner.update_camera_rect(rect)
            if int(self._current_level_index()) == int(self._viewport_runner.state.target_level):
                self._viewport_runner.mark_level_applied(self._current_level_index())
        elif self._viewport_runner is not None and int(self._current_level_index()) == int(self._viewport_runner.state.target_level):
            self._viewport_runner.mark_level_applied(self._current_level_index())

        if (
            drain_res.policy_refresh_needed
            and self._viewport_state.mode is not RenderMode.VOLUME
            and not self._level_policy_suppressed
        ):
            self._evaluate_level_policy()

        if self._level_policy_suppressed:
            ledger = self._ledger
            assert ledger is not None, "ledger must be attached before rendering"
            op_kind_entry = ledger.get("scene", "main", "op_kind")
            if op_kind_entry is not None and str(op_kind_entry.value) == "dims-update":
                self._level_policy_suppressed = False

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

        self.drain_scene_updates()
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
                runner.state.pose_reason = None
                runner.state.zoom_hint = None
            return

        if self._viewport_state.mode is RenderMode.VOLUME:
            runner = self._viewport_runner
            if runner is not None:
                runner.state.pose_reason = None
                runner.state.zoom_hint = None
            return

        source = self._scene_source or self._ensure_scene_source()

        def _resolve(level: int, _rect: tuple[float, float, float, float]) -> SliceROI:
            return viewport_roi_for_level(self, source, int(level), quiet=True)

        intent = self._viewport_runner.plan_tick(source=source, roi_resolver=_resolve)

        if intent.zoom_hint is not None:
            self._render_mailbox.record_zoom_hint(float(intent.zoom_hint))

        level_applied = False

        if intent.level_change:
            target_level = int(self._viewport_runner.state.target_level)
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
            intent.roi_change
            and self._viewport_state.mode is not RenderMode.VOLUME
            and not level_applied
        ):
            pending = self._viewport_runner.state.pending_roi
            if pending is not None:
                apply_plane_slice_roi(
                    self,
                    source,
                    int(self._viewport_runner.state.target_level),
                    pending,
                    update_contrast=False,
                )

        rect = self._current_panzoom_rect()
        if rect is not None:
            self._viewport_runner.update_camera_rect(rect)
        if intent.pose_reason is not None:
            self._emit_current_camera_pose(intent.pose_reason)


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

        step_hint = self._ledger_step()
        context = lod.build_level_context(
            decision,
            source=source,
            prev_level=current,
            last_step=step_hint,
        )
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

        requested = self._viewport_runner.request_level(int(context.level))
        if not requested:
            logger.debug("level intent suppressed (runner already pending)")
            return

        callback = self._level_intent_callback
        if callback is None:
            logger.debug("level intent callback missing; skipping emission")
            return

        callback(intent)
        self._mark_render_tick_needed()

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
        self._is_ready = False

    def viewer_model(self) -> Optional[ViewerModel]:
        """Expose the napari ``ViewerModel`` when adapter mode is active."""
        return self._viewer

    # Adapter is always used; legacy path removed
    def attach_ledger(self, ledger: ServerStateLedger) -> None:
        self._ledger = ledger

    def _ledger_step(self) -> Optional[tuple[int, ...]]:
        entry = self._ledger.get("dims", "main", "current_step")
        if entry is None:
            return None
        value = entry.value
        if isinstance(value, (list, tuple)):
            return tuple(int(v) for v in value)
        return None

    def _ledger_level(self) -> Optional[int]:
        entry = self._ledger.get("multiscale", "main", "level")
        if entry is None:
            return None
        value = entry.value
        if isinstance(value, int):
            return int(value)
        return None

    def _ledger_axis_labels(self) -> Optional[tuple[str, ...]]:
        entry = self._ledger.get("dims", "main", "axis_labels")
        if entry is None:
            return None
        value = entry.value
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        return None

    def _ledger_order(self) -> Optional[tuple[int, ...]]:
        entry = self._ledger.get("dims", "main", "order")
        if entry is None:
            return None
        value = entry.value
        if isinstance(value, (list, tuple)):
            return tuple(int(v) for v in value)
        return None

    def _ledger_ndisplay(self) -> Optional[int]:
        entry = self._ledger.get("view", "main", "ndisplay")
        if entry is None:
            return None
        value = entry.value
        if isinstance(value, int):
            return int(value)
        return None

    def _ledger_displayed(self) -> Optional[tuple[int, ...]]:
        entry = self._ledger.get("view", "main", "displayed")
        if entry is None:
            return None
        value = entry.value
        if isinstance(value, (list, tuple)):
            return tuple(int(v) for v in value)
        return None

    def _ledger_level_shapes(self) -> Optional[tuple[tuple[int, ...], ...]]:
        entry = self._ledger.get("multiscale", "main", "level_shapes")
        if entry is None:
            return None
        value = entry.value
        if isinstance(value, (list, tuple)):
            shapes: list[tuple[int, ...]] = []
            for shape in value:
                if isinstance(shape, (list, tuple)):
                    shapes.append(tuple(int(v) for v in shape))
            if shapes:
                return tuple(shapes)
        return None
