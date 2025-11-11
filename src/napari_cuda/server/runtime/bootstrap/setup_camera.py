"""Camera lifecycle helpers for the EGL render worker."""

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

from vispy import scene  # type: ignore
from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.data.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
)
from napari_cuda.server.runtime.bootstrap.interface import (
    ViewerBootstrapInterface,
)
from napari_cuda.server.runtime.bootstrap.scene_setup import (
    reset_worker_camera,
)
from napari_cuda.server.runtime.camera import CameraPoseApplied
from napari_cuda.server.runtime.ipc import LevelSwitchIntent
from napari_cuda.server.runtime.render_loop.plan.ledger_access import (
    dims_spec as ledger_dims_spec,
    step as ledger_step,
)
from napari_cuda.server.scene.viewport import (
    PlaneViewportCache,
    RenderMode,
)
from napari_cuda.shared.dims_spec import dims_spec_clamp_step

if TYPE_CHECKING:
    from napari_cuda.server.data.zarr_source import ZarrSceneSource
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


def _coarsest_level_index(source: ZarrSceneSource) -> Optional[int]:
    descriptors = source.level_descriptors
    if not descriptors:
        return None
    return len(descriptors) - 1


def _frame_volume_camera(worker: EGLRendererWorker, w: float, h: float, d: float) -> None:
    """Choose stable initial center and distance for TurntableCamera."""

    view = worker.view
    cam = view.camera if view is not None else None
    if not isinstance(cam, scene.cameras.TurntableCamera):
        return
    center = (float(w) * 0.5, float(h) * 0.5, float(d) * 0.5)
    cam.center = center  # type: ignore[attr-defined]
    fov_deg = float(getattr(cam, "fov", 60.0) or 60.0)
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


def _configure_camera_for_mode(worker: EGLRendererWorker) -> None:
    view = worker.view
    if view is None:
        return

    if worker._viewport_state.mode is RenderMode.VOLUME:
        view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
        extent = worker._volume_world_extents()
        if extent is None:
            width_px, height_px = worker._data_wh
            depth_px = worker._data_d or 1
            extent = (float(width_px), float(height_px), float(depth_px))
        view.camera.set_range(
            x=(0.0, max(1.0, extent[0])),
            y=(0.0, max(1.0, extent[1])),
            z=(0.0, max(1.0, extent[2])),
        )
        _frame_volume_camera(worker, extent[0], extent[1], extent[2])
    else:
        cam = scene.cameras.PanZoomCamera(aspect=1.0)
        view.camera = cam
        plane_state = worker.viewport_state.plane
        rect = plane_state.pose.rect
        center = plane_state.pose.center
        zoom = plane_state.pose.zoom

        if rect is not None and len(rect) >= 4:
            cam.rect = Rect(
                float(rect[0]),
                float(rect[1]),
                float(rect[2]),
                float(rect[3]),
            )
        else:
            _apply_camera_reset(worker, cam)

        if center is not None and len(center) >= 2:
            cam.center = (
                float(center[0]),
                float(center[1]),
            )
        if zoom is not None:
            cam.zoom = float(zoom)


def _bootstrap_camera_pose(
    worker: EGLRendererWorker,
    mode: RenderMode,
    source: Optional[ZarrSceneSource],
    *,
    reason: str,
) -> None:
    """Frame and emit an initial pose for the requested render mode."""

    if worker.view is None:
        return

    facade = ViewerBootstrapInterface(worker)

    original_mode = worker._viewport_state.mode
    original_camera = worker.view.camera

    if mode is RenderMode.PLANE:
        worker._viewport_state.mode = RenderMode.PLANE
        _configure_camera_for_mode(worker)
        plane_state = worker.viewport_state.plane
        rect = plane_state.pose.rect
        center = plane_state.pose.center
        zoom = plane_state.pose.zoom
        cam = worker.view.camera
        assert isinstance(cam, PanZoomCamera)
        if rect is not None and len(rect) >= 4:
            cam.rect = Rect(
                float(rect[0]),
                float(rect[1]),
                float(rect[2]),
                float(rect[3]),
            )
        else:
            _apply_camera_reset(worker, cam)
        if center is not None and len(center) >= 2:
            cam.center = (
                float(center[0]),
                float(center[1]),
            )
        if zoom is not None:
            cam.zoom = float(zoom)
        _emit_current_camera_pose(worker, reason)

    else:
        source = worker._ensure_scene_source()
        coarse_level = _coarsest_level_index(source)
        assert coarse_level is not None and coarse_level >= 0, "volume bootstrap requires multiscale levels"
        prev_mode = worker._viewport_state.mode
        prev_level = worker._current_level_index()
        prev_step = ledger_step(worker._ledger)
        spec = ledger_dims_spec(worker._ledger)
        assert spec is not None, "volume bootstrap requires dims spec"
        base_step = prev_step if prev_step is not None else spec.current_step
        step_tuple = dims_spec_clamp_step(spec, int(coarse_level), base_step)
        worker._viewport_state.mode = RenderMode.VOLUME
        worker._set_current_level_index(int(coarse_level))
        applied_context = facade.build_level_context(
            source=source,
            level=int(coarse_level),
            step=step_tuple,
        )
        facade.apply_volume_metadata(source, applied_context)
        facade.apply_volume_level(
            source,
            applied_context,
        )
        extent = worker._volume_world_extents()
        if extent is None:
            raise RuntimeError("volume bootstrap failed to determine world extents")
        _configure_camera_for_mode(worker)
        _emit_current_camera_pose(worker, reason)
        worker._set_current_level_index(int(prev_level))
        worker._viewport_state.mode = prev_mode

    worker._viewport_state.mode = original_mode
    if original_camera is not None:
        worker.view.camera = original_camera
    _configure_camera_for_mode(worker)


def _enter_volume_mode(worker: EGLRendererWorker) -> None:
    if worker._viewport_state.mode is RenderMode.VOLUME:
        return

    if worker._level_intent_callback is None:
        raise RuntimeError("level intent callback required for volume mode")

    facade = ViewerBootstrapInterface(worker)
    source = worker._ensure_scene_source()
    requested_level = _coarsest_level_index(source)
    assert requested_level is not None and requested_level >= 0, "Volume mode requires a multiscale level"
    selected_level = facade.resolve_volume_intent_level(source, int(requested_level))

    decision = lod.LevelDecision(
        desired_level=int(requested_level),
        selected_level=int(selected_level),
        reason="enter-volume",
        timestamp=time.perf_counter(),
        oversampling={},
    )
    spec = ledger_dims_spec(worker._ledger)
    assert spec is not None, "dims spec required before entering volume"
    base_step = spec.current_step
    last_step = ledger_step(worker._ledger)
    if last_step is not None:
        base_step = last_step
    step_tuple = dims_spec_clamp_step(spec, int(selected_level), base_step)
    context = facade.build_level_context(
        source=source,
        level=int(selected_level),
        step=step_tuple,
    )
    facade.apply_volume_metadata(source, context)

    worker._viewport_state.mode = RenderMode.VOLUME

    descriptor = source.level_descriptors[int(context.level)]
    shape_tuple = tuple(int(dim) for dim in descriptor.shape)

    intent = LevelSwitchIntent(
        desired_level=int(requested_level),
        selected_level=int(context.level),
        reason="enter-volume",
        previous_level=int(worker._current_level_index()),
        oversampling={},
        timestamp=decision.timestamp,
        zoom_ratio=None,
        lock_level=worker._lock_level,
        mode=worker.viewport_state.mode,
        plane_state=deepcopy(worker.viewport_state.plane),
        volume_state=deepcopy(worker.viewport_state.volume),
        level_shape=shape_tuple,
    )
    logger.info(
        "intent.level_switch: prev=%d target=%d reason=%s",
        int(worker._current_level_index()),
        int(context.level),
        intent.reason,
    )
    requested = worker._viewport_runner.request_level(int(context.level))
    if requested and worker._level_intent_callback is not None:
        worker._level_intent_callback(intent)

    volume_pose_cached = False
    if worker._ledger.get("camera_volume", "main", "center") is not None:
        center_entry = worker._ledger.get("camera_volume", "main", "center")
        angles_entry = worker._ledger.get("camera_volume", "main", "angles")
        distance_entry = worker._ledger.get("camera_volume", "main", "distance")
        fov_entry = worker._ledger.get("camera_volume", "main", "fov")
        volume_pose_cached = all(
            entry is not None and entry.value is not None
            for entry in (center_entry, angles_entry, distance_entry, fov_entry)
        )

    _configure_camera_for_mode(worker)
    if not volume_pose_cached:
        _emit_current_camera_pose(worker, "enter-3d")
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "toggle.enter_3d: requested_level=%d selected_level=%d",
            int(requested_level),
            int(selected_level),
        )
    worker._request_encoder_idr()
    worker._mark_render_tick_needed()


def _exit_volume_mode(worker: EGLRendererWorker) -> None:
    if worker._viewport_state.mode is not RenderMode.VOLUME:
        return
    facade = ViewerBootstrapInterface(worker)
    spec = ledger_dims_spec(worker._ledger)
    assert spec is not None, "plane restore requires dims spec"
    level = int(spec.current_level)
    step_tuple = tuple(int(v) for v in spec.current_step)
    if logger.isEnabledFor(logging.INFO):
        logger.info("toggle.exit_3d: restore level=%s step=%s", level, step_tuple)

    source = worker._ensure_scene_source()
    context = facade.build_level_context(
        source=source,
        level=int(level),
        step=step_tuple,
    )
    facade.apply_plane_metadata(source, context)
    descriptor = source.level_descriptors[int(context.level)]
    shape_tuple = tuple(int(dim) for dim in descriptor.shape)

    intent = LevelSwitchIntent(
        desired_level=int(level),
        selected_level=int(context.level),
        reason="exit-volume",
        previous_level=int(worker._current_level_index()),
        oversampling={},
        timestamp=time.perf_counter(),
        mode=worker.viewport_state.mode,
        plane_state=deepcopy(worker.viewport_state.plane),
        volume_state=deepcopy(worker.viewport_state.volume),
        level_shape=shape_tuple,
    )

    callback = worker._level_intent_callback
    if callback is None:
        logger.debug("plane restore intent dropped (no callback)")
        return

    requested = worker._viewport_runner.request_level(int(context.level))
    if requested:
        callback(intent)
    worker._mark_render_tick_needed()


def _apply_camera_reset(worker: EGLRendererWorker, cam) -> None:
    reset_worker_camera(worker, cam)


def _emit_current_camera_pose(worker: EGLRendererWorker, reason: str) -> None:
    """Emit the active camera pose for ledger sync."""

    cam = worker.view.camera if worker.view is not None else None
    if cam is None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("pose.emit skipped (no active camera) reason=%s", reason)
        return

    _emit_pose_from_camera(worker, cam, reason)


def _emit_pose_from_camera(worker: EGLRendererWorker, camera, reason: str) -> None:
    """Emit the pose derived from ``camera`` without mutating the render camera."""

    callback = worker._camera_pose_callback
    if callback is None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("pose.emit skipped (no callback) reason=%s", reason)
        return

    target = "main"
    base_seq = max(int(worker._pose_seq), int(worker._max_camera_command_seq))
    next_seq = base_seq + 1
    pose = _pose_from_camera(worker, camera, target, int(next_seq))
    if pose is None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("pose.emit skipped (no pose) reason=%s seq=%d", reason, int(worker._pose_seq))
        return

    if pose.angles is not None or pose.distance is not None:
        volume_key = (
            tuple(float(v) for v in pose.center) if pose.center is not None else None,
            tuple(float(v) for v in pose.angles) if pose.angles is not None else None,
            float(pose.distance) if pose.distance is not None else None,
            float(pose.fov) if pose.fov is not None else None,
        )
        if worker._last_volume_pose == volume_key:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("pose.emit skipped (unchanged volume pose) reason=%s", reason)
            return
        worker._last_volume_pose = volume_key
    else:
        plane_key = (
            tuple(float(v) for v in pose.center) if pose.center is not None else None,
            float(pose.zoom) if pose.zoom is not None else None,
            tuple(float(v) for v in pose.rect) if pose.rect is not None else None,
        )
        if worker._last_plane_pose == plane_key:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("pose.emit skipped (unchanged plane pose) reason=%s", reason)
            return
        worker._last_plane_pose = plane_key

    worker._pose_seq = int(next_seq)

    if logger.isEnabledFor(logging.INFO):
        mode = "volume" if pose.angles is not None or pose.distance is not None else "plane"
        rect = pose.rect if pose.rect is not None else None
        logger.info(
            "pose.emit: seq=%d reason=%s mode=%s rect=%s center=%s zoom=%s",
            int(worker._pose_seq),
            reason,
            mode,
            str(rect),
            str(pose.center),
            str(pose.zoom),
        )

    callback(pose)


def _pose_from_camera(
    worker: EGLRendererWorker,
    camera,
    target: str,
    seq: int,
) -> Optional[CameraPoseApplied]:
    """Convert camera state into ``CameraPoseApplied`` dataclass."""

    def _center_tuple(cam) -> Optional[tuple[float, ...]]:
        center_value = getattr(cam, "center", None)
        if center_value is None:
            return None
        return tuple(float(component) for component in center_value)

    if camera is None:
        return None

    if isinstance(camera, scene.cameras.TurntableCamera):
        center_tuple = _center_tuple(camera)
        distance_val = float(camera.distance)
        fov_val = float(camera.fov)
        azimuth = float(camera.azimuth)
        elevation = float(camera.elevation)
        roll = float(camera.roll)
        volume_state = worker._viewport_state.volume
        volume_state.update_pose(
            center=center_tuple,
            angles=(azimuth, elevation, roll),
            distance=distance_val,
            fov=fov_val,
        )
        return CameraPoseApplied(
            target=str(target or "main"),
            command_seq=int(seq),
            center=center_tuple,
            zoom=None,
            angles=(azimuth, elevation, roll),
            distance=distance_val,
            fov=fov_val,
            rect=None,
        )

    if isinstance(camera, PanZoomCamera):
        center_tuple = _center_tuple(camera)
        rect_obj = camera.rect
        if rect_obj is None:
            rect_tuple: Optional[tuple[float, float, float, float]] = None
        else:
            rect_tuple = (
                float(rect_obj.left),
                float(rect_obj.bottom),
                float(rect_obj.width),
                float(rect_obj.height),
            )
        zoom_val = float(camera.zoom_factor)
        plane_state = worker._viewport_state.plane
        update_kwargs: dict[str, Any] = {"zoom": zoom_val}
        if rect_tuple is not None:
            update_kwargs["rect"] = rect_tuple
        if center_tuple is not None:
            update_kwargs["center"] = (float(center_tuple[0]), float(center_tuple[1]))
        plane_state.update_pose(**update_kwargs)
        return CameraPoseApplied(
            target=str(target or "main"),
            command_seq=int(seq),
            center=center_tuple,
            zoom=zoom_val,
            angles=None,
            distance=None,
            fov=None,
            rect=rect_tuple,
        )

    return None


def _snapshot_camera_pose(
    worker: EGLRendererWorker,
    target: str,
    command_seq: int,
) -> Optional[CameraPoseApplied]:
    view = worker.view
    if view is None:
        return None
    return _pose_from_camera(worker, view.camera, target, command_seq)


def _current_panzoom_rect(worker: EGLRendererWorker) -> Optional[tuple[float, float, float, float]]:
    view = worker.view
    if view is None:
        return None
    cam = view.camera
    if not isinstance(cam, PanZoomCamera):
        return None
    rect = cam.rect
    if rect is None:
        return None
    return (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))


__all__ = [
    "_apply_camera_reset",
    "_bootstrap_camera_pose",
    "_coarsest_level_index",
    "_configure_camera_for_mode",
    "_current_panzoom_rect",
    "_emit_current_camera_pose",
    "_emit_pose_from_camera",
    "_enter_volume_mode",
    "_exit_volume_mode",
    "_frame_volume_camera",
    "_pose_from_camera",
    "_snapshot_camera_pose",
]
