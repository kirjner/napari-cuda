"""Camera lifecycle helpers for the EGL render worker."""

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from typing import Any, Optional, TYPE_CHECKING

from vispy import scene  # type: ignore
from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.data.roi import plane_scale_for_level, plane_wh_for_level
from napari_cuda.server.runtime.camera import CameraPoseApplied
from napari_cuda.server.runtime.core import ledger_step, reset_worker_camera
from napari_cuda.server.runtime.ipc import LevelSwitchIntent
from napari_cuda.server.runtime.viewport import RenderMode, PlaneState, VolumeState
from napari_cuda.server.runtime.worker import level_policy
from napari_cuda.server.runtime.worker.snapshots import (
    apply_plane_metadata,
    apply_volume_level,
    apply_volume_metadata,
)

if TYPE_CHECKING:
    from napari_cuda.server.data.zarr_source import ZarrSceneSource
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


def _coarsest_level_index(source: "ZarrSceneSource") -> Optional[int]:
    descriptors = source.level_descriptors
    if not descriptors:
        return None
    return len(descriptors) - 1


def _frame_volume_camera(worker: "EGLRendererWorker", w: float, h: float, d: float) -> None:
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


def _configure_camera_for_mode(worker: "EGLRendererWorker") -> None:
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
    worker: "EGLRendererWorker",
    mode: RenderMode,
    source: Optional["ZarrSceneSource"],
    *,
    reason: str,
) -> None:
    """Frame and emit an initial pose for the requested render mode."""

    if worker.view is None:
        return

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
        worker._viewport_state.mode = RenderMode.VOLUME
        worker._set_current_level_index(int(coarse_level))
        applied_context = lod.build_level_context(
            lod.LevelDecision(
                desired_level=int(coarse_level),
                selected_level=int(coarse_level),
                reason="bootstrap-volume",
                timestamp=time.perf_counter(),
                oversampling={},
                downgraded=False,
            ),
            source=source,
            prev_level=int(prev_level),
            last_step=prev_step,
        )
        apply_volume_metadata(worker, source, applied_context)
        apply_volume_level(
            worker,
            source,
            applied_context,
            downgraded=False,
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


def _enter_volume_mode(worker: "EGLRendererWorker") -> None:
    if worker._viewport_state.mode is RenderMode.VOLUME:
        return

    if worker._level_intent_callback is None:
        raise RuntimeError("level intent callback required for volume mode")

    source = worker._ensure_scene_source()
    requested_level = _coarsest_level_index(source)
    assert requested_level is not None and requested_level >= 0, "Volume mode requires a multiscale level"
    selected_level, downgraded = level_policy.resolve_volume_intent_level(
        worker,
        source,
        int(requested_level),
    )
    worker._viewport_state.volume.downgraded = bool(downgraded)

    decision = lod.LevelDecision(
        desired_level=int(requested_level),
        selected_level=int(selected_level),
        reason="ndisplay-3d",
        timestamp=time.perf_counter(),
        oversampling={},
        downgraded=bool(downgraded),
    )
    last_step = ledger_step(worker._ledger)
    context = lod.build_level_context(
        decision,
        source=source,
        prev_level=int(worker._current_level_index()),
        last_step=last_step,
    )
    apply_volume_metadata(worker, source, context)

    worker._viewport_state.mode = RenderMode.VOLUME

    intent = LevelSwitchIntent(
        desired_level=int(requested_level),
        selected_level=int(context.level),
        reason="ndisplay-3d",
        previous_level=int(worker._current_level_index()),
        context=context,
        oversampling={},
        timestamp=decision.timestamp,
        downgraded=bool(downgraded),
        zoom_ratio=None,
        lock_level=worker._lock_level,
        mode=worker.viewport_state.mode,
        plane_state=deepcopy(worker.viewport_state.plane),
        volume_state=deepcopy(worker.viewport_state.volume),
    )
    logger.info(
        "intent.level_switch: prev=%d target=%d reason=%s downgraded=%s",
        int(worker._current_level_index()),
        int(context.level),
        intent.reason,
        intent.downgraded,
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
            "toggle.enter_3d: requested_level=%d selected_level=%d downgraded=%s",
            int(requested_level),
            int(selected_level),
            bool(downgraded),
        )
    worker._request_encoder_idr()
    worker._mark_render_tick_needed()


def _exit_volume_mode(worker: "EGLRendererWorker") -> None:
    if worker._viewport_state.mode is not RenderMode.VOLUME:
        return
    worker._viewport_state.mode = RenderMode.PLANE
    lvl_entry = worker._ledger.get("view_cache", "plane", "level")
    step_entry = worker._ledger.get("view_cache", "plane", "step")
    assert lvl_entry is not None, "plane restore requires view_cache.plane.level"
    assert step_entry is not None, "plane restore requires view_cache.plane.step"
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "toggle.exit_3d: restore level=%s step=%s", str(lvl_entry.value), str(step_entry.value)
        )
    source = worker._ensure_scene_source()
    lvl_idx = int(lvl_entry.value)
    plane_entry = worker._ledger.get("viewport", "plane", "state")
    assert plane_entry is not None and isinstance(plane_entry.value, dict), "plane camera cache missing viewport state"
    plane_state = PlaneState(**dict(plane_entry.value))  # type: ignore[arg-type]
    worker._viewport_state.plane = PlaneState(**dict(plane_entry.value))
    rect_pose = plane_state.pose.rect
    assert rect_pose is not None, "plane camera cache missing rect"
    rect = tuple(float(v) for v in rect_pose)
    step_tuple = tuple(int(v) for v in step_entry.value)

    view = worker.view
    if view is not None:
        view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
        cam = view.camera
        sy, sx = plane_scale_for_level(source, lvl_idx)
        h_full, w_full = plane_wh_for_level(source, lvl_idx)
        world_w = float(w_full) * float(max(1e-12, sx))
        world_h = float(h_full) * float(max(1e-12, sy))
        cam.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
        cam.rect = Rect(*rect)
        if plane_state.pose.center is not None:
            cx, cy = plane_state.pose.center
            cam.center = (float(cx), float(cy), 0.0)  # type: ignore[attr-defined]
        if plane_state.pose.zoom is not None:
            cam.zoom = float(plane_state.pose.zoom)

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
        prev_level=int(worker._current_level_index()),
        last_step=step_tuple,
    )
    apply_plane_metadata(worker, source, context)
    intent = LevelSwitchIntent(
        desired_level=int(lvl_idx),
        selected_level=int(context.level),
        reason="ndisplay-2d",
        previous_level=int(worker._current_level_index()),
        context=context,
        oversampling={},
        timestamp=decision.timestamp,
        downgraded=False,
        mode=worker.viewport_state.mode,
        plane_state=deepcopy(worker.viewport_state.plane),
        volume_state=deepcopy(worker.viewport_state.volume),
    )

    callback = worker._level_intent_callback
    if callback is None:
        logger.debug("plane restore intent dropped (no callback)")
        return

    requested = worker._viewport_runner.request_level(int(context.level))
    if requested:
        callback(intent)
    worker._mark_render_tick_needed()


def _apply_camera_reset(worker: "EGLRendererWorker", cam) -> None:
    reset_worker_camera(worker, cam)


def _emit_current_camera_pose(worker: "EGLRendererWorker", reason: str) -> None:
    """Emit the active camera pose for ledger sync."""

    cam = worker.view.camera if worker.view is not None else None
    if cam is None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("pose.emit skipped (no active camera) reason=%s", reason)
        return

    _emit_pose_from_camera(worker, cam, reason)


def _emit_pose_from_camera(worker: "EGLRendererWorker", camera, reason: str) -> None:
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
    worker: "EGLRendererWorker",
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
    worker: "EGLRendererWorker",
    target: str,
    command_seq: int,
) -> Optional[CameraPoseApplied]:
    view = worker.view
    if view is None:
        return None
    return _pose_from_camera(worker, view.camera, target, command_seq)


def _current_panzoom_rect(worker: "EGLRendererWorker") -> Optional[tuple[float, float, float, float]]:
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
    "_configure_camera_for_mode",
    "_coarsest_level_index",
    "_current_panzoom_rect",
    "_emit_current_camera_pose",
    "_emit_pose_from_camera",
    "_enter_volume_mode",
    "_exit_volume_mode",
    "_frame_volume_camera",
    "_pose_from_camera",
    "_snapshot_camera_pose",
]
