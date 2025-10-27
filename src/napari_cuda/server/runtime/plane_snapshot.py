"""Plane snapshot application helpers."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.data.roi import plane_wh_for_level
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.roi_math import (
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
    roi_chunk_signature,
)
from napari_cuda.server.runtime.scene_types import SliceROI

from .plane_loader import apply_plane_slice_roi, viewport_roi_for_level
from .state_structs import PlaneState, RenderMode
from .viewer_stage import apply_plane_metadata


@dataclass(frozen=True)
class PlaneApplyResult:
    """Outcome of applying a plane snapshot."""

    level: int
    roi: SliceROI
    aligned_roi: SliceROI
    chunk_shape: Optional[Tuple[int, int]]
    width_px: int
    height_px: int


def apply_plane_snapshot(
    worker: Any,
    source: Any,
    snapshot: RenderLedgerSnapshot,
) -> PlaneApplyResult:
    """Apply plane metadata, camera pose, and ROI from the snapshot."""

    prev_level = int(worker._current_level_index())  # type: ignore[attr-defined]
    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]
    snapshot_level = int(snapshot.current_level) if snapshot.current_level is not None else None
    target_level = snapshot_level if snapshot_level is not None else prev_level
    if plane_state.target_ndisplay < 3:
        target_level = int(plane_state.target_level)

    step_hint: Optional[tuple[int, ...]] = None
    if plane_state.target_ndisplay < 3 and plane_state.target_step is not None:
        step_hint = tuple(int(v) for v in plane_state.target_step)
    elif snapshot.current_step is not None:
        step_hint = tuple(int(v) for v in snapshot.current_step)
    if step_hint is None:
        recorded_step = worker._ledger_step()
        if recorded_step is not None:
            step_hint = tuple(int(v) for v in recorded_step)

    was_volume = worker.viewport_state.mode is RenderMode.VOLUME  # type: ignore[attr-defined]
    stage_prev_level = target_level if was_volume else prev_level

    decision = lod.LevelDecision(
        desired_level=int(target_level),
        selected_level=int(target_level),
        reason="direct",
        timestamp=time.perf_counter(),
        oversampling={},
        downgraded=False,
    )
    applied_context = lod.build_level_context(
        decision,
        source=source,
        prev_level=stage_prev_level,
        last_step=step_hint,
    )

    if was_volume:
        worker.viewport_state.mode = RenderMode.PLANE  # type: ignore[attr-defined]
        worker._configure_camera_for_mode()
        worker._last_dims_signature = None
        if hasattr(worker, "_last_plane_pose"):
            worker._last_plane_pose = None  # type: ignore[attr-defined]

    apply_plane_metadata(worker, source, applied_context)
    worker.viewport_state.volume.downgraded = False  # type: ignore[attr-defined]
    apply_plane_camera_pose(worker, snapshot)
    return apply_slice_level(worker, source, applied_context)


def apply_plane_camera_pose(
    worker: Any,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Apply plane camera pose from the snapshot to the active view."""

    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, PanZoomCamera):
        return

    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]

    rect_source = snapshot.plane_rect
    if rect_source is None and plane_state.pose.rect is not None:
        rect_source = plane_state.pose.rect
    assert rect_source is not None and len(rect_source) >= 4, "plane snapshot missing rect"
    rect_tuple = (
        float(rect_source[0]),
        float(rect_source[1]),
        float(rect_source[2]),
        float(rect_source[3]),
    )
    cam.rect = Rect(*rect_tuple)

    center_source = snapshot.plane_center
    if center_source is None and plane_state.pose.center is not None:
        center_source = plane_state.pose.center
    assert center_source is not None and len(center_source) >= 2, "plane snapshot missing center"
    center_tuple = (float(center_source[0]), float(center_source[1]))
    cam.center = center_tuple

    zoom_source = snapshot.plane_zoom
    if zoom_source is None and plane_state.pose.zoom is not None:
        zoom_source = plane_state.pose.zoom
    assert zoom_source is not None, "plane snapshot missing zoom"
    zoom_value = float(zoom_source)
    cam.zoom = zoom_value

    plane_state.update_pose(rect=rect_tuple, center=center_tuple, zoom=zoom_value)


def apply_slice_level(
    worker: Any,
    source: Any,
    applied: lod.LevelContext,
) -> PlaneApplyResult:
    """Load the plane slice for ``applied`` and update worker metadata."""

    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]
    layer = getattr(worker, "_napari_layer", None)
    sy, sx = applied.scale_yx

    if layer is not None:
        layer.depiction = "plane"
        layer.rendering = "mip"
        layer.scale = (float(sy), float(sx))
        if hasattr(layer, "_set_view_slice"):
            layer._set_view_slice()  # type: ignore[misc]
        clear_visual = getattr(worker, "_clear_visual", None)
        if callable(clear_visual):
            clear_visual()

    view = worker.view
    assert view is not None, "VisPy view must be initialised for 2D apply"

    full_h, full_w = plane_wh_for_level(source, int(applied.level))
    roi = viewport_roi_for_level(worker, source, int(applied.level))
    chunk_shape = chunk_shape_for_level(source, int(applied.level))
    aligned_roi = roi
    if worker._roi_align_chunks and chunk_shape is not None:
        aligned_roi = align_roi_to_chunk_grid(
            roi,
            chunk_shape,
            int(worker._roi_pad_chunks),
            height=full_h,
            width=full_w,
        )
    if worker.viewport_state.mode is not RenderMode.VOLUME:  # type: ignore[attr-defined]
        worker._emit_current_camera_pose("slice-apply")
    height_px, width_px = apply_plane_slice_roi(
        worker,
        source,
        int(applied.level),
        aligned_roi,
        update_contrast=not worker._sticky_contrast,
    )

    if not worker._preserve_view_on_switch:
        world_w = float(full_w) * float(max(1e-12, sx))
        world_h = float(full_h) * float(max(1e-12, sy))
        cached_rect = plane_state.pose.rect
        if cached_rect is None:
            view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
        else:
            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.INFO):
                logger.info("plane.slice preserve view using cached rect=%s", cached_rect)
            view.camera.rect = Rect(
                float(cached_rect[0]),
                float(cached_rect[1]),
                float(cached_rect[2]),
                float(cached_rect[3]),
            )
        cached_center = plane_state.pose.center
        if cached_center is not None and len(cached_center) >= 2:
            view.camera.center = (
                float(cached_center[0]),
                float(cached_center[1]),
            )
        cached_zoom = plane_state.pose.zoom
        if cached_zoom is not None:
            view.camera.zoom = float(cached_zoom)

    worker._layer_logger.log(
        enabled=worker._log_layer_debug,
        mode="slice",
        level=applied.level,
        z_index=worker._z_index,
        shape=(int(height_px), int(width_px)),
        contrast=applied.contrast,
        downgraded=worker.viewport_state.volume.downgraded,  # type: ignore[attr-defined]
    )

    plane_state.applied_level = int(applied.level)
    plane_state.applied_step = tuple(int(v) for v in applied.step)
    plane_state.applied_roi = aligned_roi
    plane_state.applied_roi_signature = roi_chunk_signature(aligned_roi, chunk_shape)

    runner = worker._viewport_runner
    if runner is not None:
        runner.mark_roi_applied(aligned_roi, chunk_shape=chunk_shape)

    return PlaneApplyResult(
        level=int(applied.level),
        roi=roi,
        aligned_roi=aligned_roi,
        chunk_shape=chunk_shape,
        width_px=int(width_px),
        height_px=int(height_px),
    )


__all__ = [
    "PlaneApplyResult",
    "apply_plane_snapshot",
    "apply_plane_camera_pose",
    "apply_slice_level",
]
