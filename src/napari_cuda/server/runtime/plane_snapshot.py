"""Plane snapshot application helpers."""

from __future__ import annotations

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


@dataclass(frozen=True)
class PlaneApplyResult:
    """Outcome of applying a plane snapshot."""

    level: int
    roi: SliceROI
    aligned_roi: SliceROI
    chunk_shape: Optional[Tuple[int, int]]
    width_px: int
    height_px: int


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

    rect = snapshot.rect
    if rect is not None and len(rect) >= 4:
        cam.rect = Rect(
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        )
        plane_state.camera_rect = (
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        )

    center = snapshot.center
    if center is not None and len(center) >= 2:
        cam.center = (
            float(center[0]),
            float(center[1]),
        )
        plane_state.camera_center = (float(center[0]), float(center[1]))

    if snapshot.zoom is not None and hasattr(cam, "zoom"):
        cam.zoom = float(snapshot.zoom)
        plane_state.camera_zoom = float(snapshot.zoom)


def apply_slice_level(
    worker: Any,
    source: Any,
    applied: lod.LevelContext,
) -> PlaneApplyResult:
    """Load the plane slice for ``applied`` and update worker metadata."""

    layer = getattr(worker, "_napari_layer", None)
    sy, sx = applied.scale_yx

    if layer is not None:
        layer.scale = (float(sy), float(sx))

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
        view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))

    worker._layer_logger.log(
        enabled=worker._log_layer_debug,
        mode="slice",
        level=applied.level,
        z_index=worker._z_index,
        shape=(int(height_px), int(width_px)),
        contrast=applied.contrast,
        downgraded=worker.viewport_state.volume.downgraded,  # type: ignore[attr-defined]
    )

    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]
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
    "apply_plane_camera_pose",
    "apply_slice_level",
]
