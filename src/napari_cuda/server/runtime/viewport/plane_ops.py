"""Helpers for mutating :class:`PlaneState` during slice application."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera

from napari_cuda.server.runtime.render_loop.apply.snapshots.build import (
    RenderLedgerSnapshot,
)
from napari_cuda.server.runtime.data import SliceROI

from .state import PlaneState


def assign_pose_from_snapshot(
    state: PlaneState,
    snapshot: RenderLedgerSnapshot,
) -> tuple[tuple[float, float, float, float], tuple[float, float], float]:
    """Merge snapshot pose data into the cached plane state."""

    rect_source = snapshot.plane_rect
    if rect_source is None and state.pose.rect is not None:
        rect_source = state.pose.rect
    assert rect_source is not None and len(rect_source) >= 4, "plane snapshot missing rect"
    rect = (
        float(rect_source[0]),
        float(rect_source[1]),
        float(rect_source[2]),
        float(rect_source[3]),
    )

    center_source = snapshot.plane_center
    if center_source is None and state.pose.center is not None:
        center_source = state.pose.center
    assert center_source is not None and len(center_source) >= 2, "plane snapshot missing center"
    center = (float(center_source[0]), float(center_source[1]))

    zoom_source = snapshot.plane_zoom
    if zoom_source is None and state.pose.zoom is not None:
        zoom_source = state.pose.zoom
    assert zoom_source is not None, "plane snapshot missing zoom"
    zoom = float(zoom_source)

    state.update_pose(rect=rect, center=center, zoom=zoom)
    return rect, center, zoom


def mark_slice_applied(
    state: PlaneState,
    *,
    level: int,
    step: Sequence[int],
    roi: SliceROI,
    roi_signature: Optional[tuple[int, int, int, int]],
) -> None:
    """Persist newly applied slice metadata on the plane state."""

    state.applied_level = int(level)
    state.applied_step = tuple(int(v) for v in step)
    state.applied_roi = roi
    state.applied_roi_signature = roi_signature


def apply_pose_to_camera(
    camera: PanZoomCamera,
    *,
    rect: tuple[float, float, float, float],
    center: tuple[float, float],
    zoom: float,
) -> None:
    """Apply the cached plane pose to a PanZoom camera."""

    camera.rect = Rect(*rect)
    camera.center = (float(center[0]), float(center[1]))
    camera.zoom = float(zoom)


__all__ = [
    "apply_pose_to_camera",
    "assign_pose_from_snapshot",
    "mark_slice_applied",
]
