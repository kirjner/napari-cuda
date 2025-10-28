from __future__ import annotations

import pytest
from vispy.geometry import Rect

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.viewport import PlaneState
from napari_cuda.server.runtime.viewport.plane_ops import (
    assign_pose_from_snapshot,
    apply_pose_to_camera,
    mark_slice_applied,
)
from napari_cuda.server.runtime.scene_types import SliceROI


def test_assign_pose_from_snapshot_prefers_snapshot() -> None:
    state = PlaneState()
    state.update_pose(rect=(0.0, 0.0, 100.0, 100.0), center=(50.0, 50.0), zoom=1.5)

    snapshot = RenderLedgerSnapshot(
        plane_rect=(10.0, 20.0, 110.0, 120.0),
        plane_center=(40.0, 60.0),
        plane_zoom=2.0,
    )

    rect, center, zoom = assign_pose_from_snapshot(state, snapshot)

    assert rect == (10.0, 20.0, 110.0, 120.0)
    assert center == (40.0, 60.0)
    assert zoom == 2.0
    assert state.pose.rect == rect
    assert state.pose.center == center
    assert state.pose.zoom == zoom


def test_assign_pose_from_snapshot_falls_back_to_cached_pose() -> None:
    state = PlaneState()
    state.update_pose(rect=(0.0, 0.0, 115.0, 215.0), center=(57.0, 107.0), zoom=0.75)

    snapshot = RenderLedgerSnapshot()

    rect, center, zoom = assign_pose_from_snapshot(state, snapshot)

    assert rect == (0.0, 0.0, 115.0, 215.0)
    assert center == (57.0, 107.0)
    assert zoom == 0.75


def test_mark_slice_applied_updates_state() -> None:
    state = PlaneState()
    roi = SliceROI(2, 6, 4, 12)

    mark_slice_applied(
        state,
        level=2,
        step=(3, 1, 0),
        roi=roi,
        roi_signature=(1, 5, 0, 3),
    )

    assert state.applied_level == 2
    assert state.applied_step == (3, 1, 0)
    assert state.applied_roi == roi
    assert state.applied_roi_signature == (1, 5, 0, 3)


def test_assign_pose_missing_rect_raises() -> None:
    state = PlaneState()
    state.clear_pose()
    snapshot = RenderLedgerSnapshot()

    with pytest.raises(AssertionError, match="plane snapshot missing rect"):
        assign_pose_from_snapshot(state, snapshot)


def test_apply_pose_to_camera_sets_panzoom_fields() -> None:
    state = PlaneState()
    state.update_pose(rect=(1.0, 2.0, 3.0, 4.0), center=(10.0, 20.0), zoom=1.5)
    snapshot = RenderLedgerSnapshot(
        plane_rect=(5.0, 6.0, 7.0, 8.0),
        plane_center=(30.0, 40.0),
        plane_zoom=4.5,
    )
    rect, center, zoom = assign_pose_from_snapshot(state, snapshot)

    class _Camera:
        def __init__(self) -> None:
            self.rect = ()
            self.center = ()
            self.zoom = 0.0

    cam = _Camera()
    apply_pose_to_camera(cam, rect=rect, center=center, zoom=zoom)

    assert cam.rect == Rect(5.0, 6.0, 7.0, 8.0)
    assert cam.center == (30.0, 40.0)
    assert cam.zoom == 4.5
