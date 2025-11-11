from __future__ import annotations

import pytest

from napari_cuda.server.control import state_reducers as reducers
from napari_cuda.server.scene import blocks as scene_blocks
from napari_cuda.server.scene.blocks import (
    axes_from_payload,
    camera_block_from_payload,
    index_block_from_payload,
    lod_block_from_payload,
    view_block_from_payload,
)
from napari_cuda.server.scene.viewport import PlaneState, RenderMode, VolumeState
from napari_cuda.server.ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import dims_spec_from_payload


def _enable_block_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reducers, "ENABLE_VIEW_AXES_INDEX_BLOCKS", True, raising=False)
    monkeypatch.setattr(scene_blocks, "ENABLE_VIEW_AXES_INDEX_BLOCKS", True, raising=False)


def _bootstrap_ledger() -> ServerStateLedger:
    ledger = ServerStateLedger()
    reducers.reduce_bootstrap_state(
        ledger,
        step=(1, 0, 2),
        axis_labels=("z", "y", "x"),
        order=(2, 1, 0),
        level_shapes=((64, 32, 16), (32, 16, 8)),
        levels=(
            {"index": 0, "shape": [64, 32, 16]},
            {"index": 1, "shape": [32, 16, 8]},
        ),
        current_level=1,
        ndisplay=2,
        origin="test.bootstrap",
    )
    return ledger


def _current_spec(ledger: ServerStateLedger):
    entry = ledger.get("dims", "main", "dims_spec")
    assert entry is not None and entry.value is not None
    spec = dims_spec_from_payload(entry.value)
    assert spec is not None
    return spec


def _assert_blocks_match_spec(ledger: ServerStateLedger) -> None:
    spec = _current_spec(ledger)

    view_entry = ledger.get("view", "main", "state")
    assert view_entry is not None and view_entry.value is not None
    view_block = view_block_from_payload(view_entry.value)
    expected_mode = "volume" if spec.ndisplay >= 3 else "plane"
    assert view_block.mode == expected_mode
    assert view_block.displayed_axes == spec.displayed
    assert view_block.ndim == spec.ndim

    axes_entry = ledger.get("axes", "main", "state")
    assert axes_entry is not None and axes_entry.value is not None
    axes_block = axes_from_payload(axes_entry.value)
    assert len(axes_block.axes) == len(spec.axes)
    level_idx = int(spec.current_level)
    for axis_block, spec_axis in zip(axes_block.axes, spec.axes):
        assert axis_block.axis_id == spec_axis.index
        assert axis_block.label == spec_axis.label
        assert axis_block.role == spec_axis.role
        assert axis_block.displayed is spec_axis.displayed
        extent = spec_axis.per_level_world[level_idx]
        assert axis_block.world_extent.start == extent.start
        assert axis_block.world_extent.stop == extent.stop
        assert axis_block.world_extent.step == extent.step
        assert axis_block.margin_left_world == spec_axis.margin_left_world
        assert axis_block.margin_right_world == spec_axis.margin_right_world

    index_entry = ledger.get("index", "main", "cursor")
    assert index_entry is not None and index_entry.value is not None
    index_block = index_block_from_payload(index_entry.value)
    assert index_block.value == spec.current_step

    lod_entry = ledger.get("lod", "main", "state")
    assert lod_entry is not None and lod_entry.value is not None
    lod_block = lod_block_from_payload(lod_entry.value)
    assert lod_block.level == spec.current_level
    assert lod_block.roi is None
    assert lod_block.policy is None


def test_view_axes_index_lod_blocks_track_dims_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_block_writes(monkeypatch)
    ledger = _bootstrap_ledger()
    _assert_blocks_match_spec(ledger)

    reducers.reduce_dims_update(
        ledger,
        axis=0,
        prop="index",
        value=3,
        origin="test.dims",
    )
    _assert_blocks_match_spec(ledger)

    reducers.reduce_view_update(
        ledger,
        ndisplay=3,
        displayed=(0, 1, 2),
        origin="test.view",
    )
    _assert_blocks_match_spec(ledger)

    reducers.reduce_level_update(
        ledger,
        level=0,
        step=(2, 1, 0),
        level_shape=(64, 32, 16),
        origin="test.level",
        mode=RenderMode.PLANE,
    )
    _assert_blocks_match_spec(ledger)


def test_lod_block_updates_when_level_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_block_writes(monkeypatch)
    ledger = _bootstrap_ledger()

    reducers.reduce_level_update(
        ledger,
        level=1,
        step=(1, 1, 1),
        level_shape=(32, 16, 8),
        origin="test.level",
        plane_state=PlaneState(),
        volume_state=VolumeState(),
    )
    lod_entry = ledger.get("lod", "main", "state")
    assert lod_entry is not None and lod_entry.value is not None
    lod_block = lod_block_from_payload(lod_entry.value)
    assert lod_block.level == 1


def test_camera_block_matches_plane_and_volume_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_block_writes(monkeypatch)
    ledger = _bootstrap_ledger()

    reducers.reduce_camera_update(
        ledger,
        center=(12.0, 24.0),
        zoom=2.5,
        origin="test.camera",
    )
    entry = ledger.get("camera", "main", "state")
    assert entry is not None and entry.value is not None
    block = camera_block_from_payload(entry.value)
    assert tuple(block.plane.center or ()) == (12.0, 24.0)
    assert block.plane.zoom == 2.5

    reducers.reduce_camera_update(
        ledger,
        center=(1.0, 2.0, 3.0),
        angles=(30.0, 10.0, 5.0),
        distance=15.0,
        fov=45.0,
        origin="test.camera",
    )
    entry = ledger.get("camera", "main", "state")
    assert entry is not None and entry.value is not None
    block = camera_block_from_payload(entry.value)
    assert tuple(block.volume.center or ()) == (1.0, 2.0, 3.0)
    assert tuple(block.volume.angles or ()) == (30.0, 10.0, 5.0)
    assert block.volume.distance == 15.0
    assert block.volume.fov == 45.0
