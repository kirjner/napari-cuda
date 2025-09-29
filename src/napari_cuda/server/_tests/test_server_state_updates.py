"""Tests for server state update helpers."""

from __future__ import annotations

import threading

import pytest

from napari_cuda.server.server_scene import (
    create_server_scene_data,
    prune_control_metadata,
)
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server import server_state_updates as updates


def _lock() -> threading.RLock:
    return threading.RLock()


def test_apply_dims_delta_clamps_to_range():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(1, 2, 3))
    meta = {"ndim": 3, "range": [(0, 5), (0, 10), (0, 4)]}
    step = updates.apply_dims_delta(scene, _lock(), meta, axis=1, step_delta=15, set_value=None)
    assert step == [1, 10, 3]
    assert scene.latest_state.current_step == (1, 10, 3)


def test_apply_dims_delta_with_label_axis():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(0, 0, 0))
    meta = {"ndim": 3, "order": ["t", "z", "y"]}
    step = updates.apply_dims_delta(scene, _lock(), meta, axis="z", step_delta=None, set_value=7)
    assert step == [0, 7, 0]
    assert scene.latest_state.current_step == (0, 7, 0)


def test_volume_helpers_update_scene_state():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(0,), volume_mode=None)
    lock = _lock()

    updates.update_volume_mode(scene, lock, "mip")
    assert scene.volume_state["mode"] == "mip"
    assert scene.latest_state.volume_mode == "mip"

    updates.update_volume_clim(scene, lock, 1.0, 5.0)
    assert scene.volume_state["clim"] == [1.0, 5.0]
    assert scene.latest_state.volume_clim == (1.0, 5.0)

    updates.update_volume_colormap(scene, lock, "viridis")
    assert scene.volume_state["colormap"] == "viridis"
    assert scene.latest_state.volume_colormap == "viridis"

    updates.update_volume_opacity(scene, lock, 0.25)
    assert scene.volume_state["opacity"] == 0.25
    assert scene.latest_state.volume_opacity == 0.25

    updates.update_volume_sample_step(scene, lock, 0.75)
    assert scene.volume_state["sample_step"] == 0.75
    assert scene.latest_state.volume_sample_step == 0.75


@pytest.mark.parametrize(
    "alpha,expected",
    [(0.5, 0.5), (-1.0, 0.0), (5.0, 1.0), ("0.2", 0.2)],
)
def test_clamp_opacity(alpha, expected):
    assert updates.clamp_opacity(alpha) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,expected",
    [(0.5, 0.5), (0.05, 0.1), (8.0, 4.0), ("1.5", 1.5)],
)
def test_clamp_sample_step(value, expected):
    assert updates.clamp_sample_step(value) == pytest.approx(expected)


def test_clamp_level_uses_level_count():
    levels = [{"path": "a"}, {"path": "b"}]
    assert updates.clamp_level(5, levels) == 1
    assert updates.clamp_level(-2, levels) == 0
    assert updates.clamp_level("1", levels) == 1
    assert updates.clamp_level("bad", levels) is None


def test_apply_layer_state_update_updates_state():
    scene = create_server_scene_data()
    lock = _lock()

    result = updates.apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-0",
        prop="opacity",
        value=0.4,
    )

    assert result is not None
    assert result.value == 0.4
    assert result.server_seq == 1
    assert scene.layer_controls["layer-0"].opacity == 0.4
    assert scene.latest_state.layer_updates == {"layer-0": {"opacity": 0.4}}
    meta = scene.control_meta[("layer", "layer-0", "opacity")]
    assert meta.last_server_seq == 1


@pytest.mark.parametrize("value", ["I Blue", "i blue", " I Blue "])
def test_apply_layer_state_update_normalizes_colormap(value):
    scene = create_server_scene_data()
    lock = _lock()

    result = updates.apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-0",
        prop="colormap",
        value=value,
    )

    assert result is not None
    assert result.value == "I Blue"
    assert scene.layer_controls["layer-0"].colormap == "I Blue"
    assert scene.latest_state.layer_updates == {"layer-0": {"colormap": "I Blue"}}


def test_apply_layer_state_update_rejects_stale_sequence():
    scene = create_server_scene_data()
    lock = _lock()

    first = updates.apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-0",
        prop="gamma",
        value=1.2,
        client_id="client-a",
        client_seq=5,
    )

    assert first is not None
    assert first.server_seq == 1

    stale = updates.apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-0",
        prop="gamma",
        value=0.9,
        client_id="client-a",
        client_seq=4,
    )

    assert stale is None
    # Latest state should remain unchanged
    assert scene.layer_controls["layer-0"].gamma == 1.2
    meta = scene.control_meta[("layer", "layer-0", "gamma")]
    assert meta.last_server_seq == 1


def test_apply_layer_state_update_rejects_invalid_property():
    scene = create_server_scene_data()
    lock = _lock()

    with pytest.raises(KeyError):
        updates.apply_layer_state_update(scene, lock, layer_id="layer-0", prop="unknown", value="x")


def test_apply_dims_state_update_tracks_metadata_for_index():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(0, 0, 0))
    lock = _lock()
    meta = {
        "ndim": 3,
        "order": ["z", "y", "x"],
        "axis_labels": ["z", "y", "x"],
        "sizes": [10, 10, 10],
        "range": [(0, 9), (0, 9), (0, 9)],
    }

    result = updates.apply_dims_state_update(
        scene,
        lock,
        meta,
        axis="z",
        prop="index",
        value=None,
        set_value=5,
        client_id="client-z",
        client_seq=3,
        interaction_id="drag-1",
        phase="update",
    )

    assert result is not None
    assert result.value == 5
    assert scene.latest_state.current_step[0] == 5
    meta_entry = scene.control_meta[("dims", "z", "index")]
    assert meta_entry.last_client_id == "client-z"
    assert meta_entry.last_client_seq == 3
    assert meta_entry.last_interaction_id == "drag-1"
    assert meta_entry.last_phase == "update"
    payload_meta = result.extras["meta"]
    assert payload_meta["ndim"] == 3
    assert payload_meta["order"] == ["z", "y", "x"]
    assert payload_meta["axis_labels"] == ["z", "y", "x"]
    assert payload_meta["sizes"] == [10, 10, 10]
    assert tuple(payload_meta["range"][0]) == (0, 9)


def test_apply_dims_state_update_handles_step_delta():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(10, 0, 0))
    lock = _lock()
    meta = {
        "ndim": 3,
        "order": ["z", "y", "x"],
        "axis_labels": ["z", "y", "x"],
        "sizes": [21, 10, 10],
        "range": [(0, 20), (0, 9), (0, 9)],
    }

    result = updates.apply_dims_state_update(
        scene,
        lock,
        meta,
        axis="z",
        prop="step",
        value=None,
        step_delta=3,
        client_id="client-z",
        client_seq=4,
        interaction_id="drag-2",
        phase="update",
    )

    assert result is not None
    assert result.value == 13
    assert scene.latest_state.current_step[0] == 13
    meta_entry = scene.control_meta[("dims", "z", "step")]
    assert meta_entry.last_client_seq == 4


def test_prune_control_metadata_removes_stale_layer_entries() -> None:
    scene = create_server_scene_data()
    lock = _lock()

    updates.apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-0",
        prop="opacity",
        value=0.4,
    )
    updates.apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-ghost",
        prop="opacity",
        value=0.9,
    )

    assert ("layer", "layer-ghost", "opacity") in scene.control_meta
    assert "layer-ghost" in scene.layer_controls

    prune_control_metadata(
        scene,
        layer_ids=["layer-0"],
        dims_meta={},
        current_step=None,
    )

    assert ("layer", "layer-0", "opacity") in scene.control_meta
    assert ("layer", "layer-ghost", "opacity") not in scene.control_meta
    assert "layer-ghost" not in scene.layer_controls


def test_prune_control_metadata_trims_removed_dims_axes() -> None:
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(0, 0))
    lock = _lock()
    meta = {"ndim": 2, "order": ["z", "t"], "range": [(0, 5), (0, 5)]}

    updates.apply_dims_state_update(
        scene,
        lock,
        meta,
        axis="z",
        prop="step",
        value=1,
    )
    updates.apply_dims_state_update(
        scene,
        lock,
        meta,
        axis="t",
        prop="step",
        value=2,
    )

    assert ("dims", "z", "step") in scene.control_meta
    assert ("dims", "t", "step") in scene.control_meta

    prune_control_metadata(
        scene,
        layer_ids=["layer-0"],
        dims_meta={"ndim": 1, "order": ["z"]},
        current_step=(0,),
    )

    assert ("dims", "z", "step") in scene.control_meta
    assert ("dims", "t", "step") not in scene.control_meta


def test_prune_control_metadata_uses_step_fallback_when_meta_missing() -> None:
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(0, 0))
    lock = _lock()
    meta = {"ndim": 2, "order": ["z", "t"], "range": [(0, 5), (0, 5)]}

    updates.apply_dims_state_update(
        scene,
        lock,
        meta,
        axis="z",
        prop="step",
        value=3,
    )
    updates.apply_dims_state_update(
        scene,
        lock,
        meta,
        axis="t",
        prop="step",
        value=4,
    )

    prune_control_metadata(
        scene,
        layer_ids=["layer-0"],
        dims_meta=None,
        current_step=(0, 0),
    )

    assert ("dims", "z", "step") in scene.control_meta
    assert ("dims", "t", "step") in scene.control_meta
