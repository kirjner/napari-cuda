"""Tests for server scene intent helpers."""

from __future__ import annotations

import threading

import pytest

from napari_cuda.server.server_scene import create_server_scene_data
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server import server_scene_intents as intents


def _lock() -> threading.Lock:
    return threading.Lock()


def test_apply_dims_intent_clamps_to_range():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(1, 2, 3))
    meta = {"ndim": 3, "range": [(0, 5), (0, 10), (0, 4)]}
    step = intents.apply_dims_intent(scene, _lock(), meta, axis=1, step_delta=15, set_value=None)
    assert step == [1, 10, 3]
    assert scene.latest_state.current_step == (1, 10, 3)


def test_apply_dims_intent_with_label_axis():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(0, 0, 0))
    meta = {"ndim": 3, "order": ["t", "z", "y"]}
    step = intents.apply_dims_intent(scene, _lock(), meta, axis="z", step_delta=None, set_value=7)
    assert step == [0, 7, 0]
    assert scene.latest_state.current_step == (0, 7, 0)


def test_volume_helpers_update_scene_state():
    scene = create_server_scene_data()
    scene.latest_state = ServerSceneState(current_step=(0,), volume_mode=None)
    lock = _lock()

    intents.update_volume_mode(scene, lock, "mip")
    assert scene.volume_state["mode"] == "mip"
    assert scene.latest_state.volume_mode == "mip"

    intents.update_volume_clim(scene, lock, 1.0, 5.0)
    assert scene.volume_state["clim"] == [1.0, 5.0]
    assert scene.latest_state.volume_clim == (1.0, 5.0)

    intents.update_volume_colormap(scene, lock, "viridis")
    assert scene.volume_state["colormap"] == "viridis"
    assert scene.latest_state.volume_colormap == "viridis"

    intents.update_volume_opacity(scene, lock, 0.25)
    assert scene.volume_state["opacity"] == 0.25
    assert scene.latest_state.volume_opacity == 0.25

    intents.update_volume_sample_step(scene, lock, 0.75)
    assert scene.volume_state["sample_step"] == 0.75
    assert scene.latest_state.volume_sample_step == 0.75


@pytest.mark.parametrize(
    "alpha,expected",
    [(0.5, 0.5), (-1.0, 0.0), (5.0, 1.0), ("0.2", 0.2)],
)
def test_clamp_opacity(alpha, expected):
    assert intents.clamp_opacity(alpha) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,expected",
    [(0.5, 0.5), (0.05, 0.1), (8.0, 4.0), ("1.5", 1.5)],
)
def test_clamp_sample_step(value, expected):
    assert intents.clamp_sample_step(value) == pytest.approx(expected)


def test_clamp_level_uses_level_count():
    levels = [{"path": "a"}, {"path": "b"}]
    assert intents.clamp_level(5, levels) == 1
    assert intents.clamp_level(-2, levels) == 0
    assert intents.clamp_level("1", levels) == 1
    assert intents.clamp_level("bad", levels) is None


def test_apply_layer_intent_updates_state():
    scene = create_server_scene_data()
    lock = _lock()

    applied = intents.apply_layer_intent(
        scene,
        lock,
        layer_id="layer-0",
        prop="opacity",
        value=0.4,
    )

    assert applied == {"opacity": 0.4}
    assert scene.layer_controls["layer-0"].opacity == 0.4
    assert scene.latest_state.layer_updates == {"layer-0": {"opacity": 0.4}}


def test_apply_layer_intent_rejects_invalid_property():
    scene = create_server_scene_data()
    lock = _lock()

    with pytest.raises(KeyError):
        intents.apply_layer_intent(scene, lock, layer_id="layer-0", prop="unknown", value="x")
