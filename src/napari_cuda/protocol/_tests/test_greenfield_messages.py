from __future__ import annotations

import json

import pytest

from napari_cuda.protocol.greenfield.messages import (
    LAYER_REMOVE_TYPE,
    LAYER_UPDATE_TYPE,
    SCENE_SPEC_TYPE,
    LayerRemoveMessage,
    LayerSpec,
    LayerUpdateMessage,
    SceneSpec,
    SceneSpecMessage,
    StateUpdateMessage,
)


@pytest.fixture()
def sample_layer_dict() -> dict[str, object]:
    return {
        "layer_id": "layer-001",
        "layer_type": "image",
        "name": "demo",
        "ndim": 3,
        "shape": [32, 64, 128],
        "dtype": "float32",
        "axis_labels": ["z", "y", "x"],
        "scale": [2.0, 0.5, 0.5],
        "translate": [0.0, 10.0, 20.0],
        "channel_axis": 1,
        "channel_names": ["red", "green"],
        "contrast_limits": [0.0, 1.0],
        "render": {"mode": "mip", "colormap": "gray", "opacity": 0.85},
        "controls": {"visible": True, "opacity": 0.85, "blending": "opaque"},
        "multiscale": {
            "levels": [
                {"shape": [32, 64, 128], "downsample": [1.0, 1.0, 1.0], "path": "level_0"},
                {"shape": [16, 32, 64], "downsample": [1.0, 2.0, 2.0], "path": "level_1"},
            ],
            "current_level": 0,
        },
        "metadata": {"source": "zarr"},
        "extras": {"note": "integration"},
    }


def test_greenfield_layer_spec_round_trip(sample_layer_dict: dict[str, object]) -> None:
    spec = LayerSpec.from_dict(sample_layer_dict)
    assert spec.layer_id == "layer-001"
    assert spec.render is not None and spec.render.mode == "mip"
    assert spec.multiscale is not None and len(spec.multiscale.levels) == 2

    payload = spec.to_dict()
    assert payload["layer_id"] == "layer-001"
    assert payload["render"]["colormap"] == "gray"
    assert payload["multiscale"]["levels"][1]["downsample"] == [1.0, 2.0, 2.0]


def test_greenfield_scene_spec_message(sample_layer_dict: dict[str, object]) -> None:
    scene = SceneSpec(layers=[LayerSpec.from_dict(sample_layer_dict)])
    message = SceneSpecMessage(scene=scene, capabilities=["notify.layers"], timestamp=1.23)

    payload = message.to_dict()
    assert payload["type"] == SCENE_SPEC_TYPE
    assert payload["scene"]["layers"][0]["name"] == "demo"

    parsed = SceneSpecMessage.from_dict(payload)
    assert parsed.scene.layers[0].name == "demo"
    assert parsed.capabilities == ["notify.layers"]


def test_greenfield_layer_update_round_trip(sample_layer_dict: dict[str, object]) -> None:
    layer = LayerSpec.from_dict(sample_layer_dict)
    message = LayerUpdateMessage(layer=layer, partial=True, intent_seq=42, controls=layer.controls)

    payload = message.to_dict()
    assert payload["type"] == LAYER_UPDATE_TYPE
    assert payload["partial"] is True

    parsed = LayerUpdateMessage.from_dict(payload)
    assert parsed.layer.name == "demo"
    assert parsed.intent_seq == 42


def test_greenfield_layer_remove_round_trip() -> None:
    message = LayerRemoveMessage(layer_id="layer-001", reason="deleted")
    payload = message.to_dict()
    assert payload["type"] == LAYER_REMOVE_TYPE

    parsed = LayerRemoveMessage.from_dict(payload)
    assert parsed.layer_id == "layer-001"
    assert parsed.reason == "deleted"


def test_greenfield_state_update_round_trip() -> None:
    message = StateUpdateMessage(
        scope="dims",
        target="z",
        key="step",
        value=5,
        client_id="client-a",
        client_seq=10,
        server_seq=12,
        intent_seq=99,
    )

    payload = message.to_dict()
    assert payload["scope"] == "dims"
    assert payload["client_seq"] == 10

    parsed = StateUpdateMessage.from_dict(payload)
    assert parsed.scope == "dims"
    assert parsed.value == 5
    assert parsed.intent_seq == 99

    # Ensure JSON round-trip remains valid
    encoded = json.loads(message.to_json())
    assert encoded["key"] == "step"
