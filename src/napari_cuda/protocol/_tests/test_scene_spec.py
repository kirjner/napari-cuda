import json

import pytest

from napari_cuda.protocol.messages import (
    LAYER_REMOVE_TYPE,
    LAYER_UPDATE_TYPE,
    SCENE_SPEC_TYPE,
    LayerRemoveMessage,
    LayerSpec,
    LayerUpdateMessage,
    SceneSpec,
    SceneSpecMessage,
    StreamProtocol,
)


@pytest.fixture()
def sample_layer_dict():
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
        "render": {
            "mode": "mip",
            "colormap": "gray",
            "opacity": 0.85,
        },
        "controls": {
            "visible": True,
            "opacity": 0.85,
            "blending": "opaque",
        },
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


def test_layer_spec_round_trip(sample_layer_dict):
    spec = LayerSpec.from_dict(sample_layer_dict)
    assert spec.layer_id == "layer-001"
    assert spec.axis_labels == ["z", "y", "x"]
    assert spec.render is not None
    assert spec.render.mode == "mip"
    assert spec.multiscale is not None
    assert len(spec.multiscale.levels) == 2

    payload = spec.to_dict()
    assert payload["layer_id"] == "layer-001"
    assert payload["render"]["colormap"] == "gray"
    assert payload["multiscale"]["levels"][1]["downsample"] == [1.0, 2.0, 2.0]
    assert payload["controls"]["opacity"] == pytest.approx(0.85)


def test_scene_spec_message(sample_layer_dict):
    scene = SceneSpec(layers=[LayerSpec.from_dict(sample_layer_dict)])
    msg = SceneSpecMessage(scene=scene, capabilities=["layer.update"], timestamp=1.23)

    payload = msg.to_dict()
    assert payload["type"] == SCENE_SPEC_TYPE
    assert payload["version"] >= 1
    assert payload["scene"]["layers"][0]["name"] == "demo"

    parsed = SceneSpecMessage.from_dict(payload)
    assert isinstance(parsed.scene.layers[0], LayerSpec)
    assert parsed.scene.layers[0].shape == [32, 64, 128]

    parsed_via_protocol = StreamProtocol.parse_message(json.dumps(payload))
    assert isinstance(parsed_via_protocol, SceneSpecMessage)


def test_layer_update_message(sample_layer_dict):
    layer = LayerSpec.from_dict(sample_layer_dict)
    msg = LayerUpdateMessage(layer=layer, partial=True, timestamp=4.56, controls=layer.controls)

    payload = msg.to_dict()
    assert payload["type"] == LAYER_UPDATE_TYPE
    assert payload["partial"] is True
    assert payload["layer"]["layer_id"] == "layer-001"

    parsed = StreamProtocol.parse_message(json.dumps(payload))
    assert isinstance(parsed, LayerUpdateMessage)
    assert parsed.layer is not None
    assert parsed.layer.name == "demo"
    assert parsed.controls == {"visible": True, "opacity": 0.85, "blending": "opaque"}


def test_layer_remove_message():
    msg = LayerRemoveMessage(layer_id="layer-001", reason="deleted", timestamp=7.89)
    payload = msg.to_dict()
    assert payload["type"] == LAYER_REMOVE_TYPE
    assert payload["layer_id"] == "layer-001"
    assert payload["reason"] == "deleted"

    parsed = StreamProtocol.parse_message(json.dumps(payload))
    assert isinstance(parsed, LayerRemoveMessage)
    assert parsed.layer_id == "layer-001"
