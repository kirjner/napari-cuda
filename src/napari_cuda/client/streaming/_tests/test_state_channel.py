from __future__ import annotations

import pytest

pytest.importorskip("websockets")

from napari_cuda.client.streaming.state import StateChannel
from napari_cuda.protocol.messages import (
    LayerRemoveMessage,
    LayerSpec,
    LayerUpdateMessage,
    SceneSpec,
    SceneSpecMessage,
)


@pytest.fixture()
def state_channel() -> StateChannel:
    return StateChannel('localhost', 8081)


def test_state_channel_scene_callback(state_channel):
    received = []
    state_channel.on_scene_spec = received.append

    layer = LayerSpec(layer_id='a', layer_type='image', name='demo', ndim=2, shape=[32, 32])
    scene_msg = SceneSpecMessage(scene=SceneSpec(layers=[layer]))

    state_channel._handle_message(scene_msg.to_dict())  # type: ignore[arg-type]

    assert len(received) == 1
    assert isinstance(received[0], SceneSpecMessage)
    assert received[0].scene.layers[0].layer_id == 'a'


def test_state_channel_layer_update_callback(state_channel):
    received = []
    state_channel.on_layer_update = received.append

    layer = LayerSpec(layer_id='layer-1', layer_type='image', name='demo', ndim=2, shape=[16, 16])
    update_msg = LayerUpdateMessage(layer=layer, partial=False)

    state_channel._handle_message(update_msg.to_dict())  # type: ignore[arg-type]

    assert len(received) == 1
    assert isinstance(received[0], LayerUpdateMessage)
    assert received[0].layer is not None and received[0].layer.layer_id == 'layer-1'


def test_state_channel_layer_remove_callback(state_channel):
    received = []
    state_channel.on_layer_remove = received.append

    removal = LayerRemoveMessage(layer_id='layer-2', reason='test')

    state_channel._handle_message(removal.to_dict())  # type: ignore[arg-type]

    assert len(received) == 1
    assert isinstance(received[0], LayerRemoveMessage)
    assert received[0].layer_id == 'layer-2'
