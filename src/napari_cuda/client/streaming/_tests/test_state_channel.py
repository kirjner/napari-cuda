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


def test_state_channel_dims_update_normalisation():
    captured: list[dict] = []
    sc = StateChannel('localhost', 8081, on_dims_update=captured.append)

    payload = {
        'type': 'dims.update',
        'current_step': [12, 3, 4],
        'meta': {
            'ndim': 3,
            'axes': [
                {'label': 'z', 'index': 0},
                {'label': 'y', 'index': 1},
                {'label': 'x', 'index': 2},
            ],
            'ranges': [(0, 270, 1), (0, 615, 1), (0, 462, 1)],
            'displayed_axes': [1, 2],
            'level': 2,
            'level_shape': [272, 616, 463],
            'dtype': 'uint16',
            'normalized': True,
        },
        'ack': True,
        'intent_seq': 41,
        'seq': 99,
        'last_client_id': 'client-a',
    }

    sc._handle_message(payload)

    assert len(captured) == 1
    dims = captured[0]
    assert dims['current_step'] == [12, 3, 4]
    assert dims['ndim'] == 3
    assert dims['axis_labels'] == ['z', 'y', 'x']
    assert dims['order'] == [0, 1, 2]
    assert dims['range'] == [(0, 270, 1), (0, 615, 1), (0, 462, 1)]
    assert dims['displayed'] == [1, 2]
    assert dims['level'] == 2
    assert dims['level_shape'] == [272, 616, 463]
    assert dims['dtype'] == 'uint16'
    assert dims['normalized'] is True
    assert dims['ack'] is True
    assert dims['intent_seq'] == 41
    assert dims['seq'] == 99
    assert dims['last_client_id'] == 'client-a'
