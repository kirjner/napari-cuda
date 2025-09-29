from __future__ import annotations

import pytest

pytest.importorskip("websockets")

from napari_cuda.client.control.control_channel_client import StateChannel
from napari_cuda.protocol import (
    NotifyScene,
    NotifyScenePayload,
    NotifyState,
    NotifyStream,
    NotifyStreamPayload,
)
from napari_cuda.protocol.messages import (
    LayerRemoveMessage,
    LayerSpec,
    LayerUpdateMessage,
    SceneSpec,
    SceneSpecMessage,
    StateUpdateMessage,
)


@pytest.fixture()
def state_channel() -> StateChannel:
    return StateChannel('localhost', 8081)


def test_state_channel_scene_callback(state_channel):
    received = []
    state_channel.handle_scene_spec = received.append

    layer = LayerSpec(layer_id='a', layer_type='image', name='demo', ndim=2, shape=[32, 32])
    scene_msg = SceneSpecMessage(scene=SceneSpec(layers=[layer]))

    state_channel._handle_message(scene_msg.to_dict())  # type: ignore[arg-type]

    assert len(received) == 1
    assert isinstance(received[0], SceneSpecMessage)
    assert received[0].scene.layers[0].layer_id == 'a'


def test_state_channel_layer_update_callback(state_channel):
    received = []
    state_channel.handle_layer_update = received.append

    layer = LayerSpec(layer_id='layer-1', layer_type='image', name='demo', ndim=2, shape=[16, 16])
    update_msg = LayerUpdateMessage(layer=layer, partial=False)

    state_channel._handle_message(update_msg.to_dict())  # type: ignore[arg-type]

    assert len(received) == 1
    assert isinstance(received[0], LayerUpdateMessage)
    assert received[0].layer is not None and received[0].layer.layer_id == 'layer-1'


def test_state_channel_layer_remove_callback(state_channel):
    received = []
    state_channel.handle_layer_remove = received.append

    removal = LayerRemoveMessage(layer_id='layer-2', reason='test')

    state_channel._handle_message(removal.to_dict())  # type: ignore[arg-type]

    assert len(received) == 1
    assert isinstance(received[0], LayerRemoveMessage)
    assert received[0].layer_id == 'layer-2'


def test_state_channel_dims_update_normalisation():
    captured: list[dict] = []
    sc = StateChannel('localhost', 8081, handle_dims_update=captured.append)

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


def test_state_channel_notify_state_dispatch() -> None:
    captured: list[StateUpdateMessage] = []
    sc = StateChannel('localhost', 8081, handle_state_update=captured.append)

    envelope = NotifyState(
        payload=StateUpdateMessage(scope='dims', target='z', key='step', value=3),
        timestamp=1.23,
    )

    sc._handle_message(envelope.to_dict())

    assert captured and captured[0].key == 'step'
    assert captured[0].value == 3


def test_state_channel_notify_scene_dispatch(state_channel: StateChannel) -> None:
    received: list[SceneSpecMessage] = []
    state_channel.handle_scene_spec = received.append

    layer = LayerSpec(layer_id='layer-10', layer_type='image', name='demo', ndim=2, shape=[8, 8])
    envelope = NotifyScene(
        payload=NotifyScenePayload(
            version=1,
            scene=SceneSpec(layers=[layer]),
            state={'capabilities': ['notify.state']},
        ),
        timestamp=2.0,
    )

    state_channel._handle_message(envelope.to_dict())  # type: ignore[arg-type]

    assert received and isinstance(received[0], SceneSpecMessage)
    assert received[0].scene.layers[0].layer_id == 'layer-10'


def test_state_channel_notify_stream_dispatch() -> None:
    configs: list[dict] = []
    sc = StateChannel('localhost', 8081, handle_video_config=configs.append)

    envelope = NotifyStream(
        payload=NotifyStreamPayload(
            codec='h264',
            fps=30.0,
            width=1920,
            height=1080,
            extras={'format': 'avcc', 'data': 'AAA='},
        ),
        timestamp=3.14,
    )

    sc._handle_message(envelope.to_dict())

    assert configs and configs[0]['type'] == 'video_config'
    assert configs[0]['codec'] == 'h264'
    assert configs[0]['format'] == 'avcc'
