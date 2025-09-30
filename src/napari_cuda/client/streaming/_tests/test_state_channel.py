from __future__ import annotations

import pytest

pytest.importorskip("websockets")

from napari_cuda.client.control.control_channel_client import StateChannel
from napari_cuda.protocol import build_ack_state, build_notify_dims, build_notify_stream
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


def test_state_channel_dims_update_dispatch() -> None:
    captured: list[dict] = []
    sc = StateChannel('localhost', 8081, handle_dims_update=captured.append)

    frame = build_notify_dims(
        session_id='session-42',
        payload={'current_step': (12, 3, 4), 'ndisplay': 2, 'mode': 'slice', 'source': 'test'},
        timestamp=9.5,
        frame_id='dims-007',
    )

    sc._handle_message(frame.to_dict())

    assert len(captured) == 1
    dims = captured[0]
    assert dims['frame_id'] == 'dims-007'
    assert dims['session'] == 'session-42'
    assert dims['timestamp'] == pytest.approx(frame.envelope.timestamp)
    assert dims['current_step'] == (12, 3, 4)
    assert dims['ndisplay'] == 2
    assert dims['mode'] == 'slice'
    assert dims['source'] == 'test'


def test_state_channel_ack_dispatch() -> None:
    received = []
    sc = StateChannel('localhost', 8081, handle_ack_state=received.append)

    frame = build_ack_state(
        session_id='session-5',
        frame_id='ack-123',
        payload={
            'intent_id': 'intent-55',
            'in_reply_to': 'state-88',
            'status': 'accepted',
            'applied_value': {'value': 1},
        },
        timestamp=2.5,
    )

    sc._handle_message(frame.to_dict())

    assert len(received) == 1
    ack = received[0]
    assert ack.payload.intent_id == 'intent-55'
    assert ack.payload.in_reply_to == 'state-88'
    assert ack.payload.status == 'accepted'


def test_state_channel_notify_scene_dispatch(state_channel: StateChannel) -> None:
    received: list[SceneSpecMessage] = []
    state_channel.handle_scene_spec = received.append

    layer = LayerSpec(layer_id='layer-10', layer_type='image', name='demo', ndim=2, shape=[8, 8])
    envelope = SceneSpecMessage(
        scene=SceneSpec(layers=[layer]),
        timestamp=2.0,
    )

    state_channel._handle_message(envelope.to_dict())  # type: ignore[arg-type]

    assert received and isinstance(received[0], SceneSpecMessage)
    assert received[0].scene.layers[0].layer_id == 'layer-10'


def test_state_channel_notify_stream_dispatch() -> None:
    configs: list[dict] = []
    sc = StateChannel('localhost', 8081, handle_video_config=configs.append)

    stream_payload = {
        'codec': 'h264',
        'format': 'avcc',
        'fps': 30.0,
        'frame_size': [1920, 1080],
        'nal_length_size': 4,
        'avcc': 'AAA=',
        'latency_policy': {'max_buffer_ms': 120, 'grace_keyframe_ms': 500},
    }
    frame = build_notify_stream(
        session_id='sess',
        seq=1,
        delta_token='tok-stream',
        payload=stream_payload,
        timestamp=3.14,
    )

    sc._handle_message(frame.to_dict())

    assert configs and configs[0]['type'] == 'video_config'
    assert configs[0]['codec'] == 'h264'
    assert configs[0]['format'] == 'avcc'
