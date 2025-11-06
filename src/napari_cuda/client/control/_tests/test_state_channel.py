from __future__ import annotations

import asyncio
import time
import types

import pytest

pytest.importorskip("websockets")

from napari_cuda.client.control.control_channel_client import (
    HeartbeatAckError,
    ResumeCursor,
    StateChannel,
)
from napari_cuda.protocol import (
    FeatureResumeState,
    FeatureToggle,
    build_ack_state,
    build_notify_layers_delta,
    build_notify_scene_snapshot,
    build_notify_stream,
    build_session_heartbeat,
    build_session_reject,
    build_session_welcome,
)
from napari_cuda.protocol.messages import (
    NotifyDimsFrame,
    NotifyLayersFrame,
    NotifySceneFrame,
    NotifyStreamFrame,
)
from napari_cuda.client.control._tests.test_state_update_actions import _make_notify_dims_frame


@pytest.fixture
def state_channel() -> StateChannel:
    return StateChannel('localhost', 8081)


def test_state_channel_scene_callback(state_channel):
    received = []
    state_channel.ingest_notify_scene_snapshot = received.append

    layer_block = {
        "layer_id": "a",
        "layer_type": "image",
        "name": "demo",
        "ndim": 2,
        "shape": [32, 32],
        "dtype": "float32",
        "axis_labels": ["y", "x"],
        "contrast_limits": [0.0, 1.0],
        "metadata": {},
        "render": {},
        "controls": {},
        "source": {},
    }
    frame = build_notify_scene_snapshot(
        session_id="session-1",
        viewer={"dims": {"ndim": 2}, "camera": {}},
        layers=[layer_block],
        metadata={'capabilities': ['layer.update']},
        delta_token='tok-scene-1',
    )

    state_channel._handle_message(frame.to_dict())

    assert len(received) == 1
    frame = received[0]
    assert isinstance(frame, NotifySceneFrame)
    assert frame.payload.layers[0]['layer_id'] == 'a'


def test_state_channel_layer_delta_callback(state_channel):
    received = []
    state_channel.ingest_notify_layers = received.append

    frame = build_notify_layers_delta(
        session_id='session-4',
        payload={'layer_id': 'layer-1', 'controls': {'opacity': 0.5}},
        timestamp=1.0,
        delta_token='tok-layer-1',
        seq=3,
    )

    state_channel._handle_message(frame.to_dict())

    assert len(received) == 1
    message = received[0]
    assert isinstance(message, NotifyLayersFrame)
    assert message.payload.layer_id == 'layer-1'
    assert message.payload.controls is not None
    assert message.payload.controls['opacity'] == 0.5


def test_state_channel_layer_remove_callback(state_channel):
    received = []
    state_channel.ingest_notify_layers = received.append

    frame = build_notify_layers_delta(
        session_id='session-5',
        payload={'layer_id': 'layer-2', 'removed': True},
        timestamp=2.0,
        delta_token='tok-layer-2',
        seq=4,
    )

    state_channel._handle_message(frame.to_dict())

    assert len(received) == 1
    message = received[0]
    assert isinstance(message, NotifyLayersFrame)
    assert message.payload.layer_id == 'layer-2'
    assert bool(message.payload.removed) is True


def test_state_channel_dims_update_dispatch() -> None:
    frames: list[NotifyDimsFrame] = []
    sc = StateChannel('localhost', 8081, ingest_dims_notify=frames.append)
    frame = _make_notify_dims_frame(
        current_step=[12, 3, 4],
        ndisplay=2,
        level_shapes=[[16, 8, 4]],
        mode='plane',
        session_id='session-42',
        frame_id='dims-007',
        timestamp=9.5,
    )

    sc._handle_message(frame.to_dict())

    assert len(frames) == 1
    received = frames[0]
    assert isinstance(received, NotifyDimsFrame)
    assert received.envelope.frame_id == 'dims-007'
    assert received.envelope.session == 'session-42'
    assert received.payload.current_step == (12, 3, 4)
    assert received.payload.ndisplay == 2
    assert received.payload.mode == 'plane'
    assert received.payload.level_shapes == ((16, 8, 4),)


def test_state_channel_ack_dispatch() -> None:
    received = []
    sc = StateChannel('localhost', 8081, ingest_ack_state=received.append)

    frame = build_ack_state(
        session_id='session-5',
        frame_id='ack-123',
        payload={
            'intent_id': 'intent-55',
            'in_reply_to': 'state-88',
            'status': 'accepted',
            'applied_value': {'value': 1},
            'version': 4,
        },
        timestamp=2.5,
    )

    sc._handle_message(frame.to_dict())

    assert len(received) == 1
    ack = received[0]
    assert ack.payload.intent_id == 'intent-55'
    assert ack.payload.in_reply_to == 'state-88'
    assert ack.payload.status == 'accepted'
    assert ack.payload.version == 4


def test_state_channel_notify_scene_dispatch(state_channel: StateChannel) -> None:
    received: list[NotifySceneFrame] = []
    state_channel.ingest_notify_scene_snapshot = received.append

    layer_block = {
        "layer_id": "layer-10",
        "layer_type": "image",
        "name": "demo",
        "ndim": 2,
        "shape": [8, 8],
        "axis_labels": ["y", "x"],
    }
    frame = build_notify_scene_snapshot(
        session_id="session-2",
        viewer={"dims": {"ndim": 3}, "camera": {}},
        layers=[layer_block],
        metadata=None,
        delta_token='tok-scene-2',
    )

    state_channel._handle_message(frame.to_dict())

    assert received
    frame = received[0]
    assert isinstance(frame, NotifySceneFrame)
    assert frame.payload.layers[0]['layer_id'] == 'layer-10'


def test_state_channel_notify_scene_includes_policies() -> None:
    frames: list[NotifySceneFrame] = []
    sc = StateChannel(
        'localhost',
        8081,
        ingest_notify_scene_snapshot=frames.append,
    )

    layer_block = {
        "layer_id": "layer-20",
        "layer_type": "image",
        "name": "zarr",
        "ndim": 3,
        "shape": [32, 32, 8],
        "axis_labels": ["z", "y", "x"],
    }
    policies = {
        'multiscale': {
            'policy': 'oversampling',
            'active_level': 2,
            'downgraded': False,
            'index_space': 'zyx',
            'levels': [
                {'index': 0, 'path': 'level_00', 'shape': [32, 32, 8], 'downsample': [1, 1, 1]},
                {'index': 1, 'path': 'level_01', 'shape': [16, 16, 4], 'downsample': [2, 2, 2]},
                {'index': 2, 'path': 'level_02', 'shape': [8, 8, 2], 'downsample': [4, 4, 4]},
            ],
        }
    }
    snapshot_metadata = {'metadata': {'adapter_engine': 'napari-vispy'}}

    frame = build_notify_scene_snapshot(
        session_id='session-3',
        viewer={'dims': {'ndim': 3}, 'camera': {}},
        layers=[layer_block],
        policies=policies,
        metadata=snapshot_metadata,
        delta_token='tok-scene-3',
    )

    sc._handle_message(frame.to_dict())

    assert frames
    frame = frames[0]
    assert isinstance(frame, NotifySceneFrame)
    scene_payload = frame.payload
    assert scene_payload.policies is not None
    multiscale = scene_payload.policies['multiscale']
    assert isinstance(multiscale, dict)
    assert multiscale['policy'] == 'oversampling'
    assert multiscale['active_level'] == 2
    assert multiscale['downgraded'] is False
    levels = multiscale['levels']
    assert isinstance(levels, list)
    assert levels[0]['path'] == 'level_00'



def test_state_channel_notify_stream_dispatch() -> None:
    frames: list[NotifyStreamFrame] = []
    sc = StateChannel('localhost', 8081, ingest_notify_stream=frames.append)

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

    assert frames and isinstance(frames[0], NotifyStreamFrame)
    payload = frames[0].payload
    assert payload.codec == 'h264'
    assert payload.format == 'avcc'


def _welcome_with_features(session_id: str, resume_tokens: dict[str, str]) -> object:
    features: dict[str, FeatureToggle] = {}
    for name, token in (
        ("notify.scene", resume_tokens.get("notify.scene")),
        ("notify.layers", resume_tokens.get("notify.layers")),
        ("notify.stream", resume_tokens.get("notify.stream")),
    ):
        resume_state = (
            FeatureResumeState(seq=1, delta_token=token)
            if token is not None
            else None
        )
        features[name] = FeatureToggle(enabled=True, resume=True, resume_state=resume_state)
    features.setdefault("notify.dims", FeatureToggle(enabled=True, resume=False))
    return build_session_welcome(
        session_id=session_id,
        heartbeat_s=2.5,
        ack_timeout_ms=250,
        features=features,
        timestamp=time.time(),
    )


def test_session_welcome_records_resume_tokens() -> None:
    channel = StateChannel('localhost', 8081)
    welcome = _welcome_with_features(
        session_id='sess-1',
        resume_tokens={
            'notify.scene': 'scene-token',
            'notify.layers': 'layers-token',
            'notify.stream': 'stream-token',
        },
    )

    metadata = channel._record_session_metadata(welcome)

    assert metadata.session_id == 'sess-1'
    assert metadata.resume_tokens['notify.scene'] == ResumeCursor(seq=1, delta_token='scene-token')
    assert metadata.resume_tokens['notify.layers'] == ResumeCursor(seq=1, delta_token='layers-token')
    assert metadata.resume_tokens['notify.stream'] == ResumeCursor(seq=1, delta_token='stream-token')

    payload = channel._resume_token_payload()
    assert payload['notify.scene'] == 'scene-token'
    assert payload['notify.layers'] == 'layers-token'
    assert payload['notify.stream'] == 'stream-token'


def test_handshake_reject_resets_invalid_resume_token() -> None:
    channel = StateChannel('localhost', 8081)
    channel._resume_tokens['notify.scene'] = ResumeCursor(seq=4, delta_token='old-scene')
    channel._resume_tokens['notify.layers'] = ResumeCursor(seq=2, delta_token='keep-me')

    reject = build_session_reject(
        code='invalid_resume_token',
        message='bad token',
        details={'topic': 'notify.scene'},
        timestamp=time.time(),
    )

    with pytest.raises(RuntimeError):
        channel._handle_handshake_reject(reject)

    assert channel._resume_tokens['notify.scene'] is None
    assert channel._resume_tokens['notify.layers'] == ResumeCursor(seq=2, delta_token='keep-me')


def test_handshake_reject_without_topic_resets_all_tokens() -> None:
    channel = StateChannel('localhost', 8081)
    channel._resume_tokens['notify.scene'] = ResumeCursor(seq=5, delta_token='scene-old')
    channel._resume_tokens['notify.layers'] = ResumeCursor(seq=6, delta_token='layers-old')
    channel._resume_tokens['notify.stream'] = ResumeCursor(seq=4, delta_token='stream-old')

    reject = build_session_reject(
        code='invalid_resume_token',
        message='bad token',
        details={},
        timestamp=time.time(),
    )

    with pytest.raises(RuntimeError):
        channel._handle_handshake_reject(reject)

    assert channel._resume_tokens['notify.scene'] is None
    assert channel._resume_tokens['notify.layers'] is None
    assert channel._resume_tokens['notify.stream'] is None


def test_handle_session_heartbeat_sends_ack(monkeypatch) -> None:
    channel = StateChannel('localhost', 8081)
    welcome = _welcome_with_features(
        session_id='sess-heartbeat',
        resume_tokens={
            'notify.scene': 's1',
            'notify.layers': 'l1',
            'notify.stream': 't1',
        },
    )
    channel._record_session_metadata(welcome)

    captured: dict[str, object] = {}

    def _fake_send(self, frame):
        captured['frame'] = frame
        return True

    channel.send_frame = types.MethodType(_fake_send, channel)  # type: ignore[assignment]

    heartbeat = build_session_heartbeat(session_id='sess-heartbeat', timestamp=time.time())
    before = channel._last_heartbeat_ts

    channel._handle_session_heartbeat(heartbeat.to_dict())

    assert 'frame' in captured
    ack = captured['frame']
    assert ack.envelope.session == 'sess-heartbeat'
    assert channel._last_heartbeat_ts is not None
    assert channel._last_heartbeat_ts >= before


def test_handle_session_heartbeat_propagates_send_failure() -> None:
    channel = StateChannel('localhost', 8081)
    welcome = _welcome_with_features(
        session_id='sess-fail',
        resume_tokens={'notify.scene': None, 'notify.layers': None, 'notify.stream': None},
    )
    channel._record_session_metadata(welcome)

    def _fail_send(self, frame):
        return False

    channel.send_frame = types.MethodType(_fail_send, channel)  # type: ignore[assignment]

    heartbeat = build_session_heartbeat(session_id='sess-fail', timestamp=time.time())

    with pytest.raises(HeartbeatAckError):
        channel._handle_session_heartbeat(heartbeat.to_dict())


def test_monitor_heartbeat_timeout_triggers_close() -> None:
    channel = StateChannel('localhost', 8081)
    channel._last_heartbeat_ts = time.time() - 5.0

    closed = {'called': False}

    class DummyWS:
        async def close(self, code=None, reason=None):
            closed['called'] = True

    async def _run() -> None:
        await channel._monitor_heartbeat(DummyWS(), interval=0.1)

    with pytest.raises(HeartbeatAckError):
        asyncio.run(_run())

    assert closed['called']
