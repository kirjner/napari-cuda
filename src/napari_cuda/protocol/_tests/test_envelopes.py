from __future__ import annotations

import json

import pytest

from napari_cuda.protocol import (
    NOTIFY_CAMERA_TYPE,
    NOTIFY_DIMS_TYPE,
    NOTIFY_ERROR_TYPE,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    NOTIFY_TELEMETRY_TYPE,
    SESSION_ACK_TYPE,
    SESSION_GOODBYE_TYPE,
    SESSION_HEARTBEAT_TYPE,
    SESSION_HELLO_TYPE,
    STATE_UPDATE_TYPE,
    EnvelopeParser,
    HelloClientInfo,
    NotifyCamera,
    NotifyDims,
    NotifyError,
    NotifyLayers,
    NotifyScene,
    NotifyStream,
    NotifyTelemetry,
    ResumableTopicSequencer,
    SessionAck,
    SessionGoodbye,
    SessionHello,
    SessionHeartbeat,
    StateUpdate,
    build_notify_camera,
    build_notify_dims,
    build_notify_error,
    build_notify_layers_delta,
    build_notify_scene_snapshot,
    build_notify_stream,
    build_notify_telemetry,
    build_session_ack,
    build_session_goodbye,
    build_session_hello,
    build_session_heartbeat,
    build_state_update,
)


def test_session_hello_roundtrip() -> None:
    hello = build_session_hello(
        client=HelloClientInfo(
            name="test-client",
            version="0.1.0",
            platform="linux",
        ),
        features={"notify.scene": True, "notify.layers": True, "notify.stream": True},
        resume_tokens={"notify.scene": None, "notify.layers": None, "notify.stream": None},
        frame_id="hello-1",
        timestamp=1.23,
    )
    encoded = hello.to_dict()

    assert encoded["type"] == SESSION_HELLO_TYPE
    decoded = SessionHello.from_dict(encoded)

    assert decoded.envelope.frame_id == "hello-1"
    assert decoded.envelope.timestamp == 1.23
    assert decoded.payload.client.name == "test-client"
    assert decoded.payload.protocols == (1,)


def test_notify_scene_roundtrip() -> None:
    sequencer = ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE)
    notify_scene = build_notify_scene_snapshot(
        session_id="session-1",
        viewer={"dims": {"ndim": 2}},
        layers=[{"layer_id": "layer-1", "kind": "image"}],
        timestamp=2.0,
        sequencer=sequencer,
    )
    encoded = notify_scene.to_dict()

    assert encoded["type"] == NOTIFY_SCENE_TYPE
    decoded = NotifyScene.from_dict(encoded)

    assert decoded.envelope.session == "session-1"
    assert decoded.envelope.seq == 0
    assert decoded.payload.viewer["dims"]["ndim"] == 2
    assert decoded.payload.layers[0]["layer_id"] == "layer-1"


def test_parser_dispatch() -> None:
    parser = EnvelopeParser()
    sequencer = ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE)

    scene_frame = build_notify_scene_snapshot(
        session_id="sess",
        viewer={"dims": {"ndim": 3}},
        layers=[{"layer_id": "layer"}],
        sequencer=sequencer,
    )
    state_frame = build_state_update(
        session_id="sess",
        intent_id="intent-1",
        frame_id="state-1",
        payload={"scope": "dims", "target": "ndisplay", "key": "value", "value": 3},
    )

    scene_env = parser.parse_notify_scene(scene_frame.to_dict())
    state_env = parser.parse_state_update(state_frame.to_dict())

    assert scene_env.payload.viewer["dims"]["ndim"] == 3
    assert state_env.payload.key == "value"


def test_parser_json_entrypoint() -> None:
    frame = build_state_update(
        session_id="sess",
        intent_id="intent-9",
        frame_id="state-9",
        payload={"scope": "layer", "target": "layer-1", "key": "gamma", "value": 1.2},
        timestamp=42.0,
    )
    raw = json.dumps(frame.to_dict())

    parser = EnvelopeParser()
    envelope = parser.parse_json(raw)

    assert envelope.type == STATE_UPDATE_TYPE
    assert envelope.frame_id == "state-9"
    assert envelope.timestamp == 42.0
    assert envelope.payload["key"] == "gamma"


def test_session_heartbeat_roundtrip() -> None:
    heartbeat = build_session_heartbeat(
        session_id="sess",
        frame_id="hb-1",
        timestamp=3.0,
    )
    encoded = heartbeat.to_dict()

    assert encoded["type"] == SESSION_HEARTBEAT_TYPE
    decoded = SessionHeartbeat.from_dict(encoded)

    assert decoded.envelope.frame_id == "hb-1"
    assert decoded.envelope.timestamp == 3.0


def test_session_ack_roundtrip() -> None:
    ack = build_session_ack(
        session_id="sess",
        frame_id="ack-1",
        timestamp=4.0,
    )
    encoded = ack.to_dict()

    assert encoded["type"] == SESSION_ACK_TYPE
    decoded = SessionAck.from_dict(encoded)

    assert decoded.envelope.frame_id == "ack-1"
    assert decoded.envelope.timestamp == 4.0


def test_session_goodbye_from_mapping() -> None:
    goodbye = build_session_goodbye(
        session_id="sess",
        payload={"code": "normal", "message": "done", "reason": "shutdown"},
        frame_id="bye-1",
        timestamp=5.0,
    )
    encoded = goodbye.to_dict()

    assert encoded["type"] == SESSION_GOODBYE_TYPE
    decoded = SessionGoodbye.from_dict(encoded)

    assert decoded.payload.code == "normal"
    assert decoded.envelope.frame_id == "bye-1"


def test_resumable_sequencer_requires_snapshot() -> None:
    sequencer = ResumableTopicSequencer(topic=NOTIFY_STREAM_TYPE)
    with pytest.raises(ValueError):
        sequencer.delta()


def test_resumable_sequencer_snapshot_and_delta() -> None:
    sequencer = ResumableTopicSequencer(topic=NOTIFY_LAYERS_TYPE)
    snapshot_cursor = sequencer.snapshot()
    delta_cursor = sequencer.delta()

    assert snapshot_cursor.seq == 0
    assert delta_cursor.seq == 1
    assert delta_cursor.delta_token != snapshot_cursor.delta_token


def test_notify_layers_delta_uses_sequencer() -> None:
    sequencer = ResumableTopicSequencer(topic=NOTIFY_LAYERS_TYPE)
    sequencer.snapshot()
    frame = build_notify_layers_delta(
        session_id="sess",
        payload={"layer_id": "layer-1", "changes": {"opacity": 0.5}},
        sequencer=sequencer,
    )

    assert frame.envelope.seq == 1
    assert frame.envelope.delta_token == sequencer.delta_token


def test_notify_stream_delta_uses_sequencer() -> None:
    sequencer = ResumableTopicSequencer(topic=NOTIFY_STREAM_TYPE)
    sequencer.snapshot()
    frame = build_notify_stream(
        session_id="sess",
        payload={
            "codec": "h264",
            "format": "avc",
            "fps": 60.0,
            "frame_size": [1920, 1080],
            "nal_length_size": 4,
            "avcc": "AAAA",
            "latency_policy": {"max_buffer_ms": 33, "grace_keyframe_ms": 66},
        },
        sequencer=sequencer,
    )

    assert frame.envelope.type == NOTIFY_STREAM_TYPE
    assert frame.envelope.seq == 1


def test_notify_dims_short_frame_id_when_no_intent() -> None:
    frame = build_notify_dims(
        session_id="sess",
        payload={
            "current_step": [0, 1],
            "ndisplay": 2,
            "mode": "2d",
            "source": "server",
        },
    )

    assert frame.envelope.type == NOTIFY_DIMS_TYPE
    assert frame.envelope.intent_id is None
    assert frame.envelope.frame_id is not None
    assert len(frame.envelope.frame_id) <= 8


def test_notify_camera_respects_intent_id() -> None:
    frame = build_notify_camera(
        session_id="sess",
        payload={"mode": "3d", "delta": {"zoom": 1.0}, "origin": "server"},
        intent_id="intent-1",
        frame_id="camera-frame-long",
    )

    assert frame.envelope.type == NOTIFY_CAMERA_TYPE
    assert frame.envelope.intent_id == "intent-1"
    assert frame.envelope.frame_id == "camera-frame-long"


def test_notify_telemetry_roundtrip() -> None:
    frame = build_notify_telemetry(
        session_id="sess",
        payload={"presenter": 55.5, "decode": 3.2, "queue_depth": 1.0},
        timestamp=7.0,
    )
    encoded = frame.to_dict()

    assert encoded["type"] == NOTIFY_TELEMETRY_TYPE
    decoded = NotifyTelemetry.from_dict(encoded)

    assert decoded.envelope.timestamp == 7.0
    assert decoded.payload.presenter == 55.5
    assert decoded.payload.decode == 3.2
    assert decoded.payload.queue_depth == 1.0


def test_notify_error_roundtrip() -> None:
    frame = build_notify_error(
        session_id="sess",
        payload={
            "domain": "control",
            "code": "fatal",
            "message": "boom",
            "severity": "critical",
            "context": {"detail": "stack"},
        },
        timestamp=8.0,
    )
    encoded = frame.to_dict()

    assert encoded["type"] == NOTIFY_ERROR_TYPE
    decoded = NotifyError.from_dict(encoded)

    assert decoded.payload.severity == "critical"
    assert decoded.envelope.timestamp == 8.0
