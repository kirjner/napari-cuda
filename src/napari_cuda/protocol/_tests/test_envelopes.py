from __future__ import annotations

import json

from napari_cuda.protocol import (
    NOTIFY_STATE_TYPE,
    SESSION_HELLO_TYPE,
    EnvelopeParser,
    NotifyScene,
    NotifyState,
    NotifyStream,
    SessionHello,
    SessionHelloPayload,
    StateUpdateMessage,
)


def test_session_hello_roundtrip() -> None:
    payload = SessionHelloPayload(
        protocol=1,
        client={"name": "test", "version": "0.1.0"},
        auth={"token": "abc"},
    )
    message = SessionHello(id="hello-1", timestamp=1.23, payload=payload)
    encoded = message.to_dict()

    assert encoded["type"] == SESSION_HELLO_TYPE
    decoded = SessionHello.from_dict(encoded)

    assert decoded.id == "hello-1"
    assert decoded.timestamp == 1.23
    assert decoded.payload.protocol == 1
    assert decoded.payload.client["name"] == "test"
    assert decoded.payload.auth == {"token": "abc"}


def test_notify_state_roundtrip() -> None:
    state_payload = {
        "scope": "dims",
        "target": "z",
        "key": "step",
        "value": 10,
        "server_seq": 5,
    }
    message = NotifyState(
        id="state-1",
        timestamp=5.0,
        payload=StateUpdateMessage.from_dict(state_payload),
    )
    encoded = message.to_dict()

    assert encoded["type"] == NOTIFY_STATE_TYPE
    decoded = NotifyState.from_dict(encoded)

    assert decoded.payload.scope == "dims"
    assert decoded.payload.key == "step"
    assert decoded.payload.server_seq == 5


def test_parser_dispatch() -> None:
    parser = EnvelopeParser()

    notify_scene = NotifyScene.from_dict(
        {
            "type": "notify.scene",
            "payload": {
                "version": 1,
                "scene": {"layers": []},
            },
        }
    )
    notify_stream = NotifyStream.from_dict(
        {
            "type": "notify.stream",
            "payload": {"codec": "h264", "fps": 30, "width": 1920, "height": 1080},
        }
    )
    notify_state = NotifyState.from_dict(
        {
            "type": "notify.state",
            "payload": {
                "scope": "dims",
                "target": "z",
                "key": "step",
                "value": 1,
            },
        }
    )

    scene_env = parser.parse_notify_scene(notify_scene.to_dict())
    stream_env = parser.parse_notify_stream(notify_stream.to_dict())
    state_env = parser.parse_notify_state(notify_state.to_dict())

    assert scene_env.payload.version == 1
    assert stream_env.payload.codec == "h264"
    assert state_env.payload.key == "step"


def test_parser_json_entrypoint() -> None:
    parser = EnvelopeParser()
    raw = json.dumps(
        {
            "type": "notify.state",
            "id": "state-10",
            "timestamp": 42.0,
            "payload": {
                "scope": "layer",
                "target": "layer-1",
                "key": "gamma",
                "value": 1.2,
            },
        }
    )

    envelope = parser.parse_json(raw)
    assert envelope.type == NOTIFY_STATE_TYPE
    assert envelope.id == "state-10"
    assert envelope.timestamp == 42.0
    assert envelope.payload["key"] == "gamma"

