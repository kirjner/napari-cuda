"""Parser helpers for greenfield protocol envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, TypeVar

from .envelopes import (
    ACK_STATE_TYPE,
    CALL_COMMAND_TYPE,
    ERROR_COMMAND_TYPE,
    NOTIFY_CAMERA_TYPE,
    NOTIFY_DIMS_TYPE,
    NOTIFY_ERROR_TYPE,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_STREAM_TYPE,
    NOTIFY_TELEMETRY_TYPE,
    STATE_UPDATE_TYPE,
    REPLY_COMMAND_TYPE,
    SESSION_ACK_TYPE,
    SESSION_GOODBYE_TYPE,
    SESSION_HEARTBEAT_TYPE,
    SESSION_HELLO_TYPE,
    SESSION_REJECT_TYPE,
    SESSION_WELCOME_TYPE,
    AckState,
    CallCommand,
    ErrorCommand,
    NotifyCamera,
    NotifyDims,
    NotifyError,
    NotifyLayers,
    NotifyScene,
    NotifySceneLevel,
    NotifyStream,
    NotifyTelemetry,
    ReplyCommand,
    SessionAck,
    SessionGoodbye,
    SessionHeartbeat,
    SessionHello,
    SessionReject,
    SessionWelcome,
    StateUpdate,
)

FrameT = TypeVar("FrameT")


@dataclass(slots=True)
class Envelope:
    """Minimal representation of any envelope."""

    type: str
    payload: Mapping[str, Any]
    version: int | None = None
    session: str | None = None
    frame_id: str | None = None
    timestamp: float | None = None
    seq: int | None = None
    delta_token: str | None = None
    intent_id: str | None = None


class EnvelopeParser:
    """Parse JSON/mapping payloads into typed envelopes."""

    def parse(self, data: Mapping[str, Any]) -> Envelope:
        envelope_type = data.get("type")
        if not envelope_type:
            raise ValueError("Envelope missing 'type'")
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("Envelope 'payload' must be a mapping")
        return Envelope(
            type=str(envelope_type),
            payload=payload,
            version=data.get("version"),
            session=data.get("session"),
            frame_id=data.get("frame_id"),
            timestamp=data.get("timestamp"),
            seq=data.get("seq"),
            delta_token=data.get("delta_token"),
            intent_id=data.get("intent_id"),
        )

    def parse_json(self, raw: str | bytes | bytearray) -> Envelope:
        mapping = json.loads(raw)
        if not isinstance(mapping, MutableMapping):
            raise ValueError("Decoded envelope must be a JSON object")
        return self.parse(mapping)

    def parse_hello(self, data: Mapping[str, Any]) -> SessionHello:
        return self._ensure_type(data, SESSION_HELLO_TYPE, SessionHello.from_dict)

    def parse_welcome(self, data: Mapping[str, Any]) -> SessionWelcome:
        return self._ensure_type(data, SESSION_WELCOME_TYPE, SessionWelcome.from_dict)

    def parse_reject(self, data: Mapping[str, Any]) -> SessionReject:
        return self._ensure_type(data, SESSION_REJECT_TYPE, SessionReject.from_dict)

    def parse_heartbeat(self, data: Mapping[str, Any]) -> SessionHeartbeat:
        return self._ensure_type(data, SESSION_HEARTBEAT_TYPE, SessionHeartbeat.from_dict)

    def parse_ack(self, data: Mapping[str, Any]) -> SessionAck:
        return self._ensure_type(data, SESSION_ACK_TYPE, SessionAck.from_dict)

    def parse_goodbye(self, data: Mapping[str, Any]) -> SessionGoodbye:
        return self._ensure_type(data, SESSION_GOODBYE_TYPE, SessionGoodbye.from_dict)

    def parse_notify_scene(self, data: Mapping[str, Any]) -> NotifyScene:
        return self._ensure_type(data, NOTIFY_SCENE_TYPE, NotifyScene.from_dict)

    def parse_notify_scene_level(self, data: Mapping[str, Any]) -> NotifySceneLevel:
        return self._ensure_type(data, NOTIFY_SCENE_LEVEL_TYPE, NotifySceneLevel.from_dict)

    def parse_notify_layers(self, data: Mapping[str, Any]) -> NotifyLayers:
        return self._ensure_type(data, NOTIFY_LAYERS_TYPE, NotifyLayers.from_dict)

    def parse_notify_stream(self, data: Mapping[str, Any]) -> NotifyStream:
        return self._ensure_type(data, NOTIFY_STREAM_TYPE, NotifyStream.from_dict)

    def parse_notify_dims(self, data: Mapping[str, Any]) -> NotifyDims:
        return self._ensure_type(data, NOTIFY_DIMS_TYPE, NotifyDims.from_dict)

    def parse_notify_camera(self, data: Mapping[str, Any]) -> NotifyCamera:
        return self._ensure_type(data, NOTIFY_CAMERA_TYPE, NotifyCamera.from_dict)

    def parse_notify_telemetry(self, data: Mapping[str, Any]) -> NotifyTelemetry:
        return self._ensure_type(data, NOTIFY_TELEMETRY_TYPE, NotifyTelemetry.from_dict)

    def parse_notify_error(self, data: Mapping[str, Any]) -> NotifyError:
        return self._ensure_type(data, NOTIFY_ERROR_TYPE, NotifyError.from_dict)

    def parse_state_update(self, data: Mapping[str, Any]) -> StateUpdate:
        return self._ensure_type(data, STATE_UPDATE_TYPE, StateUpdate.from_dict)

    # Temporary compatibility alias while callers migrate away from legacy helper name.
    def parse_notify_state(self, data: Mapping[str, Any]) -> StateUpdate:
        return self.parse_state_update(data)

    def parse_ack_state(self, data: Mapping[str, Any]) -> AckState:
        return self._ensure_type(data, ACK_STATE_TYPE, AckState.from_dict)

    def parse_call_command(self, data: Mapping[str, Any]) -> CallCommand:
        return self._ensure_type(data, CALL_COMMAND_TYPE, CallCommand.from_dict)

    def parse_reply_command(self, data: Mapping[str, Any]) -> ReplyCommand:
        return self._ensure_type(data, REPLY_COMMAND_TYPE, ReplyCommand.from_dict)

    def parse_error_command(self, data: Mapping[str, Any]) -> ErrorCommand:
        return self._ensure_type(data, ERROR_COMMAND_TYPE, ErrorCommand.from_dict)

    def _ensure_type(
        self,
        data: Mapping[str, Any],
        expected: str,
        loader: Callable[[Mapping[str, Any]], FrameT],
    ) -> FrameT:
        envelope_type = data.get("type") or expected
        if envelope_type != expected:
            raise ValueError(f"Expected envelope type '{expected}', got '{envelope_type}'")
        return loader(data)


__all__ = ["Envelope", "EnvelopeParser"]
