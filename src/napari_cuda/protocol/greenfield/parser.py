"""Parser helpers for greenfield protocol envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

from .envelopes import (
    CALL_COMMAND_TYPE,
    ERROR_STATE_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STATE_TYPE,
    NOTIFY_STREAM_TYPE,
    REPLY_COMMAND_TYPE,
    SESSION_HELLO_TYPE,
    SESSION_REJECT_TYPE,
    SESSION_WELCOME_TYPE,
    CallCommand,
    ErrorMessage,
    NotifyScene,
    NotifyState,
    NotifyStream,
    ReplyCommand,
    SessionHello,
    SessionReject,
    SessionWelcome,
)


@dataclass(slots=True)
class Envelope:
    """Minimal representation of any envelope."""

    type: str
    payload: Mapping[str, Any]
    id: str | None = None
    timestamp: float | None = None


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
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=payload,
        )

    def parse_json(self, raw: str | bytes | bytearray) -> Envelope:
        mapping = json.loads(raw)
        if not isinstance(mapping, MutableMapping):
            raise ValueError("Decoded envelope must be a JSON object")
        return self.parse(mapping)

    def parse_hello(self, data: Mapping[str, Any]) -> SessionHello:
        return SessionHello.from_dict(data)

    def parse_welcome(self, data: Mapping[str, Any]) -> SessionWelcome:
        return SessionWelcome.from_dict(data)

    def parse_reject(self, data: Mapping[str, Any]) -> SessionReject:
        return SessionReject.from_dict(data)

    def parse_notify_scene(self, data: Mapping[str, Any]) -> NotifyScene:
        return self._ensure_type(data, NOTIFY_SCENE_TYPE, NotifyScene.from_dict)

    def parse_notify_state(self, data: Mapping[str, Any]) -> NotifyState:
        return self._ensure_type(data, NOTIFY_STATE_TYPE, NotifyState.from_dict)

    def parse_notify_stream(self, data: Mapping[str, Any]) -> NotifyStream:
        return self._ensure_type(data, NOTIFY_STREAM_TYPE, NotifyStream.from_dict)

    def parse_call_command(self, data: Mapping[str, Any]) -> CallCommand:
        return self._ensure_type(data, CALL_COMMAND_TYPE, CallCommand.from_dict)

    def parse_reply_command(self, data: Mapping[str, Any]) -> ReplyCommand:
        return self._ensure_type(data, REPLY_COMMAND_TYPE, ReplyCommand.from_dict)

    def parse_error(self, data: Mapping[str, Any]) -> ErrorMessage:
        return self._ensure_type(data, ERROR_STATE_TYPE, ErrorMessage.from_dict)

    def _ensure_type(
        self,
        data: Mapping[str, Any],
        expected: str,
        loader: Callable[[Mapping[str, Any]], Any],
    ):
        envelope_type = data.get("type") or expected
        if envelope_type != expected:
            raise ValueError(f"Expected envelope type '{expected}', got '{envelope_type}'")
        return loader(data)


__all__ = ["Envelope", "EnvelopeParser"]
