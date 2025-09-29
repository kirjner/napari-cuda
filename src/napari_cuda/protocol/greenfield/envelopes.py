"""Greenfield control protocol envelope dataclasses.

These shapes describe the new notification/handshake flows while the runtime
continues to emit the legacy payloads. They intentionally avoid inheritance to
keep the data model minimal and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .messages import (
    CameraSpec,
    LayerRenderHints,
    LayerSpec,
    MultiscaleLevelSpec,
    MultiscaleSpec,
    SceneSpec,
    StateUpdateMessage,
)

PROTO_VERSION = 1

SESSION_HELLO_TYPE = "session.hello"
SESSION_WELCOME_TYPE = "session.welcome"
SESSION_REJECT_TYPE = "session.reject"

NOTIFY_SCENE_TYPE = "notify.scene"
NOTIFY_STATE_TYPE = "notify.state"
NOTIFY_STREAM_TYPE = "notify.stream"

CALL_COMMAND_TYPE = "call.command"
REPLY_COMMAND_TYPE = "reply.command"

ERROR_STATE_TYPE = "error.state"


def _strip_none(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of *mapping* without ``None`` values."""

    return {key: value for key, value in mapping.items() if value is not None}


def _copy_mutable(mapping: Optional[Any]) -> Optional[Any]:
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        return dict(mapping)
    return mapping


@dataclass(slots=True)
class SessionHelloPayload:
    protocol: int
    client: Mapping[str, Any]
    auth: Optional[Mapping[str, Any]] = None
    extras: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "protocol": int(self.protocol),
            "client": dict(self.client),
            "auth": _copy_mutable(self.auth),
            "extras": _copy_mutable(self.extras),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionHelloPayload":
        protocol = data.get("protocol")
        if protocol is None:
            raise ValueError("session.hello payload requires 'protocol'")
        client = data.get("client")
        if not isinstance(client, Mapping):
            raise ValueError("session.hello payload requires 'client' mapping")
        auth = data.get("auth")
        extras = data.get("extras")
        return cls(
            protocol=int(protocol),
            client=dict(client),
            auth=dict(auth) if isinstance(auth, MutableMapping) else auth,
            extras=dict(extras) if isinstance(extras, MutableMapping) else extras,
        )


@dataclass(slots=True)
class SessionWelcomePayload:
    protocol: int
    session: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": int(self.protocol),
            "session": dict(self.session),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionWelcomePayload":
        protocol = data.get("protocol")
        if protocol is None:
            raise ValueError("session.welcome payload requires 'protocol'")
        session = data.get("session")
        if not isinstance(session, Mapping):
            raise ValueError("session.welcome payload requires 'session' mapping")
        return cls(protocol=int(protocol), session=dict(session))


@dataclass(slots=True)
class SessionRejectPayload:
    code: str
    message: str
    details: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "details": _copy_mutable(self.details),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionRejectPayload":
        code = data.get("code")
        message = data.get("message")
        if not code or not message:
            raise ValueError("session.reject payload requires 'code' and 'message'")
        details = data.get("details")
        return cls(code=str(code), message=str(message), details=_copy_mutable(details))


@dataclass(slots=True)
class SessionHello:
    payload: SessionHelloPayload
    type: str = SESSION_HELLO_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionHello":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("session.hello requires mapping payload")
        return cls(
            type=str(data.get("type", SESSION_HELLO_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=SessionHelloPayload.from_dict(payload),
        )


@dataclass(slots=True)
class SessionWelcome:
    payload: SessionWelcomePayload
    type: str = SESSION_WELCOME_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionWelcome":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("session.welcome requires mapping payload")
        return cls(
            type=str(data.get("type", SESSION_WELCOME_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=SessionWelcomePayload.from_dict(payload),
        )


@dataclass(slots=True)
class SessionReject:
    payload: SessionRejectPayload
    type: str = SESSION_REJECT_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionReject":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("session.reject requires mapping payload")
        return cls(
            type=str(data.get("type", SESSION_REJECT_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=SessionRejectPayload.from_dict(payload),
        )


@dataclass(slots=True)
class NotifyScenePayload:
    version: int
    scene: SceneSpec
    state: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "version": int(self.version),
            "scene": self.scene.to_dict(),
            "state": _copy_mutable(self.state),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyScenePayload":
        version = data.get("version")
        if version is None:
            raise ValueError("notify.scene payload requires 'version'")
        scene_data = data.get("scene")
        if not isinstance(scene_data, Mapping):
            raise ValueError("notify.scene payload requires 'scene' mapping")
        state_data = data.get("state")
        return cls(
            version=int(version),
            scene=SceneSpec.from_dict(scene_data),
            state=_copy_mutable(state_data),
        )


@dataclass(slots=True)
class NotifyScene:
    payload: NotifyScenePayload
    type: str = NOTIFY_SCENE_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyScene":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("notify.scene requires mapping payload")
        return cls(
            type=str(data.get("type", NOTIFY_SCENE_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=NotifyScenePayload.from_dict(payload),
        )


@dataclass(slots=True)
class NotifyState:
    payload: StateUpdateMessage
    type: str = NOTIFY_STATE_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyState":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("notify.state requires mapping payload")
        return cls(
            type=str(data.get("type", NOTIFY_STATE_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=StateUpdateMessage.from_dict(dict(payload)),
        )


@dataclass(slots=True)
class NotifyStreamPayload:
    codec: str
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bitrate: Optional[int] = None
    idr_interval: Optional[int] = None
    extras: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "codec": self.codec,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "bitrate": self.bitrate,
            "idr_interval": self.idr_interval,
            "extras": _copy_mutable(self.extras),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyStreamPayload":
        codec = data.get("codec")
        if codec is None:
            raise ValueError("notify.stream payload requires 'codec'")
        return cls(
            codec=str(codec),
            fps=float(data["fps"]) if data.get("fps") is not None else None,
            width=int(data["width"]) if data.get("width") is not None else None,
            height=int(data["height"]) if data.get("height") is not None else None,
            bitrate=int(data["bitrate"]) if data.get("bitrate") is not None else None,
            idr_interval=int(data["idr_interval"]) if data.get("idr_interval") is not None else None,
            extras=_copy_mutable(data.get("extras")),
        )


@dataclass(slots=True)
class NotifyStream:
    payload: NotifyStreamPayload
    type: str = NOTIFY_STREAM_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyStream":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("notify.stream requires mapping payload")
        return cls(
            type=str(data.get("type", NOTIFY_STREAM_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=NotifyStreamPayload.from_dict(payload),
        )


@dataclass(slots=True)
class CallCommandPayload:
    name: str
    args: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "args": _copy_mutable(self.args),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallCommandPayload":
        name = data.get("name")
        if not name:
            raise ValueError("call.command payload requires 'name'")
        return cls(name=str(name), args=_copy_mutable(data.get("args")))


@dataclass(slots=True)
class CallCommand:
    payload: CallCommandPayload
    type: str = CALL_COMMAND_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallCommand":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("call.command requires mapping payload")
        return cls(
            type=str(data.get("type", CALL_COMMAND_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=CallCommandPayload.from_dict(payload),
        )


@dataclass(slots=True)
class ReplyCommandPayload:
    status: str
    result: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "result": _copy_mutable(self.result),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplyCommandPayload":
        status = data.get("status")
        if not status:
            raise ValueError("reply.command payload requires 'status'")
        return cls(status=str(status), result=_copy_mutable(data.get("result")))


@dataclass(slots=True)
class ReplyCommand:
    payload: ReplyCommandPayload
    type: str = REPLY_COMMAND_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplyCommand":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("reply.command requires mapping payload")
        return cls(
            type=str(data.get("type", REPLY_COMMAND_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=ReplyCommandPayload.from_dict(payload),
        )


@dataclass(slots=True)
class ErrorMessagePayload:
    status: str
    code: str
    message: str
    details: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "details": _copy_mutable(self.details),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorMessagePayload":
        status = data.get("status")
        code = data.get("code")
        message = data.get("message")
        if not status or not code or not message:
            raise ValueError("error payload requires 'status', 'code', and 'message'")
        return cls(
            status=str(status),
            code=str(code),
            message=str(message),
            details=_copy_mutable(data.get("details")),
        )


@dataclass(slots=True)
class ErrorMessage:
    payload: ErrorMessagePayload
    type: str = ERROR_STATE_TYPE
    id: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "id": self.id,
            "timestamp": self.timestamp,
            "payload": self.payload.to_dict(),
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorMessage":
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("error message requires mapping payload")
        return cls(
            type=str(data.get("type", ERROR_STATE_TYPE)),
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            payload=ErrorMessagePayload.from_dict(payload),
        )


__all__ = [
    "PROTO_VERSION",
    "SESSION_HELLO_TYPE",
    "SESSION_WELCOME_TYPE",
    "SESSION_REJECT_TYPE",
    "NOTIFY_SCENE_TYPE",
    "NOTIFY_STATE_TYPE",
    "NOTIFY_STREAM_TYPE",
    "CALL_COMMAND_TYPE",
    "REPLY_COMMAND_TYPE",
    "ERROR_STATE_TYPE",
    "SessionHelloPayload",
    "SessionHello",
    "SessionWelcomePayload",
    "SessionWelcome",
    "SessionRejectPayload",
    "SessionReject",
    "NotifyScenePayload",
    "NotifyScene",
    "NotifyState",
    "NotifyStreamPayload",
    "NotifyStream",
    "CallCommandPayload",
    "CallCommand",
    "ReplyCommandPayload",
    "ReplyCommand",
    "ErrorMessagePayload",
    "ErrorMessage",
    "StateUpdateMessage",
    "SceneSpec",
    "CameraSpec",
    "LayerSpec",
    "LayerRenderHints",
    "MultiscaleSpec",
    "MultiscaleLevelSpec",
]
