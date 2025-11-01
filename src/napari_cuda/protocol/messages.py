"""Spec-compliant message dataclasses for the greenfield control protocol.

These dataclasses are the canonical, schema-derived representations for the
greenfield envelopes described in :mod:`docs/protocol_greenfield.md`.  They avoid
inheritance entirely so every frame remains a simple, explicit type composed of
an envelope and payload pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, Mapping, Sequence, Tuple

PROTO_VERSION = 2

# Session frame types
SESSION_HELLO_TYPE = "session.hello"
SESSION_WELCOME_TYPE = "session.welcome"
SESSION_REJECT_TYPE = "session.reject"
SESSION_HEARTBEAT_TYPE = "session.heartbeat"
SESSION_ACK_TYPE = "session.ack"
SESSION_GOODBYE_TYPE = "session.goodbye"

# Notify frame types
NOTIFY_SCENE_TYPE = "notify.scene"
NOTIFY_LAYERS_TYPE = "notify.layers"
NOTIFY_SCENE_LEVEL_TYPE = "notify.scene.level"
NOTIFY_STREAM_TYPE = "notify.stream"
NOTIFY_DIMS_TYPE = "notify.dims"
NOTIFY_CAMERA_TYPE = "notify.camera"
NOTIFY_TELEMETRY_TYPE = "notify.telemetry"
NOTIFY_ERROR_TYPE = "notify.error"

# State / command frame types
STATE_UPDATE_TYPE = "state.update"
ACK_STATE_TYPE = "ack.state"
CALL_COMMAND_TYPE = "call.command"
REPLY_COMMAND_TYPE = "reply.command"
ERROR_COMMAND_TYPE = "error.command"

_ACK_STATUSES = {"accepted", "rejected"}
_COMMAND_SUCCESS_STATUS = "ok"
_COMMAND_ERROR_STATUS = "error"
_ERROR_SEVERITIES = {"info", "warning", "critical"}


def _strip_none(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Return *mapping* without keys whose value is ``None``."""

    return {key: value for key, value in mapping.items() if value is not None}


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _as_mutable_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _as_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a JSON array")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_mapping(value: Any, field_name: str) -> Dict[str, Any] | None:
    if value is None:
        return None
    return _as_mutable_mapping(value, field_name)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ensure_keyset(
    mapping: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
    context: str,
) -> None:
    required_set = {str(key) for key in required}
    optional_set = {str(key) for key in optional}
    missing = sorted(key for key in required_set if key not in mapping)
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(missing)}")
    unexpected = sorted(
        key for key in mapping if key not in required_set and key not in optional_set
    )
    if unexpected:
        raise ValueError(f"{context} received unknown fields: {', '.join(unexpected)}")


@dataclass(slots=True)
class FrameEnvelope:
    """Envelope shared by all frames."""

    type: str
    version: int
    session: str | None = None
    frame_id: str | None = None
    timestamp: float | None = None
    seq: int | None = None
    delta_token: str | None = None
    intent_id: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "version": int(self.version),
            "session": self.session,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "seq": self.seq,
            "delta_token": self.delta_token,
            "intent_id": self.intent_id,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameEnvelope":
        if "type" not in data:
            raise ValueError("frame envelope requires 'type'")
        if "version" not in data:
            raise ValueError("frame envelope requires 'version'")
        if "timestamp" not in data:
            raise ValueError("frame envelope requires 'timestamp'")
        seq_value = data.get("seq")
        seq_int: int | None = None
        if seq_value is not None:
            try:
                seq_int = int(seq_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("envelope 'seq' must be an integer") from exc
            if seq_int < 0:
                raise ValueError("envelope 'seq' must be >= 0")
        return cls(
            type=str(data["type"]),
            version=int(data["version"]),
            session=_optional_str(data.get("session")),
            frame_id=_optional_str(data.get("frame_id")),
            timestamp=float(data["timestamp"]),
            seq=seq_int,
            delta_token=_optional_str(data.get("delta_token")),
            intent_id=_optional_str(data.get("intent_id")),
        )


def _pull_payload(data: Mapping[str, Any]) -> tuple[FrameEnvelope, Mapping[str, Any]]:
    payload = data.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("frame requires mapping payload")
    return FrameEnvelope.from_dict(data), payload


def _frame_dict(envelope: FrameEnvelope, payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = envelope.to_dict()
    data["payload"] = dict(payload)
    return data


@dataclass(slots=True)
class HelloClientInfo:
    name: str
    version: str
    platform: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HelloClientInfo":
        mapping = _as_mapping(data, "session.hello payload.client")
        _ensure_keyset(mapping, required=("name", "version", "platform"), context="session.hello client")
        return cls(
            name=str(mapping["name"]),
            version=str(mapping["version"]),
            platform=str(mapping["platform"]),
        )


@dataclass(slots=True)
class HelloAuthInfo:
    type: str
    token: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({"type": self.type, "token": self.token})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HelloAuthInfo":
        mapping = _as_mapping(data, "session.hello payload.auth")
        _ensure_keyset(mapping, required=("type",), optional=("token",), context="session.hello auth")
        return cls(type=str(mapping["type"]), token=_optional_str(mapping.get("token")))


@dataclass(slots=True)
class SessionHelloPayload:
    protocols: Tuple[int, ...]
    client: HelloClientInfo
    features: Dict[str, bool]
    resume_tokens: Dict[str, str | None]
    auth: HelloAuthInfo | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "protocols": [int(value) for value in self.protocols],
                "client": self.client.to_dict(),
                "features": dict(self.features),
                "resume_tokens": {key: value for key, value in self.resume_tokens.items()},
                "auth": self.auth.to_dict() if self.auth else None,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionHelloPayload":
        mapping = _as_mapping(data, "session.hello payload")
        _ensure_keyset(
            mapping,
            required=("protocols", "client", "features", "resume_tokens"),
            optional=("auth",),
            context="session.hello payload",
        )
        protocols_seq = _as_sequence(mapping["protocols"], "session.hello payload.protocols")
        protocols = tuple(int(value) for value in protocols_seq)
        if not protocols:
            raise ValueError("session.hello payload requires at least one protocol version")
        features: Dict[str, bool] = {}
        for key, value in _as_mapping(mapping["features"], "session.hello payload.features").items():
            features[str(key)] = bool(value)
        resume_tokens: Dict[str, str | None] = {}
        for key, value in _as_mapping(mapping["resume_tokens"], "session.hello payload.resume_tokens").items():
            resume_tokens[str(key)] = None if value is None else str(value)
        auth_payload = mapping.get("auth")
        auth = HelloAuthInfo.from_dict(auth_payload) if isinstance(auth_payload, Mapping) else None
        return cls(
            protocols=protocols,
            client=HelloClientInfo.from_dict(mapping["client"]),
            features=features,
            resume_tokens=resume_tokens,
            auth=auth,
        )


@dataclass(slots=True)
class WelcomeSessionInfo:
    id: str
    heartbeat_s: float
    ack_timeout_ms: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "id": self.id,
                "heartbeat_s": float(self.heartbeat_s),
                "ack_timeout_ms": int(self.ack_timeout_ms) if self.ack_timeout_ms is not None else None,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WelcomeSessionInfo":
        mapping = _as_mapping(data, "session.welcome payload.session")
        _ensure_keyset(
            mapping,
            required=("id", "heartbeat_s"),
            optional=("ack_timeout_ms",),
            context="session.welcome session",
        )
        return cls(
            id=str(mapping["id"]),
            heartbeat_s=float(mapping["heartbeat_s"]),
            ack_timeout_ms=int(mapping["ack_timeout_ms"]) if mapping.get("ack_timeout_ms") is not None else None,
        )


@dataclass(slots=True)
class FeatureResumeState:
    """Sequencer cursor advertised during ``session.welcome``."""

    seq: int
    delta_token: str

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": int(self.seq), "delta_token": str(self.delta_token)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureResumeState":
        mapping = _as_mapping(data, "session.welcome payload.features.<topic>.resume_state")
        _ensure_keyset(
            mapping,
            required=("seq", "delta_token"),
            context="session.welcome feature resume_state",
        )
        return cls(seq=int(mapping["seq"]), delta_token=str(mapping["delta_token"]))


@dataclass(slots=True)
class FeatureToggle:
    enabled: bool
    version: int | None = None
    resume: bool | None = None
    commands: Tuple[str, ...] | None = None
    resume_state: FeatureResumeState | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "enabled": bool(self.enabled),
                "version": int(self.version) if self.version is not None else None,
                "resume": self.resume,
                "commands": list(self.commands) if self.commands is not None else None,
                "resume_state": self.resume_state.to_dict() if self.resume_state is not None else None,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureToggle":
        mapping = _as_mapping(data, "session.welcome payload.features.<topic>")
        _ensure_keyset(
            mapping,
            required=("enabled",),
            optional=("version", "resume", "commands", "resume_state"),
            context="session.welcome feature",
        )
        commands_payload = mapping.get("commands")
        commands: Tuple[str, ...] | None = None
        if commands_payload is not None:
            commands = tuple(
                str(item) for item in _as_sequence(commands_payload, "session.welcome feature.commands")
            )
        resume_value = mapping.get("resume")
        resume: bool | None
        if resume_value is None:
            resume = None
        else:
            resume = bool(resume_value)
        resume_state_payload = mapping.get("resume_state")
        resume_state: FeatureResumeState | None
        if resume_state_payload is None:
            resume_state = None
        else:
            resume_state = FeatureResumeState.from_dict(resume_state_payload)
        return cls(
            enabled=bool(mapping["enabled"]),
            version=int(mapping["version"]) if mapping.get("version") is not None else None,
            resume=resume,
            commands=commands,
            resume_state=resume_state,
        )


@dataclass(slots=True)
class SessionWelcomePayload:
    protocol_version: int
    session: WelcomeSessionInfo
    features: Dict[str, FeatureToggle]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": int(self.protocol_version),
            "session": self.session.to_dict(),
            "features": {name: toggle.to_dict() for name, toggle in self.features.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionWelcomePayload":
        mapping = _as_mapping(data, "session.welcome payload")
        _ensure_keyset(
            mapping,
            required=("protocol_version", "session", "features"),
            context="session.welcome payload",
        )
        features: Dict[str, FeatureToggle] = {}
        for key, value in _as_mapping(mapping["features"], "session.welcome payload.features").items():
            features[str(key)] = FeatureToggle.from_dict(value)
        return cls(
            protocol_version=int(mapping["protocol_version"]),
            session=WelcomeSessionInfo.from_dict(mapping["session"]),
            features=features,
        )


@dataclass(slots=True)
class SessionRejectPayload:
    code: str
    message: str
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({"code": self.code, "message": self.message, "details": self.details})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionRejectPayload":
        mapping = _as_mapping(data, "session.reject payload")
        if "code" not in mapping or "message" not in mapping:
            raise ValueError("session.reject payload requires 'code' and 'message'")
        return cls(
            code=str(mapping["code"]),
            message=str(mapping["message"]),
            details=_optional_mapping(mapping.get("details"), "session.reject payload.details"),
        )


@dataclass(slots=True)
class SessionGoodbyePayload:
    code: str | None = None
    message: str | None = None
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({"code": self.code, "message": self.message, "reason": self.reason})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionGoodbyePayload":
        mapping = _as_mapping(data, "session.goodbye payload")
        _ensure_keyset(
            mapping,
            required=(),
            optional=("code", "message", "reason"),
            context="session.goodbye payload",
        )
        return cls(
            code=_optional_str(mapping.get("code")),
            message=_optional_str(mapping.get("message")),
            reason=_optional_str(mapping.get("reason")),
        )


@dataclass(slots=True)
class SessionHeartbeatPayload:
    def to_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionHeartbeatPayload":
        mapping = _as_mapping(data, "session.heartbeat payload")
        _require(not mapping, "session.heartbeat payload must be empty")
        return cls()


@dataclass(slots=True)
class SessionAckPayload:
    def to_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionAckPayload":
        mapping = _as_mapping(data, "session.ack payload")
        _require(not mapping, "session.ack payload must be empty")
        return cls()


@dataclass(slots=True)
class NotifyScenePayload:
    viewer: Dict[str, Any]
    layers: Tuple[Dict[str, Any], ...]
    metadata: Dict[str, Any] | None = None
    policies: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "viewer": dict(self.viewer),
                "layers": [dict(item) for item in self.layers],
                "metadata": dict(self.metadata) if self.metadata is not None else None,
                "policies": dict(self.policies) if self.policies is not None else None,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyScenePayload":
        mapping = _as_mapping(data, "notify.scene payload")
        _ensure_keyset(
            mapping,
            required=("viewer", "layers"),
            optional=("metadata", "policies"),
            context="notify.scene payload",
        )
        layers = tuple(
            _as_mutable_mapping(item, "notify.scene payload.layers[]")
            for item in _as_sequence(mapping["layers"], "notify.scene payload.layers")
        )
        return cls(
            viewer=_as_mutable_mapping(mapping["viewer"], "notify.scene payload.viewer"),
            layers=layers,
            metadata=_optional_mapping(mapping.get("metadata"), "notify.scene payload.metadata"),
            policies=_optional_mapping(mapping.get("policies"), "notify.scene payload.policies"),
        )


@dataclass(slots=True)
class NotifySceneLevelPayload:
    current_level: int
    downgraded: bool | None = None
    levels: Tuple[Dict[str, Any], ...] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "current_level": int(self.current_level),
                "downgraded": bool(self.downgraded)
                if self.downgraded is not None
                else None,
                "levels": [dict(entry) for entry in self.levels] if self.levels else None,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifySceneLevelPayload":
        mapping = _as_mapping(data, "notify.scene.level payload")
        _ensure_keyset(
            mapping,
            required=("current_level",),
            optional=("downgraded", "levels"),
            context="notify.scene.level payload",
        )
        try:
            current_level = int(mapping["current_level"])
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("notify.scene.level current_level must be an int") from exc
        levels_payload = mapping.get("levels")
        levels: Tuple[Dict[str, Any], ...] | None = None
        if levels_payload is not None:
            levels = tuple(
                _as_mutable_mapping(entry, "notify.scene.level payload.levels[]")
                for entry in _as_sequence(levels_payload, "notify.scene.level payload.levels")
            )
        downgraded_raw = mapping.get("downgraded")
        downgraded: bool | None
        if downgraded_raw is None:
            downgraded = None
        else:
            downgraded = bool(downgraded_raw)
        return cls(current_level=current_level, downgraded=downgraded, levels=levels)


@dataclass(slots=True)
class NotifyLayersPayload:
    """Structured per-layer delta.

    At least one of controls, metadata, data, thumbnail, or removed must be provided.
    """

    layer_id: str
    controls: Dict[str, Any] | None = None
    metadata: Dict[str, Any] | None = None
    data: Dict[str, Any] | None = None
    thumbnail: Dict[str, Any] | None = None
    removed: bool | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"layer_id": self.layer_id}
        if self.controls:
            payload["controls"] = dict(self.controls)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.data:
            payload["data"] = dict(self.data)
        if self.thumbnail:
            payload["thumbnail"] = dict(self.thumbnail)
        if self.removed:
            payload["removed"] = True
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyLayersPayload":
        mapping = _as_mapping(data, "notify.layers payload")
        _ensure_keyset(
            mapping,
            required=("layer_id",),
            optional=("controls", "metadata", "data", "thumbnail", "removed"),
            context="notify.layers payload",
        )
        provided = any(name in mapping for name in ("controls", "metadata", "data", "thumbnail", "removed"))
        if not provided:
            raise ValueError("notify.layers payload requires at least one section or 'removed'")
        return cls(
            layer_id=str(mapping["layer_id"]),
            controls=_optional_mapping(mapping.get("controls"), "notify.layers payload.controls"),
            metadata=_optional_mapping(mapping.get("metadata"), "notify.layers payload.metadata"),
            data=_optional_mapping(mapping.get("data"), "notify.layers payload.data"),
            thumbnail=_optional_mapping(mapping.get("thumbnail"), "notify.layers payload.thumbnail"),
            removed=bool(mapping.get("removed")) if mapping.get("removed") is not None else None,
        )


@dataclass(slots=True)
class NotifyStreamPayload:
    codec: str
    format: str
    fps: float
    frame_size: Tuple[int, int]
    nal_length_size: int
    avcc: str
    latency_policy: Dict[str, Any]
    vt_hint: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "codec": self.codec,
                "format": self.format,
                "fps": float(self.fps),
                "frame_size": list(self.frame_size),
                "nal_length_size": int(self.nal_length_size),
                "avcc": self.avcc,
                "latency_policy": dict(self.latency_policy),
                "vt_hint": dict(self.vt_hint) if self.vt_hint is not None else None,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyStreamPayload":
        mapping = _as_mapping(data, "notify.stream payload")
        _ensure_keyset(
            mapping,
            required=(
                "codec",
                "format",
                "fps",
                "frame_size",
                "nal_length_size",
                "avcc",
                "latency_policy",
            ),
            optional=("vt_hint",),
            context="notify.stream payload",
        )
        frame_size_seq = _as_sequence(mapping["frame_size"], "notify.stream payload.frame_size")
        if len(frame_size_seq) != 2:
            raise ValueError("notify.stream payload.frame_size must have length 2")
        return cls(
            codec=str(mapping["codec"]),
            format=str(mapping["format"]),
            fps=float(mapping["fps"]),
            frame_size=(int(frame_size_seq[0]), int(frame_size_seq[1])),
            nal_length_size=int(mapping["nal_length_size"]),
            avcc=str(mapping["avcc"]),
            latency_policy=_as_mutable_mapping(mapping["latency_policy"], "notify.stream payload.latency_policy"),
            vt_hint=_optional_mapping(mapping.get("vt_hint"), "notify.stream payload.vt_hint"),
        )


@dataclass(slots=True)
class NotifyDimsPayload:
    current_step: Tuple[int, ...]
    level_shapes: Tuple[Tuple[int, ...], ...]
    levels: Tuple[Dict[str, Any], ...]
    current_level: int
    downgraded: bool | None
    mode: str
    ndisplay: int
    axis_labels: Tuple[str, ...] | None
    order: Tuple[int, ...] | None
    displayed: Tuple[int, ...] | None
    labels: Tuple[str, ...] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step": [int(v) for v in self.current_step],
            "current_step": [int(v) for v in self.current_step],
            "levels": [dict(level) for level in self.levels],
            "current_level": int(self.current_level),
            "mode": str(self.mode),
            "ndisplay": int(self.ndisplay),
        }

        payload["level_shapes"] = [[int(v) for v in shape] for shape in self.level_shapes]

        if self.downgraded is not None:
            payload["downgraded"] = bool(self.downgraded)
        if self.axis_labels is not None:
            payload["axis_labels"] = [str(lbl) for lbl in self.axis_labels]
        if self.order is not None:
            payload["order"] = [int(idx) for idx in self.order]
        if self.displayed is not None:
            payload["displayed"] = [int(idx) for idx in self.displayed]
        if self.labels is not None:
            payload["labels"] = [str(label) for label in self.labels]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyDimsPayload":
        mapping = _as_mapping(data, "notify.dims payload")
        _ensure_keyset(
            mapping,
            required=("step", "levels", "current_level", "mode", "ndisplay", "level_shapes"),
            optional=("downgraded", "axis_labels", "order", "displayed", "labels", "current_step"),
            context="notify.dims payload",
        )

        step_seq = _as_sequence(mapping["step"], "notify.dims payload.step")
        current_step_seq = _as_sequence(mapping.get("current_step", step_seq), "notify.dims payload.current_step")
        levels_seq = _as_sequence(mapping["levels"], "notify.dims payload.levels")

        downgraded_value = mapping.get("downgraded")
        downgraded = bool(downgraded_value) if downgraded_value is not None else None

        axis_labels_value = mapping.get("axis_labels")
        axis_labels = (
            tuple(str(lbl) for lbl in _as_sequence(axis_labels_value, "notify.dims payload.axis_labels"))
            if axis_labels_value is not None
            else None
        )

        order_value = mapping.get("order")
        order = (
            tuple(int(idx) for idx in _as_sequence(order_value, "notify.dims payload.order"))
            if order_value is not None
            else None
        )

        displayed_value = mapping.get("displayed")
        displayed = (
            tuple(int(idx) for idx in _as_sequence(displayed_value, "notify.dims payload.displayed"))
            if displayed_value is not None
            else None
        )

        labels_value = mapping.get("labels")
        labels = (
            tuple(str(lbl) for lbl in _as_sequence(labels_value, "notify.dims payload.labels"))
            if labels_value is not None
            else None
        )

        level_shapes_value = mapping["level_shapes"]
        level_shapes_seq = _as_sequence(level_shapes_value, "notify.dims payload.level_shapes")
        parsed_shapes: list[tuple[int, ...]] = []
        for idx, entry in enumerate(level_shapes_seq):
            shape_seq = _as_sequence(entry, f"notify.dims payload.level_shapes[{idx}]")
            parsed_shapes.append(tuple(int(dim) for dim in shape_seq))
        level_shapes = tuple(parsed_shapes)

        levels: list[Dict[str, Any]] = []
        for idx, entry in enumerate(levels_seq):
            levels.append(dict(_as_mapping(entry, f"notify.dims payload.levels[{idx}]")))

        return cls(
            current_step=tuple(int(v) for v in current_step_seq),
            level_shapes=level_shapes,
            levels=tuple(levels),
            current_level=int(mapping["current_level"]),
            downgraded=downgraded,
            mode=str(mapping["mode"]),
            ndisplay=int(mapping["ndisplay"]),
            axis_labels=axis_labels,
            order=order,
            displayed=displayed,
            labels=labels,
        )


@dataclass(slots=True)
class NotifyCameraPayload:
    mode: str
    origin: str
    delta: Dict[str, Any] | None = None
    state: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"mode": self.mode, "origin": self.origin}
        if self.delta is not None:
            result["delta"] = dict(self.delta)
        if self.state is not None:
            result["state"] = dict(self.state)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyCameraPayload":
        mapping = _as_mapping(data, "notify.camera payload")
        _ensure_keyset(
            mapping,
            required=("mode", "origin"),
            optional=("delta", "state"),
            context="notify.camera payload",
        )
        delta_obj: Dict[str, Any] | None = None
        if "delta" in mapping and mapping["delta"] is not None:
            delta_obj = _as_mutable_mapping(mapping["delta"], "notify.camera payload.delta")
        state_obj: Dict[str, Any] | None = None
        if "state" in mapping and mapping["state"] is not None:
            state_obj = _as_mutable_mapping(mapping["state"], "notify.camera payload.state")
        if delta_obj is None and state_obj is None:
            raise ValueError("notify.camera payload requires delta or state field")
        return cls(
            mode=str(mapping["mode"]),
            origin=str(mapping["origin"]),
            delta=delta_obj,
            state=state_obj,
        )


@dataclass(slots=True)
class NotifyTelemetryPayload:
    presenter: float
    decode: float
    queue_depth: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "presenter": float(self.presenter),
            "decode": float(self.decode),
            "queue_depth": float(self.queue_depth),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyTelemetryPayload":
        mapping = _as_mapping(data, "notify.telemetry payload")
        _ensure_keyset(
            mapping,
            required=("presenter", "decode", "queue_depth"),
            context="notify.telemetry payload",
        )
        return cls(
            presenter=float(mapping["presenter"]),
            decode=float(mapping["decode"]),
            queue_depth=float(mapping["queue_depth"]),
        )


@dataclass(slots=True)
class NotifyErrorPayload:
    domain: str
    code: str
    message: str
    severity: str
    context: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "domain": self.domain,
                "code": self.code,
                "message": self.message,
                "severity": self.severity,
                "context": dict(self.context) if self.context is not None else None,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyErrorPayload":
        mapping = _as_mapping(data, "notify.error payload")
        _ensure_keyset(
            mapping,
            required=("domain", "code", "message", "severity"),
            optional=("context",),
            context="notify.error payload",
        )
        severity = str(mapping["severity"])
        if severity not in _ERROR_SEVERITIES:
            raise ValueError(f"notify.error severity must be one of {sorted(_ERROR_SEVERITIES)}")
        return cls(
            domain=str(mapping["domain"]),
            code=str(mapping["code"]),
            message=str(mapping["message"]),
            severity=severity,
            context=_optional_mapping(mapping.get("context"), "notify.error payload.context"),
        )


@dataclass(slots=True)
class StateUpdatePayload:
    scope: str
    target: str
    key: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "target": self.target,
            "key": self.key,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateUpdatePayload":
        mapping = _as_mapping(data, "state.update payload")
        required = {"scope", "target", "key", "value"}
        missing = [key for key in required if key not in mapping]
        if missing:
            raise ValueError(f"state.update payload missing fields: {', '.join(missing)}")
        return cls(
            scope=str(mapping["scope"]),
            target=str(mapping["target"]),
            key=str(mapping["key"]),
            value=mapping["value"],
        )


@dataclass(slots=True)
class StateErrorPayload:
    code: str
    message: str
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({"code": self.code, "message": self.message, "details": self.details})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateErrorPayload":
        mapping = _as_mapping(data, "ack.state payload.error")
        if "code" not in mapping or "message" not in mapping:
            raise ValueError("ack.state error requires 'code' and 'message'")
        return cls(
            code=str(mapping["code"]),
            message=str(mapping["message"]),
            details=_optional_mapping(mapping.get("details"), "ack.state payload.error.details"),
        )


@dataclass(slots=True)
class AckStatePayload:
    intent_id: str
    in_reply_to: str
    status: str
    applied_value: Any | None = None
    error: StateErrorPayload | None = None
    version: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "intent_id": self.intent_id,
            "in_reply_to": self.in_reply_to,
            "status": self.status,
            "applied_value": self.applied_value,
            "error": self.error.to_dict() if self.error else None,
            "version": self.version,
        }
        return _strip_none(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AckStatePayload":
        mapping = _as_mapping(data, "ack.state payload")
        required = {"intent_id", "in_reply_to", "status"}
        missing = [key for key in required if key not in mapping]
        if missing:
            raise ValueError(f"ack.state payload missing fields: {', '.join(missing)}")
        status_value = str(mapping["status"])
        if status_value not in _ACK_STATUSES:
            raise ValueError("ack.state status must be 'accepted' or 'rejected'")
        error_payload = mapping.get("error")
        error = StateErrorPayload.from_dict(error_payload) if isinstance(error_payload, Mapping) else None
        if status_value == "accepted" and error is not None:
            raise ValueError("ack.state accepted payload cannot include error details")
        if status_value == "rejected" and error is None:
            raise ValueError("ack.state rejected payload requires error details")
        raw_version = mapping.get("version")
        if status_value == "accepted":
            if raw_version is None:
                raise ValueError("ack.state accepted payload requires version")
            if not isinstance(raw_version, Integral):
                raise ValueError("ack.state version must be integer")
            version = int(raw_version)
        else:
            version = None
            if raw_version is not None:
                if not isinstance(raw_version, Integral):
                    raise ValueError("ack.state version must be integer")
                version = int(raw_version)
        return cls(
            intent_id=str(mapping["intent_id"]),
            in_reply_to=str(mapping["in_reply_to"]),
            status=status_value,
            applied_value=mapping.get("applied_value"),
            error=error,
            version=version,
        )


@dataclass(slots=True)
class CallCommandPayload:
    command: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    origin: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "command": self.command,
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "origin": self.origin,
        }
        return _strip_none(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallCommandPayload":
        mapping = _as_mapping(data, "call.command payload")
        if "command" not in mapping:
            raise ValueError("call.command payload requires 'command'")
        args_payload = mapping.get("args")
        kwargs_payload = mapping.get("kwargs")
        args: Tuple[Any, ...] = ()
        if args_payload is not None:
            args = tuple(_as_sequence(args_payload, "call.command payload.args"))
        kwargs: Dict[str, Any] = {}
        if kwargs_payload is not None:
            kwargs = _as_mutable_mapping(kwargs_payload, "call.command payload.kwargs")
        return cls(
            command=str(mapping["command"]),
            args=args,
            kwargs=kwargs,
            origin=_optional_str(mapping.get("origin")),
        )


@dataclass(slots=True)
class ReplyCommandPayload:
    in_reply_to: str
    status: str
    result: Any | None = None
    idempotency_key: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "in_reply_to": self.in_reply_to,
            "status": self.status,
            "result": self.result,
            "idempotency_key": self.idempotency_key,
        }
        return _strip_none(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplyCommandPayload":
        mapping = _as_mapping(data, "reply.command payload")
        required = {"in_reply_to", "status"}
        missing = [key for key in required if key not in mapping]
        if missing:
            raise ValueError(f"reply.command payload missing fields: {', '.join(missing)}")
        status_value = str(mapping["status"])
        if status_value != _COMMAND_SUCCESS_STATUS:
            raise ValueError("reply.command status must be 'ok'")
        return cls(
            in_reply_to=str(mapping["in_reply_to"]),
            status=status_value,
            result=mapping.get("result"),
            idempotency_key=_optional_str(mapping.get("idempotency_key")),
        )


@dataclass(slots=True)
class ErrorCommandPayload:
    in_reply_to: str
    status: str
    code: str
    message: str
    details: Dict[str, Any] | None = None
    idempotency_key: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "in_reply_to": self.in_reply_to,
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "idempotency_key": self.idempotency_key,
        }
        return _strip_none(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorCommandPayload":
        mapping = _as_mapping(data, "error.command payload")
        required = {"in_reply_to", "status", "code", "message"}
        missing = [key for key in required if key not in mapping]
        if missing:
            raise ValueError(f"error.command payload missing fields: {', '.join(missing)}")
        status_value = str(mapping["status"])
        if status_value != _COMMAND_ERROR_STATUS:
            raise ValueError("error.command status must be 'error'")
        return cls(
            in_reply_to=str(mapping["in_reply_to"]),
            status=status_value,
            code=str(mapping["code"]),
            message=str(mapping["message"]),
            details=_optional_mapping(mapping.get("details"), "error.command payload.details"),
            idempotency_key=_optional_str(mapping.get("idempotency_key")),
        )


@dataclass(slots=True)
class SessionHelloFrame:
    envelope: FrameEnvelope
    payload: SessionHelloPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == SESSION_HELLO_TYPE, "session.hello frame must have type=session.hello")
        _require(
            env.version == PROTO_VERSION,
            f"session.hello must advertise protocol version {PROTO_VERSION}",
        )
        _require(env.session is None, "session.hello must not include session id")
        _require(env.frame_id is not None, "session.hello requires frame_id")
        _require(env.timestamp is not None, "session.hello requires timestamp")
        _require(env.seq is None, "session.hello forbids seq")
        _require(env.delta_token is None, "session.hello forbids delta_token")
        _require(env.intent_id is None, "session.hello forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionHelloFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=SessionHelloPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class SessionWelcomeFrame:
    envelope: FrameEnvelope
    payload: SessionWelcomePayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == SESSION_WELCOME_TYPE, "session.welcome frame must have type=session.welcome")
        _require(
            env.version == PROTO_VERSION,
            f"session.welcome must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "session.welcome requires session id")
        _require(env.frame_id is not None, "session.welcome requires frame_id")
        _require(env.timestamp is not None, "session.welcome requires timestamp")
        _require(env.seq is None and env.delta_token is None, "session.welcome forbids seq/delta_token")
        _require(env.intent_id is None, "session.welcome forbids intent_id")
        _require(env.session == self.payload.session.id, "envelope.session must match payload.session.id")
        _require(
            self.payload.protocol_version == PROTO_VERSION,
            f"session.welcome payload.protocol_version must equal {PROTO_VERSION}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionWelcomeFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=SessionWelcomePayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class SessionRejectFrame:
    envelope: FrameEnvelope
    payload: SessionRejectPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == SESSION_REJECT_TYPE, "session.reject frame must have type=session.reject")
        _require(
            env.version == PROTO_VERSION,
            f"session.reject must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is None, "session.reject must not include session id")
        _require(env.frame_id is not None, "session.reject requires frame_id")
        _require(env.timestamp is not None, "session.reject requires timestamp")
        _require(env.seq is None and env.delta_token is None, "session.reject forbids seq/delta_token")
        _require(env.intent_id is None, "session.reject forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionRejectFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=SessionRejectPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class SessionHeartbeatFrame:
    envelope: FrameEnvelope
    payload: SessionHeartbeatPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == SESSION_HEARTBEAT_TYPE, "session.heartbeat frame must have type=session.heartbeat")
        _require(
            env.version == PROTO_VERSION,
            f"session.heartbeat must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "session.heartbeat requires session id")
        _require(env.frame_id is not None, "session.heartbeat requires frame_id")
        _require(env.timestamp is not None, "session.heartbeat requires timestamp")
        _require(env.seq is None and env.delta_token is None, "session.heartbeat forbids seq/delta_token")
        _require(env.intent_id is None, "session.heartbeat forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionHeartbeatFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=SessionHeartbeatPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class SessionAckFrame:
    envelope: FrameEnvelope
    payload: SessionAckPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == SESSION_ACK_TYPE, "session.ack frame must have type=session.ack")
        _require(
            env.version == PROTO_VERSION,
            f"session.ack must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "session.ack requires session id")
        _require(env.frame_id is not None, "session.ack requires frame_id")
        _require(env.timestamp is not None, "session.ack requires timestamp")
        _require(env.seq is None and env.delta_token is None, "session.ack forbids seq/delta_token")
        _require(env.intent_id is None, "session.ack forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionAckFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=SessionAckPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class SessionGoodbyeFrame:
    envelope: FrameEnvelope
    payload: SessionGoodbyePayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == SESSION_GOODBYE_TYPE, "session.goodbye frame must have type=session.goodbye")
        _require(
            env.version == PROTO_VERSION,
            f"session.goodbye must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "session.goodbye requires session id")
        _require(env.frame_id is not None, "session.goodbye requires frame_id")
        _require(env.timestamp is not None, "session.goodbye requires timestamp")
        _require(env.seq is None and env.delta_token is None, "session.goodbye forbids seq/delta_token")
        _require(env.intent_id is None, "session.goodbye forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionGoodbyeFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=SessionGoodbyePayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifySceneFrame:
    envelope: FrameEnvelope
    payload: NotifyScenePayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == NOTIFY_SCENE_TYPE, "notify.scene frame must have type=notify.scene")
        _require(
            env.version == PROTO_VERSION,
            f"notify.scene must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.scene requires session id")
        _require(env.timestamp is not None, "notify.scene requires timestamp")
        _require(env.seq is not None, "notify.scene requires seq")
        _require(env.seq == 0, "notify.scene snapshot must use seq=0")
        _require(env.delta_token is not None, "notify.scene requires delta_token")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifySceneFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifyScenePayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifySceneLevelFrame:
    envelope: FrameEnvelope
    payload: NotifySceneLevelPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(
            env.type == NOTIFY_SCENE_LEVEL_TYPE,
            "notify.scene.level frame must have type=notify.scene.level",
        )
        _require(
            env.version == PROTO_VERSION,
            f"notify.scene.level must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.scene.level requires session id")
        _require(env.timestamp is not None, "notify.scene.level requires timestamp")
        _require(env.seq is not None, "notify.scene.level requires seq")
        _require(env.delta_token is not None, "notify.scene.level requires delta_token")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifySceneLevelFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifySceneLevelPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifyLayersFrame:
    envelope: FrameEnvelope
    payload: NotifyLayersPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == NOTIFY_LAYERS_TYPE, "notify.layers frame must have type=notify.layers")
        _require(
            env.version == PROTO_VERSION,
            f"notify.layers must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.layers requires session id")
        _require(env.timestamp is not None, "notify.layers requires timestamp")
        _require(env.seq is not None, "notify.layers requires seq")
        _require(env.delta_token is not None, "notify.layers requires delta_token")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyLayersFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifyLayersPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifyStreamFrame:
    envelope: FrameEnvelope
    payload: NotifyStreamPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == NOTIFY_STREAM_TYPE, "notify.stream frame must have type=notify.stream")
        _require(
            env.version == PROTO_VERSION,
            f"notify.stream must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.stream requires session id")
        _require(env.timestamp is not None, "notify.stream requires timestamp")
        _require(env.seq is not None, "notify.stream requires seq")
        _require(env.delta_token is not None, "notify.stream requires delta_token")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyStreamFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifyStreamPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifyDimsFrame:
    envelope: FrameEnvelope
    payload: NotifyDimsPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == NOTIFY_DIMS_TYPE, "notify.dims frame must have type=notify.dims")
        _require(
            env.version == PROTO_VERSION,
            f"notify.dims must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.dims requires session id")
        _require(env.timestamp is not None, "notify.dims requires timestamp")
        _require(env.seq is None and env.delta_token is None, "notify.dims forbids seq/delta_token")
        _require(env.intent_id is not None or env.frame_id is not None, "notify.dims requires intent_id or frame_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyDimsFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifyDimsPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifyCameraFrame:
    envelope: FrameEnvelope
    payload: NotifyCameraPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == NOTIFY_CAMERA_TYPE, "notify.camera frame must have type=notify.camera")
        _require(
            env.version == PROTO_VERSION,
            f"notify.camera must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.camera requires session id")
        _require(env.timestamp is not None, "notify.camera requires timestamp")
        _require(env.seq is None and env.delta_token is None, "notify.camera forbids seq/delta_token")
        _require(env.intent_id is not None or env.frame_id is not None, "notify.camera requires intent_id or frame_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyCameraFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifyCameraPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifyTelemetryFrame:
    envelope: FrameEnvelope
    payload: NotifyTelemetryPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == NOTIFY_TELEMETRY_TYPE, "notify.telemetry frame must have type=notify.telemetry")
        _require(
            env.version == PROTO_VERSION,
            f"notify.telemetry must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.telemetry requires session id")
        _require(env.timestamp is not None, "notify.telemetry requires timestamp")
        _require(env.seq is None and env.delta_token is None, "notify.telemetry forbids seq/delta_token")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyTelemetryFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifyTelemetryPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class NotifyErrorFrame:
    envelope: FrameEnvelope
    payload: NotifyErrorPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == NOTIFY_ERROR_TYPE, "notify.error frame must have type=notify.error")
        _require(
            env.version == PROTO_VERSION,
            f"notify.error must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "notify.error requires session id")
        _require(env.timestamp is not None, "notify.error requires timestamp")
        _require(env.seq is None and env.delta_token is None, "notify.error forbids seq/delta_token")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotifyErrorFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=NotifyErrorPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class StateUpdateFrame:
    envelope: FrameEnvelope
    payload: StateUpdatePayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == STATE_UPDATE_TYPE, "state.update frame must have type=state.update")
        _require(
            env.version == PROTO_VERSION,
            f"state.update must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "state.update requires session id")
        _require(env.frame_id is not None, "state.update requires frame_id")
        _require(env.timestamp is not None, "state.update requires timestamp")
        _require(env.intent_id is not None, "state.update requires intent_id")
        _require(env.seq is None and env.delta_token is None, "state.update forbids seq/delta_token")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateUpdateFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=StateUpdatePayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class AckStateFrame:
    envelope: FrameEnvelope
    payload: AckStatePayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == ACK_STATE_TYPE, "ack.state frame must have type=ack.state")
        _require(
            env.version == PROTO_VERSION,
            f"ack.state must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "ack.state requires session id")
        _require(env.frame_id is not None, "ack.state requires frame_id")
        _require(env.timestamp is not None, "ack.state requires timestamp")
        _require(env.seq is None and env.delta_token is None, "ack.state forbids seq/delta_token")
        _require(env.intent_id is None, "ack.state must carry intent only in payload")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AckStateFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=AckStatePayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class CallCommandFrame:
    envelope: FrameEnvelope
    payload: CallCommandPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == CALL_COMMAND_TYPE, "call.command frame must have type=call.command")
        _require(
            env.version == PROTO_VERSION,
            f"call.command must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "call.command requires session id")
        _require(env.frame_id is not None, "call.command requires frame_id")
        _require(env.timestamp is not None, "call.command requires timestamp")
        _require(env.seq is None and env.delta_token is None, "call.command forbids seq/delta_token")
        _require(env.intent_id is None, "call.command forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallCommandFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=CallCommandPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class ReplyCommandFrame:
    envelope: FrameEnvelope
    payload: ReplyCommandPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == REPLY_COMMAND_TYPE, "reply.command frame must have type=reply.command")
        _require(
            env.version == PROTO_VERSION,
            f"reply.command must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "reply.command requires session id")
        _require(env.frame_id is not None, "reply.command requires frame_id")
        _require(env.timestamp is not None, "reply.command requires timestamp")
        _require(env.seq is None and env.delta_token is None, "reply.command forbids seq/delta_token")
        _require(env.intent_id is None, "reply.command forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplyCommandFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=ReplyCommandPayload.from_dict(payload))
        frame._validate()
        return frame


@dataclass(slots=True)
class ErrorCommandFrame:
    envelope: FrameEnvelope
    payload: ErrorCommandPayload

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        env = self.envelope
        _require(env.type == ERROR_COMMAND_TYPE, "error.command frame must have type=error.command")
        _require(
            env.version == PROTO_VERSION,
            f"error.command must use protocol version {PROTO_VERSION}",
        )
        _require(env.session is not None, "error.command requires session id")
        _require(env.frame_id is not None, "error.command requires frame_id")
        _require(env.timestamp is not None, "error.command requires timestamp")
        _require(env.seq is None and env.delta_token is None, "error.command forbids seq/delta_token")
        _require(env.intent_id is None, "error.command forbids intent_id")

    def to_dict(self) -> Dict[str, Any]:
        return _frame_dict(self.envelope, self.payload.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorCommandFrame":
        envelope, payload = _pull_payload(data)
        frame = cls(envelope=envelope, payload=ErrorCommandPayload.from_dict(payload))
        frame._validate()
        return frame


# Backwards-compatible aliases matching historical envelope names.  These will
# disappear once all call sites migrate to the explicit *Frame classes.
SessionHello = SessionHelloFrame
SessionWelcome = SessionWelcomeFrame
SessionReject = SessionRejectFrame
SessionHeartbeat = SessionHeartbeatFrame
SessionAck = SessionAckFrame
SessionGoodbye = SessionGoodbyeFrame
NotifyScene = NotifySceneFrame
NotifySceneLevel = NotifySceneLevelFrame
NotifyLayers = NotifyLayersFrame
NotifyStream = NotifyStreamFrame
NotifyDims = NotifyDimsFrame
NotifyCamera = NotifyCameraFrame
NotifyTelemetry = NotifyTelemetryFrame
NotifyError = NotifyErrorFrame
StateUpdate = StateUpdateFrame
AckState = AckStateFrame
CallCommand = CallCommandFrame
ReplyCommand = ReplyCommandFrame
ErrorCommand = ErrorCommandFrame


__all__ = [
    # Protocol constants
    "PROTO_VERSION",
    "SESSION_HELLO_TYPE",
    "SESSION_WELCOME_TYPE",
    "SESSION_REJECT_TYPE",
    "SESSION_HEARTBEAT_TYPE",
    "SESSION_ACK_TYPE",
    "SESSION_GOODBYE_TYPE",
    "NOTIFY_SCENE_TYPE",
    "NOTIFY_SCENE_LEVEL_TYPE",
    "NOTIFY_LAYERS_TYPE",
    "NOTIFY_STREAM_TYPE",
    "NOTIFY_DIMS_TYPE",
    "NOTIFY_CAMERA_TYPE",
    "NOTIFY_TELEMETRY_TYPE",
    "NOTIFY_ERROR_TYPE",
    "STATE_UPDATE_TYPE",
    "ACK_STATE_TYPE",
    "CALL_COMMAND_TYPE",
    "REPLY_COMMAND_TYPE",
    "ERROR_COMMAND_TYPE",
    # Envelope helpers
    "FrameEnvelope",
    # Session payloads/frames
    "HelloClientInfo",
    "HelloAuthInfo",
    "SessionHelloPayload",
    "WelcomeSessionInfo",
    "FeatureResumeState",
    "FeatureToggle",
    "SessionWelcomePayload",
    "SessionRejectPayload",
    "SessionGoodbyePayload",
    "SessionHeartbeatPayload",
    "SessionAckPayload",
    "SessionHelloFrame",
    "SessionWelcomeFrame",
    "SessionRejectFrame",
    "SessionHeartbeatFrame",
    "SessionAckFrame",
    "SessionGoodbyeFrame",
    "SessionHello",
    "SessionWelcome",
    "SessionReject",
    "SessionHeartbeat",
    "SessionAck",
    "SessionGoodbye",
    # Notify payloads/frames
    "NotifyScenePayload",
    "NotifySceneLevelPayload",
    "NotifyLayersPayload",
    "NotifyStreamPayload",
    "NotifyDimsPayload",
    "NotifyCameraPayload",
    "NotifyTelemetryPayload",
    "NotifyErrorPayload",
    "NotifySceneFrame",
    "NotifySceneLevelFrame",
    "NotifyLayersFrame",
    "NotifyStreamFrame",
    "NotifyDimsFrame",
    "NotifyCameraFrame",
    "NotifyTelemetryFrame",
    "NotifyErrorFrame",
    "NotifyScene",
    "NotifySceneLevel",
    "NotifyLayers",
    "NotifyStream",
    "NotifyDims",
    "NotifyCamera",
    "NotifyTelemetry",
    "NotifyError",
    # State lane payloads/frames
    "StateUpdatePayload",
    "StateErrorPayload",
    "AckStatePayload",
    "StateUpdateFrame",
    "AckStateFrame",
    "StateUpdate",
    "AckState",
    # Command lane payloads/frames
    "CallCommandPayload",
    "ReplyCommandPayload",
    "ErrorCommandPayload",
    "CallCommandFrame",
    "ReplyCommandFrame",
    "ErrorCommandFrame",
    "CallCommand",
    "ReplyCommand",
    "ErrorCommand",
]
