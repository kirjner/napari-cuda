"""Helper builders for spec-compliant greenfield protocol envelopes.

This module wraps the dataclasses defined in :mod:`napari_cuda.protocol.greenfield.messages`
with convenience constructors that enforce the envelope contract captured in
``docs/protocol_greenfield.md`` (see Appendix B).

It deliberately re-exports every frame/payload class so legacy imports that used
``napari_cuda.protocol.envelopes`` keep working while we migrate call sites to
the new builders.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .messages import *  # noqa: F401,F403 - compatibility surface
from .messages import __all__ as _messages_all
from .messages import (
    PROTO_VERSION,
    ACK_STATE_TYPE,
    CALL_COMMAND_TYPE,
    ERROR_COMMAND_TYPE,
    FrameEnvelope,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_STREAM_TYPE,
    REPLY_COMMAND_TYPE,
    SESSION_HEARTBEAT_TYPE,
    SESSION_HELLO_TYPE,
    SESSION_REJECT_TYPE,
    SESSION_WELCOME_TYPE,
    AckState,
    AckStatePayload,
    CallCommand,
    CallCommandPayload,
    ErrorCommand,
    ErrorCommandPayload,
    FeatureToggle,
    HelloAuthInfo,
    HelloClientInfo,
    NotifyLayers,
    NotifyLayersPayload,
    NotifyScene,
    NotifyScenePayload,
    NotifySceneLevel,
    NotifySceneLevelPayload,
    NotifyStream,
    NotifyStreamPayload,
    ReplyCommand,
    ReplyCommandPayload,
    SessionHeartbeat,
    SessionHeartbeatPayload,
    SessionHello,
    SessionHelloPayload,
    SessionReject,
    SessionRejectPayload,
    SessionWelcome,
    SessionWelcomePayload,
    StateUpdate,
    StateUpdatePayload,
    WelcomeSessionInfo,
)


@dataclass(slots=True)
class ResumableCursor:
    """Sequence/token pair returned by :class:`ResumableTopicSequencer`."""

    seq: int
    delta_token: str


class ResumableTopicSequencer:
    """Centralise seq/delta-token bookkeeping for resumable lanes."""

    __slots__ = ("_topic", "_seq", "_delta_token", "_token_factory")

    def __init__(
        self,
        *,
        topic: str,
        token_factory: Callable[[], str] | None = None,
        initial_seq: int | None = None,
        initial_token: str | None = None,
    ) -> None:
        self._topic = str(topic)
        self._token_factory = token_factory or (lambda: uuid.uuid4().hex)
        self._seq = int(initial_seq) if initial_seq is not None else -1
        self._delta_token = str(initial_token) if initial_token is not None else None

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def seq(self) -> int | None:
        return None if self._seq < 0 else self._seq

    @property
    def delta_token(self) -> str | None:
        return self._delta_token

    def snapshot(self, *, token: str | None = None) -> ResumableCursor:
        """Reset to seq=0 and return the new cursor."""

        new_token = token or self._next_token()
        self._seq = 0
        self._delta_token = new_token
        return ResumableCursor(seq=self._seq, delta_token=self._delta_token)

    def delta(self, *, token: str | None = None) -> ResumableCursor:
        """Advance the sequence for a delta frame and return the cursor."""

        if self._delta_token is None or self._seq < 0:
            raise ValueError(
                f"Cannot emit delta for topic '{self._topic}' before a snapshot resets the epoch",
            )
        new_token = token or self._next_token()
        self._seq += 1
        self._delta_token = new_token
        return ResumableCursor(seq=self._seq, delta_token=self._delta_token)

    def resume(self, *, seq: int, delta_token: str) -> None:
        """Seed the sequencer with server-provided resume state."""

        self._seq = int(seq)
        self._delta_token = str(delta_token)

    def clear(self) -> None:
        """Drop any cached state (e.g. when tearing down a session)."""

        self._seq = -1
        self._delta_token = None

    def _next_token(self) -> str:
        return str(self._token_factory())


def _now(timestamp: float | None) -> float:
    """Return *timestamp* if provided, otherwise the current time."""

    return float(timestamp) if timestamp is not None else time.time()


def _frame_id(frame_id: str | None, prefix: str | None = None) -> str:
    """Return *frame_id* or generate a deterministic UUID hex (optionally prefixed)."""

    if frame_id:
        return str(frame_id)
    base = uuid.uuid4().hex
    return f"{prefix}-{base}" if prefix else base


def _short_frame_id(frame_id: str | None = None) -> str:
    """Return a short identifier (<= 8 chars) used by hot-path notify lanes."""

    value = frame_id or uuid.uuid4().hex
    return value[:8]


def _resolve_resumable_cursor(
    *,
    topic: str,
    sequencer: ResumableTopicSequencer | None,
    seq: int | None,
    delta_token: str | None,
    snapshot: bool,
) -> ResumableCursor:
    """Get the (seq, delta_token) pair for a resumable frame."""

    if sequencer is not None:
        if snapshot:
            return sequencer.snapshot(token=delta_token)
        return sequencer.delta(token=delta_token)
    if delta_token is None:
        raise ValueError(
            f"{topic} requires a delta_token when no sequencer is provided",
        )
    if snapshot:
        return ResumableCursor(seq=0, delta_token=str(delta_token))
    if seq is None:
        raise ValueError(f"{topic} delta requires seq when no sequencer is provided")
    return ResumableCursor(seq=int(seq), delta_token=str(delta_token))


def _coerce_feature_toggles(
    features: Mapping[str, FeatureToggle | Mapping[str, Any] | bool],
) -> dict[str, FeatureToggle]:
    """Normalise feature toggle inputs into :class:`FeatureToggle` objects."""

    toggles: dict[str, FeatureToggle] = {}
    for name, value in features.items():
        key = str(name)
        if isinstance(value, FeatureToggle):
            toggles[key] = value
            continue
        if isinstance(value, Mapping):
            toggles[key] = FeatureToggle.from_dict(value)
            continue
        toggles[key] = FeatureToggle(enabled=bool(value))
    return toggles


def build_session_hello(
    *,
    client: HelloClientInfo,
    features: Mapping[str, bool],
    resume_tokens: Mapping[str, str | None] | None = None,
    protocols: Iterable[int] | None = None,
    auth: HelloAuthInfo | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
    frame_id_prefix: str | None = "hello",
) -> SessionHello:
    """Construct a spec-compliant ``session.hello`` frame."""

    proto_versions = tuple(int(value) for value in (protocols or (PROTO_VERSION,)))
    feature_map = {str(name): bool(enabled) for name, enabled in features.items()}
    resume_map: dict[str, str | None]
    if resume_tokens is None:
        resume_map = {name: None for name in feature_map}
    else:
        resume_map = {
            str(name): (None if token is None else str(token))
            for name, token in resume_tokens.items()
        }
    payload = SessionHelloPayload(
        protocols=proto_versions,
        client=client,
        features=feature_map,
        resume_tokens=resume_map,
        auth=auth,
    )
    envelope = FrameEnvelope(
        type=SESSION_HELLO_TYPE,
        version=PROTO_VERSION,
        frame_id=_frame_id(frame_id, frame_id_prefix),
        timestamp=_now(timestamp),
    )
    frame = SessionHello(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_session_welcome(
    *,
    session_id: str,
    heartbeat_s: float,
    features: Mapping[str, FeatureToggle | Mapping[str, Any] | bool],
    ack_timeout_ms: int | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
    frame_id_prefix: str | None = "welcome",
) -> SessionWelcome:
    """Construct a ``session.welcome`` frame with negotiated capabilities."""

    payload = SessionWelcomePayload(
        protocol_version=PROTO_VERSION,
        session=WelcomeSessionInfo(
            id=str(session_id),
            heartbeat_s=float(heartbeat_s),
            ack_timeout_ms=int(ack_timeout_ms) if ack_timeout_ms is not None else None,
        ),
        features=_coerce_feature_toggles(features),
    )
    envelope = FrameEnvelope(
        type=SESSION_WELCOME_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, frame_id_prefix),
        timestamp=_now(timestamp),
    )
    frame = SessionWelcome(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_session_reject(
    *,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
    frame_id_prefix: str | None = "reject",
) -> SessionReject:
    """Construct a ``session.reject`` frame before closing the socket."""

    payload = SessionRejectPayload(
        code=str(code),
        message=str(message),
        details=dict(details) if details is not None else None,
    )
    envelope = FrameEnvelope(
        type=SESSION_REJECT_TYPE,
        version=PROTO_VERSION,
        frame_id=_frame_id(frame_id, frame_id_prefix),
        timestamp=_now(timestamp),
    )
    frame = SessionReject(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_session_heartbeat(
    *,
    session_id: str,
    frame_id: str | None = None,
    timestamp: float | None = None,
    frame_id_prefix: str | None = "heartbeat",
) -> SessionHeartbeat:
    """Construct a ``session.heartbeat`` frame."""

    payload = SessionHeartbeatPayload()
    envelope = FrameEnvelope(
        type=SESSION_HEARTBEAT_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, frame_id_prefix),
        timestamp=_now(timestamp),
    )
    frame = SessionHeartbeat(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_session_ack(
    *,
    session_id: str,
    frame_id: str | None = None,
    timestamp: float | None = None,
    frame_id_prefix: str | None = "ack",
) -> SessionAck:
    """Construct a client ``session.ack`` frame."""

    payload = SessionAckPayload()
    envelope = FrameEnvelope(
        type=SESSION_ACK_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, frame_id_prefix),
        timestamp=_now(timestamp),
    )
    frame = SessionAck(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_session_goodbye(
    *,
    session_id: str,
    payload: SessionGoodbyePayload | Mapping[str, Any] | None = None,
    code: str | None = None,
    message: str | None = None,
    reason: str | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
    frame_id_prefix: str | None = "goodbye",
) -> SessionGoodbye:
    """Construct a ``session.goodbye`` frame when closing a session."""

    if payload is None:
        payload = SessionGoodbyePayload(
            code=str(code) if code is not None else None,
            message=str(message) if message is not None else None,
            reason=str(reason) if reason is not None else None,
        )
    elif not isinstance(payload, SessionGoodbyePayload):
        payload = SessionGoodbyePayload.from_dict(payload)

    envelope = FrameEnvelope(
        type=SESSION_GOODBYE_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, frame_id_prefix),
        timestamp=_now(timestamp),
    )
    frame = SessionGoodbye(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_scene_snapshot(
    *,
    session_id: str,
    viewer: Mapping[str, Any],
    layers: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    policies: Mapping[str, Any] | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
    intent_id: str | None = None,
    delta_token: str | None = None,
    sequencer: ResumableTopicSequencer | None = None,
) -> NotifyScene:
    """Construct a baseline ``notify.scene`` snapshot (seq always resets to zero)."""

    payload = NotifyScenePayload(
        viewer=dict(viewer),
        layers=tuple(dict(item) for item in layers),
        metadata=dict(metadata) if metadata is not None else None,
        policies=dict(policies) if policies is not None else None,
    )
    cursor = _resolve_resumable_cursor(
        topic=NOTIFY_SCENE_TYPE,
        sequencer=sequencer,
        seq=0,
        delta_token=delta_token,
        snapshot=True,
    )
    envelope = FrameEnvelope(
        type=NOTIFY_SCENE_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=frame_id,
        timestamp=_now(timestamp),
        seq=cursor.seq,
        delta_token=cursor.delta_token,
        intent_id=intent_id,
    )
    frame = NotifyScene(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_scene_level(
    *,
    session_id: str,
    payload: NotifySceneLevelPayload | Mapping[str, Any],
    timestamp: float | None = None,
    frame_id: str | None = None,
    intent_id: str | None = None,
    seq: int | None = None,
    delta_token: str | None = None,
    sequencer: ResumableTopicSequencer | None = None,
) -> NotifySceneLevel:
    """Construct a resumable ``notify.scene.level`` update."""

    if not isinstance(payload, NotifySceneLevelPayload):
        payload = NotifySceneLevelPayload.from_dict(payload)
    cursor = _resolve_resumable_cursor(
        topic=NOTIFY_SCENE_LEVEL_TYPE,
        sequencer=sequencer,
        seq=seq,
        delta_token=delta_token,
        snapshot=False,
    )
    envelope = FrameEnvelope(
        type=NOTIFY_SCENE_LEVEL_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=frame_id,
        timestamp=_now(timestamp),
        seq=cursor.seq,
        delta_token=cursor.delta_token,
        intent_id=intent_id,
    )
    frame = NotifySceneLevel(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_layers_delta(
    *,
    session_id: str,
    payload: NotifyLayersPayload | Mapping[str, Any],
    timestamp: float | None = None,
    frame_id: str | None = None,
    intent_id: str | None = None,
    seq: int | None = None,
    delta_token: str | None = None,
    sequencer: ResumableTopicSequencer | None = None,
) -> NotifyLayers:
    """Construct a resumable ``notify.layers`` delta frame."""

    if not isinstance(payload, NotifyLayersPayload):
        payload = NotifyLayersPayload.from_dict(payload)
    cursor = _resolve_resumable_cursor(
        topic=NOTIFY_LAYERS_TYPE,
        sequencer=sequencer,
        seq=seq,
        delta_token=delta_token,
        snapshot=False,
    )
    envelope = FrameEnvelope(
        type=NOTIFY_LAYERS_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=frame_id,
        timestamp=_now(timestamp),
        seq=cursor.seq,
        delta_token=cursor.delta_token,
        intent_id=intent_id,
    )
    frame = NotifyLayers(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_dims(
    *,
    session_id: str,
    payload: NotifyDimsPayload | Mapping[str, Any],
    timestamp: float | None = None,
    frame_id: str | None = None,
    intent_id: str | None = None,
) -> NotifyDims:
    """Construct a ``notify.dims`` hot-path frame."""

    if not isinstance(payload, NotifyDimsPayload):
        payload = NotifyDimsPayload.from_dict(payload)
    resolved_frame_id = frame_id
    if intent_id is None:
        resolved_frame_id = _short_frame_id(resolved_frame_id)
    envelope = FrameEnvelope(
        type=NOTIFY_DIMS_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=resolved_frame_id,
        timestamp=_now(timestamp),
        intent_id=intent_id,
    )
    frame = NotifyDims(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_camera(
    *,
    session_id: str,
    payload: NotifyCameraPayload | Mapping[str, Any],
    timestamp: float | None = None,
    frame_id: str | None = None,
    intent_id: str | None = None,
) -> NotifyCamera:
    """Construct a ``notify.camera`` hot-path frame."""

    if not isinstance(payload, NotifyCameraPayload):
        payload = NotifyCameraPayload.from_dict(payload)
    resolved_frame_id = frame_id
    if intent_id is None:
        resolved_frame_id = _short_frame_id(resolved_frame_id)
    envelope = FrameEnvelope(
        type=NOTIFY_CAMERA_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=resolved_frame_id,
        timestamp=_now(timestamp),
        intent_id=intent_id,
    )
    frame = NotifyCamera(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_telemetry(
    *,
    session_id: str,
    payload: NotifyTelemetryPayload | Mapping[str, Any],
    timestamp: float | None = None,
    frame_id: str | None = None,
) -> NotifyTelemetry:
    """Construct a ``notify.telemetry`` diagnostics frame."""

    if not isinstance(payload, NotifyTelemetryPayload):
        payload = NotifyTelemetryPayload.from_dict(payload)
    envelope = FrameEnvelope(
        type=NOTIFY_TELEMETRY_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=frame_id,
        timestamp=_now(timestamp),
    )
    frame = NotifyTelemetry(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_error(
    *,
    session_id: str,
    payload: NotifyErrorPayload | Mapping[str, Any],
    timestamp: float | None = None,
    frame_id: str | None = None,
) -> NotifyError:
    """Construct a ``notify.error`` frame."""

    if not isinstance(payload, NotifyErrorPayload):
        payload = NotifyErrorPayload.from_dict(payload)
    envelope = FrameEnvelope(
        type=NOTIFY_ERROR_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=frame_id,
        timestamp=_now(timestamp),
    )
    frame = NotifyError(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_notify_stream(
    *,
    session_id: str,
    payload: NotifyStreamPayload | Mapping[str, Any],
    timestamp: float | None = None,
    frame_id: str | None = None,
    seq: int | None = None,
    delta_token: str | None = None,
    sequencer: ResumableTopicSequencer | None = None,
) -> NotifyStream:
    """Construct a resumable ``notify.stream`` frame with stream capabilities."""

    if not isinstance(payload, NotifyStreamPayload):
        payload = NotifyStreamPayload.from_dict(payload)
    cursor = _resolve_resumable_cursor(
        topic=NOTIFY_STREAM_TYPE,
        sequencer=sequencer,
        seq=seq,
        delta_token=delta_token,
        snapshot=False,
    )
    envelope = FrameEnvelope(
        type=NOTIFY_STREAM_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=frame_id,
        timestamp=_now(timestamp),
        seq=cursor.seq,
        delta_token=cursor.delta_token,
    )
    frame = NotifyStream(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_state_update(
    *,
    session_id: str,
    intent_id: str,
    frame_id: str | None,
    payload: StateUpdatePayload | Mapping[str, Any],
    timestamp: float | None = None,
) -> StateUpdate:
    """Construct a client-issued ``state.update`` request."""

    if not isinstance(payload, StateUpdatePayload):
        payload = StateUpdatePayload.from_dict(payload)
    envelope = FrameEnvelope(
        type=STATE_UPDATE_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, prefix="state"),
        timestamp=_now(timestamp),
        intent_id=str(intent_id),
    )
    frame = StateUpdate(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_ack_state(
    *,
    session_id: str,
    frame_id: str | None,
    payload: AckStatePayload | Mapping[str, Any],
    timestamp: float | None = None,
) -> AckState:
    """Construct an ``ack.state`` reply."""

    if not isinstance(payload, AckStatePayload):
        payload = AckStatePayload.from_dict(payload)
    envelope = FrameEnvelope(
        type=ACK_STATE_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, prefix="ack"),
        timestamp=_now(timestamp),
    )
    frame = AckState(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_call_command(
    *,
    session_id: str,
    frame_id: str | None,
    payload: CallCommandPayload | Mapping[str, Any],
    timestamp: float | None = None,
) -> CallCommand:
    """Construct a ``call.command`` request."""

    if not isinstance(payload, CallCommandPayload):
        payload = CallCommandPayload.from_dict(payload)
    envelope = FrameEnvelope(
        type=CALL_COMMAND_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, prefix="cmd"),
        timestamp=_now(timestamp),
    )
    frame = CallCommand(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_reply_command(
    *,
    session_id: str,
    frame_id: str | None,
    payload: ReplyCommandPayload | Mapping[str, Any],
    timestamp: float | None = None,
) -> ReplyCommand:
    """Construct a ``reply.command`` success frame."""

    if not isinstance(payload, ReplyCommandPayload):
        payload = ReplyCommandPayload.from_dict(payload)
    envelope = FrameEnvelope(
        type=REPLY_COMMAND_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, prefix="reply"),
        timestamp=_now(timestamp),
    )
    frame = ReplyCommand(envelope=envelope, payload=payload)
    frame._validate()
    return frame


def build_error_command(
    *,
    session_id: str,
    frame_id: str | None,
    payload: ErrorCommandPayload | Mapping[str, Any],
    timestamp: float | None = None,
) -> ErrorCommand:
    """Construct an ``error.command`` failure frame."""

    if not isinstance(payload, ErrorCommandPayload):
        payload = ErrorCommandPayload.from_dict(payload)
    envelope = FrameEnvelope(
        type=ERROR_COMMAND_TYPE,
        version=PROTO_VERSION,
        session=str(session_id),
        frame_id=_frame_id(frame_id, prefix="error"),
        timestamp=_now(timestamp),
    )
    frame = ErrorCommand(envelope=envelope, payload=payload)
    frame._validate()
    return frame


__all__ = [
    *_messages_all,
    "ResumableCursor",
    "ResumableTopicSequencer",
    "build_session_hello",
    "build_session_welcome",
    "build_session_reject",
    "build_session_heartbeat",
    "build_session_ack",
    "build_session_goodbye",
    "build_notify_scene_snapshot",
    "build_notify_scene_level",
    "build_notify_layers_delta",
    "build_notify_dims",
    "build_notify_camera",
    "build_notify_telemetry",
    "build_notify_error",
    "build_notify_stream",
    "build_state_update",
    "build_ack_state",
    "build_call_command",
    "build_reply_command",
    "build_error_command",
]
