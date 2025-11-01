"""Control-channel handshake and resume helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Optional

from websockets.exceptions import ConnectionClosed

from napari_cuda.protocol import (
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    PROTO_VERSION,
    SESSION_HELLO_TYPE,
    FeatureToggle,
    EnvelopeParser,
    ResumableTopicSequencer,
    SessionHello,
    SessionReject,
    SessionWelcome,
    build_session_reject,
    build_session_welcome,
)
from napari_cuda.server.control.command_registry import COMMAND_REGISTRY
from napari_cuda.server.control.protocol.io import await_state_send
from napari_cuda.server.control.protocol.runtime import history_store
from napari_cuda.server.control.resumable_history_store import (
    ResumeDecision,
    ResumePlan,
)

logger = logging.getLogger(__name__)

_HANDSHAKE_TIMEOUT_S = 5.0
_REQUIRED_NOTIFY_FEATURES = ("notify.scene", "notify.layers", "notify.stream")
_RESUMABLE_TOPICS = (NOTIFY_SCENE_TYPE, NOTIFY_LAYERS_TYPE, NOTIFY_STREAM_TYPE)
_ENVELOPE_PARSER = EnvelopeParser()

_SERVER_FEATURES: dict[str, FeatureToggle] = {
    "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
    "notify.camera": FeatureToggle(enabled=True, version=1, resume=False),
    "notify.telemetry": FeatureToggle(enabled=False),
    "call.command": FeatureToggle(
        enabled=True,
        version=1,
        resume=False,
    ),
}


async def perform_state_handshake(server: Any, ws: Any) -> bool:
    """Negotiate the notify-only protocol handshake with the client."""

    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=_HANDSHAKE_TIMEOUT_S)
    except TimeoutError:
        await _send_handshake_reject(
            server,
            ws,
            code="timeout",
            message="session.hello not received",
        )
        return False
    except ConnectionClosed:
        logger.debug("state client closed during handshake")
        return False
    except Exception:
        logger.debug("state client handshake recv failed", exc_info=True)
        return False

    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            await _send_handshake_reject(
                server,
                ws,
                code="invalid_payload",
                message="session.hello must be UTF-8 text",
            )
            return False

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await _send_handshake_reject(
            server,
            ws,
            code="invalid_json",
            message="session.hello must be valid JSON",
        )
        return False

    msg_type = str(data.get("type") or "").lower()
    if msg_type != SESSION_HELLO_TYPE:
        await _send_handshake_reject(
            server,
            ws,
            code="unexpected_message",
            message="expected session.hello",
            details={"received": msg_type},
        )
        return False

    try:
        hello = _ENVELOPE_PARSER.parse_hello(data)
    except Exception as exc:
        await _send_handshake_reject(
            server,
            ws,
            code="invalid_payload",
            message="session.hello payload rejected",
            details={"error": str(exc)},
        )
        return False

    client_protocols = hello.payload.protocols
    if PROTO_VERSION not in client_protocols:
        await _send_handshake_reject(
            server,
            ws,
            code="protocol_mismatch",
            message=f"server requires protocol {PROTO_VERSION}",
            details={"client_protocols": list(client_protocols)},
        )
        return False

    negotiated_features = _resolve_handshake_features(hello)
    missing = [name for name in _REQUIRED_NOTIFY_FEATURES if not negotiated_features[name].enabled]
    if missing:
        await _send_handshake_reject(
            server,
            ws,
            code="unsupported_client",
            message="client missing notify features",
            details={"missing": missing},
        )
        return False

    store = history_store(server)
    client_tokens = hello.payload.resume_tokens
    resume_plan: dict[str, ResumePlan] = {}

    for topic in _RESUMABLE_TOPICS:
        toggle = negotiated_features.get(topic)
        if toggle is None or not toggle.enabled or not toggle.resume:
            continue
        token = client_tokens.get(topic)
        if store is None:
            resume_plan[topic] = ResumePlan(topic=topic, decision=ResumeDecision.RESET, deltas=[])
            continue
        plan = store.plan_resume(topic, token)
        resume_plan[topic] = plan
        if plan.decision == ResumeDecision.REJECT:
            await _send_handshake_reject(
                server,
                ws,
                code="invalid_resume_token",
                message=f"resume token for {topic} rejected",
                details={"topic": topic},
            )
            return False
        if plan.decision == ResumeDecision.REPLAY:
            cursor = store.latest_resume_state(topic)
            if cursor is not None:
                negotiated_features[topic] = replace(toggle, resume_state=cursor)
            continue
        cursor = store.latest_resume_state(topic)
        if cursor is not None:
            negotiated_features[topic] = replace(toggle, resume_state=cursor)

    ws._napari_cuda_resume_plan = resume_plan

    session_id = str(uuid.uuid4())
    heartbeat_s = getattr(server, "state_heartbeat_s", 15.0)
    ack_timeout_ms = getattr(server, "state_ack_timeout_ms", 250)
    welcome = build_session_welcome(
        session_id=session_id,
        heartbeat_s=heartbeat_s,
        ack_timeout_ms=ack_timeout_ms,
        features=negotiated_features,
        timestamp=time.time(),
    )

    try:
        await _send_handshake_envelope(server, ws, welcome)
    except ConnectionClosed:
        logger.debug("session.welcome send failed (connection closed)")
        return False

    ws._napari_cuda_session = session_id
    ws._napari_cuda_features = negotiated_features
    ws._napari_cuda_heartbeat_interval = float(heartbeat_s)
    ws._napari_cuda_shutdown = False
    ws._napari_cuda_goodbye_sent = False
    ws._napari_cuda_heartbeat_task = None
    ws._napari_cuda_sequencers = {
        NOTIFY_SCENE_TYPE: ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE),
        NOTIFY_LAYERS_TYPE: ResumableTopicSequencer(topic=NOTIFY_LAYERS_TYPE),
        NOTIFY_STREAM_TYPE: ResumableTopicSequencer(topic=NOTIFY_STREAM_TYPE),
    }
    ws._napari_cuda_last_activity = time.time()
    ws._napari_cuda_waiting_ack = False
    ws._napari_cuda_missed_heartbeats = 0

    client_info = hello.payload.client
    client_name = client_info.name
    client_version = client_info.version
    logger.info(
        "state handshake accepted name=%s version=%s features=%s",
        client_name,
        client_version,
        sorted(name for name, toggle in negotiated_features.items() if toggle.enabled),
    )

    return True


def _resolve_handshake_features(hello: SessionHello) -> dict[str, FeatureToggle]:
    client_features = hello.payload.features
    negotiated: dict[str, FeatureToggle] = {}
    for name, server_toggle in _SERVER_FEATURES.items():
        client_enabled = client_features.get(name, False)
        enabled = server_toggle.enabled and (client_enabled or name not in _REQUIRED_NOTIFY_FEATURES)
        negotiated[name] = replace(server_toggle, enabled=enabled)

    command_toggle = negotiated.get("call.command")
    if command_toggle is not None:
        commands = COMMAND_REGISTRY.command_names()
        negotiated["call.command"] = replace(
            command_toggle,
            commands=commands if (command_toggle.enabled and commands) else (),
        )
    return negotiated


async def _send_handshake_envelope(server: Any, ws: Any, envelope: SessionWelcome | SessionReject) -> None:
    text = json.dumps(envelope.to_dict(), separators=(",", ":"))
    await await_state_send(server, ws, text)


async def _send_handshake_reject(
    server: Any,
    ws: Any,
    *,
    code: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    reject = build_session_reject(
        code=str(code),
        message=str(message),
        details=dict(details) if details else None,
        timestamp=time.time(),
    )
    try:
        await _send_handshake_envelope(server, ws, reject)
    except ConnectionClosed:
        logger.debug("session.reject send failed (connection closed)")
    try:
        await ws.close()
    except ConnectionClosed:
        logger.debug("state ws close after reject failed (connection closed)")


__all__ = ["perform_state_handshake"]
