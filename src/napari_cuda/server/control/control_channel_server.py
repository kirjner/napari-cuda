"""State-channel orchestration helpers for `EGLHeadlessServer`."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Optional

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

# Protocol builders & dataclasses
from napari_cuda.protocol import (
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    PROTO_VERSION,
    SESSION_ACK_TYPE,
    SESSION_GOODBYE_TYPE,
    SESSION_HELLO_TYPE,
    CallCommand,
    EnvelopeParser,
    FeatureToggle,
    ResumableTopicSequencer,
    SessionHello,
    SessionReject,
    SessionWelcome,
    build_ack_state,
    build_error_command,
    build_reply_command,
    build_session_goodbye,
    build_session_heartbeat,
    build_session_reject,
    build_session_welcome,
)
from napari_cuda.protocol.messages import (
    STATE_UPDATE_TYPE,
)
from napari_cuda.server.control.command_registry import (
    COMMAND_REGISTRY,
    register_command,
)
from napari_cuda.server.control.protocol_io import (
    await_state_send,
    send_frame,
)
from napari_cuda.server.control.protocol_runtime import (
    feature_enabled,
    history_store,
    state_session,
)
from napari_cuda.server.control.resumable_history_store import (
    ResumeDecision,
    ResumePlan,
)
from napari_cuda.server.control.topics.baseline import orchestrate_connect
from napari_cuda.server.control.state_update_handlers.registry import STATE_UPDATE_HANDLERS

logger = logging.getLogger(__name__)

_HANDSHAKE_TIMEOUT_S = 5.0
_REQUIRED_NOTIFY_FEATURES = ("notify.scene", "notify.layers", "notify.stream")
_COMMAND_KEYFRAME = "napari.pixel.request_keyframe"
_COMMAND_LISTDIR = "fs.listdir"
_COMMAND_ZARR_LOAD = "napari.zarr.load"


@dataclass(slots=True)
class CommandResult:
    result: Any | None = None
    idempotency_key: str | None = None


class CommandRejected(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details) if details else None
        self.idempotency_key = idempotency_key


_RESUMABLE_TOPICS = (NOTIFY_SCENE_TYPE, NOTIFY_LAYERS_TYPE, NOTIFY_STREAM_TYPE)
_ENVELOPE_PARSER = EnvelopeParser()

async def _command_request_keyframe(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    if not hasattr(server, "_ensure_keyframe"):
        raise CommandRejected(
            code="command.forbidden",
            message="Keyframe request not supported",
        )
    await server._ensure_keyframe()
    return CommandResult()


register_command(_COMMAND_KEYFRAME, _command_request_keyframe)


async def _command_fs_listdir(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    kwargs = dict(frame.payload.kwargs or {})
    path = kwargs.get("path")
    show_hidden = bool(kwargs.get("show_hidden", False))
    only = kwargs.get("only")
    filters: Optional[Sequence[str]] = None
    if only is not None:
        if isinstance(only, (list, tuple)):
            filters = tuple(str(item) for item in only if item is not None)
        else:
            raise CommandRejected(
                code="command.invalid",
                message="'only' filter must be a list of suffixes",
            )
    if only is None:
        filters = (".zarr",)
    try:
        listing = server._list_directory(
            path,
            only=filters,
            show_hidden=show_hidden,
        )
    except RuntimeError as exc:
        raise CommandRejected(code="fs.forbidden", message=str(exc)) from exc
    except FileNotFoundError:
        raise CommandRejected(
            code="fs.not_found",
            message="Directory not found",
            details={"path": path},
        )
    except NotADirectoryError:
        raise CommandRejected(
            code="fs.not_found",
            message="Path is not a directory",
            details={"path": path},
        )
    return CommandResult(result=listing)


async def _command_zarr_load(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    kwargs = dict(frame.payload.kwargs or {})
    path = kwargs.get("path")
    if not isinstance(path, str) or not path:
        raise CommandRejected(
            code="command.invalid",
            message="'path' must be a non-empty string",
        )
    try:
        await server._handle_zarr_load(path)
    except RuntimeError as exc:
        raise CommandRejected(code="fs.forbidden", message=str(exc)) from exc
    except FileNotFoundError:
        raise CommandRejected(
            code="fs.not_found",
            message="Dataset not found",
            details={"path": path},
        )
    except NotADirectoryError:
        raise CommandRejected(
            code="fs.not_found",
            message="Dataset path is not a directory",
            details={"path": path},
        )
    except ValueError as exc:
        raise CommandRejected(
            code="zarr.invalid",
            message=str(exc),
            details={"path": path},
        ) from exc
    return CommandResult(result={"ok": True})


register_command(_COMMAND_LISTDIR, _command_fs_listdir)
register_command(_COMMAND_ZARR_LOAD, _command_zarr_load)


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


@dataclass(slots=True)
class StateUpdateContext:
    """Envelope metadata shared across state.update handlers."""

    server: Any
    ws: Any
    scope: str
    key: str
    target: str
    value: Any
    session_id: str
    frame_id: str
    intent_id: str
    timestamp: float | None

    async def reject(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        error_payload: dict[str, Any] = {"code": code, "message": message}
        if details:
            error_payload["details"] = dict(details)
        await _send_state_ack(
            self.server,
            self.ws,
            session_id=self.session_id,
            intent_id=self.intent_id,
            in_reply_to=self.frame_id,
            status="rejected",
            error=error_payload,
        )

    async def ack(
        self,
        *,
        applied_value: Any,
        version: int,
    ) -> None:
        await _send_state_ack(
            self.server,
            self.ws,
            session_id=self.session_id,
            intent_id=self.intent_id,
            in_reply_to=self.frame_id,
            status="accepted",
            applied_value=applied_value,
            version=int(version),
        )


async def _send_command_error(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    in_reply_to: str,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "in_reply_to": in_reply_to,
        "status": "error",
        "code": str(code),
        "message": str(message),
    }
    if details:
        payload["details"] = dict(details)
    if idempotency_key is not None:
        payload["idempotency_key"] = str(idempotency_key)
    frame = build_error_command(
        session_id=session_id,
        frame_id=None,
        payload=payload,
        timestamp=time.time(),
    )
    await send_frame(server, ws, frame)


async def _ingest_call_command(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    session_id = state_session(ws)
    if not session_id:
        logger.debug("call.command ignored: missing session id")
        return True

    try:
        frame = _ENVELOPE_PARSER.parse_call_command(data)
    except (ValueError, TypeError) as exc:
        logger.debug("call.command parse failed", exc_info=True)
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=str(data.get("frame_id") or "unknown"),
            code="command.forbidden",
            message=str(exc) or "invalid call.command payload",
        )
        return True

    envelope = frame.envelope
    payload = frame.payload
    in_reply_to = envelope.frame_id or "unknown"

    if not feature_enabled(ws, "call.command"):
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code="command.forbidden",
            message="call.command feature disabled",
        )
        return True

    handler = COMMAND_REGISTRY.get_handler(payload.command)
    if handler is None:
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code="command.not_found",
            message=f"unknown command '{payload.command}'",
        )
        return True

    try:
        result = await handler(server, frame, ws)
    except CommandRejected as exc:
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            idempotency_key=exc.idempotency_key,
        )
        return True
    except Exception:
        logger.exception("command handler failed", extra={"command": payload.command})
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code="command.retryable",
            message="command handler raised an exception",
        )
        return True

    response_payload: dict[str, Any] = {
        "in_reply_to": in_reply_to,
        "status": "ok",
    }
    if result.result is not None:
        response_payload["result"] = result.result
    if result.idempotency_key is not None:
        response_payload["idempotency_key"] = result.idempotency_key

    frame_out = build_reply_command(
        session_id=session_id,
        frame_id=None,
        payload=response_payload,
        timestamp=time.time(),
    )
    await send_frame(server, ws, frame_out)
    return True

StateMessageHandler = Callable[[Any, Mapping[str, Any], Any], Awaitable[bool]]

def _mark_client_activity(ws: Any) -> None:
    now = time.time()
    ws._napari_cuda_last_activity = now
    ws._napari_cuda_waiting_ack = False
    ws._napari_cuda_missed_heartbeats = 0


def _heartbeat_shutting_down(ws: Any) -> bool:
    return bool(getattr(ws, "_napari_cuda_shutdown", False))


def _flag_heartbeat_shutdown(ws: Any) -> None:
    ws._napari_cuda_shutdown = True


async def _send_session_goodbye(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    code: str,
    message: str,
    reason: Optional[str] = None,
) -> None:
    if getattr(ws, "_napari_cuda_goodbye_sent", False):
        return
    frame = build_session_goodbye(
        session_id=session_id,
        code=code,
        message=message,
        reason=reason,
        timestamp=time.time(),
    )
    await send_frame(server, ws, frame)
    ws._napari_cuda_goodbye_sent = True


async def _state_heartbeat_loop(server: Any, ws: Any) -> None:
    try:
        interval = float(getattr(ws, "_napari_cuda_heartbeat_interval", 0.0))
    except Exception:
        interval = 0.0
    if interval <= 0.0:
        return

    log_debug = logger.debug
    session_id = state_session(ws)
    missed = int(getattr(ws, "_napari_cuda_missed_heartbeats", 0))

    try:
        while True:
            await asyncio.sleep(interval)
            if ws.state is not State.OPEN or _heartbeat_shutting_down(ws):
                break
            session_id = state_session(ws)
            if not session_id:
                break

            waiting_ack = bool(getattr(ws, "_napari_cuda_waiting_ack", False))
            if waiting_ack:
                missed += 1
            else:
                missed = 0
            ws._napari_cuda_missed_heartbeats = missed

            if missed >= 2:
                logger.warning(
                    "heartbeat timeout for session %s; closing connection",
                    session_id,
                )
                with suppress(Exception):
                    await _send_session_goodbye(
                        server,
                        ws,
                        session_id=session_id,
                        code="timeout",
                        message="heartbeat timeout",
                        reason="heartbeat_timeout",
                    )
                _flag_heartbeat_shutdown(ws)
                with suppress(Exception):
                    await ws.close(code=1011, reason="heartbeat timeout")
                break

            try:
                frame = build_session_heartbeat(
                    session_id=session_id,
                    timestamp=time.time(),
                )
                await send_frame(server, ws, frame)
                ws._napari_cuda_waiting_ack = True
                ws._napari_cuda_last_heartbeat = time.time()
            except Exception:
                log_debug("session.heartbeat send failed", exc_info=True)
                break
    except asyncio.CancelledError:
        pass



def _state_features(ws: Any) -> Mapping[str, FeatureToggle]:
    features = getattr(ws, "_napari_cuda_features", None)
    if isinstance(features, Mapping):
        return features
    return {}


## moved to protocol_runtime: feature_enabled, state_sequencer


def _resolve_dims_mode_from_ndisplay(ndisplay: int) -> str:
    return "volume" if int(ndisplay) >= 3 else "plane"


## broadcast_layers_delta moved to topics.layers
## broadcast_dims_state moved to topics.dims


def _normalize_camera_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_camera_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_camera_value(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _float_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _positive_float(value: Any) -> Optional[float]:
    result = _float_value(value)
    if result is None or result <= 0.0:
        return None
    return result


def _float_pair(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    first = _float_value(value[0])
    second = _float_value(value[1])
    if first is None or second is None:
        return None
    return float(first), float(second)


def _float_triplet(value: Any) -> Optional[tuple[float, float, float]]:
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    x = _float_value(value[0])
    y = _float_value(value[1])
    z = _float_value(value[2])
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def _float_rect(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, Sequence) or len(value) < 4:
        return None
    left = _float_value(value[0])
    bottom = _float_value(value[1])
    width = _float_value(value[2])
    height = _float_value(value[3])
    if None in (left, bottom, width, height):
        return None
    return float(left), float(bottom), float(width), float(height)


def _float_sequence(value: Any) -> Optional[tuple[float, ...]]:
    if not isinstance(value, Sequence) or not value:
        return None
    components: list[float] = []
    for item in value:
        component = _float_value(item)
        if component is None:
            return None
        components.append(component)
    return tuple(components)


def _string_value(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return str(value)
    return None


## moved to topics.camera: broadcast_camera_update


## moved to topics.stream: send_stream_frame


## moved to topics.scene: send_scene_baseline

async def ingest_state(server: Any, ws: Any) -> None:
    """Ingest a state-channel websocket connection."""

    heartbeat_task: asyncio.Task[Any] | None = None
    try:
        _disable_nagle(ws)
        handshake_ok = await _perform_state_handshake(server, ws)
        if not handshake_ok:
            return
        try:
            heartbeat_task = asyncio.create_task(_state_heartbeat_loop(server, ws))

            def _log_heartbeat_completion(task: asyncio.Task[Any]) -> None:
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.debug("heartbeat task completion probe failed", exc_info=True)
                    return
                if exc is not None:
                    logger.exception("Heartbeat task failed", exc_info=exc)

            heartbeat_task.add_done_callback(_log_heartbeat_completion)
            ws._napari_cuda_heartbeat_task = heartbeat_task
        except Exception:
            logger.debug("failed to start heartbeat loop", exc_info=True)
        server._state_clients.add(ws)
        server.metrics.inc('napari_cuda_state_connects')
        server._update_client_gauges()
        resume_map = getattr(ws, "_napari_cuda_resume_plan", {}) or {}
        await orchestrate_connect(server, ws, resume_map)
        remote = getattr(ws, 'remote_address', None)
        log = logger.info if server._log_state_traces else logger.debug
        log("state client loop start remote=%s id=%s", remote, id(ws))
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    logger.debug('state client sent invalid JSON; ignoring')
                    continue
                await process_state_message(server, data, ws)
        except ConnectionClosed as exc:
            logger.info(
                "state client closed remote=%s id=%s code=%s reason=%s",
                remote,
                id(ws),
                getattr(exc, 'code', None),
                getattr(exc, 'reason', None),
            )
        except Exception:
            logger.exception("state client error remote=%s id=%s", remote, id(ws))
    finally:
        _flag_heartbeat_shutdown(ws)
        if heartbeat_task is None:
            heartbeat_task = getattr(ws, "_napari_cuda_heartbeat_task", None)
        if isinstance(heartbeat_task, asyncio.Task):
            heartbeat_task.cancel()
            with suppress(Exception):
                await heartbeat_task
        if hasattr(ws, "_napari_cuda_heartbeat_task"):
            delattr(ws, "_napari_cuda_heartbeat_task")
        try:
            await ws.close()
        except Exception as exc:
            logger.debug("State WS close error: %s", exc)
        server._state_clients.discard(ws)
        server._update_client_gauges()



async def _ingest_session_ack(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    _ENVELOPE_PARSER.parse_ack(data)
    return True


async def _ingest_session_goodbye(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    reason = None
    code = None
    message = "client requested shutdown"
    try:
        goodbye = _ENVELOPE_PARSER.parse_goodbye(data)
        payload = goodbye.payload
        reason = payload.reason
        if payload.message:
            message = payload.message
        if payload.code:
            code = payload.code
    except Exception:
        logger.debug("session.goodbye payload rejected", exc_info=True)

    session_id = state_session(ws)
    if session_id:
        with suppress(Exception):
            await _send_session_goodbye(
                server,
                ws,
                session_id=session_id,
                code=code or "goodbye",
                message=message,
                reason=reason or "client_goodbye",
            )
    _flag_heartbeat_shutdown(ws)
    with suppress(Exception):
        await ws.close()
    return True


async def _ingest_state_update(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        frame = _ENVELOPE_PARSER.parse_state_update(data)
    except Exception:
        logger.debug("state.update payload rejected by parser", exc_info=True)
        frame_id = data.get("frame_id")
        intent_id = data.get("intent_id")
        session_id = data.get("session") or state_session(ws)
        if frame_id and intent_id and session_id:
            await _send_state_ack(
                server,
                ws,
                session_id=str(session_id),
                intent_id=str(intent_id),
                in_reply_to=str(frame_id),
                status="rejected",
                error={"code": "state.invalid", "message": "state.update payload invalid"},
            )
        return True

    envelope = frame.envelope
    payload = frame.payload

    session_id = envelope.session or state_session(ws)
    frame_id = envelope.frame_id
    intent_id = envelope.intent_id
    timestamp = envelope.timestamp

    if session_id is None or frame_id is None or intent_id is None:
        raise ValueError("state.update envelope missing session/frame/intent identifiers")

    scope = str(payload.scope or "").strip().lower()
    key = str(payload.key or "").strip().lower()
    target = str(payload.target or "").strip()
    value = payload.value

    ctx = StateUpdateContext(
        server=server,
        ws=ws,
        scope=scope,
        key=key,
        target=target,
        value=value,
        session_id=str(session_id),
        frame_id=str(frame_id),
        intent_id=str(intent_id),
        timestamp=timestamp,
    )

    if not scope or not key:
        await ctx.reject(
            "state.invalid",
            "scope/key required",
            details={"scope": scope, "key": key, "target": target},
        )
        return True

    handler = STATE_UPDATE_HANDLERS.get(f"{scope}:{key}")
    if handler is None:
        handler = STATE_UPDATE_HANDLERS.get(f"{scope}:*")
    if handler is None:
        await ctx.reject(
            "state.invalid",
            f"unsupported {scope} key {key}",
            details={"scope": scope, "key": key},
        )
        return True

    return await handler(ctx)

    return True

MESSAGE_HANDLERS: dict[str, StateMessageHandler] = {
    STATE_UPDATE_TYPE: _ingest_state_update,
    'call.command': _ingest_call_command,
    SESSION_ACK_TYPE: _ingest_session_ack,
    SESSION_GOODBYE_TYPE: _ingest_session_goodbye,
}


async def process_state_message(server: Any, data: dict, ws: Any) -> None:
    msg_type = data.get('type')
    frame_id = data.get('frame_id')
    _mark_client_activity(ws)
    if server._log_state_traces:
        logger.info("state message start type=%s frame=%s", msg_type, frame_id)
    handled = False
    try:
        handler: StateMessageHandler | None = None
        if isinstance(msg_type, str):
            handler = MESSAGE_HANDLERS.get(msg_type)
        if handler is None:
            if server._log_state_traces:
                logger.info("state message ignored type=%s", msg_type)
            return
        handled = bool(await handler(server, data, ws))
        return
    finally:
        if server._log_state_traces:
            logger.info("state message end type=%s frame=%s handled=%s", msg_type, frame_id, handled)


def _log_volume_event(server: Any, fmt: str, *args: Any) -> None:
    handler = getattr(server, "_log_volume_update", None)
    if callable(handler):
        try:
            handler(fmt, *args)
            return
        except Exception:
            logger.debug("volume update log failed", exc_info=True)
    try:
        logger.debug(fmt, *args)
    except Exception:
        logger.debug("volume fallback log failed", exc_info=True)


def _ndim_from_meta(meta: Mapping[str, Any]) -> int:
    try:
        ndim = int(meta.get("ndim") or 0)
    except Exception:
        ndim = 0

    if ndim <= 0:
        for key in ("order", "axes", "level_shapes"):
            values = meta.get(key)
            if not isinstance(values, Sequence):
                continue
            if key == "level_shapes":
                assert values, "level_shapes must contain at least one entry"
                first = values[0]
                assert isinstance(first, Sequence), "level_shapes entries must be sequences"
                length = len(first)
            else:
                length = len(values)
            ndim = max(ndim, length)

    if ndim <= 0:
        current = meta.get("current_step")
        if isinstance(current, Sequence):
            try:
                ndim = max(ndim, len(tuple(current)))
            except Exception:
                ndim = max(ndim, 0)

    return max(1, int(ndim))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _disable_nagle(ws: Any) -> None:
    try:
        sock = ws.transport.get_extra_info('socket')  # type: ignore[attr-defined]
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        logger.debug('state ws: TCP_NODELAY toggle failed', exc_info=True)


async def _perform_state_handshake(server: Any, ws: Any) -> bool:
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
    except Exception:
        logger.debug("session.welcome send failed", exc_info=True)
        return False

    ws._napari_cuda_session = session_id
    ws._napari_cuda_features = negotiated_features
    ws._napari_cuda_heartbeat_interval = float(heartbeat_s)
    ws._napari_cuda_shutdown = False
    ws._napari_cuda_goodbye_sent = False
    ws._napari_cuda_sequencers = {NOTIFY_SCENE_TYPE: ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE), NOTIFY_LAYERS_TYPE: ResumableTopicSequencer(topic=NOTIFY_LAYERS_TYPE), NOTIFY_STREAM_TYPE: ResumableTopicSequencer(topic=NOTIFY_STREAM_TYPE)}

    _mark_client_activity(ws)

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
        # Required features must be explicitly true; optional ones inherit server toggle.
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
    except Exception:
        logger.debug("session.reject send failed", exc_info=True)
    try:
        await ws.close()
    except Exception:
        logger.debug("state ws close after reject failed", exc_info=True)


## moved to protocol_io: await_state_send


async def _send_state_ack(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    intent_id: str,
    in_reply_to: str,
    status: str,
    applied_value: Any | None = None,
    error: Mapping[str, Any] | Any | None = None,
    version: int | None = None,
) -> None:
    """Build and emit an ``ack.state`` frame that mirrors the incoming update."""

    if not intent_id or not in_reply_to:
        raise ValueError("ack.state requires intent_id and in_reply_to identifiers")

    normalized_status = str(status).lower()
    if normalized_status not in {"accepted", "rejected"}:
        raise ValueError("ack.state status must be 'accepted' or 'rejected'")

    payload: dict[str, Any] = {
        "intent_id": str(intent_id),
        "in_reply_to": str(in_reply_to),
        "status": normalized_status,
    }

    if normalized_status == "accepted":
        if error is not None:
            raise ValueError("accepted ack.state payload cannot include error details")
        if version is None:
            raise ValueError("accepted ack.state payload requires version")
        if not isinstance(version, Integral):
            raise ValueError("ack.state version must be integer")
        if applied_value is not None:
            payload["applied_value"] = applied_value
        payload["version"] = int(version)
    else:
        if not isinstance(error, Mapping):
            raise ValueError("rejected ack.state payload requires {code, message}")
        if "code" not in error or "message" not in error:
            raise ValueError("ack.state error payload must include 'code' and 'message'")
        payload["error"] = dict(error)
        if version is not None:
            if not isinstance(version, Integral):
                raise ValueError("ack.state version must be integer")
            payload["version"] = int(version)

    frame = build_ack_state(
        session_id=str(session_id),
        frame_id=None,
        payload=payload,
        timestamp=time.time(),
    )

    await send_frame(server, ws, frame)


## moved to protocol_io / topics.layers: send_frame, send_layer_snapshot


## moved to topics.stream: send_stream_snapshot


## moved to topics.scene: send_scene_snapshot
