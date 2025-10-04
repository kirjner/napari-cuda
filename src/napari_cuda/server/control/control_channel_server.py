"""State-channel orchestration helpers for `EGLHeadlessServer`."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid
from contextlib import suppress
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from numbers import Integral

# Protocol builders & dataclasses
from napari_cuda.protocol import (
    EnvelopeParser,
    FeatureToggle,
    PROTO_VERSION,
    NOTIFY_CAMERA_TYPE,
    NOTIFY_DIMS_TYPE,
    NOTIFY_ERROR_TYPE,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_STREAM_TYPE,
    NOTIFY_TELEMETRY_TYPE,
    SESSION_ACK_TYPE,
    SESSION_GOODBYE_TYPE,
    SESSION_HELLO_TYPE,
    SessionHello,
    SessionReject,
    SessionWelcome,
    CallCommand,
    ResumableTopicSequencer,
    build_ack_state,
    build_notify_camera,
    build_notify_dims,
    build_notify_error,
    build_notify_layers_delta,
    build_notify_scene_snapshot,
    build_notify_scene_level,
    build_notify_stream,
    build_notify_telemetry,
    build_error_command,
    build_reply_command,
    build_session_ack,
    build_session_goodbye,
    build_session_heartbeat,
    build_session_reject,
    build_session_welcome,
)
from napari_cuda.protocol.messages import (
    NotifyLayersPayload,
    NotifyScenePayload,
    NotifySceneLevelPayload,
    NotifyStreamPayload,
)
from napari_cuda.protocol.messages import STATE_UPDATE_TYPE
from napari_cuda.server.control import state_update_engine as state_updates
from napari_cuda.server.control.state_update_engine import (
    StateUpdateResult,
    apply_dims_state_update,
    apply_layer_state_update,
    axis_label_from_meta,
)
from napari_cuda.server.rendering.render_mailbox import RenderDelta
from napari_cuda.server.state.scene_state import ServerSceneState
from napari_cuda.server.state.server_scene import (
    ServerSceneCommand,
    get_control_meta,
    increment_server_sequence,
    layer_controls_to_dict,
)
from napari_cuda.server.rendering.worker_notifications import WorkerSceneNotification
from napari_cuda.server.control.control_payload_builder import (
    build_notify_dims_from_result,
    build_notify_dims_payload,
    build_notify_layers_delta_payload,
    build_notify_layers_payload,
    build_notify_scene_level_payload,
    build_notify_scene_payload,
)
from napari_cuda.protocol.snapshots import LayerDelta
from napari_cuda.server.rendering.pixel import pixel_channel_server as pixel_channel
from napari_cuda.server.control.resumable_history_store import (
    EnvelopeSnapshot,
    ResumableHistoryStore,
    ResumeDecision,
    ResumePlan,
)
from napari_cuda.server.control.command_registry import (
    COMMAND_REGISTRY,
    CommandHandler,
    register_command,
)

logger = logging.getLogger(__name__)

_HANDSHAKE_TIMEOUT_S = 5.0
_REQUIRED_NOTIFY_FEATURES = ("notify.scene", "notify.layers", "notify.stream")
_COMMAND_KEYFRAME = "napari.pixel.request_keyframe"


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


_RESUMABLE_TOPICS = (
    NOTIFY_SCENE_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_STREAM_TYPE,
)
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


_SERVER_FEATURES: dict[str, FeatureToggle] = {
    "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.scene.level": FeatureToggle(enabled=True, version=1, resume=True),
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
    payload: Dict[str, Any] = {
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
    await _send_frame(server, ws, frame)


async def _handle_call_command(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    session_id = _state_session(ws)
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

    if not _feature_enabled(ws, "call.command"):
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

    response_payload: Dict[str, Any] = {
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
    await _send_frame(server, ws, frame_out)
    return True

StateMessageHandler = Callable[[Any, Mapping[str, Any], Any], Awaitable[bool]]


def _state_session(ws: Any) -> Optional[str]:
    return getattr(ws, "_napari_cuda_session", None)


def _history_store(server: Any) -> Optional[ResumableHistoryStore]:
    store = getattr(server, "_resumable_store", None)
    if isinstance(store, ResumableHistoryStore):
        return store
    return None


def _mark_client_activity(ws: Any) -> None:
    now = time.time()
    setattr(ws, "_napari_cuda_last_activity", now)
    setattr(ws, "_napari_cuda_waiting_ack", False)
    setattr(ws, "_napari_cuda_missed_heartbeats", 0)


def _heartbeat_shutting_down(ws: Any) -> bool:
    return bool(getattr(ws, "_napari_cuda_shutdown", False))


def _flag_heartbeat_shutdown(ws: Any) -> None:
    setattr(ws, "_napari_cuda_shutdown", True)


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
    await _send_frame(server, ws, frame)
    setattr(ws, "_napari_cuda_goodbye_sent", True)


async def _state_heartbeat_loop(server: Any, ws: Any) -> None:
    try:
        interval = float(getattr(ws, "_napari_cuda_heartbeat_interval", 0.0))
    except Exception:
        interval = 0.0
    if interval <= 0.0:
        return

    log_debug = logger.debug
    session_id = _state_session(ws)
    missed = int(getattr(ws, "_napari_cuda_missed_heartbeats", 0))

    try:
        while True:
            await asyncio.sleep(interval)
            if ws.state is not State.OPEN or _heartbeat_shutting_down(ws):
                break
            session_id = _state_session(ws)
            if not session_id:
                break

            waiting_ack = bool(getattr(ws, "_napari_cuda_waiting_ack", False))
            if waiting_ack:
                missed += 1
            else:
                missed = 0
            setattr(ws, "_napari_cuda_missed_heartbeats", missed)

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
                await _send_frame(server, ws, frame)
                setattr(ws, "_napari_cuda_waiting_ack", True)
                setattr(ws, "_napari_cuda_last_heartbeat", time.time())
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


def _feature_enabled(ws: Any, name: str) -> bool:
    toggle = _state_features(ws).get(name)
    return bool(getattr(toggle, "enabled", False))


def _state_sequencer(ws: Any, topic: str) -> ResumableTopicSequencer:
    sequencers = getattr(ws, "_napari_cuda_sequencers", None)
    if not isinstance(sequencers, dict):
        sequencers = {}
        setattr(ws, "_napari_cuda_sequencers", sequencers)
    sequencer = sequencers.get(topic)
    if not isinstance(sequencer, ResumableTopicSequencer):
        sequencer = ResumableTopicSequencer(topic=topic)
        sequencers[topic] = sequencer
    return sequencer


def _viewer_settings(server: Any) -> Dict[str, Any]:
    width = int(server.width)
    height = int(server.height)
    fps = float(server.cfg.fps)
    use_volume = bool(server.use_volume)

    return {
        "fps_target": fps,
        "canvas_size": [width, height],
        "volume_enabled": use_volume,
    }


def _default_layer_id(server: Any) -> Optional[str]:
    snapshot = server._scene_manager.scene_snapshot()
    if snapshot is None or not snapshot.layers:
        return None
    return snapshot.layers[0].layer_id


def _layer_changes_from_result(server: Any, result: StateUpdateResult) -> tuple[str | None, Dict[str, Any]]:
    key = str(result.key)
    if result.scope == "layer":
        return str(result.target), {key: result.value}
    if result.scope in {"volume", "multiscale"}:
        namespaced = f"{result.scope}.{key}"
        target = str(result.target) if result.target else None
        if not target or not target.startswith("layer"):
            target = _default_layer_id(server)
        return target, {namespaced: result.value}
    return None, {}


def _current_dims_meta(server: Any) -> Mapping[str, Any]:
    meta = server._scene.last_dims_payload
    assert meta is not None, "dims metadata not initialized"
    meta_copy: Dict[str, Any] = dict(meta)
    assert "ndisplay" in meta_copy, "dims metadata missing ndisplay"
    assert "mode" in meta_copy, "dims metadata missing mode"
    return meta_copy


def _resolve_dims_mode_from_ndisplay(ndisplay: int) -> str:
    return "volume" if int(ndisplay) >= 3 else "plane"


async def _broadcast_layers_delta(
    server: Any,
    *,
    layer_id: Optional[str],
    changes: Mapping[str, Any],
    intent_id: Optional[str],
    timestamp: Optional[float],
    targets: Optional[Sequence[Any]] = None,
) -> None:
    if not changes:
        return

    delta = LayerDelta(layer_id=layer_id or "layer-0", changes=dict(changes))
    payload = delta.to_payload()
    clients = list(targets) if targets is not None else list(server._state_clients)
    now = time.time() if timestamp is None else float(timestamp)

    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None and targets is None:
        snapshot = store.delta_envelope(
            NOTIFY_LAYERS_TYPE,
            payload=payload.to_dict(),
            timestamp=now,
            intent_id=intent_id,
        )

    tasks: list[Awaitable[None]] = []
    for ws in clients:
        if not _feature_enabled(ws, "notify.layers"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        kwargs: Dict[str, Any] = {
            "session_id": session_id,
            "payload": payload,
            "timestamp": snapshot.timestamp if snapshot is not None else now,
            "intent_id": intent_id,
        }
        if snapshot is not None:
            kwargs["seq"] = snapshot.seq
            kwargs["delta_token"] = snapshot.delta_token
            kwargs["frame_id"] = snapshot.frame_id
        else:
            kwargs["sequencer"] = _state_sequencer(ws, NOTIFY_LAYERS_TYPE)
        frame = build_notify_layers_delta(**kwargs)
        tasks.append(_send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    if snapshot is not None:
        for ws in clients:
            sequencer = _state_sequencer(ws, NOTIFY_LAYERS_TYPE)
            sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def _broadcast_dims_state(
    server: Any,
    *,
    current_step: Sequence[int],
    source: str,
    intent_id: Optional[str],
    timestamp: Optional[float],
    targets: Optional[Sequence[Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    assert meta is not None, "notify.dims requires worker metadata"

    meta_dict = dict(meta)

    ndisplay_raw = meta_dict["ndisplay"]
    mode_raw = meta_dict["mode"]

    ndisplay = int(ndisplay_raw)
    mode_text = str(mode_raw).strip().lower()
    assert mode_text in {"volume", "plane"}, f"invalid dims mode: {raw_mode!r}"
    mode = "volume" if mode_text == "volume" else "plane"

    step_list = [int(value) for value in current_step]

    meta_dict["current_step"] = list(step_list)
    meta_dict["ndisplay"] = ndisplay
    meta_dict["mode"] = mode

    server._scene.last_dims_payload = dict(meta_dict)

    payload = build_notify_dims_payload(
        current_step=step_list,
        ndisplay=ndisplay,
        mode=mode,
        source=str(source),
    )

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not _feature_enabled(ws, "notify.dims"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        frame = build_notify_dims(
            session_id=session_id,
            payload=payload,
            timestamp=now,
            intent_id=intent_id,
        )
        tasks.append(_send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _broadcast_worker_dims(
    server: Any,
    *,
    current_step: Sequence[int],
    meta: Mapping[str, Any],
) -> None:
    server._update_scene_manager()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "worker dims: step=%s ndisplay=%s ndim=%s",
            [int(x) for x in current_step],
            meta.get("ndisplay"),
            meta.get("ndim"),
        )

    await _broadcast_dims_state(
        server,
        current_step=current_step,
        source="worker",
        intent_id=None,
        timestamp=None,
        meta=meta,
    )


def _normalize_camera_delta(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_camera_delta(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_camera_delta(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


async def _broadcast_camera_delta(
    server: Any,
    *,
    mode: str,
    delta: Mapping[str, Any],
    intent_id: Optional[str],
    origin: str,
    timestamp: Optional[float] = None,
    targets: Optional[Sequence[Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    payload = {
        "mode": str(mode),
        "delta": _normalize_camera_delta(delta),
        "origin": str(origin),
    }

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not _feature_enabled(ws, "notify.camera"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        frame = build_notify_camera(
            session_id=session_id,
            payload=payload,
            timestamp=now,
            intent_id=intent_id,
        )
        tasks.append(_send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_stream_frame(
    server: Any,
    ws: Any,
    *,
    payload: NotifyStreamPayload | Mapping[str, Any],
    timestamp: Optional[float],
    snapshot: EnvelopeSnapshot | None = None,
    force_snapshot: bool = False,
) -> EnvelopeSnapshot | None:
    session_id = _state_session(ws)
    if not session_id:
        return snapshot
    if not isinstance(payload, NotifyStreamPayload):
        payload = NotifyStreamPayload.from_dict(payload)
    now = time.time() if timestamp is None else float(timestamp)
    store = _history_store(server)
    if snapshot is None and store is not None:
        payload_dict = payload.to_dict()
        if force_snapshot:
            snapshot = store.snapshot_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )
        else:
            snapshot = store.delta_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )

    kwargs: Dict[str, Any] = {
        "session_id": session_id,
        "payload": payload,
        "timestamp": snapshot.timestamp if snapshot is not None else now,
    }
    sequencer = _state_sequencer(ws, NOTIFY_STREAM_TYPE)
    if snapshot is not None:
        kwargs["seq"] = snapshot.seq
        kwargs["delta_token"] = snapshot.delta_token
        kwargs["frame_id"] = snapshot.frame_id
        sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    else:
        if force_snapshot or sequencer.seq is None:
            cursor = sequencer.snapshot()
            kwargs["seq"] = cursor.seq
            kwargs["delta_token"] = cursor.delta_token
        else:
            kwargs["sequencer"] = sequencer
    frame = build_notify_stream(**kwargs)
    await _send_frame(server, ws, frame)
    return snapshot


async def _send_scene_level_frame(
    server: Any,
    ws: Any,
    *,
    payload: NotifySceneLevelPayload | Mapping[str, Any],
    timestamp: Optional[float],
    snapshot: EnvelopeSnapshot | None = None,
) -> EnvelopeSnapshot | None:
    session_id = _state_session(ws)
    if not session_id:
        return snapshot
    if not isinstance(payload, NotifySceneLevelPayload):
        payload = NotifySceneLevelPayload.from_dict(payload)
    now = time.time() if timestamp is None else float(timestamp)
    store = _history_store(server)
    if snapshot is None and store is not None:
        snapshot = store.delta_envelope(
            NOTIFY_SCENE_LEVEL_TYPE,
            payload=payload.to_dict(),
            timestamp=now,
        )

    kwargs: Dict[str, Any] = {
        "session_id": session_id,
        "payload": payload,
        "timestamp": snapshot.timestamp if snapshot is not None else now,
    }
    sequencer = _state_sequencer(ws, NOTIFY_SCENE_LEVEL_TYPE)
    if snapshot is not None:
        kwargs["seq"] = snapshot.seq
        kwargs["delta_token"] = snapshot.delta_token
        kwargs["frame_id"] = snapshot.frame_id
        sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    else:
        if sequencer.seq is None:
            cursor = sequencer.snapshot()
            kwargs["seq"] = cursor.seq
            kwargs["delta_token"] = cursor.delta_token
        else:
            kwargs["sequencer"] = sequencer
    frame = build_notify_scene_level(**kwargs)
    await _send_frame(server, ws, frame)
    return snapshot


async def _emit_scene_baseline(
    server: Any,
    ws: Any,
    *,
    payload: NotifyScenePayload | None,
    plan: ResumePlan | None,
    reason: str,
) -> None:
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    timestamp = time.time()

    if store is not None:
        snapshot = store.current_snapshot(NOTIFY_SCENE_TYPE)
        need_reset = plan is not None and plan.decision == ResumeDecision.RESET
        if snapshot is None or need_reset:
            if payload is None:
                payload = build_notify_scene_payload(
                    server._scene,
                    server._scene_manager,
                    viewer_settings=_viewer_settings(server),
                )
            snapshot = store.snapshot_envelope(
                NOTIFY_SCENE_TYPE,
                payload=payload.to_dict(),
                timestamp=timestamp,
            )
            store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=timestamp)
            store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=timestamp)
            store.reset_epoch(NOTIFY_SCENE_LEVEL_TYPE, timestamp=timestamp)
            _state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
            _state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()
            _state_sequencer(ws, NOTIFY_SCENE_LEVEL_TYPE).clear()
        await _send_scene_snapshot_from_cache(server, ws, snapshot)
    else:
        if payload is None:
            payload = build_notify_scene_payload(
                server._scene,
                server._scene_manager,
                viewer_settings=_viewer_settings(server),
            )
        session_id = _state_session(ws)
        if not session_id:
            return
        frame = build_notify_scene_snapshot(
            session_id=session_id,
            viewer=payload.viewer,
            layers=payload.layers,
            policies=payload.policies,
            metadata=payload.metadata,
            timestamp=timestamp,
            sequencer=_state_sequencer(ws, NOTIFY_SCENE_TYPE),
        )
        await _send_frame(server, ws, frame)

    if server._log_dims_info:
        logger.info("%s: notify.scene sent", reason)
    else:
        logger.debug("%s: notify.scene sent", reason)


async def _emit_scene_level_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
) -> None:
    store = _history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await _send_scene_level_frame(
                    server,
                    ws,
                    payload=snapshot.payload,
                    timestamp=snapshot.timestamp,
                    snapshot=snapshot,
                )
        return

    timestamp = time.time()
    payload: NotifySceneLevelPayload | None = None

    if store is not None:
        snapshot = store.current_snapshot(NOTIFY_SCENE_LEVEL_TYPE)
        need_reset = plan is not None and plan.decision == ResumeDecision.RESET
        if snapshot is None or need_reset:
            payload = build_notify_scene_level_payload(server._scene, server._scene_manager)
            snapshot = store.snapshot_envelope(
                NOTIFY_SCENE_LEVEL_TYPE,
                payload=payload.to_dict(),
                timestamp=timestamp,
            )
        if payload is None:
            payload = build_notify_scene_level_payload(server._scene, server._scene_manager)
        await _send_scene_level_frame(
            server,
            ws,
            payload=payload,
            timestamp=snapshot.timestamp,
            snapshot=snapshot,
        )
        return

    if payload is None:
        payload = build_notify_scene_level_payload(server._scene, server._scene_manager)
    await _send_scene_level_frame(server, ws, payload=payload, timestamp=timestamp)


async def _emit_layer_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
    fallback_controls: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    store = _history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await _send_layer_snapshot(server, ws, snapshot)
        return

    if not fallback_controls:
        return

    if store is not None:
        now = time.time()
        for layer_id, changes in fallback_controls:
            payload = build_notify_layers_payload(layer_id=layer_id or "layer-0", changes=changes)
            snapshot = store.delta_envelope(
                NOTIFY_LAYERS_TYPE,
                payload=payload.to_dict(),
                timestamp=now,
                intent_id=None,
            )
            await _send_layer_snapshot(server, ws, snapshot)
        return

    for layer_id, changes in fallback_controls:
        await _broadcast_layers_delta(
            server,
            layer_id=layer_id,
            changes=changes,
            intent_id=None,
            timestamp=time.time(),
            targets=[ws],
        )


async def _emit_stream_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
) -> None:
    store = _history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await _send_stream_snapshot(server, ws, snapshot)
        return

    channel = getattr(server, "_pixel_channel", None)
    cfg = getattr(server, "_pixel_config", None)
    if channel is None or cfg is None:
        raise AssertionError("Pixel channel not initialized")

    avcc = channel.last_avcc
    if avcc is not None:
        stream_payload = pixel_channel.build_notify_stream_payload(cfg, avcc)
        await _send_stream_frame(
            server,
            ws,
            payload=stream_payload,
            timestamp=time.time(),
        )
    else:
        pixel_channel.mark_stream_config_dirty(channel)


async def _emit_dims_baseline(
    server: Any,
    ws: Any,
    *,
    step_list: Sequence[int],
) -> None:
    await _broadcast_dims_state(
        server,
        current_step=step_list,
        source="server.bootstrap",
        intent_id=None,
        timestamp=time.time(),
        targets=[ws],
        meta=_current_dims_meta(server),
    )
    if server._log_dims_info:
        logger.info("connect: notify.dims baseline -> step=%s", step_list)
    else:
        logger.debug("connect: notify.dims baseline -> step=%s", step_list)


async def broadcast_stream_config(
    server: Any,
    *,
    payload: NotifyStreamPayload,
    timestamp: Optional[float] = None,
) -> None:
    clients = list(server._state_clients)
    if not clients:
        store = _history_store(server)
        if store is not None:
            now = time.time() if timestamp is None else float(timestamp)
            if store.current_snapshot(NOTIFY_STREAM_TYPE) is None:
                store.snapshot_envelope(
                    NOTIFY_STREAM_TYPE,
                    payload=payload.to_dict(),
                    timestamp=now,
                )
            else:
                store.delta_envelope(
                    NOTIFY_STREAM_TYPE,
                    payload=payload.to_dict(),
                    timestamp=now,
                )
        return

    now = time.time() if timestamp is None else float(timestamp)
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    snapshot_mode = False
    if store is not None:
        payload_dict = payload.to_dict()
        snapshot_mode = store.current_snapshot(NOTIFY_STREAM_TYPE) is None
        if snapshot_mode:
            snapshot = store.snapshot_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )
        else:
            snapshot = store.delta_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )

    tasks: list[Awaitable[None]] = []
    for ws in clients:
        if not _feature_enabled(ws, "notify.stream"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        tasks.append(
            _send_stream_frame(
                server,
                ws,
                payload=payload,
                timestamp=now,
                snapshot=snapshot,
                force_snapshot=snapshot_mode,
            )
        )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def broadcast_scene_level(
    server: Any,
    *,
    payload: NotifySceneLevelPayload | Mapping[str, Any] | None = None,
    timestamp: Optional[float] = None,
    reason: str = "scene.level",
) -> None:
    clients = list(server._state_clients)
    if not clients:
        store = _history_store(server)
        if store is not None:
            payload_obj = (
                payload
                if isinstance(payload, NotifySceneLevelPayload)
                else build_notify_scene_level_payload(server._scene, server._scene_manager)
            )
            if not isinstance(payload_obj, NotifySceneLevelPayload):
                payload_obj = NotifySceneLevelPayload.from_dict(payload_obj)
            store.delta_envelope(
                NOTIFY_SCENE_LEVEL_TYPE,
                payload=payload_obj.to_dict(),
                timestamp=time.time() if timestamp is None else float(timestamp),
            )
        return

    if isinstance(payload, NotifySceneLevelPayload):
        payload_obj = payload
    elif isinstance(payload, Mapping):
        payload_obj = NotifySceneLevelPayload.from_dict(payload)
    else:
        payload_obj = build_notify_scene_level_payload(server._scene, server._scene_manager)

    now = time.time() if timestamp is None else float(timestamp)
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None:
        snapshot = store.delta_envelope(
            NOTIFY_SCENE_LEVEL_TYPE,
            payload=payload_obj.to_dict(),
            timestamp=now,
        )

    tasks: list[Awaitable[EnvelopeSnapshot | None]] = []
    for ws in clients:
        if not _feature_enabled(ws, "notify.scene.level"):
            continue
        tasks.append(
            _send_scene_level_frame(
                server,
                ws,
                payload=payload_obj,
                timestamp=now,
                snapshot=snapshot,
            )
        )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        if server._log_dims_info:
            logger.info("%s: notify.scene.level broadcast to %d clients", reason, len(tasks))
        else:
            logger.debug("%s: notify.scene.level broadcast to %d clients", reason, len(tasks))


async def handle_state(server: Any, ws: Any) -> None:
    """Handle a state-channel websocket connection."""

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
            setattr(ws, "_napari_cuda_heartbeat_task", heartbeat_task)
        except Exception:
            logger.debug("failed to start heartbeat loop", exc_info=True)
        server._state_clients.add(ws)
        server.metrics.inc('napari_cuda_state_connects')
        server._update_client_gauges()
        await _send_state_baseline(server, ws)
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



async def _handle_session_ack(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    _ENVELOPE_PARSER.parse_ack(data)
    return True


async def _handle_session_goodbye(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
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

    session_id = _state_session(ws)
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


async def _handle_state_update(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        frame = _ENVELOPE_PARSER.parse_state_update(data)
    except Exception:
        logger.debug("state.update payload rejected by parser", exc_info=True)
        frame_id = data.get("frame_id")
        intent_id = data.get("intent_id")
        session_id = data.get("session") or _state_session(ws)
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

    session_id = envelope.session or _state_session(ws)
    frame_id = envelope.frame_id
    intent_id = envelope.intent_id
    timestamp = envelope.timestamp

    if session_id is None or frame_id is None or intent_id is None:
        raise ValueError("state.update envelope missing session/frame/intent identifiers")

    scope = str(payload.scope or "").strip().lower()
    key = str(payload.key or "").strip().lower()
    target = str(payload.target or "").strip()
    value = payload.value

    async def _reject(code: str, message_text: str, *, details: Mapping[str, Any] | None = None) -> None:
        error_payload: Dict[str, Any] = {"code": code, "message": message_text}
        if details:
            error_payload["details"] = dict(details)
        await _send_state_ack(
            server,
            ws,
            session_id=str(session_id),
            intent_id=intent_id,
            in_reply_to=frame_id,
            status="rejected",
            error=error_payload,
        )

    if not scope or not key or not target:
        await _reject("state.invalid", "scope/key/target required", details={"scope": scope, "key": key, "target": target})
        return True

    # ----- view scope ---------------------------------------------------
    if scope == 'view':
        if key != 'ndisplay':
            logger.debug("state.update view ignored key=%s", key)
            await _reject("state.invalid", f"unsupported view key {key}", details={"scope": scope, "key": key})
            return True
        try:
            raw_value = int(value) if value is not None else 2
        except Exception:
            logger.debug("state.update view ignored (non-integer ndisplay) value=%r", value)
            await _reject("state.invalid", "ndisplay must be integer", details={"scope": scope, "key": key})
            return True
        ndisplay = 3 if int(raw_value) >= 3 else 2
        try:
            await server._handle_set_ndisplay(ndisplay)
        except Exception:
            logger.debug("state.update view.set_ndisplay failed", exc_info=True)
            await _reject("state.error", "failed to apply ndisplay")
            return True

        await _send_state_ack(
            server,
            ws,
            session_id=str(session_id),
            intent_id=intent_id,
            in_reply_to=frame_id,
            status="accepted",
            applied_value=int(ndisplay),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "state.update view intent=%s frame=%s ndisplay=%s accepted",
                intent_id,
                frame_id,
                int(ndisplay),
            )
        return True

    # ----- camera scope -------------------------------------------------
    if scope == 'camera':
        cam_target = target.lower()
        if cam_target not in {"", "main"}:
            await _reject(
                "state.invalid",
                f"unsupported camera target {target}",
                details={"scope": scope, "target": target},
            )
            return True

        async def _ack_camera(applied_value: Mapping[str, Any] | str | None) -> None:
            await _send_state_ack(
                server,
                ws,
                session_id=str(session_id),
                intent_id=intent_id,
                in_reply_to=frame_id,
                status="accepted",
                applied_value=applied_value,
            )

        def _log_camera(fmt: str, *args: Any) -> None:
            if getattr(server, "_log_cam_info", False):
                logger.info(fmt, *args)
            elif getattr(server, "_log_cam_debug", False):
                logger.debug(fmt, *args)

        metrics = getattr(server, "metrics", None)

        if key == 'zoom':
            if not isinstance(value, Mapping):
                await _reject(
                    "state.invalid",
                    "camera.zoom requires mapping payload",
                    details={"scope": scope, "key": key},
                )
                return True
            try:
                factor = float(value.get('factor', 0.0))
            except Exception:
                factor = 0.0
            anchor_raw = value.get('anchor_px')
            if factor <= 0.0:
                await _reject(
                    "state.invalid",
                    "zoom factor must be positive",
                    details={"scope": scope, "key": key},
                )
                return True
            if not isinstance(anchor_raw, Sequence) or len(anchor_raw) < 2:
                await _reject(
                    "state.invalid",
                    "anchor_px requires [x, y]",
                    details={"scope": scope, "key": key},
                )
                return True
            try:
                anchor = (float(anchor_raw[0]), float(anchor_raw[1]))
            except Exception:
                await _reject(
                    "state.invalid",
                    "anchor_px must contain numeric values",
                    details={"scope": scope, "key": key},
                )
                return True

            ack_value = {
                "factor": float(factor),
                "anchor_px": [float(anchor[0]), float(anchor[1])],
            }

            await _ack_camera(ack_value)

            _log_camera(
                "state: camera.zoom factor=%.4f anchor=(%.1f,%.1f)",
                factor,
                anchor[0],
                anchor[1],
            )
            if metrics is not None:
                with suppress(Exception):
                    metrics.inc('napari_cuda_state_camera_updates')

            try:
                server._enqueue_camera_command(
                    ServerSceneCommand(kind='zoom', factor=float(factor), anchor_px=anchor),
                )
            except Exception:
                logger.debug("camera.zoom enqueue failed", exc_info=True)

            server._schedule_coro(
                _broadcast_camera_delta(
                    server,
                    mode='zoom',
                    delta=ack_value,
                    intent_id=intent_id,
                    origin='state.update',
                ),
                'state-camera-zoom',
            )
            return True

        if key == 'pan':
            if not isinstance(value, Mapping):
                await _reject(
                    "state.invalid",
                    "camera.pan requires mapping payload",
                    details={"scope": scope, "key": key},
                )
                return True
            try:
                dx = float(value.get('dx_px', 0.0))
            except Exception:
                dx = 0.0
            try:
                dy = float(value.get('dy_px', 0.0))
            except Exception:
                dy = 0.0

            ack_value = {"dx_px": dx, "dy_px": dy}

            await _ack_camera(ack_value)

            if dx != 0.0 or dy != 0.0:
                _log_camera("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
                if metrics is not None:
                    with suppress(Exception):
                        metrics.inc('napari_cuda_state_camera_updates')
                try:
                    server._enqueue_camera_command(
                        ServerSceneCommand(kind='pan', dx_px=float(dx), dy_px=float(dy)),
                    )
                except Exception:
                    logger.debug("camera.pan enqueue failed", exc_info=True)

                server._schedule_coro(
                    _broadcast_camera_delta(
                        server,
                        mode='pan',
                        delta=ack_value,
                        intent_id=intent_id,
                        origin='state.update',
                    ),
                    'state-camera-pan',
                )
            return True

        if key == 'orbit':
            if not isinstance(value, Mapping):
                await _reject(
                    "state.invalid",
                    "camera.orbit requires mapping payload",
                    details={"scope": scope, "key": key},
                )
                return True
            try:
                d_az = float(value.get('d_az_deg', 0.0))
            except Exception:
                d_az = 0.0
            try:
                d_el = float(value.get('d_el_deg', 0.0))
            except Exception:
                d_el = 0.0

            ack_value = {"d_az_deg": d_az, "d_el_deg": d_el}

            await _ack_camera(ack_value)

            if d_az != 0.0 or d_el != 0.0:
                _log_camera("state: camera.orbit daz=%.2f del=%.2f", d_az, d_el)
                if metrics is not None:
                    with suppress(Exception):
                        metrics.inc('napari_cuda_state_camera_updates')
                        metrics.inc('napari_cuda_orbit_events')
                try:
                    server._enqueue_camera_command(
                        ServerSceneCommand(kind='orbit', d_az_deg=float(d_az), d_el_deg=float(d_el)),
                    )
                except Exception:
                    logger.debug("camera.orbit enqueue failed", exc_info=True)

                server._schedule_coro(
                    _broadcast_camera_delta(
                        server,
                        mode='orbit',
                        delta=ack_value,
                        intent_id=intent_id,
                        origin='state.update',
                    ),
                    'state-camera-orbit',
                )
            return True

        if key == 'reset':
            reason = None
            if isinstance(value, Mapping):
                raw_reason = value.get('reason')
                if raw_reason is not None:
                    reason = str(raw_reason)
            elif isinstance(value, str) and value:
                reason = value
            else:
                reason = 'state.update'

            await _ack_camera({'reason': reason})

            _log_camera("state: camera.reset")
            if metrics is not None:
                with suppress(Exception):
                    metrics.inc('napari_cuda_state_camera_updates')
            try:
                server._enqueue_camera_command(ServerSceneCommand(kind='reset'))
            except Exception:
                logger.debug("camera.reset enqueue failed", exc_info=True)
            if getattr(server, "_idr_on_reset", False) and getattr(server, "_worker", None) is not None:
                server._schedule_coro(server._ensure_keyframe(), 'state-camera-reset-keyframe')

            server._schedule_coro(
                _broadcast_camera_delta(
                    server,
                    mode='reset',
                    delta={'reason': reason},
                    intent_id=intent_id,
                    origin='state.update',
                ),
                'state-camera-reset',
            )
            return True

        if key == 'set':
            if not isinstance(value, Mapping):
                await _reject(
                    "state.invalid",
                    "camera.set requires mapping payload",
                    details={"scope": scope, "key": key},
                )
                return True

            center_val = value.get('center')
            zoom_val = value.get('zoom')
            angles_val = value.get('angles')

            if center_val is None and zoom_val is None and angles_val is None:
                await _reject(
                    "state.invalid",
                    "camera.set payload must include center/zoom/angles",
                    details={"scope": scope, "key": key},
                )
                return True

            center_tuple: Optional[tuple[float, ...]] = None
            if center_val is not None:
                if not isinstance(center_val, Sequence) or not center_val:
                    await _reject(
                        "state.invalid",
                        "camera.center must be a sequence",
                        details={"scope": scope, "key": key},
                    )
                    return True
                try:
                    center_tuple = tuple(float(c) for c in center_val)
                except Exception:
                    await _reject(
                        "state.invalid",
                        "camera.center values must be numeric",
                        details={"scope": scope, "key": key},
                    )
                    return True

            zoom_float: Optional[float] = None
            if zoom_val is not None:
                try:
                    zoom_float = float(zoom_val)
                except Exception:
                    await _reject(
                        "state.invalid",
                        "camera.zoom must be numeric",
                        details={"scope": scope, "key": key},
                    )
                    return True

            angles_tuple: Optional[tuple[float, float, float]] = None
            if angles_val is not None:
                if not isinstance(angles_val, Sequence) or len(angles_val) < 3:
                    await _reject(
                        "state.invalid",
                        "camera.angles requires [az, el, roll]",
                        details={"scope": scope, "key": key},
                    )
                    return True
                try:
                    angles_tuple = (
                        float(angles_val[0]),
                        float(angles_val[1]),
                        float(angles_val[2]),
                    )
                except Exception:
                    await _reject(
                        "state.invalid",
                        "camera.angles must be numeric",
                        details={"scope": scope, "key": key},
                    )
                    return True

            with server._state_lock:
                latest = server._scene.latest_state
                server._scene.latest_state = replace(
                    latest,
                    center=center_tuple if center_tuple is not None else latest.center,
                    zoom=zoom_float if zoom_float is not None else latest.zoom,
                    angles=angles_tuple if angles_tuple is not None else latest.angles,
                )

            _enqueue_latest_state_for_worker(server)

            ack_components: Dict[str, Any] = {}
            if center_tuple is not None:
                ack_components['center'] = [float(c) for c in center_tuple]
            if zoom_float is not None:
                ack_components['zoom'] = float(zoom_float)
            if angles_tuple is not None:
                ack_components['angles'] = [float(a) for a in angles_tuple]

            await _ack_camera(ack_components)

            _log_camera(
                "state: camera.set center=%s zoom=%s angles=%s",
                ack_components.get('center'),
                ack_components.get('zoom'),
                ack_components.get('angles'),
            )

            if metrics is not None:
                with suppress(Exception):
                    metrics.inc('napari_cuda_state_camera_updates')

            server._schedule_coro(
                _broadcast_camera_delta(
                    server,
                    mode='set',
                    delta=ack_components,
                    intent_id=intent_id,
                    origin='state.update',
                ),
                'state-camera-set',
            )
            return True

        await _reject(
            "state.invalid",
            f"unsupported camera key {key}",
            details={"scope": scope, "key": key},
        )
        return True

    # ----- layer scope --------------------------------------------------
    if scope == 'layer':
        layer_id = target
        try:
            result = apply_layer_state_update(
                server._scene,
                server._state_lock,
                layer_id=layer_id,
                prop=key,
                value=value,
                intent_id=intent_id,
                timestamp=timestamp,
            )
        except KeyError:
            logger.debug("state.update unknown layer prop=%s", key)
            await _reject("state.invalid", f"unsupported layer key {key}", details={"scope": scope, "key": key, "target": layer_id})
            return True
        except Exception:
            logger.debug("state.update failed for layer=%s key=%s", layer_id, key, exc_info=True)
            await _reject("state.error", "layer update failed", details={"scope": scope, "key": key, "target": layer_id})
            return True

        await _send_state_ack(
            server,
            ws,
            session_id=str(session_id),
            intent_id=intent_id,
            in_reply_to=frame_id,
            status="accepted",
            applied_value=result.value,
        )

        logger.debug(
            "state.update layer intent=%s frame=%s layer_id=%s key=%s value=%s server_seq=%s",
            intent_id,
            frame_id,
            layer_id,
            key,
            result.value,
            result.server_seq,
        )

        _enqueue_latest_state_for_worker(server)
        server._schedule_coro(
            _broadcast_state_update(server, result),
            f'state-layer-{key}',
        )
        return True

    # ----- volume scope -------------------------------------------------
    if scope == 'volume':
        ts = time.time()
        if key == 'render_mode':
            mode = str(value or '').lower().strip()
            if not state_updates.is_valid_render_mode(mode, server._allowed_render_modes):
                logger.debug("state.update volume ignored invalid render_mode=%r", value)
                await _reject("state.invalid", "unknown render_mode", details={"scope": scope, "key": key, "value": value})
                return True
            state_updates.update_volume_mode(server._scene, server._state_lock, mode)
            normalized_value: Any = mode
            rebroadcast_tag = 'rebroadcast-volume-mode'
        elif key == 'contrast_limits':
            pair = value
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                logger.debug("state.update volume ignored invalid contrast_limits=%r", pair)
                await _reject("state.invalid", "contrast_limits requires [lo, hi]", details={"scope": scope, "key": key})
                return True
            norm = state_updates.normalize_clim(pair[0], pair[1])
            if norm is None:
                logger.debug("state.update volume ignored invalid clim=%r", pair)
                await _reject("state.invalid", "contrast_limits invalid", details={"scope": scope, "key": key})
                return True
            lo, hi = norm
            state_updates.update_volume_clim(server._scene, server._state_lock, lo, hi)
            normalized_value = (float(lo), float(hi))
            rebroadcast_tag = 'rebroadcast-volume-clim'
        elif key == 'colormap':
            name = value
            if not isinstance(name, str) or not name.strip():
                logger.debug("state.update volume ignored invalid colormap=%r", name)
                await _reject("state.invalid", "colormap must be non-empty", details={"scope": scope, "key": key})
                return True
            cmap = str(name)
            state_updates.update_volume_colormap(server._scene, server._state_lock, cmap)
            normalized_value = cmap
            rebroadcast_tag = 'rebroadcast-volume-colormap'
        elif key == 'opacity':
            alpha = value
            try:
                norm_alpha = state_updates.clamp_opacity(alpha)
            except Exception:
                logger.debug("state.update volume ignored invalid opacity=%r", alpha)
                await _reject("state.invalid", "opacity must be float", details={"scope": scope, "key": key})
                return True
            state_updates.update_volume_opacity(server._scene, server._state_lock, norm_alpha)
            normalized_value = float(norm_alpha)
            rebroadcast_tag = 'rebroadcast-volume-opacity'
        elif key == 'sample_step':
            rel = value
            try:
                norm_step = state_updates.clamp_sample_step(rel)
            except Exception:
                logger.debug("state.update volume ignored invalid sample_step=%r", rel)
                await _reject("state.invalid", "sample_step must be float", details={"scope": scope, "key": key})
                return True
            state_updates.update_volume_sample_step(server._scene, server._state_lock, norm_step)
            normalized_value = float(norm_step)
            rebroadcast_tag = 'rebroadcast-volume-sample-step'
        else:
            logger.debug("state.update volume ignored key=%s", key)
            await _reject("state.invalid", f"unsupported volume key {key}", details={"scope": scope, "key": key})
            return True

        await _send_state_ack(
            server,
            ws,
            session_id=str(session_id),
            intent_id=intent_id,
            in_reply_to=frame_id,
            status="accepted",
            applied_value=normalized_value,
        )

        logger.debug(
            "state.update volume intent=%s frame=%s key=%s value=%s",
            intent_id,
            frame_id,
            key,
            normalized_value,
        )

        _enqueue_latest_state_for_worker(server)
        result = _baseline_state_result(
            server,
            scope='volume',
            target=target,
            key=key,
            value=normalized_value,
            intent_id=intent_id,
            timestamp=ts,
        )
        server._schedule_coro(
            _broadcast_state_update(server, result),
            f'state-volume-{key}',
        )
        return True

    # ----- multiscale scope ---------------------------------------------
    if scope == 'multiscale':
        ts = time.time()
        log_fn = logger.info if server._log_state_traces else logger.debug

        if key == 'policy':
            policy = str(value or '').lower().strip()
            allowed = {'oversampling', 'thresholds', 'ratio'}
            if policy not in allowed:
                logger.debug("state.update multiscale ignored invalid policy=%r", value)
                await _reject("state.invalid", "unknown multiscale policy", details={"scope": scope, "key": key})
                return True
            server._scene.multiscale_state['policy'] = policy
            normalized_value = policy
            rebroadcast_tag = 'rebroadcast-policy'
            if server._worker is not None:
                try:
                    log_fn("state: multiscale policy -> worker.set_policy start")
                    server._worker.set_policy(policy)
                    log_fn("state: multiscale policy -> worker.set_policy done")
                except Exception:
                    logger.exception("worker set_policy failed for %s", policy)
        elif key == 'level':
            levels = server._scene.multiscale_state.get('levels') or []
            level = state_updates.clamp_level(value, levels)
            if level is None:
                logger.debug("state.update multiscale ignored invalid level=%r", value)
                await _reject("state.invalid", "level out of range", details={"scope": scope, "key": key})
                return True
            server._scene.multiscale_state['current_level'] = int(level)
            normalized_value = int(level)
            rebroadcast_tag = 'rebroadcast-ms-level'
            if server._worker is not None:
                try:
                    levels_meta = server._scene.multiscale_state.get('levels') or []
                    path = None
                    if isinstance(levels_meta, list) and 0 <= int(level) < len(levels_meta):
                        entry = levels_meta[int(level)]
                        if isinstance(entry, Mapping):
                            path = entry.get('path')
                    log_fn("state: multiscale level -> worker.request start level=%s", level)
                    server._worker.request_multiscale_level(int(level), path)
                    log_fn("state: multiscale level -> worker.request done")
                    log_fn("state: multiscale level -> worker.force_idr start")
                    server._worker.force_idr()
                    log_fn("state: multiscale level -> worker.force_idr done")
                    server._pixel.bypass_until_key = True
                except Exception:
                    logger.exception("multiscale level switch request failed")
        else:
            logger.debug("state.update multiscale ignored key=%s", key)
            await _reject("state.invalid", f"unsupported multiscale key {key}", details={"scope": scope, "key": key})
            return True

        await _send_state_ack(
            server,
            ws,
            session_id=str(session_id),
            intent_id=intent_id,
            in_reply_to=frame_id,
            status="accepted",
            applied_value=normalized_value,
        )

        result = _baseline_state_result(
            server,
            scope='multiscale',
            target=target,
            key=key,
            value=normalized_value,
            intent_id=intent_id,
            timestamp=timestamp or ts,
        )
        server._schedule_coro(
            _broadcast_state_update(server, result),
            f'state-multiscale-{key}',
        )
        return True

    # ----- dims scope ---------------------------------------------------
    if scope == 'dims':
        step_delta: Optional[int] = None
        set_value: Optional[int] = None
        norm_key = key
        value_obj = value
        if key == 'index':
            norm_key = 'index'
            if isinstance(value_obj, Integral):
                set_value = int(value_obj)
                value_obj = None
            else:
                logger.debug("state.update dims ignored (non-integer index) axis=%s value=%r", target, value_obj)
                await _reject("state.invalid", "index must be integer", details={"scope": scope, "key": key, "target": target})
                return True
        elif key == 'step':
            norm_key = 'step'
            if isinstance(value_obj, Integral):
                step_delta = int(value_obj)
                value_obj = None
            else:
                logger.debug("state.update dims ignored (non-integer step delta) axis=%s value=%r", target, value_obj)
                await _reject("state.invalid", "step delta must be integer", details={"scope": scope, "key": key, "target": target})
                return True
        else:
            logger.debug("state.update dims ignored (unsupported key) axis=%s key=%s", target, key)
            await _reject("state.invalid", f"unsupported dims key {key}", details={"scope": scope, "key": key, "target": target})
            return True

        meta = server._scene.last_dims_payload
        assert meta is not None, "dims metadata not initialized"
        try:
            result = apply_dims_state_update(
                server._scene,
                server._state_lock,
                dict(meta),
                axis=target,
                prop=norm_key,
                value=value_obj,
                step_delta=step_delta,
                set_value=set_value,
                intent_id=intent_id,
                timestamp=timestamp,
            )
        except Exception:
            logger.debug("state.update dims failed axis=%s key=%s", target, key, exc_info=True)
            await _reject("state.error", "dims update failed", details={"scope": scope, "key": key, "target": target})
            return True
        if result is None:
            logger.debug("state.update dims invalid axis=%s key=%s", target, key)
            await _reject("state.invalid", "dims update invalid", details={"scope": scope, "key": key, "target": target})
            return True

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "state.update dims applied: axis=%s key=%s step_delta=%s set_value=%s current_step=%s",
                target,
                key,
                step_delta,
                set_value,
                result.current_step,
            )

        await _send_state_ack(
            server,
            ws,
            session_id=str(session_id),
            intent_id=intent_id,
            in_reply_to=frame_id,
            status="accepted",
            applied_value=result.value,
        )

        _enqueue_latest_state_for_worker(server)

        server._schedule_coro(
            _broadcast_state_update(server, result),
            f'state-dims-{key}',
        )
        return True

    logger.debug("state.update unknown scope=%s", scope)
    await _reject("state.invalid", f"unknown scope {scope}", details={"scope": scope})
    return True

MESSAGE_HANDLERS: dict[str, StateMessageHandler] = {
    STATE_UPDATE_TYPE: _handle_state_update,
    'call.command': _handle_call_command,
    SESSION_ACK_TYPE: _handle_session_ack,
    SESSION_GOODBYE_TYPE: _handle_session_goodbye,
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


def _enqueue_latest_state_for_worker(server: Any) -> None:
    """Snapshot the current scene state and push it into the render mailbox."""

    worker = getattr(server, "_worker", None)
    if worker is None or not hasattr(worker, "enqueue_update"):
        return

    try:
        with server._state_lock:
            snapshot = server._scene.latest_state
    except Exception:
        logger.debug("state: failed to snapshot latest scene state", exc_info=True)
        return

    try:
        worker.enqueue_update(RenderDelta(scene_state=snapshot))
    except Exception:
        logger.debug("state: worker enqueue_update failed", exc_info=True)


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


async def _broadcast_state_update(
    server: Any,
    result: StateUpdateResult,
    *,
    include_control_versions: bool = True,
) -> None:
    if result.scope in {"layer", "volume", "multiscale"}:
        layer_id, changes = _layer_changes_from_result(server, result)
        if not layer_id:
            logger.debug(
                "layer delta ignored: unable to resolve layer id for scope=%s target=%s",
                result.scope,
                result.target,
            )
            return
        await _broadcast_layers_delta(
            server,
            layer_id=layer_id,
            changes=changes,
            intent_id=result.intent_id,
            timestamp=result.timestamp,
        )
        return

    if result.scope == "dims":
        # The render worker will emit notify.dims once it applies the update;
        # avoid duplicating frames from the control plane.
        return

    if result.scope == "view" and result.key == "ndisplay":
        # Mode switch is completed by the worker; rely on its notify.dims payload
        # to propagate authoritative metadata instead of replaying cached data.
        return

    logger.debug(
        "state update scope ignored by greenfield broadcaster scope=%s target=%s",
        result.scope,
        result.target,
    )


async def _broadcast_state_updates(
    server: Any,
    results: Sequence[StateUpdateResult],
    *,
    include_control_versions: bool = True,
) -> None:
    for result in results:
        await _broadcast_state_update(
            server,
            result,
            include_control_versions=include_control_versions,
        )


def _ndim_from_meta(meta: Mapping[str, Any]) -> int:
    try:
        ndim = int(meta.get("ndim") or 0)
    except Exception:
        ndim = 0

    if ndim <= 0:
        for key in ("order", "axes", "range", "sizes"):
            values = meta.get(key)
            if isinstance(values, Sequence):
                try:
                    length = len(tuple(values))
                except Exception:
                    length = 0
                ndim = max(ndim, length)

    if ndim <= 0:
        current = meta.get("current_step")
        if isinstance(current, Sequence):
            try:
                ndim = max(ndim, len(tuple(current)))
            except Exception:
                ndim = max(ndim, 0)

    return max(1, int(ndim))


def _baseline_state_result(
    server: Any,
    *,
    scope: str,
    target: str,
    key: str,
    value: Any,
    server_seq_override: Optional[int] = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> StateUpdateResult:
    meta = get_control_meta(server._scene, scope, target, key)
    server_seq = int(server_seq_override or meta.last_server_seq or 0)
    if server_seq <= 0:
        server_seq = increment_server_sequence(server._scene)
    meta.last_server_seq = server_seq
    ts = float(timestamp) if timestamp is not None else time.time()
    meta.last_timestamp = ts

    return StateUpdateResult(
        scope=scope,
        target=target,
        key=key,
        value=value,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
    )


def _dims_results_from_step(
    server: Any,
    step_list: Sequence[int],
    *,
    meta: Mapping[str, Any],
    timestamp: Optional[float] = None,
) -> list[StateUpdateResult]:
    results: list[StateUpdateResult] = []
    current = [int(x) for x in step_list]
    ts = float(timestamp) if timestamp is not None else time.time()
    for idx, value in enumerate(current):
        axis_label = axis_label_from_meta(meta, idx) or str(idx)
        meta_entry = get_control_meta(server._scene, "dims", axis_label, "step")
        server_seq = int(meta_entry.last_server_seq or 0)
        if server_seq <= 0:
            server_seq = increment_server_sequence(server._scene)
            meta_entry.last_server_seq = server_seq
        meta_entry.last_timestamp = ts
        results.append(
            StateUpdateResult(
                scope="dims",
                target=axis_label,
                key="step",
                value=int(value),
                server_seq=server_seq,
                timestamp=ts,
                axis_index=idx,
                current_step=tuple(current),
            )
        )
    return results


def _layer_results_from_controls(
    server: Any,
    layer_id: str,
    controls: Mapping[str, Any],
) -> list[StateUpdateResult]:
    results: list[StateUpdateResult] = []
    for key, value in controls.items():
        results.append(
            _baseline_state_result(
                server,
                scope="layer",
                target=layer_id,
                key=str(key),
                value=value,
            )
        )
    return results

async def broadcast_layer_update(
    server: Any,
    *,
    layer_id: str,
    changes: Mapping[str, Any],
    server_seq: Optional[int] = None,
    timestamp: Optional[float] = None,
) -> None:
    """Broadcast state.update payloads for the given *layer_id*."""

    if not changes:
        return

    ts = float(timestamp) if timestamp is not None else time.time()
    results: list[StateUpdateResult] = []
    for key, value in changes.items():
        results.append(
            _baseline_state_result(
                server,
                scope="layer",
                target=layer_id,
                key=str(key),
                value=value,
                server_seq_override=server_seq,
                timestamp=ts,
            )
        )

    await _broadcast_state_updates(server, results)


def _normalize_scene_level_payload(level_payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {str(key): value for key, value in level_payload.items()}
    assert "current_level" in normalized, "scene level payload missing current_level"
    normalized["current_level"] = int(normalized["current_level"])

    if "downgraded" in normalized:
        normalized["downgraded"] = bool(normalized["downgraded"])

    if "levels" in normalized:
        raw_levels = normalized["levels"]
        assert type(raw_levels) in (list, tuple), "scene level levels must be a sequence"
        level_entries: list[Dict[str, Any]] = []
        for entry in raw_levels:
            assert type(entry) is dict, "scene level entry must be a dict"
            entry_normalized: Dict[str, Any] = {str(key): value for key, value in entry.items()}
            if "index" in entry_normalized:
                entry_normalized["index"] = int(entry_normalized["index"])
            if "shape" in entry_normalized:
                shape_raw = entry_normalized["shape"]
                assert type(shape_raw) in (list, tuple), "scene level shape must be a sequence"
                entry_normalized["shape"] = [int(value) for value in shape_raw]
            if "downsample" in entry_normalized:
                down_raw = entry_normalized["downsample"]
                if type(down_raw) in (list, tuple):
                    entry_normalized["downsample"] = [float(value) for value in down_raw]
            level_entries.append(entry_normalized)
        normalized["levels"] = level_entries

    return normalized


def _verify_dims_against_level(
    meta: Dict[str, Any], level_payload: Mapping[str, Any]
) -> tuple[list[int], list[list[int]]]:
    levels = level_payload.get("levels")
    assert levels, "scene level payload missing levels"

    current_level = int(level_payload["current_level"])

    descriptor: Optional[Mapping[str, Any]] = None
    for entry in levels:
        assert type(entry) is dict, "scene level entry must be a dict"
        if "index" not in entry:
            continue
        if int(entry["index"]) == current_level:
            descriptor = entry
            break

    assert descriptor is not None, "scene level descriptor missing for active level"
    assert "shape" in descriptor, "scene level descriptor missing shape"

    shape_raw = descriptor["shape"]
    assert type(shape_raw) in (list, tuple), "scene level shape must be a sequence"
    shape = [int(value) for value in shape_raw]

    sizes_raw = meta.get("sizes")
    assert sizes_raw, "worker dims metadata missing sizes"
    sizes = [int(value) for value in sizes_raw]
    assert len(sizes) >= len(shape), "worker dims sizes shorter than level shape"

    ranges_raw = meta.get("range")
    assert ranges_raw, "worker dims metadata missing range"
    ranges: list[list[int]] = []
    for bounds in ranges_raw:
        assert type(bounds) in (list, tuple), "worker dims range entry must be a sequence"
        low = int(bounds[0])
        high = int(bounds[1])
        ranges.append([low, high])
    assert len(ranges) >= len(shape), "worker dims range shorter than level shape"

    for idx, expected in enumerate(shape):
        actual = sizes[idx]
        assert actual == expected, f"worker dims size mismatch at axis {idx}"
        low, high = ranges[idx]
        assert low == 0, f"worker dims range lower bound mismatch at axis {idx}"
        expected_high = max(expected - 1, 0)
        assert high == expected_high, f"worker dims range upper bound mismatch at axis {idx}"

    ndim = meta.get("ndim")
    if ndim is not None:
        assert int(ndim) == len(sizes), "worker dims ndim mismatch"
    else:
        meta["ndim"] = len(sizes)

    return sizes, ranges


def process_worker_notifications(
    server: Any, notifications: Sequence[WorkerSceneNotification]
) -> None:
    if not notifications:
        return

    deferred: list[WorkerSceneNotification] = []
    scene_data = server._scene
    multiscale_state = scene_data.multiscale_state

    for note in notifications:
        if note.seq <= scene_data.last_scene_seq:
            continue

        if note.kind == "dims_update":
            assert note.meta is not None, "worker dims notification missing metadata"
            meta = dict(note.meta)

            assert note.step is not None, "worker dims notification missing step"
            step_tuple = tuple(int(value) for value in note.step)

            ndim = _ndim_from_meta(meta)
            assert len(step_tuple) == ndim, "worker dims step length mismatch"

            sizes_raw = meta["sizes"]
            meta["sizes"] = [int(value) for value in sizes_raw]

            ranges_raw = meta["range"]
            coerced_ranges: list[list[int]] = []
            for bounds in ranges_raw:
                low = int(bounds[0])
                high = int(bounds[1])
                coerced_ranges.append([low, high])
            meta["range"] = coerced_ranges

            pending_level = None
            if "pending_worker_level" in multiscale_state:
                pending_level = multiscale_state["pending_worker_level"]

            if pending_level is not None:
                sizes, ranges = _verify_dims_against_level(meta, pending_level)
                meta["sizes"] = sizes
                meta["range"] = ranges
                multiscale_state["current_level"] = int(pending_level["current_level"])
                multiscale_state["level"] = int(pending_level["current_level"])
                if "downgraded" in pending_level:
                    multiscale_state["downgraded"] = bool(pending_level["downgraded"])
                if "levels" in pending_level:
                    multiscale_state["levels"] = list(pending_level["levels"])
                del multiscale_state["pending_worker_level"]

            mode_text = str(meta.get("mode", "")).strip().lower()
            if mode_text not in {"plane", "volume"}:
                raise AssertionError(f"worker dims mode invalid: {mode_text!r}")
            scene_mode = "volume" if bool(server._scene.use_volume) else "plane"
            if mode_text != scene_mode:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "worker dims ignored due to mode mismatch: worker=%s control=%s",
                        mode_text,
                        scene_mode,
                    )
                continue

            with server._state_lock:
                latest = server._scene.latest_state
                server._scene.latest_state = replace(latest, current_step=step_tuple)
            scene_data.last_dims_payload = dict(meta)
            server._schedule_coro(
                _broadcast_worker_dims(
                    server,
                    current_step=step_tuple,
                    meta=meta,
                ),
                "dims_update-worker",
            )
            scene_data.last_scene_seq = note.seq
        elif note.kind == "scene_level" and note.level is not None:
            server._update_scene_manager()
            level_payload = note.level
            assert type(level_payload) is dict, "scene level payload must be a dict"
            normalized_level = _normalize_scene_level_payload(level_payload)
            pending_level_copy: Dict[str, Any] = {}
            for key, value in normalized_level.items():
                if key == "levels" and value is not None:
                    pending_level_copy[key] = [dict(entry) for entry in value]
                else:
                    pending_level_copy[key] = value
            multiscale_state["pending_worker_level"] = pending_level_copy
            server._schedule_coro(
                broadcast_scene_level(
                    server,
                    payload=normalized_level,
                    reason="worker",
                ),
                "scene_level-worker",
            )
            scene_data.last_scene_seq = note.seq

    if deferred:
        for note in deferred:
            server._worker_notifications.push(note)
        asyncio.get_running_loop().call_later(0.05, server._process_worker_notifications)


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
    except asyncio.TimeoutError:
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

    history_store = _history_store(server)
    client_tokens = hello.payload.resume_tokens
    resume_plan: Dict[str, ResumePlan] = {}

    for topic in _RESUMABLE_TOPICS:
        toggle = negotiated_features.get(topic)
        if toggle is None or not toggle.enabled or not toggle.resume:
            continue
        token = client_tokens.get(topic)
        if history_store is None:
            resume_plan[topic] = ResumePlan(topic=topic, decision=ResumeDecision.RESET, deltas=[])
            continue
        plan = history_store.plan_resume(topic, token)
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
            cursor = history_store.latest_resume_state(topic)
            if cursor is not None:
                negotiated_features[topic] = replace(toggle, resume_state=cursor)
            continue
        cursor = history_store.latest_resume_state(topic)
        if cursor is not None:
            negotiated_features[topic] = replace(toggle, resume_state=cursor)

    setattr(ws, "_napari_cuda_resume_plan", resume_plan)

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

    setattr(ws, "_napari_cuda_session", session_id)
    setattr(ws, "_napari_cuda_features", negotiated_features)
    setattr(ws, "_napari_cuda_heartbeat_interval", float(heartbeat_s))
    setattr(ws, "_napari_cuda_shutdown", False)
    setattr(ws, "_napari_cuda_goodbye_sent", False)
    setattr(
        ws,
        "_napari_cuda_sequencers",
        {
            NOTIFY_SCENE_TYPE: ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE),
            NOTIFY_LAYERS_TYPE: ResumableTopicSequencer(topic=NOTIFY_LAYERS_TYPE),
            NOTIFY_STREAM_TYPE: ResumableTopicSequencer(topic=NOTIFY_STREAM_TYPE),
        },
    )

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


def _resolve_handshake_features(hello: SessionHello) -> Dict[str, FeatureToggle]:
    client_features = hello.payload.features
    negotiated: Dict[str, FeatureToggle] = {}
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
    await _await_state_send(server, ws, text)


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


async def _await_state_send(server: Any, ws: Any, text: str) -> None:
    try:
        result = server._state_send(ws, text)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.debug("state send helper failed", exc_info=True)


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
) -> None:
    """Build and emit an ``ack.state`` frame that mirrors the incoming update."""

    if not intent_id or not in_reply_to:
        raise ValueError("ack.state requires intent_id and in_reply_to identifiers")

    normalized_status = str(status).lower()
    if normalized_status not in {"accepted", "rejected"}:
        raise ValueError("ack.state status must be 'accepted' or 'rejected'")

    payload: Dict[str, Any] = {
        "intent_id": str(intent_id),
        "in_reply_to": str(in_reply_to),
        "status": normalized_status,
    }

    if normalized_status == "accepted":
        if error is not None:
            raise ValueError("accepted ack.state payload cannot include error details")
        if applied_value is not None:
            payload["applied_value"] = applied_value
    else:
        if not isinstance(error, Mapping):
            raise ValueError("rejected ack.state payload requires {code, message}")
        if "code" not in error or "message" not in error:
            raise ValueError("ack.state error payload must include 'code' and 'message'")
        payload["error"] = dict(error)

    frame = build_ack_state(
        session_id=str(session_id),
        frame_id=None,
        payload=payload,
        timestamp=time.time(),
    )

    await _send_frame(server, ws, frame)


async def _send_frame(server: Any, ws: Any, frame: Any) -> None:
    try:
        payload = frame.to_dict()
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError("frame must provide to_dict()") from exc
    text = json.dumps(payload, separators=(",", ":"))
    await _await_state_send(server, ws, text)


async def _send_layer_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = _state_session(ws)
    if not session_id:
        return
    payload = NotifyLayersPayload.from_dict(snapshot.payload)
    frame = build_notify_layers_delta(
        session_id=session_id,
        payload=payload,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        intent_id=snapshot.intent_id,
        seq=snapshot.seq,
        delta_token=snapshot.delta_token,
    )
    await _send_frame(server, ws, frame)
    sequencer = _state_sequencer(ws, NOTIFY_LAYERS_TYPE)
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def _send_stream_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = _state_session(ws)
    if not session_id:
        return
    payload = NotifyStreamPayload.from_dict(snapshot.payload)
    frame = build_notify_stream(
        session_id=session_id,
        payload=payload,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        seq=snapshot.seq,
        delta_token=snapshot.delta_token,
    )
    await _send_frame(server, ws, frame)
    sequencer = _state_sequencer(ws, NOTIFY_STREAM_TYPE)
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def _send_scene_snapshot_from_cache(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = _state_session(ws)
    if not session_id:
        return
    payload = NotifyScenePayload.from_dict(snapshot.payload)
    sequencer = _state_sequencer(ws, NOTIFY_SCENE_TYPE)
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    frame = build_notify_scene_snapshot(
        session_id=session_id,
        viewer=payload.viewer,
        layers=payload.layers,
        policies=payload.policies,
        metadata=payload.metadata,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        delta_token=snapshot.delta_token,
        intent_id=snapshot.intent_id,
        sequencer=sequencer,
    )
    await _send_frame(server, ws, frame)


async def _send_state_baseline(server: Any, ws: Any) -> None:
    resume_map: Dict[str, ResumePlan] = getattr(ws, "_napari_cuda_resume_plan", {}) or {}
    scene_plan = resume_map.get(NOTIFY_SCENE_TYPE)
    scene_level_plan = resume_map.get(NOTIFY_SCENE_LEVEL_TYPE)
    layers_plan = resume_map.get(NOTIFY_LAYERS_TYPE)
    stream_plan = resume_map.get(NOTIFY_STREAM_TYPE)

    scene_payload: NotifyScenePayload | None = None
    step_list: list[int] = []
    fallback_controls: list[tuple[str, Mapping[str, Any]]] = []

    try:
        await server._await_adapter_level_ready(0.5)
        try:
            if hasattr(server, "_update_scene_manager"):
                server._update_scene_manager()
        except Exception:
            logger.debug("Initial scene manager sync failed", exc_info=True)

        with server._state_lock:
            current_step = server._scene.latest_state.current_step

        step_list = list(current_step) if current_step is not None else []

        snapshot = server._scene_manager.scene_snapshot()
        assert snapshot is not None, "scene snapshot unavailable"
        dims_block = snapshot.viewer.dims
        ndim = int(dims_block.get("ndim", 0) or len(dims_block.get("sizes", [])) or 3)
        while len(step_list) < ndim:
            step_list.append(0)

        scene_payload = build_notify_scene_payload(
            server._scene,
            server._scene_manager,
            viewer_settings=_viewer_settings(server),
        )

        scene_data = server._scene
        latest_updates = scene_data.latest_state.layer_updates or {}
        for layer_snapshot in snapshot.layers:
            layer_id = layer_snapshot.layer_id
            controls: Dict[str, Any] = {}
            control_state = scene_data.layer_controls.get(layer_id)
            if control_state is not None:
                controls.update(layer_controls_to_dict(control_state))
            pending = latest_updates.get(layer_id, {})
            for key, value in pending.items():
                controls[str(key)] = value
            if controls:
                fallback_controls.append((layer_id, controls))
    except Exception:
        logger.debug("Initial scene baseline prep failed", exc_info=True)

    try:
        await _emit_scene_baseline(
            server,
            ws,
            payload=scene_payload,
            plan=scene_plan,
            reason="connect",
        )
    except Exception:
        logger.exception("Initial notify.scene send failed")

    try:
        await _emit_scene_level_baseline(
            server,
            ws,
            plan=scene_level_plan,
        )
    except Exception:
        logger.exception("Initial scene level send failed")

    try:
        await _emit_layer_baseline(
            server,
            ws,
            plan=layers_plan,
            fallback_controls=fallback_controls,
        )
    except Exception:
        logger.exception("Initial layer baseline send failed")

    try:
        await _emit_stream_baseline(server, ws, plan=stream_plan)
    except Exception:
        logger.exception("Initial state config send failed")

    if step_list:
        try:
            await _emit_dims_baseline(server, ws, step_list=step_list)
        except Exception:
            logger.exception("Initial dims baseline send failed")

    assert hasattr(server, "_ensure_keyframe"), "server must expose _ensure_keyframe"
    assert hasattr(server, "_schedule_coro"), "server must expose _schedule_coro"
    server._schedule_coro(server._ensure_keyframe(), "state-baseline-keyframe")

    if hasattr(ws, "_napari_cuda_resume_plan"):
        delattr(ws, "_napari_cuda_resume_plan")


async def _send_scene_snapshot_direct(server: Any, ws: Any, *, reason: str) -> None:
    session_id = _state_session(ws)
    if not session_id:
        logger.debug("Skipping notify.scene send without session id")
        return

    payload = build_notify_scene_payload(
        server._scene,
        server._scene_manager,
        viewer_settings=_viewer_settings(server),
    )
    timestamp = time.time()
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None:
        snapshot = store.snapshot_envelope(
            NOTIFY_SCENE_TYPE,
            payload=payload.to_dict(),
            timestamp=timestamp,
        )
    frame = build_notify_scene_snapshot(
        session_id=session_id,
        viewer=payload.viewer,
        layers=payload.layers,
        policies=payload.policies,
        metadata=payload.metadata,
        timestamp=snapshot.timestamp if snapshot is not None else timestamp,
        delta_token=snapshot.delta_token if snapshot is not None else None,
        frame_id=snapshot.frame_id if snapshot is not None else None,
    )
    await _send_frame(server, ws, frame)
    if server._log_dims_info:
        logger.info("%s: notify.scene sent", reason)
    else:
        logger.debug("%s: notify.scene sent", reason)


async def broadcast_scene_snapshot(server: Any, *, reason: str) -> None:
    clients = list(server._state_clients)
    if not clients:
        return
    payload = build_notify_scene_payload(
        server._scene,
        server._scene_manager,
        viewer_settings=_viewer_settings(server),
    )
    timestamp = time.time()
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None:
        snapshot = store.snapshot_envelope(
            NOTIFY_SCENE_TYPE,
            payload=payload.to_dict(),
            timestamp=timestamp,
        )
        store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=timestamp)
        store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=timestamp)
        store.reset_epoch(NOTIFY_SCENE_LEVEL_TYPE, timestamp=timestamp)
    tasks: list[Awaitable[None]] = []
    for ws in clients:
        session_id = _state_session(ws)
        if not session_id:
            continue
        if snapshot is not None:
            _state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
            _state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()
            _state_sequencer(ws, NOTIFY_SCENE_LEVEL_TYPE).clear()
        frame = build_notify_scene_snapshot(
            session_id=session_id,
            viewer=payload.viewer,
            layers=payload.layers,
            policies=payload.policies,
            metadata=payload.metadata,
            timestamp=snapshot.timestamp if snapshot is not None else timestamp,
            delta_token=snapshot.delta_token if snapshot is not None else None,
            frame_id=snapshot.frame_id if snapshot is not None else None,
        )
        tasks.append(_send_frame(server, ws, frame))
    if not tasks:
        return
    await asyncio.gather(*tasks, return_exceptions=True)
    if server._log_dims_info:
        logger.info("%s: notify.scene broadcast to %d clients", reason, len(tasks))
    else:
        logger.debug("%s: notify.scene broadcast to %d clients", reason, len(tasks))

    await broadcast_scene_level(
        server,
        payload=build_notify_scene_level_payload(server._scene, server._scene_manager),
        timestamp=timestamp,
        reason=reason,
    )


async def _send_layer_baseline(server: Any, ws: Any) -> None:
    """Send canonical layer controls for all known layers to *ws*."""

    scene = server._scene
    manager = server._scene_manager
    snapshot = manager.scene_snapshot()
    if snapshot is None or not snapshot.layers:
        return

    latest_updates = scene.latest_state.layer_updates or {}

    for layer in snapshot.layers:
        layer_id = layer.layer_id
        if not layer_id:
            continue

        controls: dict[str, Any] = {}
        control_state = scene.layer_controls.get(layer_id)
        if control_state is not None:
            controls.update(layer_controls_to_dict(control_state))

        pending = latest_updates.get(layer_id, {})
        if pending:
            for key, value in pending.items():
                controls[str(key)] = value

        if not controls:
            continue

        await _broadcast_layers_delta(
            server,
            layer_id=layer_id,
            changes=controls,
            intent_id=None,
            timestamp=time.time(),
            targets=[ws],
        )
