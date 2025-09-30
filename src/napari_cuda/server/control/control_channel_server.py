"""State-channel orchestration helpers for `EGLHeadlessServer`."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from websockets.exceptions import ConnectionClosed

from numbers import Integral

from napari_cuda.protocol import (
    EnvelopeParser,
    FeatureToggle,
    FeatureResumeState,
    PROTO_VERSION,
    NOTIFY_CAMERA_TYPE,
    NOTIFY_DIMS_TYPE,
    NOTIFY_ERROR_TYPE,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    NOTIFY_TELEMETRY_TYPE,
    SESSION_HELLO_TYPE,
    SessionHello,
    SessionReject,
    SessionWelcome,
    ResumableTopicSequencer,
    build_ack_state,
    build_notify_camera,
    build_notify_dims,
    build_notify_error,
    build_notify_layers_delta,
    build_notify_scene_snapshot,
    build_notify_stream,
    build_notify_telemetry,
    build_reply_command,
    build_session_ack,
    build_session_goodbye,
    build_session_heartbeat,
    build_session_reject,
    build_session_welcome,
)
from napari_cuda.protocol import NotifyStreamPayload
from napari_cuda.protocol.messages import STATE_UPDATE_TYPE
from napari_cuda.server.control import state_update_engine as state_updates
from napari_cuda.server.control.state_update_engine import (
    StateUpdateResult,
    apply_dims_state_update,
    apply_layer_state_update,
    axis_label_from_meta,
)
from napari_cuda.server.render_mailbox import RenderDelta
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.server_scene import (
    ServerSceneCommand,
    get_control_meta,
    increment_server_sequence,
    layer_controls_to_dict,
)
from napari_cuda.server.worker_notifications import WorkerSceneNotification
from napari_cuda.server.control.scene_snapshot_builder import (
    build_notify_dims_from_result,
    build_notify_dims_payload,
    build_notify_layers_delta_payload,
    build_notify_layers_payload,
    build_notify_scene_payload,
)
from napari_cuda.server.pixel import pixel_channel_server as pixel_channel

logger = logging.getLogger(__name__)

_HANDSHAKE_TIMEOUT_S = 5.0
_REQUIRED_NOTIFY_FEATURES = ("notify.scene", "notify.layers", "notify.stream")
_SERVER_FEATURES: dict[str, FeatureToggle] = {
    "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
    "notify.camera": FeatureToggle(enabled=True, version=1, resume=False),
    "notify.telemetry": FeatureToggle(enabled=False),
    "call.command": FeatureToggle(enabled=False),
}
_ENVELOPE_PARSER = EnvelopeParser()

StateMessageHandler = Callable[[Any, Mapping[str, Any], Any], Awaitable[bool]]


def _state_session(ws: Any) -> Optional[str]:
    return getattr(ws, "_napari_cuda_session", None)


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
    try:
        spec = server._scene_manager.scene_spec()
        if spec is None or not spec.layers:
            return None
        first = spec.layers[0]
        layer_id = getattr(first, "layer_id", None)
        if layer_id:
            return str(layer_id)
        layer_dict = first.to_dict() if hasattr(first, "to_dict") else None  # type: ignore[attr-defined]
        if isinstance(layer_dict, Mapping):
            candidate = layer_dict.get("layer_id")
            if candidate:
                return str(candidate)
    except Exception:
        logger.debug("default layer id resolution failed", exc_info=True)
    return None


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


def _resolve_ndisplay(server: Any) -> int:
    try:
        return 3 if bool(server._scene.use_volume) else 2
    except Exception:
        return 2


def _resolve_dims_mode(ndisplay: int) -> str:
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

    payload = build_notify_layers_payload(layer_id=layer_id or "layer-0", changes=changes)
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not _feature_enabled(ws, "notify.layers"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        kwargs: Dict[str, Any] = {
            "session_id": session_id,
            "payload": payload,
            "timestamp": now,
            "intent_id": intent_id,
        }
        sequencer = _state_sequencer(ws, NOTIFY_LAYERS_TYPE)
        if sequencer.seq is None:
            cursor = sequencer.snapshot()
            kwargs["seq"] = cursor.seq
            kwargs["delta_token"] = cursor.delta_token
        else:
            kwargs["sequencer"] = sequencer
        frame = build_notify_layers_delta(**kwargs)
        tasks.append(_send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _broadcast_dims_state(
    server: Any,
    *,
    current_step: Sequence[int],
    source: str,
    intent_id: Optional[str],
    timestamp: Optional[float],
    targets: Optional[Sequence[Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    ndisplay = _resolve_ndisplay(server)
    payload = build_notify_dims_payload(
        current_step=current_step,
        ndisplay=ndisplay,
        mode=_resolve_dims_mode(ndisplay),
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


async def _send_stream_frame(
    server: Any,
    ws: Any,
    *,
    payload: NotifyStreamPayload | Mapping[str, Any],
    timestamp: Optional[float],
    snapshot_cursor: FeatureResumeState | None = None,
) -> None:
    session_id = _state_session(ws)
    if not session_id:
        return
    if not isinstance(payload, NotifyStreamPayload):
        payload = NotifyStreamPayload.from_dict(payload)
    kwargs: Dict[str, Any] = {
        "session_id": session_id,
        "payload": payload,
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }
    sequencer = _state_sequencer(ws, NOTIFY_STREAM_TYPE)
    if snapshot_cursor is not None:
        sequencer.snapshot(token=snapshot_cursor.delta_token)
        kwargs["seq"] = snapshot_cursor.seq
        kwargs["delta_token"] = snapshot_cursor.delta_token
    elif sequencer.seq is None:
        cursor = sequencer.snapshot()
        kwargs["seq"] = cursor.seq
        kwargs["delta_token"] = cursor.delta_token
    else:
        kwargs["sequencer"] = sequencer
    frame = build_notify_stream(**kwargs)
    await _send_frame(server, ws, frame)


async def broadcast_stream_config(
    server: Any,
    *,
    payload: NotifyStreamPayload,
    timestamp: Optional[float] = None,
) -> None:
    clients = list(server._state_clients)
    if not clients:
        return

    now = time.time() if timestamp is None else float(timestamp)
    tasks: list[Awaitable[None]] = []

    for ws in clients:
        if not _feature_enabled(ws, "notify.stream"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        pending_resume = getattr(ws, "_napari_cuda_pending_resume", None)
        snapshot_cursor = None
        if pending_resume:
            snapshot_cursor = pending_resume.pop(NOTIFY_STREAM_TYPE, None)
        tasks.append(
            _send_stream_frame(
                server,
                ws,
                payload=payload,
                timestamp=now,
                snapshot_cursor=snapshot_cursor,
            )
        )
        if pending_resume is not None:
            setattr(ws, "_napari_cuda_pending_resume", pending_resume)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def handle_state(server: Any, ws: Any) -> None:
    """Handle a state-channel websocket connection."""

    try:
        _disable_nagle(ws)
        handshake_ok = await _perform_state_handshake(server, ws)
        if not handshake_ok:
            return
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
        try:
            await ws.close()
        except Exception as exc:
            logger.debug("State WS close error: %s", exc)
        server._state_clients.discard(ws)
        server._update_client_gauges()



async def _handle_set_camera(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    center = data.get('center')
    zoom = data.get('zoom')
    angles = data.get('angles')
    if server._log_cam_info:
        logger.info("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
    elif server._log_cam_debug:
        logger.debug("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
    with server._state_lock:
        server._scene.latest_state = ServerSceneState(
            center=tuple(center) if center else None,
            zoom=float(zoom) if zoom is not None else None,
            angles=tuple(angles) if angles else None,
            current_step=server._scene.latest_state.current_step,
        )
    _enqueue_latest_state_for_worker(server)
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

        result = _baseline_state_result(
            server,
            scope='view',
            target=target,
            key='ndisplay',
            value=int(ndisplay),
            intent_id=intent_id,
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

        logger.debug(
            "state.update view intent=%s frame=%s ndisplay=%s server_seq=%s",
            intent_id,
            frame_id,
            result.value,
            result.server_seq,
        )

        server._schedule_coro(
            _broadcast_state_update(server, result),
            'state-view-ndisplay',
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
        server._schedule_coro(
            rebroadcast_meta(server),
            rebroadcast_tag,
        )
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

        server._schedule_coro(
            rebroadcast_meta(server),
            rebroadcast_tag,
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

        meta = server._dims_metadata() or {}
        try:
            result = apply_dims_state_update(
                server._scene,
                server._state_lock,
                meta,
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



async def _handle_camera_zoom(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        factor = float(data.get('factor') or 0.0)
    except Exception:
        factor = 0.0
    anchor = data.get('anchor_px')
    anchor_tuple = (
        tuple(anchor) if isinstance(anchor, (list, tuple)) and len(anchor) >= 2 else None
    )
    if factor > 0.0 and anchor_tuple is not None:
        server.metrics.inc('napari_cuda_state_camera_updates')
        if server._log_cam_info:
            logger.info(
                "state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)",
                factor,
                float(anchor_tuple[0]),
                float(anchor_tuple[1]),
            )
        elif server._log_cam_debug:
            logger.debug(
                "state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)",
                factor,
                float(anchor_tuple[0]),
                float(anchor_tuple[1]),
            )
        server._enqueue_camera_command(
            ServerSceneCommand(
                kind='zoom',
                factor=float(factor),
                anchor_px=(float(anchor_tuple[0]), float(anchor_tuple[1])),
            )
        )
    return True


async def _handle_camera_pan(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        dx = float(data.get('dx_px') or 0.0)
        dy = float(data.get('dy_px') or 0.0)
    except Exception:
        dx = 0.0
        dy = 0.0
    if dx != 0.0 or dy != 0.0:
        if server._log_cam_info:
            logger.info("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
        elif server._log_cam_debug:
            logger.debug("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
        server.metrics.inc('napari_cuda_state_camera_updates')
        server._enqueue_camera_command(
            ServerSceneCommand(kind='pan', dx_px=float(dx), dy_px=float(dy))
        )
    return True


async def _handle_camera_orbit(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        d_az = float(data.get('d_az_deg') or 0.0)
        d_el = float(data.get('d_el_deg') or 0.0)
    except Exception:
        d_az = 0.0
        d_el = 0.0
    if d_az != 0.0 or d_el != 0.0:
        if server._log_cam_info:
            logger.info("state: camera.orbit daz=%.2f del=%.2f", d_az, d_el)
        elif server._log_cam_debug:
            logger.debug("state: camera.orbit daz=%.2f del=%.2f", d_az, d_el)
        server.metrics.inc('napari_cuda_state_camera_updates')
        server._enqueue_camera_command(
            ServerSceneCommand(kind='orbit', d_az_deg=float(d_az), d_el_deg=float(d_el))
        )
        server.metrics.inc('napari_cuda_orbit_events')
    return True


async def _handle_camera_reset(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    if server._log_cam_info:
        logger.info("state: camera.reset")
    elif server._log_cam_debug:
        logger.debug("state: camera.reset")
    server.metrics.inc('napari_cuda_state_camera_updates')
    server._enqueue_camera_command(ServerSceneCommand(kind='reset'))
    if server._idr_on_reset and server._worker is not None:
        logger.info("state: camera.reset -> ensure_keyframe start")
        await server._ensure_keyframe()
        logger.info("state: camera.reset -> ensure_keyframe done")
    return True


async def _handle_ping(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    await ws.send(json.dumps({'type': 'pong'}))
    return True


async def _handle_force_keyframe(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    await server._ensure_keyframe()
    return True


MESSAGE_HANDLERS: dict[str, StateMessageHandler] = {
    'set_camera': _handle_set_camera,
    STATE_UPDATE_TYPE: _handle_state_update,
    'camera.zoom_at': _handle_camera_zoom,
    'camera.pan_px': _handle_camera_pan,
    'camera.orbit': _handle_camera_orbit,
    'camera.reset': _handle_camera_reset,
    'ping': _handle_ping,
    'request_keyframe': _handle_force_keyframe,
    'force_idr': _handle_force_keyframe,
}


async def process_state_message(server: Any, data: dict, ws: Any) -> None:
    msg_type = data.get('type')
    frame_id = data.get('frame_id')
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
        step = result.current_step or ()
        await _broadcast_dims_state(
            server,
            current_step=step,
            source="state.update",
            intent_id=result.intent_id,
            timestamp=result.timestamp,
        )
        return

    if result.scope == "view" and result.key == "ndisplay":
        step = ()
        with server._state_lock:
            latest = server._scene.latest_state.current_step
            if latest is not None:
                step = tuple(int(x) for x in latest)
        await _broadcast_dims_state(
            server,
            current_step=step,
            source="state.update",
            intent_id=result.intent_id,
            timestamp=result.timestamp,
        )
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

async def rebroadcast_meta(server: Any) -> None:
    """Re-broadcast a dims.update with current_step and updated meta.

    Safe to call after mutating volume/multiscale state. Never raises.
    """
    try:
        # Gather steps from server state, worker viewer, and source
        state_step: list[int] | None = None
        with server._state_lock:
            cur = server._scene.latest_state.current_step
            state_step = list(cur) if isinstance(cur, (list, tuple)) else None

        w_step = None
        try:
            if server._worker is not None:
                vm = server._worker.viewer_model()
                if vm is not None:
                    w_step = tuple(int(x) for x in vm.dims.current_step)  # type: ignore[attr-defined]
        except Exception:
            w_step = None

        s_step = None
        try:
            src = getattr(server._worker, '_scene_source', None) if server._worker is not None else None
            if src is not None:
                s_step = tuple(int(x) for x in (src.current_step or ()))
        except Exception:
            s_step = None

        worker_volume = False
        try:
            if server._worker is not None:
                worker_volume = bool(getattr(server._worker, 'use_volume', False))
        except Exception:
            worker_volume = False

        # Choose authoritative step: favour worker viewer when volume mode is active
        chosen: list[int] = []
        source_of_truth = 'server'
        if worker_volume and w_step is not None and len(w_step) > 0:
            chosen = [int(x) for x in w_step]
            source_of_truth = 'viewer-volume'
        elif s_step is not None and len(s_step) > 0:
            chosen = [int(x) for x in s_step]
            source_of_truth = 'source'
        elif w_step is not None and len(w_step) > 0:
            chosen = [int(x) for x in w_step]
            source_of_truth = 'viewer'
        elif state_step is not None:
            chosen = [int(x) for x in state_step]
            source_of_truth = 'server'
        else:
            chosen = [0]
            source_of_truth = 'default'

        if worker_volume and chosen:
            while len(chosen) < 3:
                chosen.append(0)
        logger.debug('rebroadcast_meta: source=%s step=%s server=%s viewer=%s source_step=%s volume=%s', source_of_truth, chosen, state_step, w_step, s_step, worker_volume)

        # Synchronize server state with chosen step to keep intents consistent
        try:
            with server._state_lock:
                s = server._scene.latest_state
                server._scene.latest_state = replace(
                    s,
                    current_step=tuple(chosen),
                )
            _enqueue_latest_state_for_worker(server)
        except Exception:
            logger.debug('rebroadcast: failed to sync server state step', exc_info=True)

        # Diagnostics: compare and note source of truth
        if server._log_dims_info:
            logger.info(
                "rebroadcast: source=%s step=%s server=%s viewer=%s source_step=%s",
                source_of_truth, chosen, state_step, w_step, s_step,
            )
        else:
            logger.debug(
                "rebroadcast: source=%s step=%s server=%s viewer=%s source_step=%s",
                source_of_truth, chosen, state_step, w_step, s_step,
            )

        dims_results = _dims_results_from_step(
            server,
            chosen,
            meta=server._dims_metadata() or {},
            timestamp=time.time(),
        )
        await _broadcast_state_updates(server, dims_results)
    except Exception as e:
        logger.debug("rebroadcast meta failed: %s", e)



async def broadcast_dims_update(
    server: Any,
    step_list: Sequence[int] | list[int],
    *,
    timestamp: Optional[float] = None,
) -> None:
    """Broadcast dims state.update messages through the state channel."""

    meta = server._dims_metadata() or {}
    results = _dims_results_from_step(
        server,
        list(int(x) for x in step_list),
        meta=meta,
        timestamp=timestamp,
    )

    await _broadcast_state_updates(server, results)


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


def process_worker_notifications(
    server: Any, notifications: Sequence[WorkerSceneNotification]
) -> None:
    if not notifications:
        return

    deferred: list[WorkerSceneNotification] = []

    for note in notifications:
        if note.kind == "dims_update" and note.step is not None:
            meta = server._dims_metadata() or {}
            ndim = max(1, len(meta.get("order", [])) or len(meta.get("range", [])) or len(meta.get("axes", [])) or meta.get("ndim", 0))
            step_tuple = tuple(int(x) for x in note.step)
            if len(step_tuple) > ndim:
                deferred.append(
                    WorkerSceneNotification(
                        kind="dims_update",
                        step=step_tuple,
                    )
                )
                continue

            server._schedule_coro(
                broadcast_dims_update(
                    server,
                    list(step_tuple),
                ),
                "dims_update-worker",
            )
        elif note.kind == "meta_refresh":
            server._update_scene_manager()
            server._schedule_coro(
                rebroadcast_meta(server),
                "rebroadcast-worker",
            )

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

    pending_resume: Dict[str, FeatureResumeState] = {}
    pixel_state = getattr(server, "_pixel_channel", None)
    if pixel_state is not None and getattr(pixel_state, "last_avcc", None) is not None:
        stream_toggle = negotiated_features.get(NOTIFY_STREAM_TYPE)
        if stream_toggle is not None and stream_toggle.resume:
            stream_resume = FeatureResumeState(seq=0, delta_token=uuid.uuid4().hex)
            negotiated_features[NOTIFY_STREAM_TYPE] = replace(
                stream_toggle,
                resume_state=stream_resume,
            )
            pending_resume[NOTIFY_STREAM_TYPE] = stream_resume

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
    setattr(ws, "_napari_cuda_resume_tokens", hello.payload.resume_tokens)
    setattr(ws, "_napari_cuda_pending_resume", pending_resume)
    setattr(
        ws,
        "_napari_cuda_sequencers",
        {
            NOTIFY_SCENE_TYPE: ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE),
            NOTIFY_LAYERS_TYPE: ResumableTopicSequencer(topic=NOTIFY_LAYERS_TYPE),
            NOTIFY_STREAM_TYPE: ResumableTopicSequencer(topic=NOTIFY_STREAM_TYPE),
        },
    )

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


async def _send_state_baseline(server: Any, ws: Any) -> None:
    try:
        await server._await_adapter_level_ready(0.5)
        try:
            if hasattr(server, "_update_scene_manager"):
                server._update_scene_manager()
        except Exception:
            logger.debug("Initial scene manager sync failed", exc_info=True)

        with server._state_lock:
            current_step = server._scene.latest_state.current_step

        pending_resume: Dict[str, FeatureResumeState] = getattr(ws, "_napari_cuda_pending_resume", {})

        step_list = list(current_step) if current_step is not None else []
        spec = server._scene_manager.scene_spec()
        ndim = 0
        if spec is not None and spec.dims is not None:
            try:
                ndim = int(spec.dims.ndim or 0)
            except Exception:
                ndim = 0
            if ndim <= 0:
                try:
                    ndim = len(spec.dims.sizes or [])
                except Exception:
                    ndim = 0
        if ndim <= 0:
            ndim = len(step_list) if step_list else 3
        while len(step_list) < ndim:
            step_list.append(0)

        await _broadcast_dims_state(
            server,
            current_step=step_list,
            source="server.bootstrap",
            intent_id=None,
            timestamp=time.time(),
            targets=[ws],
        )
        if server._log_dims_info:
            logger.info("connect: notify.dims baseline -> step=%s", step_list)
        else:
            logger.debug("connect: notify.dims baseline -> step=%s", step_list)
    except Exception:
        logger.exception("Initial dims baseline send failed")

    try:
        await _send_scene_snapshot(server, ws, reason="connect")
    except Exception:
        logger.exception("Initial notify.scene send failed")

    try:
        await _send_layer_baseline(server, ws)
    except Exception:
        logger.exception("Initial layer baseline send failed")

    try:
        channel = getattr(server, "_pixel_channel", None)
        cfg = getattr(server, "_pixel_config", None)
        assert channel is not None and cfg is not None, "Pixel channel not initialized"
        avcc = channel.last_avcc
        if avcc is not None:
            stream_payload = pixel_channel.build_notify_stream_payload(cfg, avcc)
            resume_state = pending_resume.pop(NOTIFY_STREAM_TYPE, None)
            await _send_stream_frame(
                server,
                ws,
                payload=stream_payload,
                timestamp=time.time(),
                snapshot_cursor=resume_state,
            )
        else:
            pixel_channel.mark_stream_config_dirty(channel)
        setattr(ws, "_napari_cuda_pending_resume", pending_resume)
    except Exception:
        logger.exception("Initial state config send failed")


async def _send_scene_snapshot(server: Any, ws: Any, *, reason: str) -> None:
    session_id = _state_session(ws)
    if not session_id:
        logger.debug("Skipping notify.scene send without session id")
        return

    payload = build_notify_scene_payload(
        server._scene,
        server._scene_manager,
        viewer_settings=_viewer_settings(server),
    )
    frame = build_notify_scene_snapshot(
        session_id=session_id,
        viewer=payload.viewer,
        layers=payload.layers,
        policies=payload.policies,
        ancillary=payload.ancillary,
        timestamp=time.time(),
        sequencer=_state_sequencer(ws, NOTIFY_SCENE_TYPE),
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
    tasks: list[Awaitable[None]] = []
    for ws in clients:
        session_id = _state_session(ws)
        if not session_id:
            continue
        frame = build_notify_scene_snapshot(
            session_id=session_id,
            viewer=payload.viewer,
            layers=payload.layers,
            policies=payload.policies,
            ancillary=payload.ancillary,
            timestamp=timestamp,
            sequencer=_state_sequencer(ws, NOTIFY_SCENE_TYPE),
        )
        tasks.append(_send_frame(server, ws, frame))
    if not tasks:
        return
    await asyncio.gather(*tasks, return_exceptions=True)
    if server._log_dims_info:
        logger.info("%s: notify.scene broadcast to %d clients", reason, len(tasks))
    else:
        logger.debug("%s: notify.scene broadcast to %d clients", reason, len(tasks))


async def _send_layer_baseline(server: Any, ws: Any) -> None:
    """Send canonical layer controls for all known layers to *ws*."""

    scene = server._scene
    manager = server._scene_manager
    spec = manager.scene_spec()
    if spec is None or not spec.layers:
        return

    latest_updates = scene.latest_state.layer_updates or {}

    for layer in spec.layers:
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
