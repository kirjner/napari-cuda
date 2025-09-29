"""State-channel orchestration helpers for `EGLHeadlessServer`."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from collections.abc import Awaitable, Callable
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from websockets.exceptions import ConnectionClosed

from numbers import Integral

from napari_cuda.protocol.messages import STATE_UPDATE_TYPE, StateUpdateMessage
from napari_cuda.server import server_state_updates as state_updates
from napari_cuda.server.server_state_updates import (
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
from napari_cuda.server.server_scene_spec import (
    build_scene_spec_json,
    build_state_update_payload,
)
from napari_cuda.server import pixel_channel
from napari_cuda.server.protocol_bridge import encode_envelope_json

logger = logging.getLogger(__name__)

StateMessageHandler = Callable[[Any, Mapping[str, Any], Any], Awaitable[bool]]


async def handle_state(server: Any, ws: Any) -> None:
    """Handle a state-channel websocket connection."""

    server._state_clients.add(ws)
    server.metrics.inc('napari_cuda_state_connects')
    try:
        server._update_client_gauges()
        _disable_nagle(ws)
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


async def _handle_legacy_dims(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    logger.debug("state: dims.set ignored (use dims.update.*)")
    return True


async def _handle_volume_render_mode(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    mode = str(data.get('mode') or '').lower()
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    if state_updates.is_valid_render_mode(mode, server._allowed_render_modes):
        state_updates.update_volume_mode(server._scene, server._state_lock, mode)
        server._log_volume_update(
            "update: volume.set_render_mode mode=%s client_id=%s seq=%s",
            mode,
            client_id,
            client_seq,
        )
        _enqueue_latest_state_for_worker(server)
        server._schedule_coro(
            rebroadcast_meta(server, client_id),
            'rebroadcast-volume-mode',
        )
    return True


async def _handle_volume_clim(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    pair = state_updates.normalize_clim(data.get('lo'), data.get('hi'))
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    if pair is not None:
        lo, hi = pair
        server._log_volume_update(
            "update: volume.set_clim lo=%.4f hi=%.4f client_id=%s seq=%s",
            lo,
            hi,
            client_id,
            client_seq,
        )
        state_updates.update_volume_clim(server._scene, server._state_lock, lo, hi)
        _enqueue_latest_state_for_worker(server)
        server._schedule_coro(
            rebroadcast_meta(server, client_id),
            'rebroadcast-volume-clim',
        )
    return True


async def _handle_volume_colormap(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    name = data.get('name')
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    if isinstance(name, str) and name.strip():
        server._log_volume_update(
            "update: volume.set_colormap name=%s client_id=%s seq=%s",
            name,
            client_id,
            client_seq,
        )
        state_updates.update_volume_colormap(server._scene, server._state_lock, str(name))
        _enqueue_latest_state_for_worker(server)
        server._schedule_coro(
            rebroadcast_meta(server, client_id),
            'rebroadcast-volume-colormap',
        )
    return True


async def _handle_volume_opacity(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    opacity = state_updates.clamp_opacity(data.get('alpha'))
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    if opacity is not None:
        server._log_volume_update(
            "update: volume.set_opacity alpha=%.3f client_id=%s seq=%s",
            opacity,
            client_id,
            client_seq,
        )
        state_updates.update_volume_opacity(server._scene, server._state_lock, float(opacity))
        _enqueue_latest_state_for_worker(server)
        server._schedule_coro(
            rebroadcast_meta(server, client_id),
            'rebroadcast-volume-opacity',
        )
    return True


async def _handle_volume_sample_step(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    sample_step = state_updates.clamp_sample_step(data.get('relative'))
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    if sample_step is not None:
        server._log_volume_update(
            "update: volume.set_sample_step relative=%.3f client_id=%s seq=%s",
            sample_step,
            client_id,
            client_seq,
        )
        state_updates.update_volume_sample_step(server._scene, server._state_lock, float(sample_step))
        server._schedule_coro(
            rebroadcast_meta(server, client_id),
            'rebroadcast-volume-sample-step',
        )
        _enqueue_latest_state_for_worker(server)
    return True


async def _handle_state_update(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        message = StateUpdateMessage.from_dict(dict(data))
    except Exception:
        logger.debug("state.update payload invalid", exc_info=True)
        return True

    scope = str(message.scope or '').strip().lower()
    key = str(message.key or '').strip().lower()
    target = str(message.target or '').strip()

    if not scope or not key or not target:
        logger.debug("state.update missing scope/key/target")
        return True

    client_id = message.client_id or None
    client_seq = message.client_seq
    interaction_id = message.interaction_id
    phase = message.phase

    if scope == 'layer':
        layer_id = target
        try:
            result = apply_layer_state_update(
                server._scene,
                server._state_lock,
                layer_id=layer_id,
                prop=key,
                value=message.value,
                client_id=client_id,
                client_seq=client_seq,
                interaction_id=interaction_id,
                phase=phase,
            )
        except KeyError:
            logger.debug("state.update unknown layer prop=%s", key)
            return True
        except Exception:
            logger.debug("state.update failed for layer=%s key=%s", layer_id, key, exc_info=True)
            return True
        if result is None:
            logger.debug(
                "state.update stale layer=%s key=%s client_id=%s seq=%s",
                layer_id,
                key,
                client_id,
                client_seq,
            )
            return True

        log_fn = logger.info if server._log_state_traces else logger.debug
        log_fn(
            "state.update layer key=%s layer_id=%s value=%s client_id=%s seq=%s server_seq=%s phase=%s",
            key,
            layer_id,
            result.value,
            client_id,
            client_seq,
            result.server_seq,
            phase,
        )
        _enqueue_latest_state_for_worker(server)
        server._schedule_coro(
            _broadcast_state_update(server, result),
            f'state-layer-{key}',
        )
        return True

    if scope == 'dims':
        step_delta: Optional[int] = None
        set_value: Optional[int] = None
        norm_key = key
        value_obj = message.value
        if key == 'index':
            norm_key = 'index'
            if isinstance(value_obj, Integral):
                set_value = int(value_obj)
                value_obj = None
            else:
                logger.debug(
                    "state.update dims ignored (non-integer index) axis=%s value=%r",
                    target,
                    value_obj,
                )
                return True
        elif key == 'step':
            norm_key = 'step'
            if isinstance(value_obj, Integral):
                step_delta = int(value_obj)
                value_obj = None
            else:
                logger.debug(
                    "state.update dims ignored (non-integer step delta) axis=%s value=%r",
                    target,
                    value_obj,
                )
                return True
        else:
            logger.debug(
                "state.update dims ignored (unsupported key) axis=%s key=%s",
                target,
                key,
            )
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
                client_id=client_id,
                client_seq=client_seq,
                interaction_id=interaction_id,
                phase=phase,
            )
        except Exception:
            logger.debug("state.update dims failed axis=%s key=%s", target, key, exc_info=True)
            return True
        if result is None:
            logger.debug(
                "state.update dims stale axis=%s key=%s client_id=%s seq=%s",
                target,
                key,
                client_id,
                client_seq,
            )
            return True

        _enqueue_latest_state_for_worker(server)
        server._schedule_coro(
            _broadcast_state_update(server, result),
            f'state-dims-{key}',
        )
        return True

    logger.debug("state.update unknown scope=%s", scope)
    return True




async def _handle_multiscale_policy(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    policy = str(data.get('policy') or '').lower()
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    allowed = {'oversampling', 'thresholds', 'ratio'}
    if policy not in allowed:
        server._log_volume_update(
            "update: multiscale.set_policy rejected policy=%s client_id=%s seq=%s",
            policy,
            client_id,
            client_seq,
        )
        return True
    server._scene.multiscale_state['policy'] = policy
    server._log_volume_update(
        "update: multiscale.set_policy policy=%s client_id=%s seq=%s",
        policy,
        client_id,
        client_seq,
    )
    if server._worker is not None:
        try:
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_policy -> worker.set_policy start"
            )
            server._worker.set_policy(policy)
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_policy -> worker.set_policy done"
            )
        except Exception:
            logger.exception("worker set_policy failed for %s", policy)
    server._schedule_coro(
        rebroadcast_meta(server, client_id),
        'rebroadcast-policy',
    )
    return True


async def _handle_multiscale_level(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    levels = server._scene.multiscale_state.get('levels') or []
    level = state_updates.clamp_level(data.get('level'), levels)
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    if level is None:
        return True
    server._scene.multiscale_state['current_level'] = int(level)
    server._log_volume_update(
        "update: multiscale.set_level level=%d client_id=%s seq=%s",
        int(level),
        client_id,
        client_seq,
    )
    if server._worker is not None:
        try:
            levels_meta = server._scene.multiscale_state.get('levels') or []
            path = None
            if isinstance(levels_meta, list) and 0 <= int(level) < len(levels_meta):
                entry = levels_meta[int(level)]
                if isinstance(entry, Mapping):
                    path = entry.get('path')
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_level -> worker.request level=%s start",
                level,
            )
            server._worker.request_multiscale_level(int(level), path)
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_level -> worker.request done"
            )
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_level -> worker.force_idr start"
            )
            server._worker.force_idr()
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_level -> worker.force_idr done"
            )
            server._pixel.bypass_until_key = True
        except Exception:
            logger.exception("multiscale level switch request failed")
    server._schedule_coro(
        rebroadcast_meta(server, client_id),
        'rebroadcast-ms-level',
    )
    return True


async def _handle_set_ndisplay(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        raw = data.get('ndisplay')
        ndisplay = int(raw) if raw is not None else 2
    except Exception:
        ndisplay = 2
    client_seq = data.get('client_seq')
    client_id = data.get('client_id') or None
    (logger.info if server._log_state_traces else logger.debug)(
        "state: set_ndisplay start target=%s",
        ndisplay,
    )
    await server._handle_set_ndisplay(ndisplay, client_id, client_seq)
    (logger.info if server._log_state_traces else logger.debug)(
        "state: set_ndisplay done target=%s",
        ndisplay,
    )
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
    'dims.set': _handle_legacy_dims,
    'set_dims': _handle_legacy_dims,
    STATE_UPDATE_TYPE: _handle_state_update,
    'volume.update.set_render_mode': _handle_volume_render_mode,
    'volume.update.set_clim': _handle_volume_clim,
    'volume.update.set_colormap': _handle_volume_colormap,
    'volume.update.set_opacity': _handle_volume_opacity,
    'volume.update.set_sample_step': _handle_volume_sample_step,
    'multiscale.update.set_policy': _handle_multiscale_policy,
    'multiscale.update.set_level': _handle_multiscale_level,
    'view.update.set_ndisplay': _handle_set_ndisplay,
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
    seq = data.get('client_seq')
    if server._log_state_traces:
        logger.info("state message start type=%s seq=%s", msg_type, seq)
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
            logger.info("state message end type=%s seq=%s handled=%s", msg_type, seq, handled)


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


async def _broadcast_state_update(
    server: Any,
    result: StateUpdateResult,
    *,
    include_control_versions: bool = True,
) -> None:
    payload = build_state_update_payload(
        server._scene,
        server._scene_manager,
        result=result,
        include_control_versions=include_control_versions,
    )
    await server._broadcast_state_json(payload)


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
    client_id: Optional[str] = None,
    client_seq: Optional[int] = None,
    interaction_id: Optional[str] = None,
    phase: Optional[str] = None,
    timestamp: Optional[float] = None,
    ack: Optional[bool] = None,
    intent_seq: Optional[int] = None,
) -> StateUpdateResult:
    meta = get_control_meta(server._scene, scope, target, key)
    server_seq = int(server_seq_override or meta.last_server_seq or 0)
    if server_seq <= 0:
        server_seq = increment_server_sequence(server._scene)
        meta.last_server_seq = server_seq
    else:
        meta.last_server_seq = server_seq
    meta.last_client_id = client_id
    meta.last_client_seq = client_seq
    meta.last_interaction_id = interaction_id
    meta.last_phase = phase
    return StateUpdateResult(
        scope=scope,
        target=target,
        key=key,
        value=value,
        server_seq=server_seq,
        client_seq=client_seq,
        client_id=client_id,
        interaction_id=interaction_id,
        phase=phase,
        timestamp=timestamp,
        ack=ack,
        intent_seq=intent_seq,
        last_client_id=meta.last_client_id,
        last_client_seq=meta.last_client_seq,
    )


def _dims_results_from_step(
    server: Any,
    step_list: Sequence[int],
    *,
    meta: Mapping[str, Any],
    client_id: Optional[str],
    ack: Optional[bool] = None,
    intent_seq: Optional[int] = None,
    timestamp: Optional[float] = None,
) -> list[StateUpdateResult]:
    results: list[StateUpdateResult] = []
    current = list(int(x) for x in step_list)
    meta_dict = dict(meta)
    ts = timestamp if timestamp is not None else time.time()
    for idx, value in enumerate(current):
        axis_label = axis_label_from_meta(meta, idx) or str(idx)
        meta_entry = get_control_meta(server._scene, "dims", axis_label, "step")
        server_seq = int(meta_entry.last_server_seq or 0)
        if server_seq <= 0:
            server_seq = increment_server_sequence(server._scene)
            meta_entry.last_server_seq = server_seq
        meta_entry.last_client_id = client_id
        meta_entry.last_client_seq = None
        meta_entry.last_interaction_id = None
        meta_entry.last_phase = None
        results.append(
            StateUpdateResult(
                scope="dims",
                target=axis_label,
                key="step",
                value=int(value),
                server_seq=server_seq,
                client_seq=None,
                client_id=client_id,
                interaction_id=None,
                phase=None,
                timestamp=ts,
                axis_index=idx,
                current_step=list(current),
                meta=meta_dict,
                ack=ack,
                intent_seq=intent_seq,
                last_client_id=meta_entry.last_client_id,
                last_client_seq=meta_entry.last_client_seq,
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

async def rebroadcast_meta(server: Any, client_id: Optional[str]) -> None:
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
                server._scene.latest_state = ServerSceneState(
                    center=s.center,
                    zoom=s.zoom,
                    angles=s.angles,
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
            client_id=client_id,
            timestamp=time.time(),
        )
        await _broadcast_state_updates(server, dims_results)
    except Exception as e:
        logger.debug("rebroadcast meta failed: %s", e)



async def broadcast_dims_update(
    server: Any,
    step_list: Sequence[int] | list[int],
    *,
    last_client_id: Optional[str],
    ack: bool,
    intent_seq: Optional[int],
    server_seq: Optional[int] = None,
    source_client_id: Optional[str] = None,
    source_client_seq: Optional[int] = None,
    interaction_id: Optional[str] = None,
    phase: Optional[str] = None,
    control_prop: Optional[str] = None,
    control_axis: Optional[str] = None,
) -> None:
    """Broadcast dims state.update messages through the state channel."""

    meta = server._dims_metadata() or {}
    results = _dims_results_from_step(
        server,
        list(int(x) for x in step_list),
        meta=meta,
        client_id=last_client_id,
        ack=ack,
        intent_seq=intent_seq,
        timestamp=time.time(),
    )

    await _broadcast_state_updates(server, results)


async def broadcast_layer_update(
    server: Any,
    *,
    layer_id: str,
    changes: Mapping[str, Any],
    intent_seq: Optional[int],
    server_seq: Optional[int] = None,
    source_client_id: Optional[str] = None,
    source_client_seq: Optional[int] = None,
    interaction_id: Optional[str] = None,
    phase: Optional[str] = None,
    control_versions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Broadcast state.update payloads for the given *layer_id*."""

    if not changes:
        return

    ts = time.time()
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
                client_id=source_client_id,
                client_seq=source_client_seq,
                interaction_id=interaction_id,
                phase=phase,
                timestamp=ts,
                intent_seq=intent_seq,
            )
        )

    await _broadcast_state_updates(server, results)


async def broadcast_scene_spec(server: Any, *, reason: str) -> None:
    payload = build_scene_spec_json(server._scene, server._scene_manager)
    await broadcast_scene_spec_payload(server, payload, reason=reason)


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
                        last_client_id=note.last_client_id,
                        ack=note.ack,
                        intent_seq=note.intent_seq,
                    )
                )
                continue

            server._schedule_coro(
                broadcast_dims_update(
                    server,
                    list(step_tuple),
                    last_client_id=note.last_client_id,
                    ack=note.ack,
                    intent_seq=note.intent_seq,
                ),
                "dims_update-worker",
            )
        elif note.kind == "meta_refresh":
            server._update_scene_manager()
            server._schedule_coro(
                rebroadcast_meta(server, note.last_client_id),
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


def _dual_emit_enabled(server: Any) -> bool:
    return bool(getattr(server, "_protocol_dual_emit", False))


async def _send_payload(server: Any, ws: Any, payload: dict[str, Any]) -> None:
    text = json.dumps(payload)
    await server._safe_state_send(ws, text)
    if _dual_emit_enabled(server):
        envelope = encode_envelope_json(payload)
        if envelope:
            await server._safe_state_send(ws, envelope)


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
        if current_step is not None:
            step_list = list(current_step)
        else:
            meta = server._dims_metadata() or {}
            nd = int(meta.get('ndim') or 3)
            step_list = [0 for _ in range(max(1, nd))]
        meta = server._dims_metadata() or {}
        dim_results = _dims_results_from_step(
            server,
            step_list,
            meta=meta,
            client_id=None,
            ack=False,
            timestamp=time.time(),
        )
        for result in dim_results:
            payload = build_state_update_payload(
                server._scene,
                server._scene_manager,
                result=result,
            )
            await _send_payload(server, ws, payload)
        if server._log_dims_info:
            logger.info("connect: state.update dims -> step=%s", step_list)
        else:
            logger.debug("connect: state.update dims -> step=%s", step_list)
    except Exception:
        logger.exception("Initial dims baseline send failed")

    try:
        await send_scene_spec(server, ws, reason="connect")
    except Exception:
        logger.exception("Initial scene.spec send failed")

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
            msg = pixel_channel.build_video_config_payload(cfg, avcc)
            await _send_payload(server, ws, msg)
        else:
            pixel_channel.mark_config_dirty(channel)
    except Exception:
        logger.exception("Initial state config send failed")


async def send_scene_spec(server: Any, ws: Any, *, reason: str) -> None:
    payload = build_scene_spec_json(server._scene, server._scene_manager)
    await server._safe_state_send(ws, payload)
    if _dual_emit_enabled(server):
        try:
            parsed = json.loads(payload)
        except Exception:
            logger.debug("scene.spec dual emit payload decode failed", exc_info=True)
        else:
            envelope = encode_envelope_json(parsed) if isinstance(parsed, dict) else None
            if envelope:
                await server._safe_state_send(ws, envelope)
    if server._log_dims_info:
        logger.info("%s: scene.spec sent", reason)
    else:
        logger.debug("%s: scene.spec sent", reason)


async def broadcast_scene_spec_payload(server: Any, payload: str, *, reason: str) -> None:
    if not payload or not server._state_clients:
        return
    envelope: Optional[str] = None
    if _dual_emit_enabled(server):
        try:
            parsed = json.loads(payload)
        except Exception:
            logger.debug("scene.spec dual emit payload decode failed", exc_info=True)
        else:
            if isinstance(parsed, dict):
                envelope = encode_envelope_json(parsed)
    coros: list[Awaitable[None]] = []
    clients = list(server._state_clients)
    for client in clients:
        coros.append(server._safe_state_send(client, payload))
        if envelope:
            coros.append(server._safe_state_send(client, envelope))
    await asyncio.gather(*coros, return_exceptions=True)
    if server._log_dims_info:
        logger.info("%s: scene.spec broadcast to %d clients", reason, len(coros))
    else:
        logger.debug("%s: scene.spec broadcast to %d clients", reason)


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

        results = _layer_results_from_controls(server, layer_id, controls)
        for result in results:
            payload = build_state_update_payload(
                server._scene,
                server._scene_manager,
                result=result,
            )
            await _send_payload(server, ws, payload)
