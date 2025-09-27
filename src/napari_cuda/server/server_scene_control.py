"""State-channel orchestration helpers for `EGLHeadlessServer`."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any, Mapping, Optional, Sequence

from websockets.exceptions import ConnectionClosed

from napari_cuda.server import server_scene_intents as intents
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.server_scene_queue import (
    ServerSceneCommand,
    WorkerSceneNotification,
)
from napari_cuda.server.server_scene_spec import (
    build_dims_payload,
    build_layer_update_payload,
    build_scene_spec_json,
)
from napari_cuda.server import pixel_channel

logger = logging.getLogger(__name__)


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


async def process_state_message(server: Any, data: dict, ws: Any) -> None:
    t = data.get('type')
    seq = data.get('client_seq')
    if server._log_state_traces:
        logger.info("state message start type=%s seq=%s", t, seq)
    handled = False
    try:
        if t == 'set_camera':
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
            handled = True
            return

        if t == 'dims.set' or t == 'set_dims':
            logger.debug("state: dims.set ignored (use dims.intent.*)")
            handled = True
            return

        if t == 'dims.intent.step':
            axis = data.get('axis')
            delta = int(data.get('delta') or 0)
            client_seq = data.get('client_seq')
            client_id = data.get('client_id') or None
            if server._log_dims_info:
                logger.info("intent: step axis=%r delta=%d client_id=%s seq=%s", axis, delta, client_id, client_seq)
            else:
                logger.debug("intent: step axis=%r delta=%d client_id=%s seq=%s", axis, delta, client_id, client_seq)
            try:
                meta = server._dims_metadata() or {}
                new_step = intents.apply_dims_intent(
                    server._scene,
                    server._state_lock,
                    meta,
                    axis=axis,
                    step_delta=delta,
                    set_value=None,
                )
                if new_step is not None:
                    try:
                        intent_i = int(client_seq) if client_seq is not None else None
                    except Exception:
                        intent_i = None
                    server._schedule_coro(
                        broadcast_dims_update(server, new_step, last_client_id=client_id, ack=True, intent_seq=intent_i),
                        'dims_update-step',
                    )
            except Exception as e:
                logger.debug("dims.intent.step handling failed: %s", e)
            handled = True
            return

        if t == 'dims.intent.set_index':
            axis = data.get('axis')
            try:
                value = int(data.get('value'))
            except Exception:
                value = 0
            client_seq = data.get('client_seq')
            client_id = data.get('client_id') or None
            if server._log_dims_info:
                logger.info("intent: set_index axis=%r value=%d client_id=%s seq=%s", axis, value, client_id, client_seq)
            else:
                logger.debug("intent: set_index axis=%r value=%d client_id=%s seq=%s", axis, value, client_id, client_seq)
            try:
                meta = server._dims_metadata() or {}
                new_step = intents.apply_dims_intent(
                    server._scene,
                    server._state_lock,
                    meta,
                    axis=axis,
                    step_delta=None,
                    set_value=value,
                )
                if new_step is not None:
                    try:
                        intent_i = int(client_seq) if client_seq is not None else None
                    except Exception:
                        intent_i = None
                    server._schedule_coro(
                        broadcast_dims_update(server, new_step, last_client_id=client_id, ack=True, intent_seq=intent_i),
                        'dims_update-set_index',
                    )
            except Exception as e:
                logger.debug("dims.intent.set_index handling failed: %s", e)
            handled = True
            return

        if t == 'volume.intent.set_render_mode':
            mode = str(data.get('mode') or '').lower()
            client_seq = data.get('client_seq')
            client_id = data.get('client_id') or None
            if intents.is_valid_render_mode(mode, server._allowed_render_modes):
                intents.update_volume_mode(server._scene, server._state_lock, mode)
                server._log_volume_intent("intent: volume.set_render_mode mode=%s client_id=%s seq=%s", mode, client_id, client_seq)
                server._schedule_coro(
                    rebroadcast_meta(server, client_id),
                    'rebroadcast-volume-mode',
                )
            handled = True
            return

        if t == 'volume.intent.set_clim':
            pair = intents.normalize_clim(data.get('lo'), data.get('hi'))
            client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
            if pair is not None:
                lo, hi = pair
                server._log_volume_intent("intent: volume.set_clim lo=%.4f hi=%.4f client_id=%s seq=%s", lo, hi, client_id, client_seq)
                intents.update_volume_clim(server._scene, server._state_lock, lo, hi)
                server._schedule_coro(
                    rebroadcast_meta(server, client_id),
                    'rebroadcast-volume-clim',
                )
            handled = True
            return

        if t == 'volume.intent.set_colormap':
            name = data.get('name')
            client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
            if isinstance(name, str) and name.strip():
                server._log_volume_intent("intent: volume.set_colormap name=%s client_id=%s seq=%s", name, client_id, client_seq)
                intents.update_volume_colormap(server._scene, server._state_lock, str(name))
                server._schedule_coro(
                    rebroadcast_meta(server, client_id),
                    'rebroadcast-volume-colormap',
                )
            handled = True
            return

        if t == 'volume.intent.set_opacity':
            a = intents.clamp_opacity(data.get('alpha'))
            client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
            if a is not None:
                server._log_volume_intent("intent: volume.set_opacity alpha=%.3f client_id=%s seq=%s", a, client_id, client_seq)
                intents.update_volume_opacity(server._scene, server._state_lock, float(a))
                server._schedule_coro(
                    rebroadcast_meta(server, client_id),
                    'rebroadcast-volume-opacity',
                )
            handled = True
            return

        if t == 'volume.intent.set_sample_step':
            rr = intents.clamp_sample_step(data.get('relative'))
            client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
            if rr is not None:
                server._log_volume_intent("intent: volume.set_sample_step relative=%.3f client_id=%s seq=%s", rr, client_id, client_seq)
                intents.update_volume_sample_step(server._scene, server._state_lock, float(rr))
                server._schedule_coro(
                    rebroadcast_meta(server, client_id),
                    'rebroadcast-volume-sample-step',
                )
            handled = True
            return

        if isinstance(t, str) and (t.startswith('image.intent.') or t.startswith('layer.intent.')):
            prefix = 'image.intent.' if t.startswith('image.intent.') else 'layer.intent.'
            prop_token = t[len(prefix):]
            if prop_token.startswith('set_'):
                prop = prop_token[4:]
            else:
                prop = prop_token
            prop = prop.strip().lower()
            layer_id = str(data.get('layer_id') or 'layer-0')
            client_id = data.get('client_id') or None
            client_seq = data.get('client_seq')
            value = data.get('value')
            if value is None and prop in data:
                value = data[prop]
            if value is None:
                if prop == 'opacity':
                    value = data.get('alpha')
                elif prop == 'gamma':
                    value = data.get('gamma')
                elif prop == 'contrast_limits':
                    value = data.get('contrast_limits') or data.get('limits')
                elif prop == 'colormap':
                    value = data.get('name')
                elif prop == 'visible':
                    value = data.get('visible')
            if value is None:
                logger.debug("layer intent missing value for prop=%s", prop)
                handled = True
                return

            try:
                applied = intents.apply_layer_intent(
                    server._scene,
                    server._state_lock,
                    layer_id=layer_id,
                    prop=prop,
                    value=value,
                )
            except Exception as exc:
                logger.debug("layer intent failed prop=%s layer=%s error=%s", prop, layer_id, exc)
                handled = True
                return

            log_fn = logger.info if server._log_state_traces else logger.debug
            log_fn(
                "intent: layer.%s layer_id=%s value=%s client_id=%s seq=%s",
                prop,
                layer_id,
                applied.get(prop),
                client_id,
                client_seq,
            )

            try:
                intent_i = int(client_seq) if client_seq is not None else None
            except Exception:
                intent_i = None

            server._schedule_coro(
                broadcast_layer_update(
                    server,
                    layer_id=layer_id,
                    changes=applied,
                    intent_seq=intent_i,
                ),
                f'layer_update-{prop}',
            )
            handled = True
            return

        if t == 'multiscale.intent.set_policy':
            pol = str(data.get('policy') or '').lower()
            client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
            allowed = {'oversampling', 'thresholds', 'ratio'}
            if pol not in allowed:
                server._log_volume_intent(
                    "intent: multiscale.set_policy rejected policy=%s client_id=%s seq=%s",
                    pol,
                    client_id,
                    client_seq,
                )
                handled = True
                return
            server._scene.multiscale_state['policy'] = pol
            server._log_volume_intent(
                "intent: multiscale.set_policy policy=%s client_id=%s seq=%s",
                pol, client_id, client_seq,
            )
            if server._worker is not None:
                try:
                    (logger.info if server._log_state_traces else logger.debug)(
                        "state: set_policy -> worker.set_policy start"
                    )
                    server._worker.set_policy(pol)
                    (logger.info if server._log_state_traces else logger.debug)(
                        "state: set_policy -> worker.set_policy done"
                    )
                except Exception:
                    logger.exception("worker set_policy failed for %s", pol)
            server._schedule_coro(
                rebroadcast_meta(server, client_id),
                'rebroadcast-policy',
            )
            handled = True
            return

        if t == 'multiscale.intent.set_level':
            levels = server._scene.multiscale_state.get('levels') or []
            lvl = intents.clamp_level(data.get('level'), levels)
            client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
            if lvl is None:
                handled = True
                return
            server._scene.multiscale_state['current_level'] = int(lvl)
            # Keep ms_state policy unchanged
            server._log_volume_intent("intent: multiscale.set_level level=%d client_id=%s seq=%s", int(lvl), client_id, client_seq)
            if server._worker is not None:
                try:
                    levels = server._scene.multiscale_state.get('levels') or []
                    path = None
                    if isinstance(levels, list) and 0 <= int(lvl) < len(levels):
                        path = levels[int(lvl)].get('path')
                    (logger.info if server._log_state_traces else logger.debug)(
                        "state: set_level -> worker.request level=%s start", lvl
                    )
                    server._worker.request_multiscale_level(int(lvl), path)
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
            handled = True
            return

        if t == 'view.intent.set_ndisplay':
            try:
                ndisp_raw = data.get('ndisplay')
                ndisp = int(ndisp_raw) if ndisp_raw is not None else 2
            except Exception:
                ndisp = 2
            client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_ndisplay start target=%s", ndisp
            )
            await server._handle_set_ndisplay(ndisp, client_id, client_seq)
            (logger.info if server._log_state_traces else logger.debug)(
                "state: set_ndisplay done target=%s", ndisp
            )
            handled = True
            return

        if t == 'camera.zoom_at':
            try:
                factor = float(data.get('factor') or 0.0)
            except Exception:
                factor = 0.0
            anchor = data.get('anchor_px')
            anc_t = tuple(anchor) if isinstance(anchor, (list, tuple)) and len(anchor) >= 2 else None
            if factor > 0.0 and anc_t is not None:
                server.metrics.inc('napari_cuda_state_camera_intents')
                if server._log_cam_info:
                    logger.info("state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)", factor, float(anc_t[0]), float(anc_t[1]))
                elif server._log_cam_debug:
                    logger.debug("state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)", factor, float(anc_t[0]), float(anc_t[1]))
                server._enqueue_camera_command(
                    ServerSceneCommand(
                        kind='zoom',
                        factor=float(factor),
                        anchor_px=(float(anc_t[0]), float(anc_t[1])),
                    )
                )
            handled = True
            return

        if t == 'camera.pan_px':
            try:
                dx = float(data.get('dx_px') or 0.0)
                dy = float(data.get('dy_px') or 0.0)
            except Exception:
                dx = 0.0; dy = 0.0
            if dx != 0.0 or dy != 0.0:
                if server._log_cam_info:
                    logger.info("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
                elif server._log_cam_debug:
                    logger.debug("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
                server.metrics.inc('napari_cuda_state_camera_intents')
                server._enqueue_camera_command(
                    ServerSceneCommand(kind='pan', dx_px=float(dx), dy_px=float(dy))
                )
            handled = True
            return

        if t == 'camera.orbit':
            try:
                daz = float(data.get('d_az_deg') or 0.0)
                delv = float(data.get('d_el_deg') or 0.0)
            except Exception:
                daz = 0.0; delv = 0.0
            if daz != 0.0 or delv != 0.0:
                if server._log_cam_info:
                    logger.info("state: camera.orbit daz=%.2f del=%.2f", daz, delv)
                elif server._log_cam_debug:
                    logger.debug("state: camera.orbit daz=%.2f del=%.2f", daz, delv)
                server.metrics.inc('napari_cuda_state_camera_intents')
                server._enqueue_camera_command(
                    ServerSceneCommand(kind='orbit', d_az_deg=float(daz), d_el_deg=float(delv))
                )
                server.metrics.inc('napari_cuda_orbit_events')
            handled = True
            return

        if t == 'camera.reset':
            if server._log_cam_info:
                logger.info("state: camera.reset")
            elif server._log_cam_debug:
                logger.debug("state: camera.reset")
            server.metrics.inc('napari_cuda_state_camera_intents')
            server._enqueue_camera_command(ServerSceneCommand(kind='reset'))
            if server._idr_on_reset and server._worker is not None:
                logger.info("state: camera.reset -> ensure_keyframe start")
                await server._ensure_keyframe()
                logger.info("state: camera.reset -> ensure_keyframe done")
            handled = True
            return

        if t == 'ping':
            await ws.send(json.dumps({'type': 'pong'}))
            handled = True
            return

        if t in ('request_keyframe', 'force_idr'):
            await server._ensure_keyframe()
            handled = True
            return

        if server._log_state_traces:
            logger.info("state message ignored type=%s", t)
    finally:
        if server._log_state_traces:
            logger.info("state message end type=%s seq=%s handled=%s", t, seq, handled)

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

        await broadcast_dims_update(server, chosen, last_client_id=client_id, ack=True, intent_seq=None)
    except Exception as e:
        logger.debug("rebroadcast meta failed: %s", e)



async def broadcast_dims_update(
    server: Any,
    step_list: Sequence[int] | list[int],
    *,
    last_client_id: Optional[str],
    ack: bool,
    intent_seq: Optional[int],
) -> None:
    """Broadcast a dims update through the state channel."""

    meta = server._dims_metadata() or {}
    ndim = _meta_axis_count(meta)
    step = [int(x) for x in step_list]
    assert len(step) <= ndim, (
        f"dims_update step length {len(step)} exceeds metadata ndim {ndim}"
    )

    payload = build_dims_update_message(
        server,
        step_list=step,
        last_client_id=last_client_id,
        ack=ack,
        intent_seq=intent_seq,
    )
    await server._broadcast_state_json(payload)


async def broadcast_layer_update(
    server: Any,
    *,
    layer_id: str,
    changes: Mapping[str, Any],
    intent_seq: Optional[int],
) -> None:
    """Broadcast a layer.update payload for the given *layer_id*."""

    if not changes:
        return
    payload = build_layer_update_payload(
        server._scene,
        server._scene_manager,
        layer_id=layer_id,
        changes=changes,
        intent_seq=intent_seq,
    )
    await server._broadcast_state_json(payload)


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
            ndim = _meta_axis_count(meta)
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


async def _send_state_baseline(server: Any, ws: Any) -> None:
    try:
        await server._await_adapter_level_ready(0.5)
        with server._state_lock:
            current_step = server._scene.latest_state.current_step
        if current_step is not None:
            obj = build_dims_update_message(
                server,
                step_list=list(current_step),
                last_client_id=None,
                ack=False,
                intent_seq=None,
            )
        else:
            meta = server._dims_metadata() or {}
            nd = int(meta.get('ndim') or 3)
            step_list = [0 for _ in range(max(1, nd))]
            obj = build_dims_update_message(
                server,
                step_list=step_list,
                last_client_id=None,
                ack=False,
                intent_seq=None,
            )
        await ws.send(json.dumps(obj))
        if server._log_dims_info:
            logger.info("connect: dims.update -> current_step=%s", obj.get('current_step'))
        else:
            logger.debug("connect: dims.update -> current_step=%s", obj.get('current_step'))
    except Exception:
        logger.exception("Initial dims baseline send failed")

    try:
        await send_scene_spec(server, ws, reason="connect")
    except Exception:
        logger.exception("Initial scene.spec send failed")

    try:
        channel = getattr(server, "_pixel_channel", None)
        cfg = getattr(server, "_pixel_config", None)
        assert channel is not None and cfg is not None, "Pixel channel not initialized"
        avcc = channel.last_avcc
        if avcc is not None:
            msg = pixel_channel.build_video_config_payload(cfg, avcc)
            await ws.send(json.dumps(msg))
        else:
            pixel_channel.mark_config_dirty(channel)
    except Exception:
        logger.exception("Initial state config send failed")


async def send_scene_spec(server: Any, ws: Any, *, reason: str) -> None:
    payload = build_scene_spec_json(server._scene, server._scene_manager)
    await server._safe_state_send(ws, payload)
    if server._log_dims_info:
        logger.info("%s: scene.spec sent", reason)
    else:
        logger.debug("%s: scene.spec sent", reason)


async def broadcast_scene_spec_payload(server: Any, payload: str, *, reason: str) -> None:
    if not payload or not server._state_clients:
        return
    coros = [server._safe_state_send(client, payload) for client in list(server._state_clients)]
    await asyncio.gather(*coros, return_exceptions=True)
    if server._log_dims_info:
        logger.info("%s: scene.spec broadcast to %d clients", reason, len(coros))
    else:
        logger.debug("%s: scene.spec broadcast to %d clients", reason)


def build_dims_update_message(
    server: Any,
    *,
    step_list: Sequence[int],
    last_client_id: Optional[str],
    ack: bool,
    intent_seq: Optional[int],
) -> dict:
    """Create a dims.update payload for the current server context."""

    meta = server._dims_metadata() or {}
    worker = getattr(server, '_worker', None)
    src = getattr(worker, '_scene_source', None) if worker is not None else None
    use_volume = bool(getattr(worker, 'use_volume', getattr(server, 'use_volume', False)) if worker is not None else getattr(server, 'use_volume', False))
    return build_dims_payload(
        server._scene,
        step_list=step_list,
        last_client_id=last_client_id,
        meta=meta,
        worker_scene_source=src,
        use_volume=use_volume,
        ack=ack,
        intent_seq=intent_seq,
    )


def _meta_axis_count(meta: Mapping[str, Any]) -> int:
    ndim = meta.get("ndim")
    if isinstance(ndim, int) and ndim > 0:
        return ndim
    axes = meta.get("axes")
    if isinstance(axes, Sequence):
        return len(axes)
    sizes = meta.get("sizes")
    if isinstance(sizes, Sequence):
        return len(sizes)
    order = meta.get("order")
    if isinstance(order, Sequence):
        return len(order)
    ranges = meta.get("range")
    if isinstance(ranges, Sequence):
        return len(ranges)
    return max(1, len(meta.get("current_step", [])))
