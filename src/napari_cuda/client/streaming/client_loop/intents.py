"""Intent helpers for the streaming client loop."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.streaming.presenter_facade import PresenterFacade

    from .loop_state import ClientLoopState

from napari_cuda.client.streaming.control_sessions import ControlSession
from napari_cuda.protocol.messages import CONTROL_COMMAND_TYPE

logger = logging.getLogger("napari_cuda.client.streaming.client_stream_loop")


def _default_dims_meta() -> dict[str, object | None]:
    return {
        'ndim': None,
        'order': None,
        'axis_labels': None,
        'range': None,
        'sizes': None,
        'ndisplay': None,
        'volume': None,
        'render': None,
        'multiscale': None,
    }


@dataclass
class IntentState:
    """Mutable intent-related state hoisted out of the loop object."""

    dims_ready: bool = False
    dims_meta: dict[str, object | None] = field(default_factory=_default_dims_meta)
    primary_axis_index: int | None = None
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    client_seq: int = 0
    dims_min_dt: float = 0.0
    last_dims_send: float = 0.0
    wheel_px_accum: float = 0.0
    wheel_step: float = 1.0
    settings_min_dt: float = 0.0
    last_settings_send: float = 0.0
    dims_z: float | None = None
    dims_z_min: float | None = None
    dims_z_max: float | None = None
    control_sessions: Dict[str, ControlSession] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_cfg: Any) -> "IntentState":
        state = cls()
        dims_rate = getattr(env_cfg, 'dims_rate_hz', 1.0) or 1.0
        state.dims_min_dt = 1.0 / max(1.0, float(dims_rate))
        state.wheel_step = float(getattr(env_cfg, 'wheel_step', 1.0) or 1.0)
        settings_rate = getattr(env_cfg, 'settings_rate_hz', 1.0) or 1.0
        state.settings_min_dt = 1.0 / max(1.0, float(settings_rate))
        state.dims_z = getattr(env_cfg, 'dims_z', None)
        state.dims_z_min = getattr(env_cfg, 'dims_z_min', None)
        state.dims_z_max = getattr(env_cfg, 'dims_z_max', None)
        if state.dims_z is not None:
            if state.dims_z_min is not None and state.dims_z < state.dims_z_min:
                state.dims_z = state.dims_z_min
            if state.dims_z_max is not None and state.dims_z > state.dims_z_max:
                state.dims_z = state.dims_z_max
        return state

    def next_client_seq(self) -> int:
        self.client_seq = (int(self.client_seq) + 1) & 0x7FFFFFFF
        return int(self.client_seq)


def on_state_connected(state: IntentState) -> None:
    state.dims_ready = False
    state.primary_axis_index = None


def on_state_disconnected(loop_state: "ClientLoopState", state: IntentState) -> None:
    state.dims_ready = False
    state.primary_axis_index = None
    loop_state.pending_intents.clear()
    loop_state.last_dims_seq = None
    loop_state.last_dims_payload = None
    state.control_sessions.clear()


def handle_dims_update(
    state: IntentState,
    loop_state: "ClientLoopState",
    data: dict[str, object],
    *,
    presenter: "PresenterFacade",
    viewer_ref,
    ui_call,
    notify_first_dims_ready: Callable[[], None],
    log_dims_info: bool,
) -> None:
    seq_val = _int_or_none(data.get('seq'))
    if seq_val is not None:
        loop_state.last_dims_seq = seq_val

    meta = state.dims_meta
    cur = data.get('current_step')
    ndisp = data.get('ndisplay')
    ndim = data.get('ndim')
    dims_range = data.get('range')
    order = data.get('order')
    axis_labels = data.get('axis_labels')
    sizes = data.get('sizes')
    displayed = data.get('displayed')
    level = data.get('level')
    level_shape = data.get('level_shape')
    dtype = data.get('dtype')
    normalized = data.get('normalized')
    volume = data.get('volume')
    render = data.get('render')
    multiscale = data.get('multiscale')
    intent_seq = _int_or_none(data.get('intent_seq'))
    ack_val = data.get('ack') if isinstance(data.get('ack'), bool) else None

    if ndim is not None:
        meta['ndim'] = int(ndim)
    if order is not None:
        meta['order'] = order
    if axis_labels is not None:
        meta['axis_labels'] = axis_labels
    if dims_range is not None:
        meta['range'] = dims_range
    if sizes is not None:
        meta['sizes'] = sizes
    if displayed is not None:
        meta['displayed'] = displayed
    if level is not None:
        meta['level'] = level
    if level_shape is not None:
        meta['level_shape'] = level_shape
    if dtype is not None:
        meta['dtype'] = dtype
    if normalized is not None:
        meta['normalized'] = normalized
    if volume is not None:
        meta['volume'] = bool(volume)
    if render is not None:
        meta['render'] = render
    if multiscale is not None:
        meta['multiscale'] = multiscale
    if ndisp is not None:
        meta['ndisplay'] = int(ndisp)

    if not state.dims_ready and (ndim is not None or order is not None):
        state.dims_ready = True
        logger.info("dims.update: metadata received; client intents enabled")
        notify_first_dims_ready()

    state.primary_axis_index = _compute_primary_axis_index(meta)

    if isinstance(cur, (list, tuple)) and cur and log_dims_info:
        logger.info(
            "dims_update: step=%s ndisp=%s order=%s labels=%s",
            list(cur),
            meta.get('ndisplay'),
            meta.get('order'),
            meta.get('axis_labels'),
        )

    viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
    if viewer_obj is not None:
        mirror_dims_to_viewer(
            viewer_obj,
            ui_call,
            current_step=cur,
            ndisplay=ndisp,
            ndim=ndim,
            dims_range=dims_range,
            order=order,
            axis_labels=axis_labels,
            sizes=sizes,
            displayed=displayed,
        )

    server_seq_val = _int_or_none(data.get('server_seq'))
    source_client_seq = _int_or_none(data.get('source_client_seq'))
    phase_meta = str(data.get('phase') or '').lower() or None
    control_versions = (
        data.get('control_versions')
        if isinstance(data.get('control_versions'), dict)
        else None
    )

    handled_ack = False
    if control_versions:
        for prop, meta_entry in control_versions.items():
            if not isinstance(meta_entry, dict):
                continue
            client_seq_entry = _int_or_none(meta_entry.get('source_client_seq'))
            if client_seq_entry is None:
                continue
            server_seq_entry = _int_or_none(meta_entry.get('server_seq'))
            phase_entry = str(meta_entry.get('phase') or phase_meta or 'update').lower()
            pending_info = loop_state.pending_intents.get(client_seq_entry)
            if _handle_dims_ack_session(
                state,
                loop_state,
                client_seq=client_seq_entry,
                phase=phase_entry,
                server_seq=server_seq_entry,
            ):
                handled_ack = True
                if ack_val is False:
                    logger.warning(
                        "dims_update ack=false for client_seq=%s prop=%s info=%s",
                        client_seq_entry,
                        prop,
                        pending_info,
                    )
                elif log_dims_info:
                    logger.debug(
                        "dims_update ack: client_seq=%s prop=%s phase=%s info=%s",
                        client_seq_entry,
                        prop,
                        phase_entry,
                        pending_info,
                    )
    else:
        client_seq_entry = source_client_seq or intent_seq
        if client_seq_entry is not None:
            phase_entry = str(phase_meta or ('commit' if ack_val else 'update')).lower()
            pending_info = loop_state.pending_intents.get(client_seq_entry)
            if _handle_dims_ack_session(
                state,
                loop_state,
                client_seq=client_seq_entry,
                phase=phase_entry,
                server_seq=server_seq_val,
            ):
                handled_ack = True
                if ack_val is False:
                    logger.warning(
                        "dims_update ack=false for client_seq=%s info=%s",
                        client_seq_entry,
                        pending_info,
                    )
                elif log_dims_info:
                    logger.debug(
                        "dims_update ack: client_seq=%s phase=%s info=%s",
                        client_seq_entry,
                        phase_entry,
                        pending_info,
                    )
            else:
                info = loop_state.pending_intents.pop(client_seq_entry, None)
                if info is not None:
                    if ack_val is False:
                        logger.warning(
                            "dims_update ack=false for client_seq=%s info=%s",
                            client_seq_entry,
                            info,
                        )
                    elif log_dims_info:
                        logger.debug(
                            "dims_update ack (legacy): client_seq=%s info=%s",
                            client_seq_entry,
                            info,
                        )

    loop_state.last_dims_payload = {
        'current_step': cur,
        'ndisplay': ndisp,
        'ndim': ndim,
        'dims_range': dims_range,
        'order': order,
        'axis_labels': axis_labels,
        'sizes': sizes,
        'displayed': displayed,
    }
    presenter.apply_dims_update(dict(data))


def _dims_session_key(axis_idx: int, prop: str) -> str:
    return f"dims:{prop}:{int(axis_idx)}"


def _parse_dims_session_key(key: str) -> tuple[str, int]:
    parts = key.split(":", 2)
    if len(parts) != 3 or parts[0] != "dims":
        return "step", 0
    prop = parts[1]
    try:
        axis_idx = int(parts[2])
    except Exception:
        axis_idx = 0
    return prop, axis_idx


def _ensure_dims_session(state: IntentState, axis_idx: int, prop: str) -> ControlSession:
    key = _dims_session_key(axis_idx, prop)
    session = state.control_sessions.get(key)
    if session is None:
        session = ControlSession(key=key)
        state.control_sessions[key] = session
    return session


def _send_dims_command(
    state: IntentState,
    loop_state: "ClientLoopState",
    *,
    axis_idx: int,
    prop: str,
    value: Any,
    session: ControlSession,
    phase: str,
    origin: Optional[str] = None,
) -> Optional[int]:
    channel = loop_state.state_channel
    if channel is None:
        return None

    session.mark_target(value)
    interaction_id = session.ensure_interaction_id()
    seq = state.next_client_seq()
    payload: Dict[str, Any] = {
        "type": CONTROL_COMMAND_TYPE,
        "scope": "dims",
        "target": f"axis-{int(axis_idx)}",
        "prop": prop,
        "value": value,
        "client_id": state.client_id,
        "client_seq": seq,
        "interaction_id": interaction_id,
        "phase": phase,
        "timestamp": time.time(),
    }
    extras: Dict[str, Any] = {}
    if origin is not None:
        extras["origin"] = origin
    if extras:
        payload["extras"] = extras

    ok = channel.post(payload)
    if not ok:
        session.dirty = True
        return None

    loop_state.pending_intents[seq] = {
        "kind": f"dims.{prop}",
        "axis": int(axis_idx),
        "phase": phase,
        "value": value,
        "interaction_id": interaction_id,
    }
    session.push_pending(seq, value, phase=phase)
    if phase == "commit":
        session.commit_in_flight = True
        session.awaiting_commit = False
        session.last_commit_seq = seq
    else:
        session.awaiting_commit = True
        session.commit_in_flight = False
    return seq


def _send_dims_commit(
    state: IntentState,
    loop_state: "ClientLoopState",
    *,
    axis_idx: int,
    prop: str,
    session: ControlSession,
) -> None:
    if session.commit_in_flight or session.pending:
        return
    commit_value = session.confirmed_value
    if commit_value is None:
        return
    _send_dims_command(
        state,
        loop_state,
        axis_idx=axis_idx,
        prop=prop,
        value=commit_value,
        session=session,
        phase="commit",
    )


def _handle_dims_ack_session(
    state: IntentState,
    loop_state: "ClientLoopState",
    *,
    client_seq: int,
    phase: str,
    server_seq: Optional[int],
) -> bool:
    handled = False
    for key, session in state.control_sessions.items():
        pending = session.pop_pending(client_seq)
        if pending is None:
            continue
        handled = True
        loop_state.pending_intents.pop(client_seq, None)
        if (
            server_seq is not None
            and session.last_server_seq is not None
            and server_seq < session.last_server_seq
        ):
            return True
        session.mark_confirmed(
            pending.value,
            client_seq if phase != "commit" else None,
            server_seq=server_seq,
        )
        prop, axis_idx = _parse_dims_session_key(key)
        if phase == "commit":
            session.reset_interaction()
        else:
            if not session.pending and session.awaiting_commit and not session.commit_in_flight:
                _send_dims_commit(
                    state,
                    loop_state,
                    axis_idx=axis_idx,
                    prop=prop,
                    session=session,
                )
        break
    return handled


def mirror_dims_to_viewer(
    viewer_obj,
    ui_call,
    *,
    current_step,
    ndisplay,
    ndim,
    dims_range,
    order,
    axis_labels,
    sizes,
    displayed,
) -> None:
    if viewer_obj is None or not hasattr(viewer_obj, '_apply_remote_dims_update'):
        return
    apply_remote = viewer_obj._apply_remote_dims_update  # type: ignore[attr-defined]

    def _apply() -> None:
        apply_remote(
            current_step=current_step,
            ndisplay=ndisplay,
            ndim=ndim,
            dims_range=dims_range,
            order=order,
            axis_labels=axis_labels,
            sizes=sizes,
            displayed=displayed,
        )

    if ui_call is not None:
        ui_call.call.emit(_apply)
        return
    _apply()


def replay_last_dims_payload(state: IntentState, loop_state: "ClientLoopState", viewer_ref, ui_call) -> None:
    payload = loop_state.last_dims_payload
    if not payload:
        return
    viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
    if viewer_obj is None:
        return
    mirror_dims_to_viewer(
        viewer_obj,
        ui_call,
        current_step=payload.get('current_step'),
        ndisplay=payload.get('ndisplay'),
        ndim=payload.get('ndim'),
        dims_range=payload.get('dims_range'),
        order=payload.get('order'),
        axis_labels=payload.get('axis_labels'),
        sizes=payload.get('sizes'),
        displayed=payload.get('displayed'),
    )


def dims_step(
    state: IntentState,
    loop_state: "ClientLoopState",
    axis: int | str,
    delta: int,
    *,
    origin: str,
    viewer_ref,
    ui_call,
) -> bool:
    if not state.dims_ready:
        return False
    idx = _axis_to_index(state, axis)
    if idx is None:
        return False
    viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
    if _is_axis_playing(viewer_obj, idx) and origin != 'play':
        return False
    now = time.perf_counter()
    if (now - float(state.last_dims_send or 0.0)) < state.dims_min_dt:
        logger.debug("control.command dims.step gated by rate limiter (%s)", origin)
        return False
    session = _ensure_dims_session(state, int(idx), "step")
    seq = _send_dims_command(
        state,
        loop_state,
        axis_idx=int(idx),
        prop="step",
        value=int(delta),
        session=session,
        phase="update",
        origin=str(origin),
    )
    if seq is None:
        return False
    state.last_dims_send = now
    return True


def handle_wheel_for_dims(
    state: IntentState,
    loop_state: "ClientLoopState",
    data: dict,
    *,
    viewer_ref,
    ui_call,
    log_dims_info: bool,
) -> bool:
    ay = int(data.get('angle_y') or 0)
    py = int(data.get('pixel_y') or 0)
    step = 0
    if ay != 0:
        step = (1 if ay > 0 else -1) * int(state.wheel_step or 1)
    elif py != 0:
        state.wheel_px_accum += float(py)
        thr = 30.0
        while state.wheel_px_accum >= thr:
            step += int(state.wheel_step or 1)
            state.wheel_px_accum -= thr
        while state.wheel_px_accum <= -thr:
            step -= int(state.wheel_step or 1)
            state.wheel_px_accum += thr
    if step == 0:
        return False
    sent = dims_step(
        state,
        loop_state,
        'primary',
        int(step),
        origin='wheel',
        viewer_ref=viewer_ref,
        ui_call=ui_call,
    )
    if log_dims_info:
        logger.info("wheel->control.command dims.step d=%+d sent=%s", int(step), bool(sent))
    else:
        logger.debug("wheel->control.command dims.step d=%+d sent=%s", int(step), bool(sent))
    return sent


def dims_set_index(
    state: IntentState,
    loop_state: "ClientLoopState",
    axis: int | str,
    value: int,
    *,
    origin: str,
    viewer_ref,
    ui_call,
) -> bool:
    if not state.dims_ready:
        return False
    idx = _axis_to_index(state, axis)
    if idx is None:
        return False
    viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
    if _is_axis_playing(viewer_obj, idx) and origin != 'play':
        return False
    now = time.perf_counter()
    if (now - float(state.last_dims_send or 0.0)) < state.dims_min_dt:
        logger.debug("control.command dims.index gated by rate limiter (%s)", origin)
        return False
    session = _ensure_dims_session(state, int(idx), "index")
    seq = _send_dims_command(
        state,
        loop_state,
        axis_idx=int(idx),
        prop="index",
        value=int(value),
        session=session,
        phase="update",
        origin=str(origin),
    )
    if seq is None:
        return False
    state.last_dims_send = now
    return True


def current_ndisplay(state: IntentState) -> Optional[int]:
    return _int_or_none(state.dims_meta.get('ndisplay'))


def toggle_ndisplay(state: IntentState, loop_state: "ClientLoopState", *, origin: str) -> bool:
    if not state.dims_ready:
        return False
    current = current_ndisplay(state)
    target = 2 if current == 3 else 3
    return view_set_ndisplay(state, loop_state, target, origin=origin)


def handle_key_event(
    data: dict,
    *,
    reset_camera: Callable[[], None],
    step_primary: Callable[[int], None],
) -> bool:
    try:
        from qtpy import QtCore  # type: ignore
    except Exception:
        logger.debug("key handler skipped: QtCore unavailable", exc_info=True)
        return False

    key_raw = data.get('key')
    key = int(key_raw) if key_raw is not None else -1
    mods = int(data.get('mods') or 0)
    txt = str(data.get('text') or '')

    keypad_mask = int(QtCore.Qt.KeypadModifier)
    keypad_only = (mods & ~keypad_mask) == 0 and (mods & keypad_mask) != 0

    if txt == '0' and mods == 0:
        logger.info("keycb: '0' -> camera.reset")
        reset_camera()
        return True
    if key == int(QtCore.Qt.Key_0) and (mods == 0 or keypad_only):
        logger.info("keycb: Key_0 -> camera.reset")
        reset_camera()
        return True

    k_left = int(QtCore.Qt.Key_Left)
    k_right = int(QtCore.Qt.Key_Right)
    k_up = int(QtCore.Qt.Key_Up)
    k_down = int(QtCore.Qt.Key_Down)
    k_pgup = int(QtCore.Qt.Key_PageUp)
    k_pgdn = int(QtCore.Qt.Key_PageDown)

    if key in (k_left, k_right, k_up, k_down, k_pgup, k_pgdn):
        coarse = 10
        if key in (k_left, k_down):
            step_primary(-1)
        elif key in (k_right, k_up):
            step_primary(+1)
        elif key == k_pgup:
            step_primary(+coarse)
        elif key == k_pgdn:
            step_primary(-coarse)
        return True

    return False


def view_set_ndisplay(
    state: IntentState,
    loop_state: "ClientLoopState",
    ndisplay: int,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready:
        return False
    if _rate_gate_settings(state, origin):
        return False
    nd_value = int(ndisplay)
    nd_target = 3 if nd_value >= 3 else 2
    cur = state.dims_meta.get('ndisplay')
    if cur is not None and int(cur) == nd_target:
        return True
    return _send_intent(loop_state, state, 'view.intent.set_ndisplay', {'ndisplay': nd_target}, origin)


def volume_set_render_mode(
    state: IntentState,
    loop_state: "ClientLoopState",
    mode: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    return _send_intent(loop_state, state, 'volume.intent.set_render_mode', {'mode': str(mode)}, origin)


def volume_set_clim(
    state: IntentState,
    loop_state: "ClientLoopState",
    lo: float,
    hi: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    lo_f, hi_f = _ensure_lo_hi(lo, hi)
    return _send_intent(loop_state, state, 'volume.intent.set_clim', {'lo': float(lo_f), 'hi': float(hi_f)}, origin)


def volume_set_colormap(
    state: IntentState,
    loop_state: "ClientLoopState",
    name: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    return _send_intent(loop_state, state, 'volume.intent.set_colormap', {'name': str(name)}, origin)


def volume_set_opacity(
    state: IntentState,
    loop_state: "ClientLoopState",
    alpha: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    a = _clamp01(alpha)
    return _send_intent(loop_state, state, 'volume.intent.set_opacity', {'alpha': float(a)}, origin)


def volume_set_sample_step(
    state: IntentState,
    loop_state: "ClientLoopState",
    relative: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    r = _clamp_sample_step(relative)
    return _send_intent(loop_state, state, 'volume.intent.set_sample_step', {'relative': float(r)}, origin)


def multiscale_set_policy(
    state: IntentState,
    loop_state: "ClientLoopState",
    policy: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready:
        return False
    if _rate_gate_settings(state, origin):
        return False
    pol = str(policy).lower().strip()
    if pol not in {'oversampling', 'thresholds', 'ratio'}:
        logger.debug("multiscale_set_policy rejected: policy=%s", pol)
        return False
    return _send_intent(loop_state, state, 'multiscale.intent.set_policy', {'policy': pol}, origin)


def multiscale_set_level(
    state: IntentState,
    loop_state: "ClientLoopState",
    level: int,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready:
        return False
    if _rate_gate_settings(state, origin):
        return False
    lv = _clamp_level(state, level)
    return _send_intent(loop_state, state, 'multiscale.intent.set_level', {'level': int(lv)}, origin)


def hud_snapshot(state: IntentState, *, video_size: tuple[Optional[int], Optional[int]], zoom_state: dict[str, object]) -> dict[str, object]:
    meta = state.dims_meta
    snap: dict[str, object] = {}
    snap['ndisplay'] = _int_or_none(meta.get('ndisplay'))
    snap['volume'] = _bool_or_none(meta.get('volume'))
    snap['vol_mode'] = bool(_is_volume_mode(state))

    controls = meta.get('controls') if isinstance(meta.get('controls'), dict) else None
    if isinstance(controls, dict):
        snap['render_mode'] = controls.get('rendering')
        clim = controls.get('contrast_limits')
        if isinstance(clim, Sequence):
            snap['clim_lo'] = _float_or_none(clim[0] if len(clim) > 0 else None)
            snap['clim_hi'] = _float_or_none(clim[1] if len(clim) > 1 else None)
        else:
            snap['clim_lo'] = None
            snap['clim_hi'] = None
        snap['colormap'] = controls.get('colormap')
        snap['opacity'] = _float_or_none(controls.get('opacity'))
    render_meta = meta.get('render') if isinstance(meta.get('render'), dict) else None
    if isinstance(render_meta, dict):
        snap['sample_step'] = _float_or_none(render_meta.get('sample_step'))
    elif 'sample_step' not in snap:
        snap['sample_step'] = None

    multiscale = meta.get('multiscale') if isinstance(meta.get('multiscale'), dict) else None
    if isinstance(multiscale, dict):
        snap['ms_policy'] = multiscale.get('policy')
        level_value = _int_or_none(multiscale.get('current_level'))
        snap['ms_level'] = level_value
        levels_obj = multiscale.get('levels')
        if isinstance(levels_obj, Sequence):
            snap['ms_levels'] = len(levels_obj)
            if level_value is not None and 0 <= level_value < len(levels_obj):
                entry = levels_obj[level_value]
                snap['ms_path'] = entry.get('path') if isinstance(entry, dict) else None
            else:
                snap['ms_path'] = None
        else:
            snap['ms_levels'] = None
            snap['ms_path'] = None

    snap['primary_axis'] = _int_or_none(state.primary_axis_index)
    snap.update(zoom_state)
    video_w, video_h = video_size
    snap['video_w'] = _int_or_none(video_w)
    snap['video_h'] = _int_or_none(video_h)
    return snap


def _axis_to_index(state: IntentState, axis: int | str) -> Optional[int]:
    if axis == 'primary':
        return int(state.primary_axis_index) if state.primary_axis_index is not None else 0
    if isinstance(axis, (int, float)) or (isinstance(axis, str) and str(axis).isdigit()):
        return int(axis)
    labels = state.dims_meta.get('axis_labels')
    if isinstance(labels, Sequence):
        label_map = {str(lbl): i for i, lbl in enumerate(labels)}
        match = label_map.get(str(axis))
        return int(match) if match is not None else None
    return None


def _compute_primary_axis_index(meta: dict[str, object | None]) -> Optional[int]:
    order = meta.get('order')
    ndisplay = meta.get('ndisplay')
    labels = meta.get('axis_labels')
    nd = int(ndisplay) if ndisplay is not None else 2
    idx_order: list[int] | None = None
    if isinstance(order, Sequence) and len(order) > 0:
        if all(isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()) for x in order):
            idx_order = [int(x) for x in order]
        elif isinstance(labels, Sequence) and all(isinstance(x, str) for x in order):
            label_to_index = {str(lbl): i for i, lbl in enumerate(labels)}
            idx_order = [int(label_to_index.get(str(lbl), i)) for i, lbl in enumerate(order)]
    if idx_order and len(idx_order) > nd:
        return int(idx_order[0])
    return 0


def _rate_gate_settings(state: IntentState, origin: str) -> bool:
    now = time.perf_counter()
    if (now - float(state.last_settings_send or 0.0)) < state.settings_min_dt:
        logger.debug("settings intent gated by rate limiter (%s)", origin)
        return True
    state.last_settings_send = now
    return False


def _send_intent(
    loop_state: "ClientLoopState",
    state: IntentState,
    type_str: str,
    fields: dict[str, object],
    origin: str,
) -> bool:
    ch = loop_state.state_channel
    if ch is None:
        return False
    payload = {'type': type_str}
    payload.update(fields)
    payload['client_id'] = state.client_id
    payload['client_seq'] = state.next_client_seq()
    payload['origin'] = str(origin)
    ok = ch.post(payload)
    fields_to_log = {k: v for k, v in payload.items() if k not in {'type', 'client_id', 'client_seq', 'origin'}}
    logger.info("%s->%s %s sent=%s", origin, type_str, fields_to_log, bool(ok))
    return bool(ok)


def _is_volume_mode(state: IntentState) -> bool:
    vol = bool(state.dims_meta.get('volume'))
    nd = int(state.dims_meta.get('ndisplay') or 2)
    return bool(vol) and int(nd) == 3


def _is_axis_playing(viewer_obj, axis_index: int) -> bool:
    if viewer_obj is None:
        return False
    is_playing = bool(getattr(viewer_obj, '_is_playing', False))
    play_axis = getattr(viewer_obj, '_play_axis', None)
    return bool(is_playing) and play_axis is not None and int(play_axis) == int(axis_index)


def _clamp01(a: float) -> float:
    a = float(a)
    if a < 0.0:
        return 0.0
    if a > 1.0:
        return 1.0
    return a


def _clamp_sample_step(r: float) -> float:
    r = float(r)
    if r < 0.1:
        return 0.1
    if r > 4.0:
        return 4.0
    return r


def _ensure_lo_hi(lo: float, hi: float) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f <= lo_f:
        lo_f, hi_f = hi_f, lo_f
    return lo_f, hi_f


def _clamp_level(state: IntentState, level: int) -> int:
    multiscale = state.dims_meta.get('multiscale')
    if isinstance(multiscale, dict):
        levels = multiscale.get('levels')
        if isinstance(levels, Sequence) and levels:
            lo, hi = 0, len(levels) - 1
            lv = int(level)
            if lv < lo:
                return lo
            if lv > hi:
                return hi
            return lv
    return int(level)


def _int_or_none(value: object) -> Optional[int]:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _float_or_none(value: object) -> Optional[float]:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _bool_or_none(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)
from napari_cuda.client.streaming.control_sessions import ControlSession
from napari_cuda.protocol.messages import CONTROL_COMMAND_TYPE
