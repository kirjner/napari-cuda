"""Intent helpers for the streaming client loop."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.streaming.presenter_facade import PresenterFacade

    from .loop_state import ClientLoopState
    from napari_cuda.client.streaming.state_store import StateStore

from napari_cuda.protocol.messages import StateUpdateMessage

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
class ClientStateContext:
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
    control_runtimes: Dict[str, "ControlRuntime"] = field(default_factory=dict)
    dims_state: Dict[tuple[str, str], Any] = field(default_factory=dict)
    view_state: Dict[str, Any] = field(default_factory=dict)
    volume_state: Dict[str, Any] = field(default_factory=dict)
    multiscale_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_cfg: Any) -> "ClientStateContext":
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


@dataclass
class ControlRuntime:
    interaction_id: Optional[str] = None
    active: bool = False
    last_phase: Optional[str] = None
    last_send_ts: float = 0.0


def on_state_connected(state: ClientStateContext) -> None:
    state.dims_ready = False
    state.primary_axis_index = None


def on_state_disconnected(loop_state: "ClientLoopState", state: ClientStateContext) -> None:
    state.dims_ready = False
    state.primary_axis_index = None
    loop_state.pending_intents.clear()
    loop_state.last_dims_seq = None
    loop_state.last_dims_payload = None
    state.control_runtimes.clear()


def handle_dims_update(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    data: dict[str, object],
    *,
    presenter: "PresenterFacade",
    viewer_ref,
    ui_call,
    notify_first_dims_ready: Callable[[], None],
    log_dims_info: bool,
    state_store: Optional["StateStore"] = None,
) -> None:
    seq_val = _int_or_none(data.get('seq'))
    if seq_val is not None:
        loop_state.last_dims_seq = seq_val

    meta = state.dims_meta
    was_ready = bool(state.dims_ready)
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

    if cur is not None:
        try:
            meta['current_step'] = [int(x) for x in cur]
        except Exception:
            meta['current_step'] = list(cur)
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

    payload = _sync_dims_payload_from_meta(state, loop_state)

    if not was_ready and state_store is not None:
        _seed_dims_baseline(state, state_store, payload)

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

    presenter_payload = dict(payload)
    if seq_val is not None:
        presenter_payload['seq'] = seq_val
    presenter.apply_dims_update(presenter_payload)


def _axis_target_label(state: ClientStateContext, axis_idx: int) -> str:
    labels = state.dims_meta.get('axis_labels')
    if isinstance(labels, Sequence) and 0 <= axis_idx < len(labels):
        label = labels[axis_idx]
        if isinstance(label, str) and label.strip():
            return label
    return str(axis_idx)


def _axis_index_from_target(state: ClientStateContext, target: str) -> Optional[int]:
    target_lower = target.lower()
    labels = state.dims_meta.get('axis_labels')
    if isinstance(labels, Sequence):
        for idx, label in enumerate(labels):
            text = str(label)
            if text == target or text.lower() == target_lower:
                return int(idx)
    if target.startswith('axis-'):
        target = target.split('-', 1)[1]
    try:
        return int(target)
    except Exception:
        return None


def _runtime_key(scope: str, target: str, key: str) -> str:
    return f"{scope}:{target}:{key}"


def _emit_state_update(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
    *,
    scope: str,
    target: str,
    key: str,
    value: Any,
    origin: str,
) -> tuple[bool, Optional[Any]]:
    runtime_key = _runtime_key(scope, target, key)
    runtime = state.control_runtimes.setdefault(runtime_key, ControlRuntime())
    phase = "start" if not runtime.active else "update"
    interaction_id = runtime.interaction_id or uuid.uuid4().hex

    payload, projection = state_store.apply_local(
        scope,
        target,
        key,
        value,
        phase,
        interaction_id=interaction_id,
    )

    if not dispatch_state_update(payload, origin):
        return False, None

    runtime.active = True
    runtime.interaction_id = interaction_id
    runtime.last_phase = phase
    runtime.last_send_ts = time.perf_counter()
    return True, projection


def _update_runtime_from_ack(
    state: ClientStateContext,
    message: StateUpdateMessage,
    result,
) -> None:
    runtime_key = _runtime_key(str(message.scope), str(message.target), str(message.key))
    runtime = state.control_runtimes.setdefault(runtime_key, ControlRuntime())
    runtime.last_phase = message.phase or runtime.last_phase
    runtime.last_send_ts = time.perf_counter()
    if result.pending_len == 0 or not result.is_self:
        runtime.active = False
        runtime.interaction_id = None


def _sync_dims_payload_from_meta(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
) -> dict[str, Any]:
    meta = state.dims_meta

    current_step = meta.get('current_step')
    if current_step is not None:
        assert isinstance(current_step, Sequence)
        payload_current_step = list(current_step)
    else:
        payload_current_step = None

    dims_range = meta.get('range')
    if dims_range is not None:
        assert isinstance(dims_range, Sequence)
        payload_range = list(dims_range)
    else:
        payload_range = None

    order = meta.get('order')
    if order is not None:
        assert isinstance(order, Sequence)
        payload_order = list(order)
    else:
        payload_order = None

    axis_labels = meta.get('axis_labels')
    if axis_labels is not None:
        assert isinstance(axis_labels, Sequence)
        payload_axis_labels = list(axis_labels)
    else:
        payload_axis_labels = None

    sizes = meta.get('sizes')
    if sizes is not None:
        assert isinstance(sizes, Sequence)
        payload_sizes = list(sizes)
    else:
        payload_sizes = None

    displayed = meta.get('displayed')
    if displayed is not None:
        assert isinstance(displayed, Sequence)
        payload_displayed = list(displayed)
    else:
        payload_displayed = None

    payload = {
        'current_step': payload_current_step,
        'ndisplay': meta.get('ndisplay'),
        'ndim': meta.get('ndim'),
        'dims_range': payload_range,
        'order': payload_order,
        'axis_labels': payload_axis_labels,
        'sizes': payload_sizes,
        'displayed': payload_displayed,
    }

    loop_state.last_dims_payload = payload
    return payload


def _seed_dims_baseline(
    state: ClientStateContext,
    state_store: "StateStore",
    payload: dict[str, Any],
) -> None:
    current_step = payload.get('current_step') or []
    if isinstance(current_step, list):
        for idx, value in enumerate(current_step):
            if value is None:
                continue
            target_label = _axis_target_label(state, idx)
            state_store.seed_confirmed(
                'dims',
                target_label,
                'index',
                int(value),
            )

    ndisplay = payload.get('ndisplay')
    if ndisplay is not None:
        state_store.seed_confirmed('view', 'main', 'ndisplay', int(ndisplay))

    multiscale_meta = state.dims_meta.get('multiscale')
    if isinstance(multiscale_meta, dict):
        level_val = multiscale_meta.get('level')
        if level_val is not None:
            state_store.seed_confirmed('multiscale', 'main', 'level', int(level_val))
        policy_val = multiscale_meta.get('policy')
        if policy_val is not None:
            state_store.seed_confirmed('multiscale', 'main', 'policy', str(policy_val))

    render_meta = state.dims_meta.get('render')
    if isinstance(render_meta, dict):
        mode_val = render_meta.get('mode') or render_meta.get('render_mode')
        if mode_val is not None:
            state_store.seed_confirmed('volume', 'main', 'render_mode', str(mode_val))
        clim_val = render_meta.get('contrast_limits')
        if isinstance(clim_val, (list, tuple)) and len(clim_val) >= 2:
            lo = float(clim_val[0])
            hi = float(clim_val[1])
            state_store.seed_confirmed('volume', 'main', 'contrast_limits', (lo, hi))
        cmap_val = render_meta.get('colormap')
        if cmap_val is not None:
            state_store.seed_confirmed('volume', 'main', 'colormap', str(cmap_val))
        opacity_val = render_meta.get('opacity')
        if opacity_val is not None:
            state_store.seed_confirmed('volume', 'main', 'opacity', float(opacity_val))
        sample_val = render_meta.get('sample_step')
        if sample_val is not None:
            state_store.seed_confirmed('volume', 'main', 'sample_step', float(sample_val))


def _apply_dims_meta_snapshot(
    meta: dict[str, object | None],
    snapshot: Mapping[str, Any],
) -> None:
    assert isinstance(snapshot, Mapping)

    if 'ndim' in snapshot:
        value = snapshot['ndim']
        assert isinstance(value, int)
        meta['ndim'] = value

    if 'ndisplay' in snapshot:
        value = snapshot['ndisplay']
        assert isinstance(value, int)
        meta['ndisplay'] = value

    for key in ('order', 'axis_labels', 'range', 'sizes', 'displayed', 'current_step'):
        if key in snapshot:
            value = snapshot[key]
            assert isinstance(value, Sequence)
            meta[key] = list(value)

    if 'volume' in snapshot:
        meta['volume'] = bool(snapshot['volume'])

    if 'render' in snapshot:
        render = snapshot['render']
        assert isinstance(render, Mapping)
        meta['render'] = dict(render)

    if 'multiscale' in snapshot:
        multiscale = snapshot['multiscale']
        assert isinstance(multiscale, Mapping)
        meta['multiscale'] = dict(multiscale)

    if 'controls' in snapshot:
        controls = snapshot['controls']
        assert isinstance(controls, Mapping)
        meta['controls'] = dict(controls)
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


def replay_last_dims_payload(state: ClientStateContext, loop_state: "ClientLoopState", viewer_ref, ui_call) -> None:
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
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
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
    if origin == 'keys':
        min_dt = 0.0
    else:
        min_dt = float(state.dims_min_dt)
    if (now - float(state.last_dims_send or 0.0)) < min_dt:
        logger.debug("state.update dims.step gated by rate limiter (%s)", origin)
        return False
    axis_idx = int(idx)
    target_label = _axis_target_label(state, axis_idx)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="dims",
        target=target_label,
        key="step",
        value=int(delta),
        origin=origin,
    )
    if not ok:
        return False
    state.last_dims_send = now
    return True


def handle_wheel_for_dims(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
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
        state_store,
        dispatch_state_update,
        'primary',
        int(step),
        origin='wheel',
        viewer_ref=viewer_ref,
        ui_call=ui_call,
    )
    if log_dims_info:
        logger.info("wheel->state.update dims.step d=%+d sent=%s", int(step), bool(sent))
    else:
        logger.debug("wheel->state.update dims.step d=%+d sent=%s", int(step), bool(sent))
    return sent


def dims_set_index(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
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
    if origin == 'keys':
        min_dt = 0.0
    else:
        min_dt = float(state.dims_min_dt)
    if (now - float(state.last_dims_send or 0.0)) < min_dt:
        logger.debug("state.update dims.index gated by rate limiter (%s)", origin)
        return False
    axis_idx = int(idx)
    target_label = _axis_target_label(state, axis_idx)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="dims",
        target=target_label,
        key="index",
        value=int(value),
        origin=origin,
    )
    if not ok:
        return False
    state.last_dims_send = now
    return True


def handle_dims_state_update(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    message: StateUpdateMessage,
    *,
    presenter: Optional["PresenterFacade"] = None,
    viewer_ref=None,
    ui_call=None,
    log_dims_info: bool = False,
) -> None:
    if message.scope != "dims":
        return

    result = state_store.apply_remote(message)
    _update_runtime_from_ack(state, message, result)

    meta = state.dims_meta

    extras = message.extras if isinstance(message.extras, Mapping) else {}
    meta_snapshot = extras.get('meta')
    if isinstance(meta_snapshot, Mapping):
        _apply_dims_meta_snapshot(meta, meta_snapshot)

    axis_idx = extras.get('axis_index')
    if axis_idx is not None:
        assert isinstance(axis_idx, int)
    else:
        axis_idx = _axis_index_from_target(state, str(message.target))

    if axis_idx is None and message.key in {'index', 'step'}:
        logger.debug(
            "handle_dims_state_update: unknown axis target=%s; applying fallback index=0",
            message.target,
        )
        axis_idx = 0

    current_step_override = extras.get('current_step')
    if current_step_override is not None:
        assert isinstance(current_step_override, Sequence)
        meta['current_step'] = list(current_step_override)
    elif message.key in {'index', 'step'}:
        assert axis_idx is not None, f"unknown dims target {message.target!r}"
        current = list(meta.get('current_step') or [])
        while len(current) <= axis_idx:
            current.append(0)
        if message.key == 'step':
            assert isinstance(result.projection_value, (int, float)), "dims.step expects numeric value"
            current[axis_idx] = current[axis_idx] + int(result.projection_value)
        else:
            current[axis_idx] = int(result.projection_value)
        meta['current_step'] = current

    if 'ndisplay' in extras:
        value = extras['ndisplay']
        assert value is None or isinstance(value, int)
        meta['ndisplay'] = value

    state.dims_ready = True
    state.primary_axis_index = _compute_primary_axis_index(meta)

    state.dims_state[(str(message.target), str(message.key))] = result.projection_value

    payload = _sync_dims_payload_from_meta(state, loop_state)

    if presenter is not None:
        try:
            presenter.apply_dims_update(dict(payload))
        except Exception:
            logger.debug("presenter dims update failed", exc_info=True)

    viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
    if viewer_obj is not None:
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
        if log_dims_info:
            logger.info(
                "state.update dims mirrored: target=%s key=%s value=%s",
                message.target,
                message.key,
                result.projection_value,
            )

    logger.debug(
        "state.update dims applied: target=%s key=%s value=%s is_self=%s pending=%d overridden=%s",
        message.target,
        message.key,
        result.projection_value,
        result.is_self,
        result.pending_len,
        result.overridden,
    )


def handle_generic_state_update(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    message: StateUpdateMessage,
    *,
    presenter: Optional["PresenterFacade"] = None,
) -> None:
    result = state_store.apply_remote(message)
    _update_runtime_from_ack(state, message, result)

    scope = str(message.scope)
    key = str(message.key)
    value = result.projection_value

    if scope == 'view':
        state.view_state[key] = value
        if key == 'ndisplay':
            assert value is None or isinstance(value, int)
            state.dims_meta['ndisplay'] = value
            payload = _sync_dims_payload_from_meta(state, loop_state)
            if presenter is not None:
                try:
                    presenter.apply_dims_update(dict(payload))
                except Exception:
                    logger.debug("presenter dims update failed", exc_info=True)
    elif scope == 'volume':
        state.volume_state[key] = value
        render_meta = state.dims_meta.setdefault('render', {})
        assert isinstance(render_meta, dict)
        render_meta[key] = value
    elif scope == 'multiscale':
        state.multiscale_state[key] = value
        ms_meta = state.dims_meta.setdefault('multiscale', {})
        assert isinstance(ms_meta, dict)
        ms_meta[key] = value

    logger.debug(
        "state.update applied: scope=%s target=%s key=%s value=%s is_self=%s pending=%d overridden=%s",
        message.scope,
        message.target,
        message.key,
        result.projection_value,
        result.is_self,
        result.pending_len,
        result.overridden,
    )


def current_ndisplay(state: ClientStateContext) -> Optional[int]:
    return _int_or_none(state.dims_meta.get('ndisplay'))


def toggle_ndisplay(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
    *,
    origin: str,
) -> bool:
    if not state.dims_ready:
        return False
    current = current_ndisplay(state)
    target = 2 if current == 3 else 3
    return view_set_ndisplay(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        target,
        origin=origin,
    )


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
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
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
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="view",
        target="main",
        key="ndisplay",
        value=int(nd_target),
        origin=origin,
    )
    return ok


def volume_set_render_mode(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
    mode: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    mode_value = str(mode)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="render_mode",
        value=mode_value,
        origin=origin,
    )
    return ok


def volume_set_clim(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
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
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="contrast_limits",
        value=(float(lo_f), float(hi_f)),
        origin=origin,
    )
    return ok


def volume_set_colormap(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
    name: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="colormap",
        value=str(name),
        origin=origin,
    )
    return ok


def volume_set_opacity(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
    alpha: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    a = _clamp01(alpha)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="opacity",
        value=float(a),
        origin=origin,
    )
    return ok


def volume_set_sample_step(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
    relative: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    r = _clamp_sample_step(relative)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="sample_step",
        value=float(r),
        origin=origin,
    )
    return ok


def multiscale_set_policy(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
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
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="multiscale",
        target="main",
        key="policy",
        value=pol,
        origin=origin,
    )
    return ok


def multiscale_set_level(
    state: ClientStateContext,
    loop_state: "ClientLoopState",
    state_store: "StateStore",
    dispatch_state_update: Callable[[StateUpdateMessage, str], bool],
    level: int,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready:
        return False
    if _rate_gate_settings(state, origin):
        return False
    lv = _clamp_level(state, level)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_store,
        dispatch_state_update,
        scope="multiscale",
        target="main",
        key="level",
        value=int(lv),
        origin=origin,
    )
    return ok


def hud_snapshot(state: ClientStateContext, *, video_size: tuple[Optional[int], Optional[int]], zoom_state: dict[str, object]) -> dict[str, object]:
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


def _axis_to_index(state: ClientStateContext, axis: int | str) -> Optional[int]:
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


def _rate_gate_settings(state: ClientStateContext, origin: str) -> bool:
    now = time.perf_counter()
    if (now - float(state.last_settings_send or 0.0)) < state.settings_min_dt:
        logger.debug("settings intent gated by rate limiter (%s)", origin)
        return True
    state.last_settings_send = now
    return False


def _is_volume_mode(state: ClientStateContext) -> bool:
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


def _clamp_level(state: ClientStateContext, level: int) -> int:
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
