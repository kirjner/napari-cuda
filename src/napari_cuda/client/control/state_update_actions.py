"""State update helpers for the streaming client loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

from napari_cuda.protocol import NotifyCamera
from napari_cuda.shared.dims_spec import DimsSpec, dims_spec_axis_index_for_target, dims_spec_primary_axis

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.rendering.presenter_facade import PresenterFacade
    from napari_cuda.client.runtime.client_loop.loop_state import (
        ClientLoopState,
    )
from napari_cuda.client.control.client_state_ledger import (
    AckReconciliation,
    ClientStateLedger,
    IntentRecord,
)

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


def _normalize_camera_delta_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_camera_delta_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_camera_delta_value(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _normalize_camera_state_value(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        name = str(key)
        if name in {"center", "angles"}:
            normalized[name] = [float(component) for component in raw]
        elif name in {"zoom", "distance", "fov"}:
            normalized[name] = float(raw)
        else:
            normalized[name] = raw
    return normalized


@dataclass
class ControlStateContext:
    """Mutable control state hoisted out of the loop object."""

    dims_ready: bool = False
    dims_spec: DimsSpec | None = None
    primary_axis_index: int | None = None
    session_id: Optional[str] = None
    ack_timeout_ms: Optional[int] = None
    intent_counter: int = 0
    dims_min_dt: float = 0.0
    last_dims_send: float = 0.0
    wheel_px_accum: float = 0.0
    wheel_step: float = 1.0
    settings_min_dt: float = 0.0
    last_settings_send: float = 0.0
    dims_z: float | None = None
    dims_z_min: float | None = None
    dims_z_max: float | None = None
    control_runtimes: dict[str, ControlRuntime] = field(default_factory=dict)
    view_state: dict[str, Any] = field(default_factory=dict)
    volume_state: dict[str, Any] = field(default_factory=dict)
    multiscale_state: dict[str, Any] = field(default_factory=dict)
    camera_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_cfg: Any) -> ControlStateContext:
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

    def next_intent_ids(self) -> tuple[str, str]:
        self.intent_counter = (int(self.intent_counter) + 1) & 0xFFFFFFFF
        base = f"{self.intent_counter:08x}"
        intent_id = f"intent-{base}"
        frame_id = f"state-{base}"
        return intent_id, frame_id


@dataclass
class ControlRuntime:
    active: bool = False
    last_phase: Optional[str] = None
    last_send_ts: float = 0.0
    active_intent_id: Optional[str] = None
    active_frame_id: Optional[str] = None


def on_state_connected(state: ControlStateContext) -> None:
    state.dims_ready = False
    state.dims_spec = None
    state.primary_axis_index = None


def on_state_disconnected(loop_state: ClientLoopState, state: ControlStateContext) -> None:
    state.dims_ready = False
    state.dims_spec = None
    state.primary_axis_index = None
    loop_state.pending_intents.clear()
    loop_state.last_dims_spec = None
    state.control_runtimes.clear()
    state.camera_state.clear()
def handle_notify_camera(
    state: ControlStateContext,
    state_ledger: ClientStateLedger,
    frame: NotifyCamera,
    *,
    log_debug: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    payload = frame.payload
    mode = str(payload.mode or "")
    normalized_delta = _normalize_camera_delta_value(payload.delta)
    mode_key = mode or 'main'
    state.camera_state[mode_key] = normalized_delta

    timestamp = frame.envelope.timestamp
    state_ledger.record_confirmed(
        'camera',
        'main',
        mode_key,
        normalized_delta,
        timestamp=timestamp,
    )

    if log_debug:
        logger.debug(
            "notify.camera mode=%s origin=%s intent=%s delta=%s",
            mode,
            payload.origin,
            frame.envelope.intent_id,
            normalized_delta,
        )

    return mode_key, normalized_delta


def _axis_target_label(state: ControlStateContext, axis_idx: int) -> str:
    spec = state.dims_spec
    if isinstance(spec, DimsSpec):
        axes = spec.axes
        if 0 <= axis_idx < len(axes):
            return axes[axis_idx].label
    return str(axis_idx)

def _runtime_key(scope: str, target: str, key: str) -> str:
    return f"{scope}:{target}:{key}"


def viewer_update_from_dims_spec(
    spec: DimsSpec,
    *,
    step: Sequence[int] | None = None,
    ndisplay: int | None = None,
) -> dict[str, Any]:
    if step is None:
        step = spec.current_step
    if ndisplay is None:
        ndisplay = int(spec.ndisplay)
    level_shapes = spec.level_shapes
    if 0 <= spec.current_level < len(level_shapes):
        active_shape = level_shapes[spec.current_level]
    else:
        active_shape = ()
    dims_range = tuple((0, max(0, int(dim) - 1)) for dim in active_shape)
    axis_labels = tuple(axis.label for axis in spec.axes)
    order = tuple(int(idx) for idx in spec.order)
    displayed = tuple(int(idx) for idx in spec.displayed)
    return {
        'current_step': tuple(int(v) for v in step),
        'ndisplay': int(ndisplay),
        'ndim': int(spec.ndim),
        'dims_range': dims_range,
        'order': order,
        'axis_labels': axis_labels,
        'displayed': displayed,
    }


def projected_dims_step(ledger: ClientStateLedger, spec: DimsSpec) -> tuple[int, ...]:
    values: list[int] = []
    for axis in spec.axes:
        label = axis.label
        pending = ledger.latest_pending_value('dims', label, 'index')
        if pending is not None:
            values.append(int(pending))
            continue
        confirmed = ledger.confirmed_value('dims', label, 'index')
        if confirmed is not None:
            values.append(int(confirmed))
            continue
        values.append(int(axis.current_step))
    return tuple(values)


def projected_ndisplay(ledger: ClientStateLedger, spec: DimsSpec) -> int:
    pending = ledger.latest_pending_value('view', 'main', 'ndisplay')
    if pending is not None:
        return int(pending)
    confirmed = ledger.confirmed_value('view', 'main', 'ndisplay')
    if confirmed is not None:
        return int(confirmed)
    return int(spec.ndisplay)


def ensure_dims_spec(state: ControlStateContext) -> DimsSpec:
    spec = state.dims_spec
    assert spec is not None, "dims_spec must be available"
    return spec


def current_ndisplay(state: ControlStateContext, ledger: ClientStateLedger) -> Optional[int]:
    spec = state.dims_spec
    if spec is None:
        return None
    return projected_ndisplay(ledger, spec)


def _emit_state_update(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    *,
    scope: str,
    target: str,
    key: str,
    value: Any,
    origin: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, Optional[Any]]:
    runtime_key = _runtime_key(scope, target, key)
    runtime = state.control_runtimes.setdefault(runtime_key, ControlRuntime())
    phase = "start" if not runtime.active else "update"
    intent_id, frame_id = state.next_intent_ids()
    pending = state_ledger.apply_local(
        scope,
        target,
        key,
        value,
        phase,
        intent_id=intent_id,
        frame_id=frame_id,
        metadata=metadata,
    )

    if pending is None:
        projection_value = state_ledger.confirmed_value(scope, target, key)
        if projection_value is None:
            projection_value = value
        logger.debug(
            "state.update suppressed (duplicate): scope=%s target=%s key=%s value=%s runtime.active=%s",
            scope,
            target,
            key,
            value,
            runtime.active,
        )
        return False, projection_value

    if not dispatch_state_update(pending, origin):
        state_ledger.discard_pending(frame_id)
        return False, None

    runtime.active = True
    runtime.last_phase = phase
    runtime.last_send_ts = time.perf_counter()
    runtime.active_intent_id = intent_id
    runtime.active_frame_id = frame_id
    return True, pending.projection_value

def _update_runtime_from_ack_outcome(state: ControlStateContext, outcome: AckReconciliation) -> None:
    if outcome.scope is None or outcome.target is None or outcome.key is None:
        return
    runtime_key = _runtime_key(outcome.scope, outcome.target, outcome.key)
    runtime = state.control_runtimes.setdefault(runtime_key, ControlRuntime())
    runtime.last_phase = outcome.update_phase or runtime.last_phase
    runtime.last_send_ts = time.perf_counter()
    matched = outcome.in_reply_to == runtime.active_frame_id
    if matched:
        runtime.active_frame_id = None
        runtime.active_intent_id = None

    if matched and outcome.pending_len == 0:
        runtime.active = False
        runtime.last_phase = None
    else:
        runtime.active = outcome.pending_len > 0
        if not runtime.active:
            runtime.last_phase = None


def _mirror_viewer_dims(viewer_obj, ui_call, update: Mapping[str, Any]) -> None:
    apply_remote = viewer_obj._apply_remote_dims_update  # type: ignore[attr-defined]

    def _apply() -> None:
        apply_remote(**update)

    if ui_call is not None:
        ui_call.call.emit(_apply)
    else:
        _apply()


def handle_generic_ack(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    outcome: AckReconciliation,
    *,
    presenter: Optional[PresenterFacade] = None,
) -> None:
    if outcome.scope is None:
        return

    _update_runtime_from_ack_outcome(state, outcome)

    if outcome.status != "accepted":
        error = outcome.error or {}
        logger.warning(
            "ack.state rejected: scope=%s target=%s key=%s code=%s message=%s details=%s",
            outcome.scope,
            outcome.target,
            outcome.key,
            error.get("code"),
            error.get("message"),
            error.get("details"),
        )

        confirmed_value = outcome.confirmed_value
        scope = outcome.scope
        key = outcome.key or ""

        return

    logger.debug(
        "ack.state accepted: scope=%s target=%s key=%s pending=%d",
        outcome.scope,
        outcome.target,
        outcome.key,
        outcome.pending_len,
    )

    confirmed_value = outcome.confirmed_value
    scope = outcome.scope
    key = outcome.key or ""

    if scope == 'view':
        if confirmed_value is not None:
            state.view_state[key] = confirmed_value
    elif scope == 'volume' and confirmed_value is not None:
        state.volume_state[key] = confirmed_value
    elif scope == 'multiscale' and confirmed_value is not None:
        state.multiscale_state[key] = confirmed_value


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


def camera_zoom(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    *,
    factor: float,
    anchor_px: tuple[float, float],
    origin: str,
) -> bool:
    sanitized = {
        "factor": float(factor),
        "anchor_px": [float(anchor_px[0]), float(anchor_px[1])],
    }
    metadata = {
        "mode": "zoom",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="zoom",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['zoom'] = sanitized
    return ok


def camera_pan(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    *,
    dx_px: float,
    dy_px: float,
    origin: str,
) -> bool:
    sanitized = {"dx_px": float(dx_px), "dy_px": float(dy_px)}
    metadata = {
        "mode": "pan",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="pan",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['pan'] = sanitized
    return ok


def camera_orbit(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    *,
    d_az_deg: float,
    d_el_deg: float,
    origin: str,
) -> bool:
    sanitized = {"d_az_deg": float(d_az_deg), "d_el_deg": float(d_el_deg)}
    metadata = {
        "mode": "orbit",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="orbit",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['orbit'] = sanitized
    return ok


def camera_reset(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    *,
    reason: str,
    origin: str,
) -> bool:
    sanitized = {"reason": str(reason)}
    metadata = {
        "mode": "reset",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="reset",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['reset'] = sanitized
    return ok


def camera_set(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    *,
    center: Optional[Sequence[float]] = None,
    zoom: Optional[float] = None,
    angles: Optional[Sequence[float]] = None,
    origin: str,
) -> bool:
    payload: dict[str, Any] = {}
    if center is not None:
        payload['center'] = [float(c) for c in center]
    if zoom is not None:
        payload['zoom'] = float(zoom)
    if angles is not None:
        payload['angles'] = [float(a) for a in angles]
    if not payload:
        return False
    metadata = {
        "mode": "set",
        "origin": origin,
        "delta": dict(payload),
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="set",
        value=payload,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['set'] = payload
    return ok


def volume_set_rendering(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    mode: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state, state_ledger):
        return False
    if _rate_gate_settings(state, origin):
        return False
    mode_value = str(mode)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="rendering",
        value=mode_value,
        origin=origin,
    )
    return ok


def volume_set_clim(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    lo: float,
    hi: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state, state_ledger):
        return False
    if _rate_gate_settings(state, origin):
        return False
    lo_f, hi_f = _ensure_lo_hi(lo, hi)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="contrast_limits",
        value=(float(lo_f), float(hi_f)),
        origin=origin,
    )
    return ok


def volume_set_colormap(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    name: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state, state_ledger):
        return False
    if _rate_gate_settings(state, origin):
        return False
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="colormap",
        value=str(name),
        origin=origin,
    )
    return ok


def volume_set_opacity(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    alpha: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state, state_ledger):
        return False
    if _rate_gate_settings(state, origin):
        return False
    a = _clamp01(alpha)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="opacity",
        value=float(a),
        origin=origin,
    )
    return ok


def volume_set_sample_step(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
    relative: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state, state_ledger):
        return False
    if _rate_gate_settings(state, origin):
        return False
    r = _clamp_sample_step(relative)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="sample_step",
        value=float(r),
        origin=origin,
    )
    return ok


def multiscale_set_policy(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
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
        state_ledger,
        dispatch_state_update,
        scope="multiscale",
        target="main",
        key="policy",
        value=pol,
        origin=origin,
    )
    return ok


def multiscale_set_level(
    state: ControlStateContext,
    loop_state: ClientLoopState,
    state_ledger: ClientStateLedger,
    dispatch_state_update: Callable[[IntentRecord, str], bool],
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
        state_ledger,
        dispatch_state_update,
        scope="multiscale",
        target="main",
        key="level",
        value=int(lv),
        origin=origin,
    )
    return ok


def hud_snapshot(
    state: ControlStateContext,
    ledger: ClientStateLedger,
    *,
    video_size: tuple[Optional[int], Optional[int]],
    zoom_state: dict[str, object],
) -> dict[str, object]:
    spec = state.dims_spec
    snap: dict[str, object] = {}
    if spec is not None:
        ndisplay = projected_ndisplay(ledger, spec)
        snap['volume'] = bool(not spec.plane_mode)
        snap['vol_mode'] = bool(not spec.plane_mode and ndisplay >= 3)
        snap['ndisplay'] = int(ndisplay)
    else:
        snap['volume'] = None
        snap['vol_mode'] = False
        snap['ndisplay'] = None

    rendering = state.volume_state.get('rendering') if state.volume_state else None
    clim = state.volume_state.get('contrast_limits') if state.volume_state else None
    colormap = state.volume_state.get('colormap') if state.volume_state else None
    opacity = state.volume_state.get('opacity') if state.volume_state else None

    snap['rendering'] = rendering
    if isinstance(clim, Sequence):
        snap['clim_lo'] = _float_or_none(clim[0] if len(clim) > 0 else None)
        snap['clim_hi'] = _float_or_none(clim[1] if len(clim) > 1 else None)
    else:
        snap['clim_lo'] = None
        snap['clim_hi'] = None
    snap['colormap'] = colormap
    snap['opacity'] = _float_or_none(opacity)

    snap['sample_step'] = _float_or_none(state.volume_state.get('sample_step')) if state.volume_state else None

    if isinstance(state.multiscale_state, dict):
        levels_obj = state.multiscale_state.get('levels')
        snap['ms_policy'] = state.multiscale_state.get('policy')
        level_value = _int_or_none(state.multiscale_state.get('current_level'))
        snap['ms_level'] = level_value
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


def _axis_to_index(state: ControlStateContext, axis: int | str) -> Optional[int]:
    if axis == 'primary':
        return int(state.primary_axis_index) if state.primary_axis_index is not None else 0
    if isinstance(axis, (int, float)) or (isinstance(axis, str) and str(axis).isdigit()):
        return int(axis)
    spec = state.dims_spec
    if isinstance(spec, DimsSpec):
        resolved = dims_spec_axis_index_for_target(spec, str(axis))
        if resolved is not None:
            return resolved
        labels = tuple(ax.label for ax in spec.axes)
        label_map = {str(lbl): idx for idx, lbl in enumerate(labels)}
        match = label_map.get(str(axis))
        if match is not None:
            return int(match)
    return None


def _rate_gate_settings(state: ControlStateContext, origin: str) -> bool:
    now = time.perf_counter()
    if (now - float(state.last_settings_send or 0.0)) < state.settings_min_dt:
        logger.debug("settings intent gated by rate limiter (%s)", origin)
        return True
    state.last_settings_send = now
    return False


def _is_volume_mode(state: ControlStateContext, ledger: ClientStateLedger) -> bool:
    spec = state.dims_spec
    if spec is None:
        return False
    ndisplay = projected_ndisplay(ledger, spec)
    return (not spec.plane_mode) and ndisplay >= 3


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


def _clamp_level(state: ControlStateContext, level: int) -> int:
    levels = state.multiscale_state.get('levels') if isinstance(state.multiscale_state, Mapping) else None
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
