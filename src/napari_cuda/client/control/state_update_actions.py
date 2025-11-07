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
from napari_cuda.shared.dims_spec import DimsSpec, dims_spec_axis_index_for_target

from .dims_projection import (
    current_ndisplay as dims_current_ndisplay,
    is_volume_mode as dims_is_volume_mode,
    project_dims,
)

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

# Re-export shared control plumbing from the new control_state module.
from .control_state import (
    ControlRuntime,
    ControlStateContext,
    _emit_state_update,
    _mirror_viewer_dims,
    _rate_gate_settings,
    _update_runtime_from_ack_outcome,
    handle_generic_ack,
    handle_notify_camera,
)

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


"""Stateful helpers retained here until dims/camera/volume helpers migrate.

We import and re-export shared plumbing from control_state to keep existing
imports stable during the refactor.
"""


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
def current_ndisplay(state: ControlStateContext, ledger: ClientStateLedger) -> Optional[int]:
    return dims_current_ndisplay(state, ledger)




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
        projection = project_dims(spec, ledger)
        snap['volume'] = bool(not spec.plane_mode)
        snap['vol_mode'] = bool(not spec.plane_mode and projection.ndisplay >= 3)
        snap['ndisplay'] = projection.ndisplay
    else:
        projection = None
        snap['volume'] = None
        snap['vol_mode'] = False
        snap['ndisplay'] = None

    rendering = state.volume_state.get('rendering') if state.volume_state else None
    clim = state.volume_state.get('contrast_limits') if state.volume_state else None
    colormap = state.volume_state.get('colormap') if state.volume_state else None
    opacity = state.volume_state.get('opacity') if state.volume_state else None

    snap['rendering'] = rendering
    if isinstance(clim, Sequence):
        snap['clim_lo'] = clim[0] if len(clim) > 0 else None
        snap['clim_hi'] = clim[1] if len(clim) > 1 else None
    else:
        snap['clim_lo'] = None
        snap['clim_hi'] = None
    snap['colormap'] = colormap
    snap['opacity'] = opacity

    snap['sample_step'] = state.volume_state.get('sample_step') if state.volume_state else None

    if isinstance(state.multiscale_state, dict):
        levels_obj = state.multiscale_state.get('levels')
        snap['ms_policy'] = state.multiscale_state.get('policy')
        level_value = state.multiscale_state.get('current_level')
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

    primary_axis = projection.primary_axis if projection is not None else state.primary_axis_index
    snap['primary_axis'] = primary_axis
    snap.update(zoom_state)
    video_w, video_h = video_size
    snap['video_w'] = video_w
    snap['video_h'] = video_h
    return snap


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
