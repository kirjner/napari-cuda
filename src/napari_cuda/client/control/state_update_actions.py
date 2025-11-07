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


def current_ndisplay(state: ControlStateContext, ledger: ClientStateLedger) -> Optional[int]:
    return dims_current_ndisplay(state, ledger)

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
