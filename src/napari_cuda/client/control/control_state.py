"""Shared control-channel plumbing for emitters/mirrors/runtime.

This module hosts only reusable pieces with no dims/camera/volume specifics:
- ControlStateContext/ControlRuntime
- _emit_state_update (intent emission helper)
- _update_runtime_from_ack_outcome (runtime activity tracking)
- _rate_gate_settings (generic settings rate limiter)
- handle_generic_ack (generic ack applier for view/volume/multiscale)
- _mirror_viewer_dims (UI-thread bridge to viewer)
- handle_notify_camera (notify.camera ingestion helper)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from napari_cuda.protocol import NotifyCamera

from .client_state_ledger import (
    AckReconciliation,
    ClientStateLedger,
    IntentRecord,
)

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.rendering.presenter_facade import PresenterFacade
    from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


@dataclass
class ControlRuntime:
    active: bool = False
    last_phase: Optional[str] = None
    last_send_ts: float = 0.0
    active_intent_id: Optional[str] = None
    active_frame_id: Optional[str] = None


@dataclass
class ControlStateContext:
    """Mutable control state hoisted out of the loop object."""

    # Dims/session
    dims_ready: bool = False
    dims_spec: Any | None = None
    primary_axis_index: int | None = None
    session_id: Optional[str] = None
    ack_timeout_ms: Optional[int] = None
    intent_counter: int = 0

    # Rate gating and wheel
    dims_min_dt: float = 0.0
    last_dims_send: float = 0.0
    wheel_px_accum: float = 0.0
    wheel_step: float = 1.0
    settings_min_dt: float = 0.0
    last_settings_send: float = 0.0

    # Per-property runtimes and last confirmed state snapshots
    control_runtimes: dict[str, ControlRuntime] = field(default_factory=dict)
    view_state: dict[str, Any] = field(default_factory=dict)
    volume_state: dict[str, Any] = field(default_factory=dict)
    multiscale_state: dict[str, Any] = field(default_factory=dict)
    camera_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_cfg: Any) -> ControlStateContext:
        state = cls()
        dims_rate = getattr(env_cfg, "dims_rate_hz", 1.0) or 1.0
        state.dims_min_dt = 1.0 / max(1.0, float(dims_rate))
        state.wheel_step = float(getattr(env_cfg, "wheel_step", 1.0) or 1.0)
        settings_rate = getattr(env_cfg, "settings_rate_hz", 1.0) or 1.0
        state.settings_min_dt = 1.0 / max(1.0, float(settings_rate))
        return state

    def next_intent_ids(self) -> tuple[str, str]:
        self.intent_counter = (int(self.intent_counter) + 1) & 0xFFFFFFFF
        base = f"{self.intent_counter:08x}"
        intent_id = f"intent-{base}"
        frame_id = f"state-{base}"
        return intent_id, frame_id


def _runtime_key(scope: str, target: str, key: str) -> str:
    return f"{scope}:{target}:{key}"


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

    if scope == "view":
        if confirmed_value is not None:
            state.view_state[key] = confirmed_value
    elif scope == "volume" and confirmed_value is not None:
        state.volume_state[key] = confirmed_value
    elif scope == "multiscale" and confirmed_value is not None:
        state.multiscale_state[key] = confirmed_value


def handle_notify_camera(
    state: ControlStateContext,
    state_ledger: ClientStateLedger,
    frame: NotifyCamera,
    *,
    log_debug: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    payload = frame.payload
    mode = str(payload.mode or "")
    delta_payload = payload.delta
    mode_key = mode or "main"
    state.camera_state[mode_key] = delta_payload

    timestamp = frame.envelope.timestamp
    state_ledger.record_confirmed(
        "camera",
        "main",
        mode_key,
        delta_payload,
        timestamp=timestamp,
    )

    if log_debug:
        logger.debug(
            "notify.camera mode=%s origin=%s intent=%s delta=%s",
            mode,
            payload.origin,
            frame.envelope.intent_id,
            delta_payload,
        )

    return mode_key, delta_payload


def _rate_gate_settings(state: ControlStateContext, origin: str) -> bool:
    now = time.perf_counter()
    if (now - float(state.last_settings_send or 0.0)) < state.settings_min_dt:
        logger.debug("settings intent gated by rate limiter (%s)", origin)
        return True
    state.last_settings_send = now
    return False


__all__ = [
    "ControlRuntime",
    "ControlStateContext",
    "_emit_state_update",
    "_mirror_viewer_dims",
    "_rate_gate_settings",
    "_update_runtime_from_ack_outcome",
    "handle_generic_ack",
    "handle_notify_camera",
    "on_state_connected",
    "on_state_disconnected",
]


def on_state_connected(state: ControlStateContext) -> None:
    state.dims_ready = False
    state.dims_spec = None
    state.primary_axis_index = None


def on_state_disconnected(loop_state, state: ControlStateContext) -> None:
    state.dims_ready = False
    state.dims_spec = None
    state.primary_axis_index = None
    loop_state.pending_intents.clear()
    loop_state.last_dims_spec = None
    state.control_runtimes.clear()
    state.camera_state.clear()
