"""Warmup helpers for the streaming client loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.rendering.presenter import FixedLatencyPresenter

    from .loop_state import ClientLoopState


@dataclass(frozen=True)
class WarmupPolicy:
    """Compute and apply VT warmup latency ramps."""

    ms_override: float | None
    window_s: float
    margin_ms: float
    max_ms: float

    def compute_extra_latency_s(self, fps: float | None, base_latency_s: float) -> float:
        """Return the warmup latency boost in seconds."""

        if self.window_s <= 0:
            return 0.0
        if self.ms_override is not None:
            extra_ms = max(0.0, float(self.ms_override))
        else:
            frame_rate = float(fps) if (fps and fps > 0) else 60.0
            frame_ms = 1000.0 / max(1e-6, frame_rate)
            target_ms = frame_ms + float(self.margin_ms)
            base_ms = float(base_latency_s) * 1000.0
            extra_ms = max(0.0, min(float(self.max_ms), target_ms - base_ms))
        return extra_ms / 1000.0


def on_gate_lift(
    policy: WarmupPolicy,
    state: ClientLoopState,
    presenter: FixedLatencyPresenter,
    base_latency_s: float,
    fps: float | None,
) -> None:
    """Apply the warmup latency when VT becomes active."""

    extra_s = policy.compute_extra_latency_s(fps, base_latency_s)
    if extra_s <= 0.0:
        cancel(state, presenter, base_latency_s)
        return
    presenter.set_latency(base_latency_s + extra_s)
    state.warmup_extra_active_s = extra_s
    state.warmup_until = time.perf_counter() + float(max(0.0, policy.window_s))


def apply_ramp(
    policy: WarmupPolicy,
    state: ClientLoopState,
    presenter: FixedLatencyPresenter,
    base_latency_s: float,
    now_pc: float,
) -> None:
    """Gradually ramp the VT latency back to steady state."""

    if state.warmup_until <= 0.0:
        return
    if now_pc >= state.warmup_until or policy.window_s <= 0.0:
        cancel(state, presenter, base_latency_s)
        return
    remain = max(0.0, state.warmup_until - now_pc)
    frac = remain / max(1e-6, float(policy.window_s))
    cur = base_latency_s + state.warmup_extra_active_s * frac
    presenter.set_latency(cur)


def cancel(
    state: ClientLoopState,
    presenter: FixedLatencyPresenter,
    base_latency_s: float,
) -> None:
    """Reset warmup state and restore the baseline latency."""

    state.warmup_until = 0.0
    state.warmup_extra_active_s = 0.0
    presenter.set_latency(base_latency_s)
