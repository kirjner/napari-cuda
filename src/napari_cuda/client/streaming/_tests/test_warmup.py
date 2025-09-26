from __future__ import annotations

import math

import pytest

from napari_cuda.client.streaming.client_loop import warmup
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState


class StubPresenter:
    def __init__(self) -> None:
        self.latencies: list[float] = []

    def set_latency(self, value: float) -> None:
        self.latencies.append(float(value))


def test_on_gate_lift_with_override_sets_extra_latency() -> None:
    policy = warmup.WarmupPolicy(ms_override=30.0, window_s=0.5, margin_ms=10.0, max_ms=40.0)
    state = ClientLoopState()
    presenter = StubPresenter()

    warmup.on_gate_lift(policy, state, presenter, base_latency_s=0.05, fps=60.0)

    assert math.isclose(state.warmup_extra_active_s, 0.03, rel_tol=1e-6)
    assert state.warmup_until > 0.0
    assert presenter.latencies[-1] == pytest.approx(0.08)


def test_apply_ramp_progressively_returns_to_base() -> None:
    policy = warmup.WarmupPolicy(ms_override=None, window_s=1.0, margin_ms=8.0, max_ms=10.0)
    state = ClientLoopState()
    presenter = StubPresenter()
    base_latency = 0.05
    state.warmup_extra_active_s = 0.01
    state.warmup_until = 10.0

    warmup.apply_ramp(policy, state, presenter, base_latency, now_pc=9.5)
    assert presenter.latencies[-1] == pytest.approx(0.055)

    warmup.apply_ramp(policy, state, presenter, base_latency, now_pc=10.0)
    assert state.warmup_until == 0.0
    assert presenter.latencies[-1] == pytest.approx(base_latency)


def test_cancel_resets_state_and_latency() -> None:
    state = ClientLoopState()
    presenter = StubPresenter()
    state.warmup_extra_active_s = 0.02
    state.warmup_until = 5.0

    warmup.cancel(state, presenter, base_latency_s=0.06)

    assert state.warmup_extra_active_s == 0.0
    assert state.warmup_until == 0.0
    assert presenter.latencies[-1] == pytest.approx(0.06)
