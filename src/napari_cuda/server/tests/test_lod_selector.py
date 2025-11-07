from __future__ import annotations

from napari_cuda.server.data.lod import (
    LevelBudgetError,
    LevelDecision,
    LevelPolicyConfig,
    LevelPolicyInputs,
    enforce_budgets,
    select_level,
)

_BASE_CONFIG = LevelPolicyConfig(
    threshold_in=1.05,
    threshold_out=1.35,
    fine_threshold=1.05,
    hysteresis=0.0,
    cooldown_ms=150.0,
)


def test_select_level_prefers_zoom_in_hint() -> None:
    inputs = LevelPolicyInputs(
        current_level=1,
        oversampling={0: 0.95, 1: 1.6},
        zoom_ratio=0.5,
        lock_level=None,
        last_switch_ts=0.0,
        now_ts=1.0,
    )

    decision = select_level(_BASE_CONFIG, inputs)

    assert decision.selected_level == 0
    assert decision.action == "zoom-in"
    assert decision.should_switch is True


def test_zoom_in_hint_allows_relaxed_threshold() -> None:
    config = LevelPolicyConfig(
        threshold_in=1.2,
        threshold_out=1.35,
        fine_threshold=1.2,
        hysteresis=0.0,
        cooldown_ms=150.0,
    )
    inputs = LevelPolicyInputs(
        current_level=2,
        oversampling={1: 1.18, 2: 1.6},
        zoom_ratio=0.7,
        lock_level=None,
        last_switch_ts=0.0,
        now_ts=3.0,
    )

    decision = select_level(config, inputs)

    assert decision.selected_level == 1
    assert decision.action == "zoom-in"
    assert decision.should_switch is True


def test_select_level_uses_heuristic_when_no_zoom_hint() -> None:
    inputs = LevelPolicyInputs(
        current_level=2,
        oversampling={0: 0.7, 1: 0.9, 2: 1.6},
        zoom_ratio=None,
        lock_level=None,
        last_switch_ts=0.0,
        now_ts=2.0,
    )

    decision = select_level(_BASE_CONFIG, inputs)

    assert decision.selected_level == 1
    assert decision.desired_level == 1
    assert decision.action == "heuristic"
    assert decision.should_switch is True


def test_zoom_out_hint_uses_relaxed_threshold() -> None:
    config = LevelPolicyConfig(
        threshold_in=1.05,
        threshold_out=1.35,
        fine_threshold=1.05,
        hysteresis=0.0,
        cooldown_ms=150.0,
    )
    inputs = LevelPolicyInputs(
        current_level=0,
        oversampling={0: 1.28, 1: 1.6},
        zoom_ratio=1.4,
        lock_level=None,
        last_switch_ts=0.0,
        now_ts=4.0,
    )

    decision = select_level(config, inputs)

    assert decision.selected_level == 1
    assert decision.action == "zoom-out"
    assert decision.should_switch is True


def test_select_level_respects_cooldown() -> None:
    inputs = LevelPolicyInputs(
        current_level=1,
        oversampling={0: 0.8, 1: 1.8},
        zoom_ratio=None,
        lock_level=None,
        last_switch_ts=1.0,
        now_ts=1.01,
    )

    decision = select_level(_BASE_CONFIG, inputs)

    assert decision.selected_level == 0
    assert decision.should_switch is False
    assert decision.blocked_reason == "cooldown"
    assert decision.cooldown_remaining_ms > 0.0


def test_select_level_honours_lock_override() -> None:
    inputs = LevelPolicyInputs(
        current_level=0,
        oversampling={0: 1.2, 1: 1.4, 2: 1.6},
        zoom_ratio=None,
        lock_level=2,
        last_switch_ts=0.0,
        now_ts=5.0,
    )

    decision = select_level(_BASE_CONFIG, inputs)

    assert decision.selected_level == 2
    assert decision.action == "locked"
    assert decision.should_switch is True


def test_select_level_handles_empty_oversampling() -> None:
    inputs = LevelPolicyInputs(
        current_level=0,
        oversampling={},
        zoom_ratio=None,
        lock_level=None,
        last_switch_ts=0.0,
        now_ts=6.0,
    )

    decision = select_level(_BASE_CONFIG, inputs)

    assert decision.should_switch is False
    assert decision.blocked_reason == "no-oversampling"


def test_select_level_filters_none_entries() -> None:
    inputs = LevelPolicyInputs(
        current_level=1,
        oversampling={0: 0.95, 1: None, 2: 1.6},  # type: ignore[arg-type]
        zoom_ratio=None,
        lock_level=None,
        last_switch_ts=0.0,
        now_ts=7.0,
    )

    decision = select_level(_BASE_CONFIG, inputs)

    assert decision.selected_level == 0
    assert decision.should_switch is True


class _StubLevelDescriptor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _StubSource:
    def __init__(self, level_shapes: tuple[tuple[int, ...], ...]) -> None:
        self.level_descriptors = [_StubLevelDescriptor(shape) for shape in level_shapes]
        self.axes = ("z", "y", "x")
        self.dtype = "float32"

    def ensure_contrast(self, *, level: int) -> tuple[float, float]:
        return (0.0, 1.0)

    def level_scale(self, level: int) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)


def test_enforce_budgets_downgrades_when_needed() -> None:
    source = _StubSource(((8, 8, 8), (4, 4, 4), (2, 2, 2)))

    decision = LevelDecision(
        desired_level=0,
        selected_level=0,
        reason="policy",
        timestamp=0.0,
        oversampling={0: 1.0, 1: 1.2, 2: 1.5},
    )

    def _budget_check(_scene: _StubSource, level: int) -> None:
        if level == 0:
            raise LevelBudgetError("reject finest")

    downgraded = enforce_budgets(
        decision,
        source=source,
        use_volume=False,
        budget_check=_budget_check,  # type: ignore[arg-type]
        log_layer_debug=False,
    )

    assert downgraded.selected_level == 1
    assert downgraded.downgraded is True
