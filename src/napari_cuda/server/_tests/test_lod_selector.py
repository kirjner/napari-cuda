from __future__ import annotations

from typing import Optional

from napari_cuda.server.lod import (
    AppliedLevel,
    LevelBudgetError,
    LevelPolicyConfig,
    LevelPolicyInputs,
    apply_level_with_context,
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
    def __init__(self, level_shapes: tuple[tuple[int, ...], ...], *, current_level: int) -> None:
        self.level_descriptors = [_StubLevelDescriptor(shape) for shape in level_shapes]
        self.current_level = current_level
        self.level_history: list[int] = []


def _make_applied(level: int) -> AppliedLevel:
    return AppliedLevel(
        level=level,
        step=(0, 0, 0),
        z_index=0,
        shape=(8, 8, 8),
        scale_yx=(1.0, 1.0),
        contrast=(0.0, 1.0),
        axes="zyx",
        dtype="float32",
    )


def test_apply_level_with_context_clears_roi_caches() -> None:
    source = _StubSource(((8, 8, 8), (4, 4, 4)), current_level=1)
    roi_cache = {0: object(), 1: object()}
    roi_log_state = {0: object(), 1: object()}
    applied_levels: list[int] = []
    switches: list[tuple[int, int, float]] = []

    def _budget_check(_scene: _StubSource, _level: int) -> None:
        return None

    def _apply(scene: _StubSource, level: int, prev: Optional[int]) -> AppliedLevel:
        scene.level_history.append(level)
        applied_levels.append(level)
        return _make_applied(level)

    def _on_switch(prev: int, new: int, elapsed_ms: float) -> None:
        switches.append((prev, new, elapsed_ms))

    snapshot, downgraded = apply_level_with_context(
        desired_level=0,
        use_volume=False,
        source=source,  # type: ignore[arg-type]
        current_level=1,
        log_layer_debug=False,
        budget_check=_budget_check,  # type: ignore[arg-type]
        apply_level_fn=_apply,  # type: ignore[arg-type]
        on_switch=_on_switch,
        roi_cache=roi_cache,
        roi_log_state=roi_log_state,
    )

    assert snapshot.level == 0
    assert downgraded is False
    assert applied_levels == [0]
    assert switches and switches[0][0] == 1 and switches[0][1] == 0
    assert 0 not in roi_cache
    assert 0 not in roi_log_state


def test_apply_level_with_context_falls_back_on_budget_error() -> None:
    source = _StubSource(((8, 8, 8), (4, 4, 4), (2, 2, 2)), current_level=2)
    switches: list[tuple[int, int, float]] = []

    def _budget_check(_scene: _StubSource, level: int) -> None:
        if level == 0:
            raise LevelBudgetError("reject coarse")

    def _apply(scene: _StubSource, level: int, prev: Optional[int]) -> AppliedLevel:
        scene.level_history.append(level)
        return _make_applied(level)

    def _on_switch(prev: int, new: int, elapsed_ms: float) -> None:
        switches.append((prev, new, elapsed_ms))

    snapshot, downgraded = apply_level_with_context(
        desired_level=0,
        use_volume=False,
        source=source,  # type: ignore[arg-type]
        current_level=2,
        log_layer_debug=False,
        budget_check=_budget_check,  # type: ignore[arg-type]
        apply_level_fn=_apply,  # type: ignore[arg-type]
        on_switch=_on_switch,
    )

    assert snapshot.level == 1
    assert downgraded is True
    assert source.level_history == [1]
    assert switches and switches[0][0] == 2 and switches[0][1] == 1
