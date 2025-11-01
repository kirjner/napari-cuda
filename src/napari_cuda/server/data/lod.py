"""Pure helpers for multiscale (level-of-detail) selection."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Optional

from napari_cuda.server.data.level_budget import LevelBudgetError
from napari_cuda.server.data.zarr_source import ZarrSceneSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelContext:
    """Immutable snapshot describing a multiscale level."""

    level: int
    step: tuple[int, ...]
    z_index: Optional[int]
    shape: tuple[int, ...]
    scale_yx: tuple[float, float]
    contrast: tuple[float, float]
    axes: str
    dtype: str


@dataclass(frozen=True)
class LevelPolicyConfig:
    threshold_in: float
    threshold_out: float
    fine_threshold: float
    hysteresis: float
    cooldown_ms: float


@dataclass(frozen=True)
class LevelPolicyInputs:
    current_level: int
    oversampling: Mapping[int, float]
    zoom_ratio: Optional[float]
    lock_level: Optional[int]
    last_switch_ts: float
    now_ts: float


@dataclass(frozen=True)
class LevelPolicyDecision:
    desired_level: int
    selected_level: int
    action: str
    should_switch: bool
    blocked_reason: Optional[str]
    cooldown_remaining_ms: float = 0.0


@dataclass(frozen=True)
class LevelDecision:
    """Policy decision annotated with oversampling and timestamp."""

    desired_level: int
    selected_level: int
    reason: str
    timestamp: float
    oversampling: Mapping[int, float]
    blocked_reason: Optional[str] = None
    cooldown_remaining_ms: float = 0.0
    downgraded: bool = False


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def stabilize_level(napari_level: int, prev_level: int, *, hysteresis: float = 0.0) -> int:
    """Return a stabilized level around ``prev_level`` honouring hysteresis."""

    cur = int(prev_level)
    nxt = int(napari_level)
    if hysteresis <= 0.0:
        return nxt
    if abs(nxt - cur) == 1:
        return cur
    return nxt


def _clamp_level(value: int, *, max_level: int) -> int:
    return max(0, min(int(value), int(max_level)))


def _cooldown_remaining(config: LevelPolicyConfig, inputs: LevelPolicyInputs) -> float:
    if inputs.last_switch_ts <= 0.0 or config.cooldown_ms <= 0.0:
        return 0.0
    elapsed_ms = (inputs.now_ts - inputs.last_switch_ts) * 1000.0
    return max(0.0, float(config.cooldown_ms) - elapsed_ms)


def select_level(
    config: LevelPolicyConfig,
    inputs: LevelPolicyInputs,
) -> LevelPolicyDecision:
    overs_map = {int(k): float(v) for k, v in inputs.oversampling.items() if v is not None}
    if not overs_map:
        return LevelPolicyDecision(
            desired_level=int(inputs.current_level),
            selected_level=int(inputs.current_level),
            action="no-oversampling",
            should_switch=False,
            blocked_reason="no-oversampling",
        )

    current = int(inputs.current_level)
    max_level = max(overs_map.keys())

    desired: Optional[int] = None
    action = "heuristic"
    zoom_ratio = inputs.zoom_ratio
    if zoom_ratio is not None:
        eps = 1e-3
        if zoom_ratio < 1.0 - eps and current > 0:
            desired = current - 1
            action = "zoom-in"
        elif zoom_ratio > 1.0 + eps and current < max_level:
            desired = current + 1
            action = "zoom-out"

    if desired is None:
        desired = current
        if current > 0 and overs_map.get(current - 1, float("inf")) <= config.fine_threshold:
            desired = current - 1
        elif current < max_level and overs_map.get(current, 0.0) > config.threshold_out:
            desired = current + 1
        action = "heuristic"
    else:
        desired = _clamp_level(desired, max_level=max_level)
        desired = max(desired, current - 1) if desired < current else min(desired, current + 1)

    selected = stabilize_level(int(desired), current, hysteresis=config.hysteresis)
    sel = int(selected)

    thr_in = float(config.threshold_in)
    thr_out = float(config.threshold_out)
    if action == "zoom-in":
        thr_in = min(1.20, thr_in)
    elif action == "zoom-out":
        thr_out = min(1.20, thr_out)

    if sel < current:
        if overs_map.get(sel, float("inf")) > thr_in:
            sel = current
    elif sel > current:
        if overs_map.get(current, 0.0) <= thr_out:
            sel = current
    selected = sel

    reason = action
    if inputs.lock_level is not None:
        selected = int(inputs.lock_level)
        reason = "locked"

    if selected == current:
        return LevelPolicyDecision(
            desired_level=int(desired),
            selected_level=int(selected),
            action=reason,
            should_switch=False,
            blocked_reason=None,
        )

    remaining = _cooldown_remaining(config, inputs)
    if selected != current and remaining > 0.0:
        return LevelPolicyDecision(
            desired_level=int(desired),
            selected_level=int(selected),
            action=reason,
            should_switch=False,
            blocked_reason="cooldown",
            cooldown_remaining_ms=remaining,
        )

    return LevelPolicyDecision(
        desired_level=int(desired),
        selected_level=int(selected),
        action=reason,
        should_switch=True,
        blocked_reason=None,
    )


def evaluate_policy(
    *,
    source: ZarrSceneSource,
    current_level: int,
    oversampling_for_level: Callable[[ZarrSceneSource, int], float],
    zoom_ratio: Optional[float],
    lock_level: Optional[int],
    last_switch_ts: float,
    config: LevelPolicyConfig,
    now_ts: Optional[float] = None,
    log_policy_eval: bool = False,
    select_level_fn: Callable[[LevelPolicyConfig, LevelPolicyInputs], LevelPolicyDecision] = select_level,
    logger_ref: logging.Logger = logger,
) -> Optional[LevelDecision]:
    """Evaluate policy inputs and return a candidate level decision."""

    now = float(now_ts) if now_ts is not None else time.perf_counter()
    overs_map: dict[int, float] = {}
    for index, descriptor in enumerate(source.level_descriptors):
        try:
            overs_map[int(index)] = float(oversampling_for_level(source, int(index)))
        except Exception:
            continue
    if not overs_map:
        return None

    inputs = LevelPolicyInputs(
        current_level=int(current_level),
        oversampling=overs_map,
        zoom_ratio=zoom_ratio,
        lock_level=lock_level,
        last_switch_ts=last_switch_ts,
        now_ts=now,
    )
    decision = select_level_fn(config, inputs)

    if not decision.should_switch:
        if decision.blocked_reason == "cooldown":
            if log_policy_eval and logger_ref.isEnabledFor(logging.DEBUG):
                logger_ref.debug(
                    "lod.cooldown: level=%d target=%d remaining=%.1fms",
                    int(current_level),
                    int(decision.selected_level),
                    decision.cooldown_remaining_ms,
                )
        elif logger_ref.isEnabledFor(logging.DEBUG):
            logger_ref.debug(
                "lod.hold: level=%d desired=%d selected=%d overs=%s",
                int(current_level),
                int(decision.desired_level),
                int(decision.selected_level),
                format_oversampling(overs_map),
            )
        return None

    if logger_ref.isEnabledFor(logging.INFO):
        if int(decision.selected_level) < int(current_level):
            logger_ref.info(
                "lod.zoom_in: current=%d -> selected=%d overs=%.3f reason=%s",
                int(current_level),
                int(decision.selected_level),
                overs_map.get(int(decision.selected_level), float("nan")),
                decision.action,
            )
        elif int(decision.selected_level) > int(current_level):
            logger_ref.info(
                "lod.zoom_out: current=%d -> selected=%d overs=%.3f reason=%s",
                int(current_level),
                int(decision.selected_level),
                overs_map.get(int(decision.selected_level), float("nan")),
                decision.action,
            )

    return LevelDecision(
        desired_level=int(decision.desired_level),
        selected_level=int(decision.selected_level),
        reason=str(decision.action),
        timestamp=now,
        oversampling=overs_map,
        blocked_reason=decision.blocked_reason,
        cooldown_remaining_ms=float(decision.cooldown_remaining_ms),
        downgraded=False,
    )


def enforce_budgets(
    decision: LevelDecision,
    *,
    source: ZarrSceneSource,
    use_volume: bool,
    budget_check: Callable[[ZarrSceneSource, int], None],
    log_layer_debug: bool = False,
    logger_ref: logging.Logger = logger,
) -> LevelDecision:
    """Clamp ``decision`` to the finest level allowed by budgets."""

    level_count = len(source.level_descriptors)
    if level_count == 0:
        return decision

    desired = int(decision.desired_level)
    start = int(decision.selected_level)
    max_level = level_count - 1

    for level in range(start, max_level + 1):
        try:
            budget_check(source, level)
        except LevelBudgetError:
            if level == max_level:
                raise
            if log_layer_debug:
                logger_ref.info(
                    "budget reject: mode=%s level=%d",
                    "volume" if use_volume else "slice",
                    int(level),
                )
            continue

        downgraded = level != desired
        if downgraded and log_layer_debug:
            logger_ref.info(
                "level downgrade: requested=%d active=%d",
                desired,
                level,
            )
        return replace(decision, selected_level=int(level), downgraded=downgraded)

    raise RuntimeError("Unable to select multiscale level within budget")


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def format_oversampling(overs_map: Mapping[int, float]) -> str:
    return "{" + ", ".join(f"{k}:{overs_map[k]:.2f}" for k in sorted(overs_map)) + "}"


AppliedLevel = LevelContext

__all__ = [
    "AppliedLevel",
    "LevelBudgetError",
    "LevelContext",
    "LevelDecision",
    "LevelPolicyConfig",
    "LevelPolicyDecision",
    "LevelPolicyInputs",
    "enforce_budgets",
    "evaluate_policy",
    "format_oversampling",
    "select_level",
    "stabilize_level",
]
