"""Policy surface for multiscale level selection via oversampling thresholds."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .zarr_source import LevelDescriptor

logger = logging.getLogger(__name__)


ThresholdMap = Mapping[int, float]
PolicyFunc = Callable[["LevelSelectionContext"], int | None]


def _parse_thresholds(env_value: str | None, fallback: ThresholdMap) -> dict[int, float]:
    """Parse level:threshold pairs from an env string."""
    if not env_value:
        return dict(fallback)
    result: dict[int, float] = {}
    for item in env_value.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            logger.warning("Invalid threshold token '%s'; expected level:value", item)
            continue
        level_part, value_part = item.split(':', 1)
        try:
            level = int(level_part)
            value = float(value_part)
        except ValueError:
            logger.warning("Invalid threshold entry '%s'; skipping", item)
            continue
        if value <= 0:
            logger.warning("Threshold for level %d must be > 0; skipping", level)
            continue
        result[level] = float(value)
    if not result:
        return dict(fallback)
    return result


@dataclass(frozen=True)
class LevelSelectionContext:
    levels: Sequence[LevelDescriptor]
    current_level: int
    intent_level: int | None
    level_oversampling: Mapping[int, float]
    thresholds: ThresholdMap | None = None


def select_by_oversampling(ctx: LevelSelectionContext) -> int | None:
    """Pick the finest level whose oversampling stays within configured thresholds."""
    if not ctx.levels:
        return ctx.intent_level

    default_thresholds = {0: 1.25, 1: 2.5}
    thresholds = dict(default_thresholds)
    thresholds.update(_parse_thresholds(os.getenv('NAPARI_CUDA_LEVEL_THRESHOLDS'), thresholds))
    if ctx.thresholds:
        thresholds.update({int(k): float(v) for k, v in ctx.thresholds.items()})

    current = int(ctx.current_level)
    intent = ctx.intent_level if ctx.intent_level is not None else current
    overs_map = {int(lvl): float(val) for lvl, val in ctx.level_oversampling.items() if val is not None}

    if ctx.levels:
        max_level = max(int(desc.index) for desc in ctx.levels)
        thresholds.setdefault(max_level, float('inf'))

    ordered = sorted({int(desc.index) for desc in ctx.levels})
    hysteresis = 0.1
    for level in ordered:
        overs = overs_map.get(level)
        if overs is None:
            continue
        limit = thresholds.get(level)
        if limit is None:
            continue
        if level < current:
            effective_limit = max(0.0, limit - hysteresis)
        elif level == current:
            effective_limit = limit + hysteresis
        else:
            effective_limit = limit
        if overs <= effective_limit:
            return level

    if intent in overs_map:
        return intent
    return current


def resolve_policy(name: str) -> PolicyFunc:
    # Map legacy and simplified names to the same selector
    return select_by_oversampling
