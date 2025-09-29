"""Policy surface for multiscale level selection via oversampling thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .zarr_source import LevelDescriptor

logger = logging.getLogger(__name__)


ThresholdMap = Mapping[int, float]
PolicyFunc = Callable[["LevelSelectionContext"], int | None]


_DEFAULT_OVERSAMPLING_THRESHOLDS: dict[int, float] = {0: 1.25, 1: 2.5}


@dataclass(frozen=True)
class LevelSelectionContext:
    levels: Sequence[LevelDescriptor]
    current_level: int
    requested_level: int | None
    level_oversampling: Mapping[int, float]
    thresholds: ThresholdMap | None = None
    hysteresis: float = 0.1


def select_by_oversampling(ctx: LevelSelectionContext) -> int | None:
    """Pick the finest level whose oversampling stays within configured thresholds."""
    if not ctx.levels:
        return ctx.requested_level

    thresholds = dict(_DEFAULT_OVERSAMPLING_THRESHOLDS)
    if ctx.thresholds:
        thresholds.update({int(k): float(v) for k, v in ctx.thresholds.items()})

    current = int(ctx.current_level)
    requested = ctx.requested_level if ctx.requested_level is not None else current
    overs_map = {int(lvl): float(val) for lvl, val in ctx.level_oversampling.items() if val is not None}

    if ctx.levels:
        max_level = max(int(desc.index) for desc in ctx.levels)
        thresholds.setdefault(max_level, float('inf'))

    ordered = sorted({int(desc.index) for desc in ctx.levels})
    hysteresis = float(ctx.hysteresis)
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

    if requested in overs_map:
        return requested
    return current


def resolve_policy(name: str) -> PolicyFunc:
    # Map legacy and simplified names to the same selector
    return select_by_oversampling
