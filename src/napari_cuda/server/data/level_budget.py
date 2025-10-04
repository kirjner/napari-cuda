"""Budget helpers for multiscale level switching."""

from __future__ import annotations

from typing import Callable, Optional
import logging
import time

from napari_cuda.server.data.zarr_source import ZarrSceneSource


logger = logging.getLogger(__name__)


class LevelBudgetError(RuntimeError):
    """Raised when a multiscale level exceeds configured budgets."""


def apply_level_with_budget(
    *,
    desired_level: int,
    use_volume: bool,
    source: ZarrSceneSource,
    current_level: int,
    log_layer_debug: bool,
    budget_check: Callable[[ZarrSceneSource, int], None],
    apply_level_cb: Callable[[ZarrSceneSource, int, Optional[int]], None],
    on_switch: Callable[[int, int, float], None],
    logger_ref: logging.Logger = logger,
) -> tuple[int, bool]:
    """Pick the finest level allowed by budgets, apply it, and report downgrade."""

    level_count = len(source.level_descriptors)
    if level_count == 0:
        return current_level, False

    desired_level = max(0, min(int(desired_level), level_count - 1))
    candidates = range(desired_level, level_count)
    total = len(candidates)

    for idx, level in enumerate(candidates):
        try:
            budget_check(source, level)
        except LevelBudgetError:
            if idx == total - 1:
                raise
            if log_layer_debug:
                logger_ref.info(
                    "budget reject: mode=%s level=%d",
                    'volume' if use_volume else 'slice',
                    int(level),
                )
            continue

        start = time.perf_counter()
        apply_level_cb(source, level, current_level)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        on_switch(current_level, level, elapsed_ms)
        downgraded = level != desired_level
        if downgraded and log_layer_debug:
            logger_ref.info(
                "level downgrade: requested=%d active=%d",
                desired_level,
                level,
            )
        return level, downgraded

    raise RuntimeError("Unable to select multiscale level within budget")


__all__ = [
    "LevelBudgetError",
    "apply_level_with_budget",
]
