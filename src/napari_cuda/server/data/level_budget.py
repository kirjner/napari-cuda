"""Budget helpers for multiscale level switching."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Optional

import numpy as np

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
) -> int:
    """Pick the finest level allowed by budgets and apply it."""

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
        return level

    raise RuntimeError("Unable to select multiscale level within budget")


def select_volume_level(
    source: ZarrSceneSource,
    requested_level: int,
    *,
    max_voxels: Optional[int],
    max_bytes: Optional[int],
    error_cls: type[Exception] = LevelBudgetError,
) -> int:
    """Clamp ``requested_level`` against voxel/byte budgets using source metadata."""

    level_shapes = [descriptor.shape for descriptor in source.level_descriptors]
    if not level_shapes:
        raise error_cls("select_volume_level requires at least one multiscale level")

    max_index = len(level_shapes) - 1
    clamped_request = max(0, min(int(requested_level), max_index))
    itemsize = int(np.dtype(source.dtype).itemsize)

    def _fits(level_idx: int) -> tuple[bool, int, int]:
        shape = level_shapes[level_idx]
        voxels = 1
        for dim in shape:
            voxels *= max(1, int(dim))
        voxels = int(voxels)
        bytes_est = int(voxels * itemsize)
        vox_cap = int(max_voxels) if max_voxels else 0
        bytes_cap = int(max_bytes) if max_bytes else 0
        if vox_cap and voxels > vox_cap:
            return False, voxels, vox_cap
        if bytes_cap and bytes_est > bytes_cap:
            return False, bytes_est, bytes_cap
        return True, voxels, bytes_est

    fits_request, estimate, cap = _fits(clamped_request)
    if fits_request:
        return int(clamped_request)

    coarsest = max_index
    fits_coarse, coarse_estimate, coarse_cap = _fits(coarsest)
    if fits_coarse:
        return int(coarsest)

    if max_voxels and estimate > max_voxels:
        msg = f"voxels={estimate} exceeds cap={int(max_voxels)}"
    elif max_bytes and estimate > max_bytes:
        msg = f"bytes={estimate} exceeds cap={int(max_bytes)}"
    else:
        msg = "requested level exceeds configured budgets"
    raise error_cls(msg)


__all__ = [
    "LevelBudgetError",
    "apply_level_with_budget",
    "select_volume_level",
]
