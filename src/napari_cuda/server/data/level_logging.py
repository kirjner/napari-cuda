"""Logging helpers for multiscale level operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerAssignmentLogger:
    """Deduplicate verbose layer assignment logs."""

    logger_ref: logging.Logger
    _last_key: Optional[tuple[str, int, int]] = field(default=None, init=False)

    def log(
        self,
        *,
        enabled: bool,
        mode: str,
        level: int,
        z_index: Optional[int],
        shape: tuple[int, ...],
        contrast: tuple[float, float],
    ) -> None:
        if not enabled or not self.logger_ref.isEnabledFor(logging.INFO):
            return
        key = (mode, int(level), int(z_index) if z_index is not None else -1)
        if key == self._last_key:
            return
        self._last_key = key
        self.logger_ref.info(
            "layer assign: mode=%s level=%d z=%s shape=%s contrast=(%.4f, %.4f)",
            mode,
            int(level),
            z_index if z_index is not None else "na",
            "x".join(str(int(x)) for x in shape),
            float(contrast[0]),
            float(contrast[1]),
        )


@dataclass
class LevelSwitchLogger:
    """Log multiscale switch events."""

    logger_ref: logging.Logger

    def log(
        self,
        *,
        enabled: bool,
        previous: int,
        applied: int,
        roi_desc: str,
        reason: str,
        elapsed_ms: float,
    ) -> None:
        if not enabled or int(previous) == int(applied):
            return
        self.logger_ref.info(
            "ms.switch: level=%d->%d roi=%s reason=%s elapsed=%.2fms",
            int(previous),
            int(applied),
            roi_desc,
            reason,
            float(elapsed_ms),
        )


__all__ = [
    "LayerAssignmentLogger",
    "LevelSwitchLogger",
]
