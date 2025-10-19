"""Render worker intent messages delivered to the control thread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class LevelSwitchIntent:
    """Request controller to stage a multiscale level change."""

    desired_level: int
    step: Tuple[int, ...]
    reason: str
    previous_level: int
    oversampling: Mapping[int, float]
    timestamp: float
    zoom_ratio: float | None = None
    lock_level: int | None = None


__all__ = ["LevelSwitchIntent"]
