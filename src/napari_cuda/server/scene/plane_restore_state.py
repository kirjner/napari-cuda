"""Immutable snapshot of the 2D plane state for mode toggles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from napari_cuda.server.runtime.scene_types import SliceROI


@dataclass(frozen=True)
class PlaneRestoreState:
    """Capture the 2D plane context prior to switching into volume mode."""

    step: tuple[int, ...]
    level: int
    roi_level: Optional[int] = None
    roi: Optional[SliceROI] = None
    rect: Optional[tuple[float, float, float, float]] = None
    zoom: Optional[float] = None
    center: Optional[tuple[float, float]] = None
    data_wh: Optional[tuple[int, int]] = None


__all__ = ["PlaneRestoreState"]
