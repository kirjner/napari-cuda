"""Immutable snapshot of the 2D plane state for mode toggles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from napari_cuda.server.state.scene_types import SliceROI


@dataclass(frozen=True)
class PlaneRestoreState:
    """Capture the 2D plane context prior to switching into volume mode."""

    step: Tuple[int, ...]
    level: int
    roi_level: Optional[int] = None
    roi: Optional[SliceROI] = None
    camera_state: Optional[Mapping[str, Any]] = None
    data_wh: Optional[Tuple[int, int]] = None


__all__ = ["PlaneRestoreState"]
