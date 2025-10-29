"""Render worker intent messages delivered to the control thread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from napari_cuda.server.data.lod import LevelContext
from napari_cuda.server.runtime.viewport import (
    PlaneState,
    RenderMode,
    VolumeState,
)


@dataclass(frozen=True)
class LevelSwitchIntent:
    """Request controller to stage a multiscale level change."""

    desired_level: int
    selected_level: int
    reason: str
    previous_level: int
    context: LevelContext
    oversampling: Mapping[int, float]
    timestamp: float
    downgraded: bool
    zoom_ratio: float | None = None
    lock_level: int | None = None
    mode: RenderMode = RenderMode.PLANE
    plane_state: PlaneState | None = None
    volume_state: VolumeState | None = None


__all__ = ["LevelSwitchIntent"]
