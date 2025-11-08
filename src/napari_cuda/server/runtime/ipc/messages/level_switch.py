"""Render worker intent messages delivered to the control thread."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from napari_cuda.server.scene.viewport import (
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
    oversampling: Mapping[int, float]
    timestamp: float
    zoom_ratio: float | None = None
    lock_level: int | None = None
    mode: RenderMode = RenderMode.PLANE
    plane_state: PlaneState | None = None
    volume_state: VolumeState | None = None
    level_shape: tuple[int, ...] | None = None


__all__ = ["LevelSwitchIntent"]
