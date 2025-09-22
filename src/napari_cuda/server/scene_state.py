from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ServerSceneState:
    """Minimal scene state snapshot applied atomically per frame."""

    center: Optional[tuple[float, float, float]] = None
    zoom: Optional[float] = None
    angles: Optional[tuple[float, float, float]] = None
    current_step: Optional[tuple[int, ...]] = None
    volume_mode: Optional[str] = None
    volume_colormap: Optional[str] = None
    volume_clim: Optional[tuple[float, float]] = None
    volume_opacity: Optional[float] = None
    volume_sample_step: Optional[float] = None


__all__ = ["ServerSceneState"]
