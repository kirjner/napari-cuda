"""Shared payload structures for camera pose callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CameraPoseApplied:
    """Camera pose reported by the render worker after applying deltas."""

    target: str
    command_seq: int
    center: Optional[tuple[float, ...]]
    zoom: Optional[float]
    angles: Optional[tuple[float, ...]]
    distance: Optional[float]
    fov: Optional[float]
    rect: Optional[tuple[float, float, float, float]] = None


__all__ = ["CameraPoseApplied"]
