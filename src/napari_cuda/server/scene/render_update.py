"""Shared payloads for communicating render updates to the worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import RenderLedgerSnapshot
from .viewport import PlaneViewportCache, RenderMode, VolumeViewportCache


@dataclass(frozen=True)
class RenderUpdate:
    """Latest-wins state drained by the render worker."""

    scene_state: Optional[RenderLedgerSnapshot]
    mode: Optional[RenderMode] = None
    plane_state: Optional[PlaneViewportCache] = None
    volume_state: Optional[VolumeViewportCache] = None
    op_seq: Optional[int] = None


__all__ = ["RenderUpdate"]
