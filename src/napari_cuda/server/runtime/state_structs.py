"""Shared viewport state structures for plane and volume rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from napari_cuda.server.runtime.scene_types import SliceROI


class RenderMode(Enum):
    """Viewport render mode."""

    PLANE = auto()
    VOLUME = auto()


@dataclass
class PlaneState:
    """Controller targets, applied state, and reload flags for plane rendering."""

    # Controller intent (latest snapshot)
    target_level: int = 0
    target_ndisplay: int = 2
    target_step: Optional[tuple[int, ...]] = None
    snapshot_level: Optional[int] = None
    awaiting_level_confirm: bool = False

    # Camera metadata
    camera_rect: Optional[tuple[float, float, float, float]] = None
    camera_center: Optional[tuple[float, float]] = None
    camera_zoom: Optional[float] = None

    # Applied state
    applied_level: int = 0
    applied_step: Optional[tuple[int, ...]] = None
    applied_roi: Optional[SliceROI] = None
    applied_roi_signature: Optional[tuple[int, int, int, int]] = None

    # Pending reload work
    pending_roi: Optional[SliceROI] = None
    pending_roi_signature: Optional[tuple[int, int, int, int]] = None
    level_reload_required: bool = False
    roi_reload_required: bool = False

    # Camera pose signals
    pose_reason: Optional[str] = None
    zoom_hint: Optional[float] = None
    camera_pose_dirty: bool = False


@dataclass
class VolumeState:
    """Applied state for volume rendering."""

    level: int = 0
    downgraded: bool = False
    pose_center: Optional[tuple[float, float, float]] = None
    pose_angles: Optional[tuple[float, float, float]] = None
    pose_distance: Optional[float] = None
    pose_fov: Optional[float] = None
    scale: Optional[tuple[float, float, float]] = None
    world_extents: Optional[tuple[float, float, float]] = None


@dataclass
class ViewportState:
    """Aggregated plane/volume state for the render worker."""

    plane: PlaneState = field(default_factory=PlaneState)
    volume: VolumeState = field(default_factory=VolumeState)
    mode: RenderMode = RenderMode.PLANE
    op_seq: int = -1


__all__ = ["PlaneState", "RenderMode", "ViewportState", "VolumeState"]
