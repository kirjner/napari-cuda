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
class PlanePose:
    """Cached plane camera pose (rect/center/zoom)."""

    rect: Optional[tuple[float, float, float, float]] = None
    center: Optional[tuple[float, float]] = None
    zoom: Optional[float] = None


@dataclass
class PlaneState:
    """Controller targets, applied state, and reload flags for plane rendering."""

    # Controller intent (latest snapshot)
    target_level: int = 0
    target_ndisplay: int = 2
    target_step: Optional[tuple[int, ...]] = None
    snapshot_level: Optional[int] = None
    awaiting_level_confirm: bool = False
    pose: PlanePose = field(default_factory=PlanePose)

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

    def __post_init__(self) -> None:
        if not isinstance(self.pose, PlanePose):
            payload = dict(self.pose) if isinstance(self.pose, dict) else {}
            self.pose = PlanePose(**payload)

    def update_pose(
        self,
        *,
        rect: Optional[tuple[float, float, float, float]] = None,
        center: Optional[tuple[float, float]] = None,
        zoom: Optional[float] = None,
    ) -> None:
        if rect is not None:
            self.pose.rect = tuple(float(v) for v in rect)
        if center is not None:
            self.pose.center = tuple(float(v) for v in center)
        if zoom is not None:
            self.pose.zoom = float(zoom)

    def clear_pose(self) -> None:
        self.pose = PlanePose()


@dataclass
class VolumePose:
    """Cached volume camera pose (center/angles/distance/fov)."""

    center: Optional[tuple[float, float, float]] = None
    angles: Optional[tuple[float, float, float]] = None
    distance: Optional[float] = None
    fov: Optional[float] = None


@dataclass
class VolumeState:
    """Applied state for volume rendering."""

    level: int = 0
    downgraded: bool = False
    scale: Optional[tuple[float, float, float]] = None
    world_extents: Optional[tuple[float, float, float]] = None
    pose: VolumePose = field(default_factory=VolumePose)

    def __post_init__(self) -> None:
        if not isinstance(self.pose, VolumePose):
            payload = dict(self.pose) if isinstance(self.pose, dict) else {}
            self.pose = VolumePose(**payload)

    def update_pose(
        self,
        *,
        center: Optional[tuple[float, float, float]] = None,
        angles: Optional[tuple[float, float, float]] = None,
        distance: Optional[float] = None,
        fov: Optional[float] = None,
    ) -> None:
        if center is not None:
            self.pose.center = tuple(float(v) for v in center)
        if angles is not None:
            self.pose.angles = tuple(float(v) for v in angles)
        if distance is not None:
            self.pose.distance = float(distance)
        if fov is not None:
            self.pose.fov = float(fov)

    def clear_pose(self) -> None:
        self.pose = VolumePose()


@dataclass
class ViewportState:
    """Aggregated plane/volume state for the render worker."""

    plane: PlaneState = field(default_factory=PlaneState)
    volume: VolumeState = field(default_factory=VolumeState)
    mode: RenderMode = RenderMode.PLANE
    op_seq: int = -1


__all__ = ["PlanePose", "PlaneState", "RenderMode", "ViewportState", "VolumePose", "VolumeState"]
