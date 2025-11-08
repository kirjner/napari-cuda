"""Shared viewport state structures for control and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from napari_cuda.server.data import SliceROI


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


class PoseEvent(Enum):
    """Signals that the current camera pose should be emitted."""

    LEVEL_RELOAD = "level-reload"
    ROI_RELOAD = "roi-reload"
    CAMERA_DELTA = "camera-delta"


@dataclass
class PlaneRequest:
    """Controller request for plane rendering."""

    level: int = 0
    step: Optional[tuple[int, ...]] = None
    ndisplay: int = 2
    snapshot_level: Optional[int] = None
    awaiting_level_confirm: bool = False


@dataclass
class PlaneResult:
    """Last applied plane state on the worker."""

    level: int = 0
    step: Optional[tuple[int, ...]] = None
    roi_signature: Optional[tuple[int, int, int, int]] = None


@dataclass
class PlaneState:
    """Controller targets, applied state, and reload flags for plane rendering."""

    request: PlaneRequest = field(default_factory=PlaneRequest)
    applied: PlaneResult = field(default_factory=PlaneResult)
    pose: PlanePose = field(default_factory=PlanePose)
    zoom_hint: Optional[float] = None
    camera_dirty: bool = False
    _last_roi: Optional[SliceROI] = None

    def __post_init__(self) -> None:
        if not isinstance(self.pose, PlanePose):
            payload = dict(self.pose) if isinstance(self.pose, dict) else {}
            self.pose = PlanePose(**payload)
        if not isinstance(self.request, PlaneRequest):
            if not isinstance(self.request, dict):
                raise TypeError(f"Expected mapping for PlaneRequest, got {type(self.request)!r}")
            self.request = PlaneRequest(**dict(self.request))
        if not isinstance(self.applied, PlaneResult):
            payload = dict(self.applied) if isinstance(self.applied, dict) else {}
            self.applied = PlaneResult(**payload)

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

    # Convenience accessors for legacy call sites ---------------------------------
    @property
    def target_level(self) -> int:
        return int(self.request.level)

    @target_level.setter
    def target_level(self, value: int) -> None:
        self.request.level = int(value)

    @property
    def target_ndisplay(self) -> int:
        return int(self.request.ndisplay)

    @target_ndisplay.setter
    def target_ndisplay(self, value: int) -> None:
        self.request.ndisplay = int(value)

    @property
    def target_step(self) -> Optional[tuple[int, ...]]:
        return self.request.step

    @target_step.setter
    def target_step(self, value: Optional[tuple[int, ...]]) -> None:
        self.request.step = tuple(int(v) for v in value) if value is not None else None

    @property
    def snapshot_level(self) -> Optional[int]:
        return self.request.snapshot_level

    @snapshot_level.setter
    def snapshot_level(self, value: Optional[int]) -> None:
        self.request.snapshot_level = int(value) if value is not None else None

    @property
    def awaiting_level_confirm(self) -> bool:
        return bool(self.request.awaiting_level_confirm)

    @awaiting_level_confirm.setter
    def awaiting_level_confirm(self, value: bool) -> None:
        self.request.awaiting_level_confirm = bool(value)

    @property
    def applied_level(self) -> int:
        return int(self.applied.level)

    @applied_level.setter
    def applied_level(self, value: int) -> None:
        self.applied.level = int(value)

    @property
    def applied_step(self) -> Optional[tuple[int, ...]]:
        return self.applied.step

    @applied_step.setter
    def applied_step(self, value: Optional[tuple[int, ...]]) -> None:
        self.applied.step = tuple(int(v) for v in value) if value is not None else None

    @property
    def applied_roi_signature(self) -> Optional[tuple[int, int, int, int]]:
        return self.applied.roi_signature

    @applied_roi_signature.setter
    def applied_roi_signature(self, value: Optional[tuple[int, int, int, int]]) -> None:
        self.applied.roi_signature = tuple(int(v) for v in value) if value is not None else None

    @property
    def applied_roi(self) -> Optional[SliceROI]:
        return self._last_roi

    @applied_roi.setter
    def applied_roi(self, value: Optional[SliceROI]) -> None:
        self._last_roi = value

    @property
    def camera_pose_dirty(self) -> bool:
        return bool(self.camera_dirty)

    @camera_pose_dirty.setter
    def camera_pose_dirty(self, value: bool) -> None:
        self.camera_dirty = bool(value)


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


__all__ = [
    "PlanePose",
    "PlaneRequest",
    "PlaneResult",
    "PlaneState",
    "PoseEvent",
    "RenderMode",
    "ViewportState",
    "VolumePose",
    "VolumeState",
]
