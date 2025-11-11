from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypedDict


# Plane restore cache ---------------------------------------------------------


@dataclass(frozen=True)
class PlaneRestoreCachePose:
    rect: tuple[float, float, float, float] | None
    center: tuple[float, float] | None
    zoom: float | None


@dataclass(frozen=True)
class PlaneRestoreCacheBlock:
    level: int
    index: tuple[int, ...]
    pose: PlaneRestoreCachePose


class PlaneRestoreCachePosePayload(TypedDict):
    rect: Sequence[float] | None
    center: Sequence[float] | None
    zoom: float | None


class PlaneRestoreCacheBlockPayload(TypedDict):
    level: int
    index: Sequence[int]
    pose: PlaneRestoreCachePosePayload


def plane_restore_cache_block_to_payload(state: PlaneRestoreCacheBlock) -> PlaneRestoreCacheBlockPayload:
    return {
        "level": int(state.level),
        "index": tuple(int(v) for v in state.index),
        "pose": {
            "rect": list(state.pose.rect) if state.pose.rect is not None else None,
            "center": list(state.pose.center) if state.pose.center is not None else None,
            "zoom": state.pose.zoom,
        },
    }


def plane_restore_cache_block_from_payload(
    data: dict[str, Any] | PlaneRestoreCacheBlockPayload,
) -> PlaneRestoreCacheBlock:
    level_value = data["level"]
    assert isinstance(level_value, int), "plane restore level missing"

    pose_payload = data["pose"]
    assert isinstance(pose_payload, dict), "plane restore pose missing"

    rect_value = pose_payload.get("rect")
    center_value = pose_payload.get("center")
    zoom_value = pose_payload.get("zoom")

    rect_tuple = _rect4(rect_value) if rect_value is not None else None
    center_tuple = _center2(center_value) if center_value is not None else None
    zoom_float = float(zoom_value) if zoom_value is not None else None

    index_value = data["index"]
    assert isinstance(index_value, Sequence), "plane restore index missing"
    for component in index_value:
        assert isinstance(component, int), "plane restore index entries must be integers"

    return PlaneRestoreCacheBlock(
        level=level_value,
        index=tuple(index_value),
        pose=PlaneRestoreCachePose(
            rect=rect_tuple,
            center=center_tuple,
            zoom=zoom_float,
        ),
    )


# Volume restore cache --------------------------------------------------------


@dataclass(frozen=True)
class VolumeRestoreCachePose:
    center: tuple[float, float, float] | None
    angles: tuple[float, float, float] | None
    distance: float | None
    fov: float | None


@dataclass(frozen=True)
class VolumeRestoreCacheBlock:
    level: int
    index: tuple[int, ...]
    pose: VolumeRestoreCachePose


class VolumeRestoreCachePosePayload(TypedDict):
    center: Sequence[float] | None
    angles: Sequence[float] | None
    distance: float | None
    fov: float | None


class VolumeRestoreCacheBlockPayload(TypedDict):
    level: int
    index: Sequence[int]
    pose: VolumeRestoreCachePosePayload


def volume_restore_cache_block_to_payload(state: VolumeRestoreCacheBlock) -> VolumeRestoreCacheBlockPayload:
    return {
        "level": int(state.level),
        "index": tuple(int(v) for v in state.index),
        "pose": {
            "center": list(state.pose.center) if state.pose.center is not None else None,
            "angles": list(state.pose.angles) if state.pose.angles is not None else None,
            "distance": state.pose.distance,
            "fov": state.pose.fov,
        },
    }


def volume_restore_cache_block_from_payload(
    data: dict[str, Any] | VolumeRestoreCacheBlockPayload,
) -> VolumeRestoreCacheBlock:
    level_value = data["level"]
    assert isinstance(level_value, int), "volume restore level missing"

    pose_payload = data["pose"]
    assert isinstance(pose_payload, dict), "volume restore pose missing"

    center_value = pose_payload.get("center")
    angles_value = pose_payload.get("angles")
    distance_value = pose_payload.get("distance")
    fov_value = pose_payload.get("fov")

    center_tuple = _center3(center_value) if center_value is not None else None
    angles_tuple = _angles3(angles_value) if angles_value is not None else None
    distance_float = float(distance_value) if distance_value is not None else None
    fov_float = float(fov_value) if fov_value is not None else None

    index_value = data["index"]
    assert isinstance(index_value, Sequence), "volume restore index missing"
    for component in index_value:
        assert isinstance(component, int), "volume restore index entries must be integers"

    return VolumeRestoreCacheBlock(
        level=level_value,
        index=tuple(index_value),
        pose=VolumeRestoreCachePose(
            center=center_tuple,
            angles=angles_tuple,
            distance=distance_float,
            fov=fov_float,
        ),
    )


__all__ = [
    "PlaneRestoreCachePose",
    "PlaneRestoreCacheBlock",
    "PlaneRestoreCachePosePayload",
    "PlaneRestoreCacheBlockPayload",
    "plane_restore_cache_block_to_payload",
    "plane_restore_cache_block_from_payload",
    "VolumeRestoreCachePose",
    "VolumeRestoreCacheBlock",
    "VolumeRestoreCachePosePayload",
    "VolumeRestoreCacheBlockPayload",
    "volume_restore_cache_block_to_payload",
    "volume_restore_cache_block_from_payload",
]


def _rect4(value: Sequence[float]) -> tuple[float, float, float, float]:
    rect = _float_list(value)
    assert len(rect) == 4, "plane rect requires four components"
    return (rect[0], rect[1], rect[2], rect[3])


def _center2(value: Sequence[float]) -> tuple[float, float]:
    center = _float_list(value)
    assert len(center) == 2, "plane center requires two components"
    return (center[0], center[1])


def _center3(value: Sequence[float]) -> tuple[float, float, float]:
    center = _float_list(value)
    assert len(center) == 3, "volume center requires three components"
    return (center[0], center[1], center[2])


def _angles3(value: Sequence[float]) -> tuple[float, float, float]:
    angles = _float_list(value)
    assert len(angles) >= 2, "volume angles require at least two components"
    third = angles[2] if len(angles) >= 3 else 0.0
    return (angles[0], angles[1], third)


def _float_list(value: Sequence[float]) -> list[float]:
    assert not isinstance(value, (str, bytes, bytearray)), "sequence payload must not be string"
    return [float(component) for component in value]
