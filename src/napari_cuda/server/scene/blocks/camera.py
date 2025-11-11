from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class PlaneCameraBlock:
    rect: tuple[float, float, float, float] | None
    center: tuple[float, float] | None
    zoom: float | None


@dataclass(frozen=True)
class VolumeCameraBlock:
    center: tuple[float, float, float] | None
    angles: tuple[float, float, float] | None
    distance: float | None
    fov: float | None


@dataclass(frozen=True)
class CameraBlock:
    plane: PlaneCameraBlock
    volume: VolumeCameraBlock


class PlaneCameraPayload(TypedDict):
    # pose fields remain optional during bootstrap/reset flows
    rect: tuple[float, float, float, float] | None
    center: tuple[float, float] | None
    zoom: float | None


class VolumeCameraPayload(TypedDict):
    # turntable pose is also optional until the worker publishes it
    center: tuple[float, float, float] | None
    angles: tuple[float, float, float] | None
    distance: float | None
    fov: float | None


class CameraBlockPayload(TypedDict):
    plane: PlaneCameraPayload
    volume: VolumeCameraPayload


def plane_camera_to_payload(block: PlaneCameraBlock) -> PlaneCameraPayload:
    return {
        "rect": list(block.rect) if block.rect is not None else None,
        "center": list(block.center) if block.center is not None else None,
        "zoom": block.zoom,
    }


def volume_camera_to_payload(block: VolumeCameraBlock) -> VolumeCameraPayload:
    return {
        "center": list(block.center) if block.center is not None else None,
        "angles": list(block.angles) if block.angles is not None else None,
        "distance": block.distance,
        "fov": block.fov,
    }


def camera_block_to_payload(block: CameraBlock) -> CameraBlockPayload:
    return {
        "plane": plane_camera_to_payload(block.plane),
        "volume": volume_camera_to_payload(block.volume),
    }


def plane_camera_from_payload(data: PlaneCameraPayload) -> PlaneCameraBlock:
    return PlaneCameraBlock(rect=data["rect"], center=data["center"], zoom=data["zoom"])


def volume_camera_from_payload(data: VolumeCameraPayload) -> VolumeCameraBlock:
    return VolumeCameraBlock(center=data["center"], angles=data["angles"], distance=data["distance"], fov=data["fov"])


def camera_block_from_payload(data: CameraBlockPayload) -> CameraBlock:
    return CameraBlock(plane=plane_camera_from_payload(data["plane"]), volume=volume_camera_from_payload(data["volume"]))
