"""Helpers for mutating :class:`VolumeState` during snapshot application."""

from __future__ import annotations

from napari_cuda.server.runtime.core.snapshot_build import RenderLedgerSnapshot

from .state import VolumeState


def assign_pose_from_snapshot(
    state: VolumeState,
    snapshot: RenderLedgerSnapshot,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
]:
    """Populate volume pose fields from a controller snapshot and return the pose."""

    update_kwargs: dict[str, object] = {}
    center = snapshot.volume_center
    if center is None and state.pose.center is not None:
        center = state.pose.center
    if center is not None and len(center) >= 3:
        update_kwargs["center"] = (
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )

    angles = snapshot.volume_angles
    if angles is None and state.pose.angles is not None:
        angles = state.pose.angles
    if angles is not None and len(angles) >= 2:
        roll: float
        if len(angles) >= 3:
            roll = float(angles[2])
        elif state.pose.angles is not None and len(state.pose.angles) >= 3:
            roll = float(state.pose.angles[2])
        else:
            roll = 0.0

        update_kwargs["angles"] = (
            float(angles[0]),
            float(angles[1]),
            roll,
        )

    distance_value = snapshot.volume_distance
    if distance_value is None and state.pose.distance is not None:
        distance_value = state.pose.distance
    if distance_value is not None:
        update_kwargs["distance"] = float(distance_value)

    fov_value = snapshot.volume_fov
    if fov_value is None and state.pose.fov is not None:
        fov_value = state.pose.fov
    if fov_value is not None:
        update_kwargs["fov"] = float(fov_value)

    if update_kwargs:
        state.update_pose(**update_kwargs)
    assert state.pose.center is not None, "volume pose missing center"
    assert state.pose.angles is not None, "volume pose missing angles"
    assert state.pose.distance is not None, "volume pose missing distance"
    assert state.pose.fov is not None, "volume pose missing fov"
    return (
        state.pose.center,
        state.pose.angles,
        float(state.pose.distance),
        float(state.pose.fov),
    )


def update_level(
    state: VolumeState,
    level: int,
    *,
    downgraded: bool,
) -> None:
    """Update the level metadata stored by the volume state."""

    state.level = int(level)
    state.downgraded = bool(downgraded)


def update_scale(
    state: VolumeState,
    scale: tuple[float, float, float],
) -> None:
    """Persist the volume scale tuple on the state."""

    state.scale = (
        float(scale[0]),
        float(scale[1]),
        float(scale[2]),
    )

def apply_pose_to_camera(
    camera,
    *,
    center: tuple[float, float, float],
    angles: tuple[float, float, float],
    distance: float,
    fov: float,
) -> None:
    """Apply the provided pose data to a turntable camera."""

    camera.center = center  # type: ignore[attr-defined]
    camera.azimuth = float(angles[0])  # type: ignore[attr-defined]
    camera.elevation = float(angles[1])  # type: ignore[attr-defined]
    roll_value = float(angles[2]) if len(angles) >= 3 else 0.0
    camera.roll = roll_value  # type: ignore[attr-defined]
    camera.distance = float(distance)  # type: ignore[attr-defined]
    camera.fov = float(fov)  # type: ignore[attr-defined]


__all__ = ["apply_pose_to_camera", "assign_pose_from_snapshot", "update_level", "update_scale"]
