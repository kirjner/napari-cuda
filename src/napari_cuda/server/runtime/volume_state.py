"""Helpers for mutating :class:`VolumeState` from runtime snapshots."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot

from .state_structs import VolumeState


def assign_pose_from_snapshot(
    state: VolumeState,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Populate volume pose fields from a controller snapshot."""

    update_kwargs: dict[str, object] = {}
    # The snapshot may omit components the worker already applied (e.g. a
    # distance-only update); merge in the cached pose so we do not stomp fields
    # the controller did not touch.
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
    scale: Tuple[float, float, float],
) -> None:
    """Persist the volume scale tuple on the state."""

    state.scale = (
        float(scale[0]),
        float(scale[1]),
        float(scale[2]),
    )


__all__ = ["assign_pose_from_snapshot", "update_level", "update_scale"]
