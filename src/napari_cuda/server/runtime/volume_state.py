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

    center = snapshot.center
    if center is not None and len(center) >= 3:
        state.pose_center = (
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )

    angles = snapshot.angles
    if angles is not None and len(angles) >= 2:
        roll: float
        if len(angles) >= 3:
            roll = float(angles[2])
        elif state.pose_angles is not None and len(state.pose_angles) >= 3:
            roll = float(state.pose_angles[2])
        else:
            roll = 0.0

        state.pose_angles = (
            float(angles[0]),
            float(angles[1]),
            roll,
        )

    if snapshot.distance is not None:
        state.pose_distance = float(snapshot.distance)
    if snapshot.fov is not None:
        state.pose_fov = float(snapshot.fov)


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
