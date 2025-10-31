"""Volume snapshot application helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from vispy.scene.cameras import TurntableCamera

import napari_cuda.server.data.lod as lod
from .interface import SnapshotInterface
from napari_cuda.server.runtime.viewport.layers import apply_volume_layer_data
from napari_cuda.server.runtime.viewport.volume_ops import (
    assign_pose_from_snapshot,
    apply_pose_to_camera,
    update_level,
    update_scale,
)
from napari_cuda.server.runtime.core.snapshot_build import RenderLedgerSnapshot


@dataclass(frozen=True)
class VolumeApplyResult:
    """Outcome of applying a volume snapshot."""

    level: int
    downgraded: bool
    data_wh: Tuple[int, int]
    data_d: Optional[int]
    scale: Tuple[float, float, float]


def apply_volume_camera_pose(
    snapshot_iface: SnapshotInterface,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Apply volume camera pose from the snapshot to the active view."""

    view = snapshot_iface.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, TurntableCamera):
        return

    volume_state = snapshot_iface.viewport_state.volume
    center, angles, distance, fov = assign_pose_from_snapshot(volume_state, snapshot)
    apply_pose_to_camera(
        cam,
        center=(float(center[0]), float(center[1]), float(center[2])),
        angles=(float(angles[0]), float(angles[1]), float(angles[2])),
        distance=float(distance),
        fov=float(fov),
    )


def apply_volume_level(
    snapshot_iface: SnapshotInterface,
    source: Any,
    applied: lod.LevelContext,
    *,
    downgraded: bool,
) -> VolumeApplyResult:
    """Load the volume level for ``applied`` and update worker metadata."""

    scale_vals = list(source.level_scale(applied.level)) if hasattr(source, "level_scale") else []
    scales = [float(s) for s in scale_vals[-3:]] if scale_vals else []
    while len(scales) < 3:
        scales.insert(0, 1.0)
    scale_tuple = (float(scales[-3]), float(scales[-2]), float(scales[-1]))
    snapshot_iface.set_volume_scale(scale_tuple)

    volume = snapshot_iface.load_volume(source, applied.level)
    data_wh, data_d = apply_volume_layer_data(
        layer=snapshot_iface.napari_layer,
        volume=volume,
        contrast=applied.contrast,
        scale=scale_tuple,
        ensure_volume_visual=snapshot_iface.ensure_volume_visual,
    )
    snapshot_iface.set_data_shape(int(data_wh[0]), int(data_wh[1]))
    snapshot_iface.set_data_depth(data_d)

    shape = (
        (int(data_d), int(data_wh[1]), int(data_wh[0]))
        if data_d is not None
        else (int(data_wh[1]), int(data_wh[0]))
    )

    layer_logger = snapshot_iface.layer_logger
    if layer_logger is not None:
        layer_logger.log(
            enabled=snapshot_iface.log_layer_debug,
            mode="volume",
            level=applied.level,
            z_index=None,
            shape=shape,
            contrast=applied.contrast,
            downgraded=downgraded,
        )

    volume_state = snapshot_iface.viewport_state.volume
    update_level(volume_state, int(applied.level), downgraded=downgraded)
    update_scale(volume_state, scale_tuple)

    return VolumeApplyResult(
        level=int(applied.level),
        downgraded=bool(downgraded),
        data_wh=(int(data_wh[0]), int(data_wh[1])),
        data_d=int(data_d) if data_d is not None else None,
        scale=scale_tuple,
    )


__all__ = ["VolumeApplyResult", "apply_volume_camera_pose", "apply_volume_level"]
