"""Volume snapshot application helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from vispy.scene.cameras import TurntableCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.viewport.layers import apply_volume_layer_data
from napari_cuda.server.runtime.viewport.volume_ops import (
    assign_pose_from_snapshot,
    apply_pose_to_camera,
    update_level,
    update_scale,
)


@dataclass(frozen=True)
class VolumeApplyResult:
    """Outcome of applying a volume snapshot."""

    level: int
    downgraded: bool
    data_wh: Tuple[int, int]
    data_d: Optional[int]
    scale: Tuple[float, float, float]


def apply_volume_camera_pose(
    worker: Any,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Apply volume camera pose from the snapshot to the active view."""

    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, TurntableCamera):
        return

    volume_state = worker.viewport_state.volume  # type: ignore[attr-defined]
    center, angles, distance, fov = assign_pose_from_snapshot(volume_state, snapshot)
    apply_pose_to_camera(
        cam,
        center=(float(center[0]), float(center[1]), float(center[2])),
        angles=(float(angles[0]), float(angles[1]), float(angles[2])),
        distance=float(distance),
        fov=float(fov),
    )


def apply_volume_level(
    worker: Any,
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
    worker._volume_scale = scale_tuple

    volume = worker._get_level_volume(source, applied.level)
    cam = worker.view.camera if getattr(worker, "view", None) is not None else None
    data_wh, data_d = apply_volume_layer_data(
        layer=getattr(worker, "_napari_layer", None),
        volume=volume,
        contrast=applied.contrast,
        scale=scale_tuple,
        ensure_volume_visual=getattr(worker, "_ensure_volume_visual", None),
    )
    worker._data_wh = data_wh
    worker._data_d = data_d

    shape = (
        (int(data_d), int(data_wh[1]), int(data_wh[0]))
        if data_d is not None
        else (int(data_wh[1]), int(data_wh[0]))
    )

    worker._layer_logger.log(
        enabled=worker._log_layer_debug,
        mode="volume",
        level=applied.level,
        z_index=None,
        shape=shape,
        contrast=applied.contrast,
        downgraded=downgraded,
    )

    volume_state = worker.viewport_state.volume  # type: ignore[attr-defined]
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
