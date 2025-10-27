"""Volume snapshot application helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

from vispy.scene.cameras import TurntableCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.scene_state_applier import SceneStateApplier

from .volume_state import assign_pose_from_snapshot, update_level, update_scale


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
    assign_pose_from_snapshot(volume_state, snapshot)

    center_source = snapshot.volume_center if snapshot.volume_center is not None else volume_state.pose.center
    assert center_source is not None and len(center_source) >= 3, "volume snapshot missing center"
    center = (
        float(center_source[0]),
        float(center_source[1]),
        float(center_source[2]),
    )
    cam.center = center


    angles_source = snapshot.volume_angles if snapshot.volume_angles is not None else volume_state.pose.angles
    assert angles_source is not None and len(angles_source) >= 2, "volume snapshot missing angles"
    roll_value = float(angles_source[2]) if len(angles_source) >= 3 else float(volume_state.pose.angles[2]) if volume_state.pose.angles is not None and len(volume_state.pose.angles) >= 3 else 0.0
    angles = (float(angles_source[0]), float(angles_source[1]), roll_value)
    cam.azimuth = angles[0]
    cam.elevation = angles[1]
    cam.roll = angles[2]

    distance_source = snapshot.volume_distance if snapshot.volume_distance is not None else volume_state.pose.distance
    assert distance_source is not None, "volume snapshot missing distance"
    distance_val = float(distance_source)
    cam.distance = distance_val
    fov_source = snapshot.volume_fov if snapshot.volume_fov is not None else volume_state.pose.fov
    assert fov_source is not None, "volume snapshot missing fov"
    fov_val = float(fov_source)
    cam.fov = fov_val

    volume_state.update_pose(center=center, angles=angles, distance=distance_val, fov=fov_val)


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
    ctx = worker._build_scene_state_context(cam)
    if ctx.volume_scale is None:
        ctx = replace(ctx, volume_scale=scale_tuple)

    data_wh, data_d = SceneStateApplier.apply_volume_layer(
        ctx,
        volume=volume,
        contrast=applied.contrast,
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
