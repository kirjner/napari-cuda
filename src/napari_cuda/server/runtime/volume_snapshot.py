"""Volume snapshot application helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from vispy.scene.cameras import TurntableCamera
from napari.layers.image._image_constants import ImageRendering as NapariImageRendering

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot

from .volume_state import assign_pose_from_snapshot, update_level, update_scale

logger = logging.getLogger(__name__)


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
    data_wh, data_d = _apply_volume_to_layer(
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


def _apply_volume_to_layer(
    *,
    layer: Any,
    volume: Any,
    contrast: Tuple[float, float],
    scale: Tuple[float, float, float],
    ensure_volume_visual: Optional[Callable[[], Any]] = None,
) -> Tuple[Tuple[int, int], Optional[int]]:
    """Apply the provided volume data to the active napari layer."""

    if ensure_volume_visual is not None:
        ensure_volume_visual()
    if layer is not None:
        layer.depiction = "volume"  # type: ignore[assignment]
        layer.rendering = NapariImageRendering.MIP.value  # type: ignore[assignment]
        layer.data = volume

        lo = float(contrast[0])
        hi = float(contrast[1])
        if hi <= lo:
            hi = lo + 1.0

        layer.translate = tuple(0.0 for _ in range(int(volume.ndim)))  # type: ignore[assignment]

        data_min = float(np.nanmin(volume)) if hasattr(np, "nanmin") else float(np.min(volume))
        data_max = float(np.nanmax(volume)) if hasattr(np, "nanmax") else float(np.max(volume))
        normalized = (-0.05 <= data_min <= 1.05) and (-0.05 <= data_max <= 1.05)
        logger.debug(
            "volume.apply stats: min=%.6f max=%.6f contrast=(%.6f, %.6f) normalized=%s",
            data_min,
            data_max,
            lo,
            hi,
            normalized,
        )
        if normalized:
            layer.contrast_limits = [0.0, 1.0]  # type: ignore[assignment]
        else:
            layer.contrast_limits = [lo, hi]  # type: ignore[assignment]

        scale_vals: Tuple[float, ...] = tuple(float(s) for s in scale)
        if len(scale_vals) < int(volume.ndim):
            pad = int(volume.ndim) - len(scale_vals)
            scale_vals = tuple(1.0 for _ in range(pad)) + scale_vals
        scale_vals = tuple(scale_vals[-int(volume.ndim):])
        layer.scale = scale_vals  # type: ignore[assignment]

    depth = int(volume.shape[0])
    height = int(volume.shape[1]) if int(volume.ndim) >= 2 else int(volume.shape[-1])
    width = int(volume.shape[2]) if int(volume.ndim) >= 3 else int(volume.shape[-1])
    return (int(width), int(height)), depth
