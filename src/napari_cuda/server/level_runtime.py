"""Worker-oriented helpers for applying multiscale levels."""

from __future__ import annotations

from typing import Optional
import logging

import numpy as np

import napari_cuda.server.lod as lod
from napari_cuda.server.scene_state_applier import SceneStateApplier
from napari_cuda.server.zarr_source import ZarrSceneSource

logger = logging.getLogger(__name__)


def apply_worker_level(
    worker: object,
    source: ZarrSceneSource,
    level: int,
    *,
    prev_level: Optional[int] = None,
) -> lod.AppliedLevel:
    """Apply ``level`` through the worker's existing helpers and return the snapshot."""

    applied = lod.apply_level(
        source=source,
        target_level=int(level),
        prev_level=prev_level,
        last_step=getattr(worker, "_last_step", None),
        viewer=getattr(worker, "_viewer", None),
    )

    descriptor = source.level_descriptors[int(level)]
    worker._update_level_metadata(descriptor, applied)  # type: ignore[attr-defined]

    if getattr(worker, "use_volume", False):
        apply_worker_volume_level(worker, source, applied)
    else:
        apply_worker_slice_level(worker, source, applied)

    worker._notify_scene_refresh()  # type: ignore[attr-defined]
    return applied


def apply_worker_volume_level(
    worker: object,
    source: ZarrSceneSource,
    applied: lod.AppliedLevel,
) -> None:
    """Apply a volume level and emit layer logging via the worker hooks."""

    volume = worker._get_level_volume(source, applied.level)  # type: ignore[attr-defined]
    cam = worker.view.camera if getattr(worker, "view", None) is not None else None
    ctx = worker._build_scene_state_context(cam)  # type: ignore[attr-defined]
    data_wh, data_d = SceneStateApplier.apply_volume_layer(
        ctx,
        volume=volume,
        contrast=applied.contrast,
    )
    worker._data_wh = data_wh  # type: ignore[attr-defined]
    worker._data_d = data_d  # type: ignore[attr-defined]
    volume_shape = (
        (int(data_d), int(data_wh[1]), int(data_wh[0]))
        if data_d is not None
        else (int(data_wh[1]), int(data_wh[0]))
    )
    worker._layer_logger.log(  # type: ignore[attr-defined]
        enabled=worker._log_layer_debug,  # type: ignore[attr-defined]
        mode="volume",
        level=applied.level,
        z_index=None,
        shape=volume_shape,
        contrast=applied.contrast,
        downgraded=worker._level_downgraded,  # type: ignore[attr-defined]
    )


def apply_worker_slice_level(
    worker: object,
    source: ZarrSceneSource,
    applied: lod.AppliedLevel,
) -> None:
    """Apply a slice level and refresh the napari layer state."""

    layer = getattr(worker, "_napari_layer", None)
    sy, sx = applied.scale_yx
    if layer is not None:
        try:
            layer.scale = (sy, sx)
        except Exception:
            logger.debug("apply_level: setting 2D layer scale pre-slab failed", exc_info=True)

    z_idx = int(getattr(worker, "_z_index", 0) or 0)
    slab = worker._load_slice(source, applied.level, z_idx)  # type: ignore[attr-defined]
    roi_for_layer = None
    last_roi = getattr(worker, "_last_roi", None)
    if last_roi is not None and int(last_roi[0]) == int(applied.level):
        roi_for_layer = last_roi[1]

    if layer is not None:
        view = getattr(worker, "view", None)
        assert view is not None
        ctx = worker._build_scene_state_context(view.camera)  # type: ignore[attr-defined]
        SceneStateApplier.apply_slice_to_layer(
            ctx,
            source=source,
            slab=slab,
            roi=roi_for_layer,
            update_contrast=not getattr(worker, "_sticky_contrast", False),
        )

    h, w = int(slab.shape[0]), int(slab.shape[1])
    worker._data_wh = (w, h)  # type: ignore[attr-defined]
    worker._data_d = None  # type: ignore[attr-defined]

    view = getattr(worker, "view", None)
    if (
        not getattr(worker, "_preserve_view_on_switch", False)
        and view is not None
        and getattr(view, "camera", None) is not None
    ):
        world_w = float(w) * float(max(1e-12, sx))
        world_h = float(h) * float(max(1e-12, sy))
        view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))

    worker._layer_logger.log(  # type: ignore[attr-defined]
        enabled=worker._log_layer_debug,  # type: ignore[attr-defined]
        mode="slice",
        level=applied.level,
        z_index=getattr(worker, "_z_index", None),
        shape=(int(h), int(w)),
        contrast=applied.contrast,
        downgraded=worker._level_downgraded,  # type: ignore[attr-defined]
    )


def format_worker_level_roi(
    worker: object,
    source: ZarrSceneSource,
    level: int,
) -> str:
    """Return the stringified ROI description for logging."""

    if getattr(worker, "use_volume", False):
        return "volume"
    roi = worker._viewport_roi_for_level(source, level)  # type: ignore[attr-defined]
    if roi.is_empty():
        return "full"
    return f"y={roi.y_start}:{roi.y_stop} x={roi.x_start}:{roi.x_stop}"


__all__ = [
    "apply_worker_level",
    "apply_worker_volume_level",
    "apply_worker_slice_level",
    "format_worker_level_roi",
]
