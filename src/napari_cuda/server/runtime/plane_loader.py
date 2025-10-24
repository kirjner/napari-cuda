"""Plane slice loading helpers for render snapshot application."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from napari_cuda.server.data.roi import (
    resolve_worker_viewport_roi,
    viewport_debug_snapshot,
)
from napari_cuda.server.runtime.roi_math import chunk_shape_for_level
from napari_cuda.server.runtime.scene_state_applier import SceneStateApplier
from napari_cuda.server.runtime.scene_types import SliceROI

logger = logging.getLogger(__name__)


def viewport_roi_for_level(
    worker: Any,
    source: Any,
    level: int,
    *,
    quiet: bool = False,
    for_policy: bool = False,
) -> SliceROI:
    """Compute the viewport ROI for the requested multiscale level.

    Mirrors the legacy helper from ``render_snapshot`` so Stage B refactors can
    delegate ROI alignment decisions through this module.
    """

    view = worker.view
    align_chunks = (not for_policy) and bool(worker._roi_align_chunks)
    ensure_contains = (not for_policy) and bool(worker._roi_ensure_contains_viewport)
    edge_threshold = int(worker._roi_edge_threshold)
    chunk_pad = int(worker._roi_pad_chunks)

    roi_log = worker._roi_log_state
    log_state = roi_log if isinstance(roi_log, dict) else None

    data_wh = worker._data_wh

    def _snapshot() -> dict[str, Any]:
        return viewport_debug_snapshot(
            view=view,
            canvas_size=(int(worker.width), int(worker.height)),  # type: ignore[attr-defined]
            data_wh=data_wh,
            data_depth=worker._data_d,
        )

    reason = "policy-roi" if for_policy else "roi-request"
    roi_cache = worker._roi_cache

    roi = resolve_worker_viewport_roi(
        view=view,
        canvas_size=(int(worker.width), int(worker.height)),  # type: ignore[attr-defined]
        source=source,
        level=int(level),
        align_chunks=align_chunks,
        chunk_pad=chunk_pad,
        ensure_contains_viewport=ensure_contains,
        edge_threshold=edge_threshold,
        for_policy=for_policy,
        roi_cache=roi_cache,
        roi_log_state=log_state,
        snapshot_cb=_snapshot,
        log_layer_debug=worker._log_layer_debug,
        quiet=quiet,
        data_wh=data_wh,
        reason=reason,
        logger_ref=logger,
    )

    if log_state is not None:
        log_state["roi"] = roi
        log_state["level"] = int(level)
        log_state["requested"] = (align_chunks, ensure_contains, chunk_pad)

    return roi


def apply_plane_slice_roi(
    worker: Any,
    source: Any,
    level: int,
    roi: SliceROI,
    *,
    update_contrast: bool,
) -> Tuple[int, int]:
    """Load and apply a plane slice for the given ROI."""

    z_idx = int(worker._z_index or 0)
    slab = source.slice(int(level), z_idx, compute=True, roi=roi)

    layer = worker._napari_layer
    if layer is not None:
        view = worker.view
        assert view is not None, "VisPy view required for slice apply"
        ctx = worker._build_scene_state_context(view.camera)
        SceneStateApplier.apply_slice_to_layer(
            ctx,
            source=source,
            slab=slab,
            roi=roi,
            update_contrast=bool(update_contrast),
        )

    height_px = int(slab.shape[0])
    width_px = int(slab.shape[1])
    worker._data_wh = (int(width_px), int(height_px))
    worker._data_d = None

    runner = worker._viewport_runner
    if runner is not None:
        chunk_shape = chunk_shape_for_level(source, int(level))
        if not worker.use_volume:
            runner.mark_roi_applied(roi, chunk_shape=chunk_shape)
            runner.mark_level_applied(int(level))
            rect = worker._current_panzoom_rect()
            runner.update_camera_rect(rect)
    worker._mark_render_tick_needed()
    return height_px, width_px


__all__ = ["apply_plane_slice_roi", "viewport_roi_for_level"]
