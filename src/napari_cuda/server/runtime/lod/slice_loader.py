"""Helpers for resolving viewport ROI and loading plane slices."""

from __future__ import annotations

from typing import Any

import numpy as np

from napari_cuda.server.data import SliceROI
from napari_cuda.server.data.roi import (
    resolve_worker_viewport_roi,
    viewport_debug_snapshot,
)


def viewport_roi_for_lod(
    worker: Any,
    source: Any,
    level: int,
    *,
    quiet: bool = False,
    for_policy: bool = False,
    reason: str | None = None,
) -> SliceROI:
    """Compute the viewport ROI for the requested multiscale level."""

    view = getattr(worker, "view", None)
    width = int(getattr(worker, "width", 0))
    height = int(getattr(worker, "height", 0))

    align_chunks = (not for_policy) and bool(getattr(worker, "_roi_align_chunks", False))
    ensure_contains = (not for_policy) and bool(getattr(worker, "_roi_ensure_contains_viewport", False))
    edge_threshold = int(getattr(worker, "_roi_edge_threshold", 0))
    chunk_pad = int(getattr(worker, "_roi_pad_chunks", 0))
    data_wh = getattr(worker, "_data_wh", (width, height))
    data_wh = (int(data_wh[0]), int(data_wh[1]))
    data_depth = getattr(worker, "_data_d", None)

    # When no view exists yet, fall back to a full-frame ROI derived from source dimensions.
    if view is None:
        shape_fn = getattr(source, "level_shape", None)
        if callable(shape_fn):
            try:
                dims = shape_fn(int(level))
                if dims:
                    if len(dims) >= 2:
                        height = int(dims[-2])
                    if len(dims) >= 1:
                        width = int(dims[-1])
            except Exception:
                pass
        return SliceROI(0, max(0, int(height)), 0, max(0, int(width)))

    prev_roi = getattr(worker.viewport_state.plane, "applied_roi", None)

    def _snapshot() -> dict[str, Any]:
        return viewport_debug_snapshot(
            view=view,
            canvas_size=(width, height),
            data_wh=data_wh,
            data_depth=data_depth,
        )

    roi_reason = reason or ("policy-roi" if for_policy else "roi-request")
    return resolve_worker_viewport_roi(
        view=view,
        canvas_size=(width, height),
        source=source,
        level=int(level),
        align_chunks=align_chunks,
        chunk_pad=chunk_pad,
        ensure_contains_viewport=ensure_contains,
        edge_threshold=edge_threshold,
        for_policy=for_policy,
        prev_roi=prev_roi,
        snapshot_cb=_snapshot,
        log_layer_debug=bool(getattr(worker, "_log_layer_debug", False)),
        quiet=quiet,
        data_wh=data_wh,
        reason=roi_reason,
    )


def load_lod_slice(
    worker: Any,
    source: Any,
    level: int,
    z_index: int,
    *,
    quiet: bool = False,
    for_policy: bool = False,
    reason: str | None = None,
) -> np.ndarray:
    """Load a slice for ``level`` honouring the worker's viewport policy."""

    roi = viewport_roi_for_lod(
        worker,
        source,
        level,
        quiet=quiet,
        for_policy=for_policy,
        reason=reason,
    )
    slab = source.slice(int(level), int(z_index), compute=True, roi=roi)
    if not isinstance(slab, np.ndarray):
        slab = np.asarray(slab, dtype=np.float32)
    return slab


__all__ = ["load_lod_slice", "viewport_roi_for_lod"]
