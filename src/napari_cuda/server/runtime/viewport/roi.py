"""Viewport ROI resolution helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from napari_cuda.server.data.roi import (
    resolve_worker_viewport_roi,
    viewport_debug_snapshot,
)
from napari_cuda.server.runtime.data import SliceROI

logger = logging.getLogger(__name__)


def viewport_roi_for_level(
    worker: Any,
    source: Any,
    level: int,
    *,
    quiet: bool = False,
    for_policy: bool = False,
) -> SliceROI:
    """Compute the viewport ROI for the requested multiscale level."""

    view = worker.view
    align_chunks = (not for_policy) and bool(worker._roi_align_chunks)
    ensure_contains = (not for_policy) and bool(worker._roi_ensure_contains_viewport)
    edge_threshold = int(worker._roi_edge_threshold)
    chunk_pad = int(worker._roi_pad_chunks)

    prev_roi: Optional[SliceROI] = worker.viewport_state.plane.applied_roi  # type: ignore[attr-defined]

    data_wh = worker._data_wh

    def _snapshot() -> Dict[str, Any]:
        return viewport_debug_snapshot(
            view=view,
            canvas_size=(int(worker.width), int(worker.height)),  # type: ignore[attr-defined]
            data_wh=data_wh,
            data_depth=worker._data_d,
        )

    reason = "policy-roi" if for_policy else "roi-request"
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
        prev_roi=prev_roi,
        snapshot_cb=_snapshot,
        log_layer_debug=worker._log_layer_debug,
        quiet=quiet,
        data_wh=data_wh,
        reason=reason,
        logger_ref=logger,
    )

    return roi


__all__ = ["viewport_roi_for_level"]
