"""Viewport ROI resolution helpers."""

from __future__ import annotations

from typing import Any

from napari_cuda.server.runtime.data import SliceROI
from .slice_loader import viewport_roi_for_lod


def viewport_roi_for_level(
    worker: Any,
    source: Any,
    level: int,
    *,
    quiet: bool = False,
    for_policy: bool = False,
) -> SliceROI:
    """Compute the viewport ROI for the requested multiscale level."""

    reason = "policy-roi" if for_policy else "roi-request"
    return viewport_roi_for_lod(
        worker,
        source,
        int(level),
        quiet=quiet,
        for_policy=for_policy,
        reason=reason,
    )


__all__ = ["viewport_roi_for_level"]
