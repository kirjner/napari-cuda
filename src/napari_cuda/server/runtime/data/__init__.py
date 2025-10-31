"""Shared runtime data helpers (slice geometry, ROI math)."""

from .roi_math import (
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
    roi_chunk_signature,
)
from .scene_types import SceneSource, SliceIOMetrics, SliceROI

__all__ = [
    "SceneSource",
    "SliceIOMetrics",
    "SliceROI",
    "align_roi_to_chunk_grid",
    "chunk_shape_for_level",
    "roi_chunk_signature",
]
