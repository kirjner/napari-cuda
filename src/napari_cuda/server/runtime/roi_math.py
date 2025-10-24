"""Pure helpers for chunk-aligned ROI math.

These functions stay free of worker/viewer state so both the EGL worker and
future controller-side plumbing can reuse the exact same alignment logic.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from napari_cuda.server.runtime.scene_types import SliceROI


__all__ = [
    "chunk_shape_for_level",
    "align_roi_to_chunk_grid",
    "roi_chunk_signature",
]


def _axis_index(axes_lower: Sequence[str], axis: str, fallback: int) -> int:
    """Return the index of *axis* (case-insensitive) or the provided fallback."""

    if axis in axes_lower:
        return axes_lower.index(axis)
    return fallback


def chunk_shape_for_level(source: object, level: int) -> Optional[Tuple[int, int]]:
    """Return (cy, cx) chunk dimensions for ``source`` at ``level``.

    When chunk metadata is unavailable we return ``None`` so callers can
    gracefully skip alignment. Axis lookup mirrors the legacy worker logic:
    defaulting to the final two axes when we cannot find explicit ``"y"`` or
    ``"x"`` labels.
    """

    try:
        arr = source.get_level(level)  # type: ignore[attr-defined]
    except Exception:
        return None

    chunks = getattr(arr, "chunks", None)
    if chunks is None:
        return None

    axes = getattr(source, "axes", ())
    axes_lower = [str(ax).lower() for ax in axes] if axes else []

    fallback_y = max(0, len(chunks) - 2)
    fallback_x = max(0, len(chunks) - 1)
    y_idx = _axis_index(axes_lower, "y", fallback_y)
    x_idx = _axis_index(axes_lower, "x", fallback_x)

    try:
        cy = int(chunks[y_idx])
    except Exception:
        cy = 1
    try:
        cx = int(chunks[x_idx])
    except Exception:
        cx = 1

    cy = max(1, cy)
    cx = max(1, cx)
    return (cy, cx)


def align_roi_to_chunk_grid(
    roi: SliceROI,
    chunk_shape: Optional[Tuple[int, int]],
    pad_chunks: int,
    *,
    height: int,
    width: int,
) -> SliceROI:
    """Snap *roi* to the provided chunk grid and apply optional chunk padding."""

    if chunk_shape is None:
        return roi

    cy, cx = chunk_shape
    cy = max(1, int(cy))
    cx = max(1, int(cx))

    ys = (int(roi.y_start) // cy) * cy
    ye = ((int(roi.y_stop) + cy - 1) // cy) * cy
    xs = (int(roi.x_start) // cx) * cx
    xe = ((int(roi.x_stop) + cx - 1) // cx) * cx

    pad = max(0, int(pad_chunks))
    if pad:
        ys = max(0, ys - pad * cy)
        ye = min(height, ye + pad * cy)
        xs = max(0, xs - pad * cx)
        xe = min(width, xe + pad * cx)

    return SliceROI(ys, ye, xs, xe).clamp(height, width)


def roi_chunk_signature(
    roi: SliceROI,
    chunk_shape: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Return a stable signature describing which chunks *roi* spans.

    The signature is ``(y_start_idx, y_stop_idx, x_start_idx, x_stop_idx)`` and
    is intended for equality tests—identical signatures imply the same chunk
    footprint. When chunk metadata is missing we return ``None`` so callers can
    fall back to raw ROI comparisons.
    """

    if chunk_shape is None:
        return None

    cy, cx = chunk_shape
    cy = max(1, int(cy))
    cx = max(1, int(cx))

    return (
        int(roi.y_start) // cy,
        max(0, (int(roi.y_stop) - 1)) // cy,
        int(roi.x_start) // cx,
        max(0, (int(roi.x_stop) - 1)) // cx,
    )
