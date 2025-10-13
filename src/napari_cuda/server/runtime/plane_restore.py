"""Helpers for staging 3D→2D plane restore state.

The render transaction consumes the staged rect and lets the unified viewport
logic clamp and emit the corresponding pose + ROI during the 2D re-entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple


RectTuple = Tuple[float, float, float, float]
StepTuple = Tuple[int, ...]


@dataclass
class PlaneRestore:
    """Staged state for restoring a 2D plane after leaving volume mode."""

    level: int
    step: StepTuple
    rect: RectTuple


def stage_plane_restore(
    worker: Any,
    source: Any,  # noqa: ARG001 - source retained for future metadata uses
    level: int,
    step: Sequence[int],
    rect: RectTuple,
) -> PlaneRestore:
    """Stash plane restore data (level, step, rect) on the worker."""

    del source  # Unused for now; keep signature stable for callers.
    step_tuple: StepTuple = tuple(int(v) for v in step)
    restore = PlaneRestore(level=int(level), step=step_tuple, rect=rect)
    worker._pending_plane_restore = restore  # noqa: SLF001
    return restore


__all__ = [
    "PlaneRestore",
    "stage_plane_restore",
]
