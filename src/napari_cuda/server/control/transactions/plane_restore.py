"""Ledger transaction helper for deterministic 3D → 2D plane restore."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)
from napari_cuda.server.scene import build_render_scene_state


def _as_level(value: int | float | str) -> int:
    return int(value)


def _as_step(step: Sequence[int | float | str]) -> Tuple[int, ...]:
    return tuple(int(v) for v in step)


def _as_center(center: Sequence[float | int | str]) -> Tuple[float, float, float]:
    if len(center) < 3:
        raise ValueError("plane restore center requires three components")
    return (
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )


def _as_rect(rect: Sequence[float | int | str]) -> Tuple[float, float, float, float]:
    if len(rect) < 4:
        raise ValueError("plane restore rect requires four components")
    return (
        float(rect[0]),
        float(rect[1]),
        float(rect[2]),
        float(rect[3]),
    )


def apply_plane_restore_transaction(
    *,
    ledger: ServerStateLedger,
    scene,
    level: int | float | str,
    step: Sequence[int | float | str],
    center: Sequence[float | int | str],
    zoom: float | int | str,
    rect: Sequence[float | int | str],
    origin: str = "control.view.plane_restore",
    timestamp: Optional[float] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Batch ledger writes for a plane camera restore.

    Parameters
    ----------
    ledger
        Server ledger to update.
    scene
        Mutable scene data bag; used to rebuild the cached snapshot.
    level, step
        Target multiscale level and step indices for the restored plane.
    center, zoom, rect
        Plane camera pose expressed in world coordinates.
    origin
        Ledger origin identifier.
    timestamp
        Optional timestamp to stamp on the ledger entries.
    """

    level_idx = _as_level(level)
    step_tuple = _as_step(step)
    center_tuple = _as_center(center)
    rect_tuple = _as_rect(rect)
    zoom_value = float(zoom)

    batch_entries: list[tuple] = [
        ("multiscale", "main", "level", level_idx),
        ("dims", "main", "current_step", step_tuple),
        ("camera", "main", "center", center_tuple),
        ("camera", "main", "zoom", zoom_value),
        ("camera", "main", "rect", rect_tuple),
        ("camera_plane", "main", "center", center_tuple),
        ("camera_plane", "main", "zoom", zoom_value),
        ("camera_plane", "main", "rect", rect_tuple),
        ("view_cache", "plane", "level", level_idx),
        ("view_cache", "plane", "step", step_tuple),
    ]

    stored = ledger.batch_record_confirmed(
        batch_entries,
        origin=origin,
        timestamp=timestamp,
    )

    return stored
