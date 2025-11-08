"""Ledger transaction helper for deterministic 3D → 2D plane restore."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def _as_level(value: float | str) -> int:
    return int(value)


def _as_step(step: Sequence[int | float | str]) -> tuple[int, ...]:
    return tuple(int(v) for v in step)


def _as_center(center: Sequence[float | int | str]) -> tuple[float, float, float]:
    if len(center) < 2:
        raise ValueError("plane restore center requires at least two components")
    if len(center) >= 3:
        return (
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )
    return (float(center[0]), float(center[1]), 0.0)


def _as_rect(rect: Sequence[float | int | str]) -> tuple[float, float, float, float]:
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
    level: float | str,
    step: Sequence[int | float | str],
    center: Sequence[float | int | str],
    zoom: float | str,
    rect: Sequence[float | int | str],
    origin: str = "control.view.plane_restore",
    timestamp: Optional[float] = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Batch ledger writes for a plane camera restore.

    Parameters
    ----------
    ledger
        Server ledger to update.
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
    plane_center_tuple = (float(center_tuple[0]), float(center_tuple[1]))

    batch_entries: list[tuple] = []
    if op_seq is not None:
        batch_entries.append(("scene", "main", "op_seq", int(op_seq)))
        if op_state is not None:
            batch_entries.append(("scene", "main", "op_state", str(op_state)))
        if op_kind is not None:
            batch_entries.append(("scene", "main", "op_kind", str(op_kind)))

    # Pose-only restore: do not mutate multiscale level or dims current_step here.
    # Level/step are applied deterministically by the worker on replay.
    batch_entries.extend(
        [
            ("camera_plane", "main", "center", plane_center_tuple),
            ("camera_plane", "main", "zoom", zoom_value),
            ("camera_plane", "main", "rect", rect_tuple),
        ]
    )

    stored = ledger.batch_record_confirmed(
        batch_entries,
        origin=origin,
        timestamp=timestamp,
    )

    return stored
