"""Ledger transaction helper for deterministic 2D → 3D volume restore."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def _as_level(value: int | float | str) -> int:
    return int(value)


def _as_center(center: Sequence[float | int | str]) -> Tuple[float, float, float]:
    if len(center) < 3:
        raise ValueError("volume restore center requires three components")
    return (
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )


def _as_angles(angles: Sequence[float | int | str]) -> Tuple[float, float, float]:
    if len(angles) < 2:
        raise ValueError("volume restore angles require at least two components")
    roll = float(angles[2]) if len(angles) >= 3 else 0.0
    return (
        float(angles[0]),
        float(angles[1]),
        roll,
    )


def apply_volume_restore_transaction(
    *,
    ledger: ServerStateLedger,
    level: int | float | str,
    center: Sequence[float | int | str],
    angles: Sequence[float | int | str],
    distance: float | int | str,
    fov: float | int | str,
    origin: str = "control.view.volume_restore",
    timestamp: Optional[float] = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Batch ledger writes for a volume camera restore."""

    level_idx = _as_level(level)
    center_tuple = _as_center(center)
    angles_tuple = _as_angles(angles)
    distance_value = float(distance)
    fov_value = float(fov)

    batch_entries: list[tuple] = []
    if op_seq is not None:
        batch_entries.append(("scene", "main", "op_seq", int(op_seq)))
        if op_state is not None:
            batch_entries.append(("scene", "main", "op_state", str(op_state)))
        if op_kind is not None:
            batch_entries.append(("scene", "main", "op_kind", str(op_kind)))

    batch_entries.extend(
        [
            ("camera_volume", "main", "center", center_tuple),
            ("camera_volume", "main", "angles", angles_tuple),
            ("camera_volume", "main", "distance", distance_value),
            ("camera_volume", "main", "fov", fov_value),
        ]
    )

    return ledger.batch_record_confirmed(
        batch_entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = ["apply_volume_restore_transaction"]
