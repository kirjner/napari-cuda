"""Ledger transaction helper for camera updates."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


CameraLedgerUpdate = Tuple[str, str, str, object] | Tuple[str, str, str, object, Dict[str, object]]


def apply_camera_update_transaction(
    *,
    ledger: ServerStateLedger,
    updates: Iterable[CameraLedgerUpdate],
    origin: str = "control.camera",
    timestamp: float | None = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Record camera updates in the ledger and return stored entries."""

    entries = list(updates)
    if not entries:
        return {}

    if op_seq is not None:
        entries.insert(0, ("scene", "main", "op_seq", int(op_seq)))
        if op_state is not None:
            entries.insert(1, ("scene", "main", "op_state", str(op_state)))
        if op_kind is not None:
            entries.insert(2, ("scene", "main", "op_kind", str(op_kind)))

    return ledger.batch_record_confirmed(
        entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = [
    "CameraLedgerUpdate",
    "apply_camera_update_transaction",
]
