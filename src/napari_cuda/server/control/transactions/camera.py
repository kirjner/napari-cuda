"""Ledger transaction helper for camera updates."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


CameraLedgerUpdate = Tuple[str, str, str, object]


def apply_camera_update_transaction(
    *,
    ledger: ServerStateLedger,
    updates: Iterable[CameraLedgerUpdate],
    origin: str = "control.camera",
    timestamp: float | None = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Record camera updates in the ledger and return stored entries."""

    entries = list(updates)
    if not entries:
        return {}

    return ledger.batch_record_confirmed(
        entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = [
    "CameraLedgerUpdate",
    "apply_camera_update_transaction",
]
