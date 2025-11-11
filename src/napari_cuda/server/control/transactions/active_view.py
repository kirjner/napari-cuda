"""Ledger transaction helper for ActiveView (mode + level)."""

from __future__ import annotations

from typing import Optional

from napari_cuda.server.ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_active_view_transaction(
    *,
    ledger: ServerStateLedger,
    mode: str,
    level: int,
    origin: str,
    timestamp: Optional[float] = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Record the authoritative ActiveView selection in the ledger.

    Writes ("viewport","active","state") = {"mode": mode, "level": level} atomically.
    """
    payload = {"mode": str(mode), "level": int(level)}
    return ledger.batch_record_confirmed(
        [("viewport", "active", "state", payload)],
        origin=origin,
        timestamp=timestamp,
    )


__all__ = ["apply_active_view_transaction"]

