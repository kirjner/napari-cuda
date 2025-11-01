"""Ledger transaction helper for bootstrap seeding."""

from __future__ import annotations

from collections.abc import Iterable

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_bootstrap_transaction(
    *,
    ledger: ServerStateLedger,
    op_seq: int,
    entries: Iterable[tuple],
    origin: str = "server.bootstrap",
    timestamp: float | None = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Batch the bootstrap ledger writes and return stored entries."""

    materialized = list(entries)
    materialized[:0] = [
        ("scene", "main", "op_seq", int(op_seq)),
        ("scene", "main", "op_state", "open"),
        ("scene", "main", "op_kind", "bootstrap"),
    ]

    return ledger.batch_record_confirmed(
        materialized,
        origin=origin,
        timestamp=timestamp,
        dedupe=False,
    )


__all__ = [
    "apply_bootstrap_transaction",
]
