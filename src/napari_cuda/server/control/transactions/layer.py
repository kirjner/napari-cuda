"""Ledger transaction helper for layer property updates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

from napari_cuda.server.ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)

LayerLedgerUpdate = tuple[str, str, str, object]


def apply_layer_property_transaction(
    *,
    ledger: ServerStateLedger,
    updates: Iterable[LayerLedgerUpdate],
    origin: str = "control.layer",
    timestamp: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Record layer (and related) property updates in the ledger."""

    entries = []
    for scope, target, key, value in updates:
        if metadata is None:
            entries.append((scope, target, key, value))
        else:
            entries.append((scope, target, key, value, dict(metadata)))

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
    "LayerLedgerUpdate",
    "apply_layer_property_transaction",
]
