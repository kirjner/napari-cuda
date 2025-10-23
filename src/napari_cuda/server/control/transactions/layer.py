"""Ledger transaction helper for layer property updates."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


LayerLedgerUpdate = Tuple[str, str, str, object]


def apply_layer_property_transaction(
    *,
    ledger: ServerStateLedger,
    updates: Iterable[LayerLedgerUpdate],
    origin: str = "control.layer",
    timestamp: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Record layer (and related) property updates in the ledger."""

    entries = []
    for scope, target, key, value in updates:
        if metadata is None:
            entries.append((scope, target, key, value))
        else:
            entries.append((scope, target, key, value, dict(metadata)))

    if not entries:
        return {}

    return ledger.batch_record_confirmed(
        entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = [
    "LayerLedgerUpdate",
    "apply_layer_property_transaction",
]
