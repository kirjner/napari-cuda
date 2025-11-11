"""Ledger transaction helper for thumbnail capture ingestion."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from napari_cuda.server.ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_thumbnail_capture(
    *,
    ledger: ServerStateLedger,
    layer_id: str,
    payload: Mapping[str, Any],
    origin: str = "server.thumbnail",
    timestamp: Optional[float] = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Persist a captured thumbnail.

    Writes the normalized thumbnail payload under the layer scope.
    Internal sequencing is handled by the ledger per-key version; we do not
    persist any dedupe signatures or custom counters here.
    """

    entries: list[tuple] = [
        ("layer", str(layer_id), "thumbnail", dict(payload))
    ]
    return ledger.batch_record_confirmed(entries, origin=origin, timestamp=timestamp)


__all__ = [
    "apply_thumbnail_capture",
]
