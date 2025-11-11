"""Ledger transaction helper for axis step updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from napari_cuda.server.ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_dims_step_transaction(
    *,
    ledger: ServerStateLedger,
    step: Sequence[int],
    dims_spec_payload: Mapping[str, Any],
    origin: str = "control.dims",
    timestamp: Optional[float] = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Record the requested dims step in the ledger and return stored entries."""

    entries: list[tuple] = []
    if op_seq is not None:
        entries.append(("scene", "main", "op_seq", int(op_seq)))
        if op_state is not None:
            entries.append(("scene", "main", "op_state", str(op_state)))
        if op_kind is not None:
            entries.append(("scene", "main", "op_kind", str(op_kind)))
    spec_entry = ("dims", "main", "dims_spec", dict(dims_spec_payload))
    entries.append(spec_entry)

    return ledger.batch_record_confirmed(
        entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = ["apply_dims_step_transaction"]
