"""Ledger transaction helper for axis step updates."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_dims_step_transaction(
    *,
    ledger: ServerStateLedger,
    step: Sequence[int],
    metadata: Optional[Mapping[str, object]] = None,
    origin: str = "control.dims",
    timestamp: Optional[float] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Record the requested dims step in the ledger and return stored entries."""

    step_tuple: Tuple[int, ...] = tuple(int(v) for v in step)
    entries: list[tuple] = []
    if metadata is None:
        entries.append(("dims", "main", "current_step", step_tuple))
    else:
        entries.append(
            (
                "dims",
                "main",
                "current_step",
                step_tuple,
                dict(metadata),
            )
        )

    return ledger.batch_record_confirmed(
        entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = ["apply_dims_step_transaction"]
