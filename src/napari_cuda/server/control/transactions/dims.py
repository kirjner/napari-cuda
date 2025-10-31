"""Ledger transaction helper for axis step updates."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from napari_cuda.server.state_ledger import (
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
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Record the requested dims step in the ledger and return stored entries."""

    step_tuple: Tuple[int, ...] = tuple(int(v) for v in step)
    entries: list[tuple] = []
    if op_seq is not None:
        entries.append(("scene", "main", "op_seq", int(op_seq)))
        if op_state is not None:
            entries.append(("scene", "main", "op_state", str(op_state)))
        if op_kind is not None:
            entries.append(("scene", "main", "op_kind", str(op_kind)))
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
