"""Ledger transaction helper for axis step updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_dims_step_transaction(
    *,
    ledger: ServerStateLedger,
    step: Sequence[int],
    axes_spec_payload: Mapping[str, Any],
    metadata: Optional[Mapping[str, object]] = None,
    origin: str = "control.dims",
    timestamp: Optional[float] = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Record the requested dims step in the ledger and return stored entries."""

    step_tuple: tuple[int, ...] = tuple(int(v) for v in step)
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
    entries.append(("dims", "main", "axes_spec", dict(axes_spec_payload)))

    return ledger.batch_record_confirmed(
        entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = ["apply_dims_step_transaction"]
