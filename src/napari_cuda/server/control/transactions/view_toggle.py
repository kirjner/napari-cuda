"""Ledger transaction helper for 2D ↔ 3D view toggles."""

from __future__ import annotations

from typing import Dict, Tuple

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_view_toggle_transaction(
    *,
    ledger: ServerStateLedger,
    op_seq: int,
    target_ndisplay: int,
    order_value: Tuple[int, ...],
    displayed_value: Tuple[int, ...],
    origin: str,
    timestamp: float,
) -> Dict[PropertyKey, LedgerEntry]:
    """Record the ledger updates for a view toggle intent."""

    return ledger.batch_record_confirmed(
        (
            ("scene", "main", "op_seq", int(op_seq)),
            ("scene", "main", "op_state", "open"),
            ("scene", "main", "op_kind", "view-toggle"),
            ("view", "main", "ndisplay", int(target_ndisplay)),
            ("view", "main", "displayed", tuple(int(idx) for idx in displayed_value)),
            ("dims", "main", "order", tuple(int(idx) for idx in order_value)),
        ),
        origin=origin,
        timestamp=timestamp,
        dedupe=False,
    )


__all__ = ["apply_view_toggle_transaction"]
