"""Ledger transaction helper for 2D ↔ 3D view toggles."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)
from napari_cuda.server.scene_defaults import default_volume_state

logger = logging.getLogger(__name__)


def _primary_layer_id(snapshot: Dict[PropertyKey, LedgerEntry]) -> str:
    for scope, target, _ in snapshot:
        if scope == "layer":
            return str(target)
    # Fallback for bootstrap scenarios where no layer entries exist yet.
    return "layer-0"


def _merge_entries(
    base: Iterable[Tuple[str, str, str, object]],
    extras: Iterable[Tuple[str, str, str, object]],
) -> Tuple[Tuple[str, str, str, object], ...]:
    merged: List[Tuple[str, str, str, object]] = list(base)
    merged.extend(extras)
    return tuple(merged)


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

    snapshot = ledger.snapshot()
    layer_id = _primary_layer_id(snapshot)

    layer_depiction = "volume" if int(target_ndisplay) >= 3 else "plane"
    layer_rendering = "attenuated_mip" if layer_depiction == "volume" else "mip"

    volume_mode = "attenuated_mip" if layer_depiction == "volume" else default_volume_state().get("mode", "mip")

    base_entries: Tuple[Tuple[str, str, str, object], ...] = (
        ("scene", "main", "op_seq", int(op_seq)),
        ("scene", "main", "op_state", "open"),
        ("scene", "main", "op_kind", "view-toggle"),
        ("view", "main", "ndisplay", int(target_ndisplay)),
        ("view", "main", "displayed", tuple(int(idx) for idx in displayed_value)),
        ("dims", "main", "order", tuple(int(idx) for idx in order_value)),
    )

    extra_entries = (
        ("layer", str(layer_id), "depiction", layer_depiction),
        ("layer", str(layer_id), "rendering", layer_rendering),
        ("volume", "main", "render_mode", volume_mode),
    )

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "view.toggle ledger update: layer=%s depiction=%s rendering=%s volume_mode=%s ndisplay=%d",
            layer_id,
            layer_depiction,
            layer_rendering,
            volume_mode,
            int(target_ndisplay),
        )

    return ledger.batch_record_confirmed(
        _merge_entries(base_entries, extra_entries),
        origin=origin,
        timestamp=timestamp,
        dedupe=False,
    )


__all__ = ["apply_view_toggle_transaction"]
