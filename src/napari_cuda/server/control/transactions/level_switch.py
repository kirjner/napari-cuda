"""Ledger transaction helper for multiscale level switches."""

from __future__ import annotations

from typing import Optional

from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.control.state_reducers import reduce_level_update
from napari_cuda.server.data.lod import LevelContext
from napari_cuda.server.scene import ServerSceneData
from napari_cuda.server.control.state_models import ServerLedgerUpdate


def apply_level_switch_transaction(
    *,
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock,
    applied: LevelContext,
    downgraded: bool,
    origin: str = "worker.state.level",
    timestamp: Optional[float] = None,
) -> ServerLedgerUpdate:
    """Apply a level switch by recording the updated level/step on the ledger."""

    return reduce_level_update(
        store,
        ledger,
        lock,
        applied=applied,
        downgraded=bool(downgraded),
        intent_id=None,
        timestamp=timestamp,
        origin=str(origin),
    )


__all__ = [
    "apply_level_switch_transaction",
]
