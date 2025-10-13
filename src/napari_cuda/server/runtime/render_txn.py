"""Atomic render transaction scaffolding.

Initial scaffold: route dims application through a single entry point so the
apply path stays shallow and centralized. Subsequent phases will batch
dimension updates, level apply, and camera pose application within this
function to ensure napari never observes partial state.
"""

from __future__ import annotations

from typing import Any

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot


def apply_render_txn(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply the snapshot atomically, suppressing napari auto-fit during dims.

    This ensures that napari's fit_to_view callback does not run against a
    transiently inconsistent (order/displayed/ndim) state while we are applying
    the toggle back to 2D or 3D. Camera and level application are handled by
    the worker helpers invoked from within the dims application.
    """
    viewer = worker._viewer  # noqa: SLF001
    assert viewer is not None, "RenderTxn requires an active viewer"

    # Suppress ONLY napari's fit_to_view during dims apply so that layer
    # adapters still observe ndisplay/order events and can rebuild visuals.
    # We temporarily disconnect the specific callback, apply dims, then
    # reconnect.
    import logging as _logging
    _l = _logging.getLogger(__name__)

    nd = viewer.dims.events.ndisplay
    od = viewer.dims.events.order

    # Disconnect the fit_to_view callback from both emitters
    nd.disconnect(viewer.fit_to_view)
    od.disconnect(viewer.fit_to_view)

    if _l.isEnabledFor(_logging.INFO):
        _l.info("txn.begin: suppress fit; applying dims")
    worker._apply_dims_from_snapshot(snapshot)  # noqa: SLF001
    # Apply mode + level + slab + camera atomically after dims
    worker._apply_mode_and_level_txn(snapshot)  # noqa: SLF001

    if _l.isEnabledFor(_logging.INFO):
        _l.info("txn.end: dims applied; resuming fit callbacks")

    # Reconnect fit_to_view so subsequent dims changes behave normally
    nd.connect(viewer.fit_to_view)
    od.connect(viewer.fit_to_view)


__all__ = ["apply_render_txn"]
