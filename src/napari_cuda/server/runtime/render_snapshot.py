"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

from typing import Any

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.worker_runtime import (
    prepare_worker_level,
    apply_worker_slice_level,
)


def apply_render_snapshot(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
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
        _l.info("snapshot.apply.begin: suppress fit; applying dims")
    worker._apply_dims_from_snapshot(snapshot)  # noqa: SLF001
    _apply_level_state(worker, snapshot)

    if _l.isEnabledFor(_logging.INFO):
        _l.info("snapshot.apply.end: dims applied; resuming fit callbacks")

    # Reconnect fit_to_view so subsequent dims changes behave normally
    nd.connect(viewer.fit_to_view)
    od.connect(viewer.fit_to_view)


__all__ = ["apply_render_snapshot"]


def _apply_level_state(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply mode + level + ROI inside the transaction."""

    nd = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
    target_volume = nd >= 3

    if target_volume:
        if not worker.use_volume:
            worker._enter_volume_mode()
        return

    if worker.use_volume:
        worker._exit_volume_mode()

    source = worker._ensure_scene_source()  # noqa: SLF001
    prev_level = int(worker._active_ms_level)
    target_level = int(snapshot.current_level) if snapshot.current_level is not None else prev_level
    ledger_step = (
        tuple(int(v) for v in snapshot.current_step)
        if snapshot.current_step is not None
        else None
    )

    applied_snapshot = prepare_worker_level(
        worker,
        source,
        target_level,
        prev_level=prev_level,
        ledger_step=ledger_step,
    )
    apply_worker_slice_level(worker, source, applied_snapshot)
