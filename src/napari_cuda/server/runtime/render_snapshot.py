"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

from typing import Any

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.worker_runtime import (
    set_level_with_budget,
    stage_worker_level,
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

    if target_volume and not worker.use_volume:
        worker._pending_plane_restore = None  # noqa: SLF001
        worker._enter_volume_mode()
    elif not target_volume and worker.use_volume:
        worker._exit_volume_mode()

    source = worker._ensure_scene_source()  # noqa: SLF001

    restore = getattr(worker, "_pending_plane_restore", None)
    skip_signature = getattr(worker, "_skip_snapshot_once", None)
    if (
        not target_volume
        and restore is None
        and skip_signature is not None
    ):
        sig_level, sig_step = skip_signature
        snap_level = snapshot.current_level
        snap_step = snapshot.current_step
        level_matches = snap_level is not None and int(snap_level) == int(sig_level)
        step_matches = (
            snap_step is None
            or tuple(int(v) for v in snap_step) == tuple(int(v) for v in sig_step)
        )
        if level_matches and step_matches:
            worker._skip_snapshot_once = None  # noqa: SLF001
            return
        worker._skip_snapshot_once = None  # noqa: SLF001

    if restore is None:
        if target_volume:
            return
        current_level = int(getattr(worker, "_active_ms_level", 0))
        applied_snapshot = stage_worker_level(
            worker,
            source,
            current_level,
            prev_level=current_level,
            restoring_plane_state=False,
        )
        apply_worker_slice_level(worker, source, applied_snapshot)
        return

    applied_snapshot = set_level_with_budget(
        worker,
        int(restore.level),
        reason="plane-restore",
        budget_error=worker._budget_error_cls,  # noqa: SLF001
        restoring_plane_state=True,
        step_override=restore.step,
        stage_only=True,
    )

    apply_worker_slice_level(worker, source, applied_snapshot)

    worker._pending_plane_restore = None  # noqa: SLF001
    worker.use_volume = False
    worker._level_policy_refresh_needed = False  # noqa: SLF001
