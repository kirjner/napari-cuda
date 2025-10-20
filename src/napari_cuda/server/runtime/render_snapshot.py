"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

from typing import Any

from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera, TurntableCamera

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.worker_runtime import (
    prepare_worker_level,
    apply_worker_slice_level,
    apply_worker_volume_level,
)
from napari_cuda.server.data.level_budget import select_volume_level


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
    _apply_snapshot_multiscale(worker, snapshot)

    if _l.isEnabledFor(_logging.INFO):
        _l.info("snapshot.apply.end: dims applied; resuming fit callbacks")

    # Reconnect fit_to_view so subsequent dims changes behave normally
    nd.connect(viewer.fit_to_view)
    od.connect(viewer.fit_to_view)


__all__ = ["apply_render_snapshot"]


def _apply_snapshot_multiscale(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply multiscale state reflected in a controller-authored snapshot."""

    nd = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
    target_volume = nd >= 3

    source = worker._ensure_scene_source()  # noqa: SLF001
    prev_level = int(worker._active_ms_level)
    target_level = int(snapshot.current_level) if snapshot.current_level is not None else prev_level
    level_changed = target_level != prev_level

    ledger_step = (
        tuple(int(v) for v in snapshot.current_step)
        if snapshot.current_step is not None
        else None
    )

    if target_volume:
        entering_volume = not worker.use_volume
        if entering_volume:
            worker.use_volume = True
            worker._last_dims_signature = None  # noqa: SLF001
        requested_level = int(target_level)
        effective_level = _resolve_volume_level(worker, source, requested_level)
        worker._level_downgraded = bool(effective_level != requested_level)
        load_needed = entering_volume or (int(effective_level) != prev_level)
        if load_needed:
            applied_context = prepare_worker_level(
                worker,
                source,
                int(effective_level),
                prev_level=prev_level,
                ledger_step=ledger_step,
            )
            apply_worker_volume_level(worker, source, applied_context)
            target_level = int(effective_level)
            level_changed = int(target_level) != prev_level
        if load_needed:
            worker._configure_camera_for_mode()
        _apply_volume_camera_pose(worker, snapshot)
        return

    stage_prev_level = prev_level
    if worker.use_volume and not target_volume:
        stage_prev_level = target_level

    applied_context = prepare_worker_level(
        worker,
        source,
        target_level,
        prev_level=stage_prev_level,
        ledger_step=ledger_step,
    )

    if worker.use_volume:
        worker.use_volume = False
        worker._configure_camera_for_mode()
        worker._last_dims_signature = None  # noqa: SLF001
    worker._level_downgraded = False
    _apply_plane_camera_pose(worker, snapshot)

    apply_worker_slice_level(worker, source, applied_context)


def _apply_volume_camera_pose(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, TurntableCamera):
        return

    center = snapshot.center
    if center is not None and len(center) >= 3:
        cam.center = (
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )

    angles = snapshot.angles
    if angles is not None and len(angles) >= 2:
        cam.azimuth = float(angles[0])
        cam.elevation = float(angles[1])
        if len(angles) >= 3:
            cam.roll = float(angles[2])  # type: ignore[attr-defined]

    if snapshot.distance is not None:
        cam.distance = float(snapshot.distance)
    if snapshot.fov is not None:
        cam.fov = float(snapshot.fov)


def _apply_plane_camera_pose(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, PanZoomCamera):
        return

    rect = snapshot.rect
    if rect is not None and len(rect) >= 4:
        cam.rect = Rect(
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        )

    center = snapshot.center
    if center is not None and len(center) >= 2:
        cam.center = (
            float(center[0]),
            float(center[1]),
        )


def _resolve_volume_level(worker: Any, source: Any, requested_level: int) -> int:
    max_voxels_cfg = worker._volume_max_voxels or worker._hw_limits.volume_max_voxels
    max_bytes_cfg = worker._volume_max_bytes or worker._hw_limits.volume_max_bytes
    max_voxels = int(max_voxels_cfg) if max_voxels_cfg else None
    max_bytes = int(max_bytes_cfg) if max_bytes_cfg else None
    level, _ = select_volume_level(
        source,
        int(requested_level),
        max_voxels=max_voxels,
        max_bytes=max_bytes,
        error_cls=worker._budget_error_cls,
    )
    return int(level)
