"""Viewport-only snapshot helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from napari_cuda.server.runtime.viewport import PlaneState, RenderMode, VolumeState
from napari_cuda.server.runtime.worker.interfaces.snapshot_interface import SnapshotInterface
from ..napari_viewer.camera_ops import _current_panzoom_rect


def apply_viewport_state_snapshot(
    snapshot_iface: SnapshotInterface,
    *,
    mode: Optional[RenderMode],
    plane_state: Optional[PlaneState],
    volume_state: Optional[VolumeState],
) -> None:
    """Apply a mailbox-only viewport update (no scene snapshot)."""

    worker = snapshot_iface.worker
    runner = snapshot_iface.viewport_runner
    updated = False

    if plane_state is not None:
        snapshot_iface.viewport_state.plane = deepcopy(plane_state)
        if runner is not None:
            runner._plane = snapshot_iface.viewport_state.plane  # type: ignore[attr-defined]
        updated = True

    if volume_state is not None:
        snapshot_iface.viewport_state.volume = deepcopy(volume_state)
        updated = True

    if mode is not None and mode is not snapshot_iface.viewport_state.mode:
        snapshot_iface.viewport_state.mode = mode
        snapshot_iface.configure_camera_for_mode()
        updated = True

    if not updated:
        return

    if runner is not None and snapshot_iface.viewport_state.mode is RenderMode.PLANE:
        rect = _current_panzoom_rect(worker)
        if rect is not None:
            runner.update_camera_rect(rect)


__all__ = ["apply_viewport_state_snapshot"]
