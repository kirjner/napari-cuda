"""Viewport state synchronization helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, TYPE_CHECKING

from napari_cuda.server.runtime.viewport import PlaneState, RenderMode, VolumeState
from .camera_ops import _configure_camera_for_mode, _current_panzoom_rect

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


def _apply_viewport_state_snapshot(
    worker: "EGLRendererWorker",
    *,
    mode: Optional[RenderMode],
    plane_state: Optional[PlaneState],
    volume_state: Optional[VolumeState],
) -> None:
    """Apply a mailbox-only viewport update (no scene snapshot)."""

    runner = worker._viewport_runner  # noqa: SLF001
    updated = False

    if plane_state is not None:
        worker._viewport_state.plane = deepcopy(plane_state)  # noqa: SLF001
        if runner is not None:
            runner._plane = worker._viewport_state.plane  # type: ignore[attr-defined]  # noqa: SLF001
        updated = True

    if volume_state is not None:
        worker._viewport_state.volume = deepcopy(volume_state)  # noqa: SLF001
        updated = True

    if mode is not None and mode is not worker._viewport_state.mode:  # noqa: SLF001
        worker._viewport_state.mode = mode  # noqa: SLF001
        _configure_camera_for_mode(worker)
        updated = True

    if not updated:
        return

    if runner is not None and worker._viewport_state.mode is RenderMode.PLANE:  # noqa: SLF001
        rect = _current_panzoom_rect(worker)
        if rect is not None:
            runner.update_camera_rect(rect)


__all__ = ["_apply_viewport_state_snapshot"]
