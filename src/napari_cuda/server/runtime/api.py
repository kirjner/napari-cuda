"""Control-facing façade for interacting with the render worker."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

from napari_cuda.server.scene import (
    PlaneState,
    RenderMode,
    RenderUpdate,
    VolumeState,
)

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


@dataclass
class ViewportSnapshot:
    """Copy of the worker viewport state for control consumers."""

    mode: RenderMode
    plane: PlaneState
    volume: VolumeState


class RuntimeHandle:
    """Narrow API for control-plane code to interact with the runtime worker."""

    def __init__(
        self,
        worker_getter: Callable[[], Optional[EGLRendererWorker]],
    ) -> None:
        self._worker_getter = worker_getter

    def _resolve_worker(self) -> Optional[EGLRendererWorker]:
        try:
            return self._worker_getter()
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("RuntimeHandle: worker getter failed")
            return None

    @property
    def is_ready(self) -> bool:
        worker = self._resolve_worker()
        return bool(worker is not None and getattr(worker, "is_ready", False))

    def viewport_snapshot(self) -> Optional[ViewportSnapshot]:
        worker = self._resolve_worker()
        if worker is None or not getattr(worker, "is_ready", False):
            return None

        base_state = getattr(worker, "viewport_state", None)
        if base_state is None:
            return None

        try:
            mode = base_state.mode
            plane_copy = replace(base_state.plane)
            volume_copy = replace(base_state.volume)
        except Exception:
            logger.exception("RuntimeHandle: failed to clone viewport state")
            return None

        return ViewportSnapshot(mode=mode, plane=plane_copy, volume=volume_copy)

    def enqueue_render_update(self, update: RenderUpdate) -> bool:
        worker = self._resolve_worker()
        if worker is None:
            logger.debug("RuntimeHandle: enqueue skipped (no worker)")
            return False
        if not getattr(worker, "is_ready", False):
            logger.debug("RuntimeHandle: enqueue skipped (worker not ready)")
            return False
        try:
            worker.enqueue_update(update)  # type: ignore[attr-defined]
            return True
        except Exception:
            logger.exception("RuntimeHandle: enqueue_update failed")
            return False

    def request_level(self, level: int, path: Optional[str]) -> bool:
        logger.debug(
            "RuntimeHandle: request_level(level=%s, path=%s) ignored; manual switches are unsupported",
            level,
            path,
        )
        return False

    def force_idr(self) -> bool:
        worker = self._resolve_worker()
        if worker is None:
            return False
        try:
            worker.force_idr()  # type: ignore[attr-defined]
            return True
        except Exception:
            logger.exception("RuntimeHandle: force_idr failed")
            return False


__all__ = ["RuntimeHandle", "ViewportSnapshot"]
