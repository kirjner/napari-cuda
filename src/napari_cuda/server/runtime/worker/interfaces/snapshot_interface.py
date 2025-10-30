"""Typed interface for snapshot application helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from napari_cuda.server.runtime.viewport import ViewportState

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker
    from napari_cuda.server.runtime.viewport.runner import ViewportRunner


@dataclass(slots=True)
class SnapshotInterface:
    """Limited worker surface exposed to snapshot helpers."""

    worker: "EGLRendererWorker"

    @property
    def viewport_state(self) -> ViewportState:
        return self.worker.viewport_state

    @property
    def viewport_runner(self) -> Optional["ViewportRunner"]:
        return getattr(self.worker, "_viewport_runner", None)

    def ensure_scene_source(self):
        return self.worker._ensure_scene_source()  # type: ignore[attr-defined]

    def current_level_index(self) -> int:
        return int(self.worker._current_level_index())  # type: ignore[attr-defined]

    def set_current_level_index(self, level: int) -> None:
        self.worker._set_current_level_index(int(level))  # type: ignore[attr-defined]

    def configure_camera_for_mode(self) -> None:
        self.worker._configure_camera_for_mode()  # type: ignore[attr-defined]

    def current_panzoom_rect(self):
        return self.worker._current_panzoom_rect()  # type: ignore[attr-defined]
