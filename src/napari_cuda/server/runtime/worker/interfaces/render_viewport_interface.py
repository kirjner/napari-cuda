"""Read-only viewport/camera access for render helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


@dataclass(slots=True)
class RenderViewportInterface:
    """Expose viewport state and read helpers without touching worker internals."""

    worker: "EGLRendererWorker"

    # Basic geometry -----------------------------------------------------
    @property
    def width(self) -> int:
        return int(self.worker.width)

    @property
    def height(self) -> int:
        return int(self.worker.height)

    @property
    def view(self) -> Any:
        return getattr(self.worker, "view", None)

    @property
    def viewport_state(self):
        return self.worker.viewport_state

    # Data/cache state ---------------------------------------------------
    @property
    def data_wh(self) -> Tuple[int, int]:
        wh = getattr(self.worker, "_data_wh", (0, 0))  # type: ignore[attr-defined]
        return (int(wh[0]), int(wh[1]))

    @property
    def data_d(self) -> Optional[int]:
        depth = getattr(self.worker, "_data_d", None)  # type: ignore[attr-defined]
        return None if depth is None else int(depth)

    # ROI configuration --------------------------------------------------
    @property
    def roi_align_chunks(self) -> bool:
        return bool(getattr(self.worker, "_roi_align_chunks", False))  # type: ignore[attr-defined]

    @property
    def roi_pad_chunks(self) -> int:
        return int(getattr(self.worker, "_roi_pad_chunks", 0))  # type: ignore[attr-defined]

    @property
    def roi_ensure_contains_viewport(self) -> bool:
        return bool(getattr(self.worker, "_roi_ensure_contains_viewport", False))  # type: ignore[attr-defined]

    @property
    def roi_edge_threshold(self) -> int:
        return int(getattr(self.worker, "_roi_edge_threshold", 0))  # type: ignore[attr-defined]

    @property
    def log_layer_debug(self) -> bool:
        return bool(getattr(self.worker, "_log_layer_debug", False))  # type: ignore[attr-defined]

    # Camera helpers -----------------------------------------------------
    def current_panzoom_rect(self) -> Optional[Tuple[float, float, float, float]]:
        rect = self.worker._current_panzoom_rect()  # type: ignore[attr-defined]
        if rect is None:
            return None
        return tuple(float(v) for v in rect)

    def emit_camera_pose(self, reason: str) -> None:
        self.worker._emit_current_camera_pose(reason)  # type: ignore[attr-defined]


__all__ = ["RenderViewportInterface"]
