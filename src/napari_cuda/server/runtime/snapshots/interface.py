"""Typed interface for snapshot application helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TYPE_CHECKING

from napari_cuda.server.runtime.viewport import ViewportState

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker
    from napari_cuda.server.runtime.viewport.runner import ViewportRunner


@dataclass(slots=True)
class SnapshotInterface:
    """Explicit surface the snapshot helpers may touch on the worker."""

    worker: "EGLRendererWorker"

    # Viewport access --------------------------------------------------
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

    # Viewer / camera --------------------------------------------------
    @property
    def viewer(self):
        return getattr(self.worker, "_viewer", None)

    @property
    def view(self):
        return getattr(self.worker, "view", None)

    def ensure_plane_visual(self):
        return self.worker._ensure_plane_visual()  # type: ignore[attr-defined]

    def ensure_volume_visual(self):
        return self.worker._ensure_volume_visual()  # type: ignore[attr-defined]

    def emit_current_camera_pose(self, reason: str) -> None:
        self.worker._emit_current_camera_pose(reason)  # type: ignore[attr-defined]

    # Snapshot bookkeeping --------------------------------------------
    def last_snapshot_signature(self):
        return getattr(self.worker, "_last_snapshot_signature", None)

    def set_last_snapshot_signature(self, signature) -> None:
        self.worker._last_snapshot_signature = signature  # type: ignore[attr-defined]

    def last_dims_signature(self):
        return getattr(self.worker, "_last_dims_signature", None)

    def set_last_dims_signature(self, signature) -> None:
        self.worker._last_dims_signature = signature  # type: ignore[attr-defined]

    def last_slice_signature(self):
        return getattr(self.worker, "_last_slice_signature", None)

    def set_last_slice_signature(self, signature) -> None:
        self.worker._last_slice_signature = signature  # type: ignore[attr-defined]

    # Data / cache -----------------------------------------------------
    def z_index(self) -> Optional[int]:
        return getattr(self.worker, "_z_index", None)

    def set_z_index(self, value: int) -> None:
        self.worker._z_index = int(value)  # type: ignore[attr-defined]

    def set_data_shape(self, width_px: int, height_px: int) -> None:
        self.worker._data_wh = (int(width_px), int(height_px))  # type: ignore[attr-defined]

    def set_data_depth(self, depth: Optional[int]) -> None:
        self.worker._data_d = None if depth is None else int(depth)  # type: ignore[attr-defined]

    def set_volume_scale(self, scale: Sequence[float]) -> None:
        self.worker._volume_scale = tuple(float(v) for v in scale)  # type: ignore[attr-defined]

    def mark_render_tick_needed(self) -> None:
        self.worker._mark_render_tick_needed()  # type: ignore[attr-defined]

    def reset_last_plane_pose(self) -> None:
        self.worker._last_plane_pose = None  # type: ignore[attr-defined]

    # ROI / layer logging ---------------------------------------------
    @property
    def napari_layer(self):
        return getattr(self.worker, "_napari_layer", None)

    def set_napari_layer(self, layer: Any) -> None:
        self.worker._napari_layer = layer  # type: ignore[attr-defined]

    @property
    def roi_align_chunks(self) -> bool:
        return bool(getattr(self.worker, "_roi_align_chunks", False))

    @property
    def roi_pad_chunks(self) -> int:
        return int(getattr(self.worker, "_roi_pad_chunks", 0))

    @property
    def sticky_contrast(self) -> bool:
        return bool(getattr(self.worker, "_sticky_contrast", False))

    @property
    def layer_logger(self):
        return getattr(self.worker, "_layer_logger", None)

    @property
    def log_layer_debug(self) -> bool:
        return bool(getattr(self.worker, "_log_layer_debug", False))

    # Visual handles --------------------------------------------------
    @property
    def plane_visual_handle(self):
        return getattr(self.worker, "_plane_visual_handle", None)

    @property
    def volume_visual_handle(self):
        return getattr(self.worker, "_volume_visual_handle", None)
