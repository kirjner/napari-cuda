"""Typed interface for viewer bootstrap helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


@dataclass(slots=True)
class ViewerInterface:
    """Explicit surface the viewer helpers may touch on the worker."""

    worker: "EGLRendererWorker"

    # Read-only properties -------------------------------------------------
    @property
    def width(self) -> int:
        return int(self.worker.width)

    @property
    def height(self) -> int:
        return int(self.worker.height)

    @property
    def volume_depth(self) -> int:
        return int(getattr(self.worker, "volume_depth", 0))

    @property
    def viewport_state(self):
        return self.worker.viewport_state

    @property
    def debug_policy(self):
        return self.worker._debug_policy  # type: ignore[attr-defined]

    @property
    def zarr_init_z(self) -> Optional[int]:
        return getattr(self.worker, "_zarr_init_z", None)

    @property
    def log_layer_debug(self) -> bool:
        return bool(getattr(self.worker, "_log_layer_debug", False))

    @property
    def zarr_level(self) -> Optional[str]:
        return getattr(self.worker, "_zarr_level", None)

    # Basic setters --------------------------------------------------------
    def set_canvas(self, canvas: Any) -> None:
        self.worker.canvas = canvas

    def set_view(self, view: Any) -> None:
        self.worker.view = view

    def set_viewer(self, viewer: Any) -> None:
        self.worker._viewer = viewer  # type: ignore[attr-defined]

    def set_napari_layer(self, layer: Any) -> None:
        self.worker._napari_layer = layer  # type: ignore[attr-defined]

    def set_scene_source(self, source: Any) -> None:
        self.worker._scene_source = source  # type: ignore[attr-defined]

    def set_current_level_index(self, level: int) -> None:
        self.worker._set_current_level_index(int(level))  # type: ignore[attr-defined]

    def set_zarr_level(self, level: Optional[str]) -> None:
        self.worker._zarr_level = level  # type: ignore[attr-defined]

    def set_zarr_axes(self, axes: str) -> None:
        self.worker._zarr_axes = axes  # type: ignore[attr-defined]

    def set_zarr_shape(self, shape: Optional[Sequence[int]]) -> None:
        self.worker._zarr_shape = tuple(int(v) for v in shape) if shape is not None else None  # type: ignore[attr-defined]

    def set_zarr_dtype(self, dtype: str) -> None:
        self.worker._zarr_dtype = dtype  # type: ignore[attr-defined]

    def set_zarr_clim(self, clim: Any) -> None:
        self.worker._zarr_clim = clim  # type: ignore[attr-defined]

    def set_z_index(self, index: int) -> None:
        self.worker._z_index = int(index)  # type: ignore[attr-defined]

    def set_volume_scale(self, scale: Tuple[float, float, float]) -> None:
        self.worker._volume_scale = tuple(float(v) for v in scale)  # type: ignore[attr-defined]

    def set_data_wh(self, wh: Tuple[int, int]) -> None:
        self.worker._data_wh = (int(wh[0]), int(wh[1]))  # type: ignore[attr-defined]

    def set_data_depth(self, depth: int) -> None:
        self.worker._data_d = int(depth)  # type: ignore[attr-defined]

    def set_bootstrap_full_roi(self, value: bool) -> None:
        self.worker._bootstrap_full_roi = bool(value)  # type: ignore[attr-defined]

    def set_debug_overlay(self, overlay: Any) -> None:
        self.worker._debug_overlay = overlay  # type: ignore[attr-defined]

    # Delegated helpers ----------------------------------------------------
    def frame_volume_camera(self, w: float, h: float, d: float) -> None:
        self.worker._frame_volume_camera(w, h, d)  # type: ignore[attr-defined]

    def register_plane_visual(self, node: Any) -> None:
        self.worker._register_plane_visual(node)  # type: ignore[attr-defined]

    def register_volume_visual(self, node: Any) -> None:
        self.worker._register_volume_visual(node)  # type: ignore[attr-defined]

    def ensure_plane_visual(self) -> Any:
        return self.worker._ensure_plane_visual()  # type: ignore[attr-defined]

    def ensure_volume_visual(self) -> Any:
        return self.worker._ensure_volume_visual()  # type: ignore[attr-defined]

    # Convenience ----------------------------------------------------------
    def set_volume_metadata(
        self,
        *,
        level_index: int,
        level_path: Optional[str],
        axes: Sequence[str],
        shape: Sequence[int],
        dtype: str,
        clim: Any,
    ) -> None:
        self.set_current_level_index(level_index)
        self.set_zarr_level(level_path)
        self.set_zarr_axes("".join(str(a) for a in axes))
        self.set_zarr_shape(shape)
        self.set_zarr_dtype(dtype)
        self.set_zarr_clim(clim)
