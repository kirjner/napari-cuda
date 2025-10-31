"""Typed interface for viewer bootstrap helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from napari_cuda.server.runtime.lod.context import build_level_context as lod_build_level_context
from napari_cuda.server.runtime.lod.level_policy import (
    load_volume as lod_load_volume,
    resolve_volume_intent_level as lod_resolve_volume_intent_level,
)
from napari_cuda.server.runtime.lod.slice_loader import load_lod_slice
from napari_cuda.server.runtime.render_loop.apply_interface import RenderApplyInterface
from napari_cuda.server.runtime.render_loop.apply.snapshots.viewer_metadata import (
    apply_plane_metadata as snapshot_apply_plane_metadata,
    apply_volume_metadata as snapshot_apply_volume_metadata,
)
from napari_cuda.server.runtime.render_loop.apply.snapshots.volume import (
    VolumeApplyResult,
    apply_volume_level as snapshot_apply_volume_level,
)

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


@dataclass(slots=True)
class ViewerBootstrapInterface:
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

    def set_volume_scale(self, scale: tuple[float, float, float]) -> None:
        self.worker._volume_scale = tuple(float(v) for v in scale)  # type: ignore[attr-defined]

    def set_data_wh(self, wh: tuple[int, int]) -> None:
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

    def load_initial_slice(
        self,
        source: Any,
        level: int,
        z_index: int,
    ) -> Any:
        """Fetch the initial 2D slab using the worker's viewport policy."""

        return load_lod_slice(
            self.worker,
            source,
            int(level),
            int(z_index),
            quiet=False,
            for_policy=False,
            reason="bootstrap",
        )

    def build_level_context(
        self,
        decision: Any,
        *,
        source: Any,
        prev_level: Optional[int],
        last_step: Optional[Sequence[int]],
    ) -> Any:
        """Delegate level context construction to the runtime LOD helpers."""

        return lod_build_level_context(
            decision,
            source=source,
            prev_level=prev_level,
            last_step=last_step,
        )

    def resolve_volume_intent_level(
        self,
        source: Any,
        requested_level: int,
    ) -> tuple[int, bool]:
        """Resolve the requested volume level against worker policies."""

        return lod_resolve_volume_intent_level(
            self.worker,
            source,
            int(requested_level),
        )

    def load_volume(
        self,
        source: Any,
        level: int,
    ) -> Any:
        """Load a volume payload after enforcing worker budgets."""

        return lod_load_volume(
            self.worker,
            source,
            int(level),
        )

    def apply_volume_metadata(
        self,
        source: Any,
        context: Any,
    ) -> None:
        """Apply viewer metadata for a volume context."""

        snapshot_apply_volume_metadata(
            RenderApplyInterface(self.worker),
            source,
            context,
        )

    def apply_plane_metadata(
        self,
        source: Any,
        context: Any,
    ) -> None:
        """Apply viewer metadata for a plane context."""

        snapshot_apply_plane_metadata(
            RenderApplyInterface(self.worker),
            source,
            context,
        )

    def apply_volume_level(
        self,
        source: Any,
        context: Any,
        *,
        downgraded: bool,
    ) -> VolumeApplyResult:
        """Apply the volume level for bootstrap using shared helpers."""

        return snapshot_apply_volume_level(
            RenderApplyInterface(self.worker),
            source,
            context,
            downgraded=downgraded,
        )
