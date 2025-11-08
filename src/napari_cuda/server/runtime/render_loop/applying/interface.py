"""Mutation interface for render-loop apply helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from napari_cuda.server.data import SliceROI
from napari_cuda.server.utils.signatures import SignatureToken

if TYPE_CHECKING:
    from napari_cuda.server.scene.viewport import ViewportState
    from napari_cuda.server.runtime.render_loop.planning.viewport_planner import ViewportPlanner
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


@dataclass(slots=True)
class RenderApplyInterface:
    """Expose the worker state that slice/volume apply helpers mutate."""

    worker: EGLRendererWorker
    _slice_signature: Optional[SignatureToken] = None

    # Viewport access --------------------------------------------------
    @property
    def viewport_state(self) -> ViewportState:
        return self.worker.viewport_state

    @property
    def viewport_runner(self) -> Optional[ViewportPlanner]:
        return getattr(self.worker, "_viewport_runner", None)

    def ensure_scene_source(self):
        return self.worker._ensure_scene_source()  # type: ignore[attr-defined]

    def current_level_index(self) -> int:
        return int(self.worker._current_level_index())  # type: ignore[attr-defined]

    def set_current_level_index(self, level: int) -> None:
        self.worker._set_current_level_index(int(level))  # type: ignore[attr-defined]

    def set_plane_level_index(self, level: int) -> None:
        self.worker._set_plane_level_index(int(level))  # type: ignore[attr-defined]

    def set_volume_level_index(self, level: int) -> None:
        self.worker._set_volume_level_index(int(level))  # type: ignore[attr-defined]

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
        return self.worker.view

    def ensure_plane_visual(self):
        return self.worker._ensure_plane_visual()  # type: ignore[attr-defined]

    def ensure_volume_visual(self):
        return self.worker._ensure_volume_visual()  # type: ignore[attr-defined]

    def emit_current_camera_pose(self, reason: str) -> None:
        self.worker._emit_current_camera_pose(reason)  # type: ignore[attr-defined]

    # Snapshot bookkeeping --------------------------------------------
    def last_slice_signature(self) -> Optional[SignatureToken]:
        return self._slice_signature

    def set_last_slice_signature(self, signature: Optional[SignatureToken]) -> None:
        self._slice_signature = signature

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
        return self.worker._napari_layer  # type: ignore[attr-defined]

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

    # Runtime helpers -------------------------------------------------
    def viewport_roi_for_level(
        self,
        source: Any,
        level: int,
        *,
        quiet: bool = False,
        for_policy: bool = False,
    ) -> SliceROI:
        from napari_cuda.server.runtime.lod.slice_loader import (
            viewport_roi_for_lod,
        )

        return viewport_roi_for_lod(
            self.worker,
            source,
            int(level),
            quiet=quiet,
            for_policy=for_policy,
            reason="policy-roi" if for_policy else "roi-request",
        )

    def resolve_volume_intent_level(self, source: Any, requested_level: int):
        from napari_cuda.server.runtime.lod import level_policy

        return level_policy.resolve_volume_intent_level(
            self.worker,
            source,
            int(requested_level),
        )

    def load_volume(self, source: Any, level: int):
        from napari_cuda.server.runtime.lod import level_policy

        return level_policy.load_volume(self.worker, source, int(level))

    def ledger_step(self):
        from napari_cuda.server.runtime.render_loop.plan.ledger_access import (
            step as ledger_step,
        )

        ledger = self.worker._ledger
        return ledger_step(ledger)

    def set_dims_range_for_level(self, source: Any, level: int) -> None:
        setter = getattr(self.worker, "_set_dims_range_for_level", None)  # type: ignore[attr-defined]
        if setter is not None:
            setter(source, int(level))

    def update_level_metadata(self, descriptor: Any, context: Any) -> None:
        updater = getattr(self.worker, "_update_level_metadata", None)  # type: ignore[attr-defined]
        if updater is not None:
            updater(descriptor, context)

    def mark_level_applied(self, level: int) -> None:
        runner = self.viewport_runner
        if runner is not None:
            runner.mark_level_applied(int(level))


__all__ = ["RenderApplyInterface"]
