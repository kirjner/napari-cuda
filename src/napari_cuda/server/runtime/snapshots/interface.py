"""Typed interface for snapshot application helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from napari_cuda.server.runtime.data import SliceROI
from napari_cuda.server.runtime.viewport import ViewportState

if TYPE_CHECKING:
    from napari_cuda.server.runtime.viewport.runner import ViewportRunner
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SnapshotInterface:
    """Explicit surface the snapshot helpers may touch on the worker."""

    _worker: EGLRendererWorker

    # Viewport access --------------------------------------------------
    @property
    def viewport_state(self) -> ViewportState:
        return self._worker.viewport_state

    @property
    def viewport_runner(self) -> Optional[ViewportRunner]:
        return getattr(self._worker, "_viewport_runner", None)

    def ensure_scene_source(self):
        return self._worker._ensure_scene_source()  # type: ignore[attr-defined]

    def current_level_index(self) -> int:
        return int(self._worker._current_level_index())  # type: ignore[attr-defined]

    def set_current_level_index(self, level: int) -> None:
        self._worker._set_current_level_index(int(level))  # type: ignore[attr-defined]

    def configure_camera_for_mode(self) -> None:
        self._worker._configure_camera_for_mode()  # type: ignore[attr-defined]

    def current_panzoom_rect(self):
        return self._worker._current_panzoom_rect()  # type: ignore[attr-defined]

    # Viewer / camera --------------------------------------------------
    @property
    def viewer(self):
        return getattr(self._worker, "_viewer", None)

    @property
    def view(self):
        return getattr(self._worker, "view", None)

    def ensure_plane_visual(self):
        return self._worker._ensure_plane_visual()  # type: ignore[attr-defined]

    def ensure_volume_visual(self):
        return self._worker._ensure_volume_visual()  # type: ignore[attr-defined]

    def emit_current_camera_pose(self, reason: str) -> None:
        self._worker._emit_current_camera_pose(reason)  # type: ignore[attr-defined]

    # Snapshot bookkeeping --------------------------------------------
    def last_snapshot_signature(self):
        return getattr(self._worker, "_last_snapshot_signature", None)

    def set_last_snapshot_signature(self, signature) -> None:
        self._worker._last_snapshot_signature = signature  # type: ignore[attr-defined]

    def last_dims_signature(self):
        return getattr(self._worker, "_last_dims_signature", None)

    def set_last_dims_signature(self, signature) -> None:
        self._worker._last_dims_signature = signature  # type: ignore[attr-defined]

    def last_slice_signature(self):
        return getattr(self._worker, "_last_slice_signature", None)

    def set_last_slice_signature(self, signature) -> None:
        self._worker._last_slice_signature = signature  # type: ignore[attr-defined]

    # Data / cache -----------------------------------------------------
    def z_index(self) -> Optional[int]:
        return getattr(self._worker, "_z_index", None)

    def set_z_index(self, value: int) -> None:
        self._worker._z_index = int(value)  # type: ignore[attr-defined]

    def set_data_shape(self, width_px: int, height_px: int) -> None:
        self._worker._data_wh = (int(width_px), int(height_px))  # type: ignore[attr-defined]

    def set_data_depth(self, depth: Optional[int]) -> None:
        self._worker._data_d = None if depth is None else int(depth)  # type: ignore[attr-defined]

    def set_volume_scale(self, scale: Sequence[float]) -> None:
        self._worker._volume_scale = tuple(float(v) for v in scale)  # type: ignore[attr-defined]

    def mark_render_tick_needed(self) -> None:
        self._worker._mark_render_tick_needed()  # type: ignore[attr-defined]

    def reset_last_plane_pose(self) -> None:
        self._worker._last_plane_pose = None  # type: ignore[attr-defined]

    # ROI / layer logging ---------------------------------------------
    @property
    def napari_layer(self):
        return getattr(self._worker, "_napari_layer", None)

    def set_napari_layer(self, layer: Any) -> None:
        self._worker._napari_layer = layer  # type: ignore[attr-defined]

    @property
    def roi_align_chunks(self) -> bool:
        return bool(getattr(self._worker, "_roi_align_chunks", False))

    @property
    def roi_pad_chunks(self) -> int:
        return int(getattr(self._worker, "_roi_pad_chunks", 0))

    @property
    def sticky_contrast(self) -> bool:
        return bool(getattr(self._worker, "_sticky_contrast", False))

    @property
    def layer_logger(self):
        return getattr(self._worker, "_layer_logger", None)

    @property
    def log_layer_debug(self) -> bool:
        return bool(getattr(self._worker, "_log_layer_debug", False))

    # Visual handles --------------------------------------------------
    @property
    def plane_visual_handle(self):
        return getattr(self._worker, "_plane_visual_handle", None)

    @property
    def volume_visual_handle(self):
        return getattr(self._worker, "_volume_visual_handle", None)

    # Runtime helpers -------------------------------------------------
    def viewport_roi_for_level(
        self,
        source: Any,
        level: int,
        *,
        quiet: bool = False,
        for_policy: bool = False,
    ) -> SliceROI:
        """Resolve the viewport ROI using the worker's current policy."""

        from napari_cuda.server.runtime.lod.slice_loader import (
            viewport_roi_for_lod,
        )

        return viewport_roi_for_lod(
            self._worker,
            source,
            int(level),
            quiet=quiet,
            for_policy=for_policy,
            reason="policy-roi" if for_policy else "roi-request",
        )

    def resolve_volume_intent_level(self, source: Any, requested_level: int):
        from napari_cuda.server.runtime.lod import level_policy

        return level_policy.resolve_volume_intent_level(
            self._worker,
            source,
            int(requested_level),
        )

    def load_volume(self, source: Any, level: int):
        from napari_cuda.server.runtime.lod import level_policy

        return level_policy.load_volume(self._worker, source, int(level))

    def ledger_step(self):
        from napari_cuda.server.runtime.core import ledger_step

        ledger = getattr(self._worker, "_ledger", None)
        if ledger is None:
            return None
        return ledger_step(ledger)

    def set_dims_range_for_level(self, source: Any, level: int) -> None:
        set_range = getattr(self._worker, "_set_dims_range_for_level", None)
        if callable(set_range):
            set_range(source, int(level))

    def update_level_metadata(self, descriptor: Any, context: Any) -> None:
        updater = getattr(self._worker, "_update_level_metadata", None)
        if callable(updater):
            updater(descriptor, context)

    def mark_level_applied(self, level: int) -> None:
        runner = self.viewport_runner
        if runner is not None:
            runner.mark_level_applied(int(level))
