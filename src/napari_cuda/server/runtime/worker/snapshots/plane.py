"""Slice snapshot application helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from vispy.scene.cameras import PanZoomCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.data.roi import plane_wh_for_level
from napari_cuda.server.runtime.data import (
    SliceROI,
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
    roi_chunk_signature,
)
from napari_cuda.server.runtime.viewport.state import PlaneState, RenderMode
from napari_cuda.server.runtime.viewport.layers import apply_slice_layer_data
from napari_cuda.server.runtime.viewport.roi import viewport_roi_for_level
from napari_cuda.server.runtime.viewport.plane_ops import (
    assign_pose_from_snapshot,
    apply_pose_to_camera,
    mark_slice_applied,
)
from napari_cuda.server.runtime.core import ledger_step
from napari_cuda.server.runtime.core.snapshot_build import RenderLedgerSnapshot
from .viewer_metadata import apply_plane_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliceApplyResult:
    """Outcome of applying a slice snapshot."""

    level: int
    roi: SliceROI
    aligned_roi: SliceROI
    chunk_shape: Optional[Tuple[int, int]]
    width_px: int
    height_px: int


def load_slice(
    worker: Any,
    source: Any,
    level: int,
    z_index: int,
) -> np.ndarray:
    """Load a slice for ``level`` using the worker viewport ROI."""

    roi = viewport_roi_for_level(worker, source, int(level))
    slab = source.slice(int(level), int(z_index), compute=True, roi=roi)
    if not isinstance(slab, np.ndarray):
        slab = np.asarray(slab, dtype=np.float32)
    return slab


def aligned_roi_signature(
    worker: Any,
    source: Any,
    level: int,
    roi: Optional[SliceROI] = None,
) -> tuple[SliceROI, Optional[tuple[int, int]], Optional[tuple[int, int, int, int]]]:
    """Align ``roi`` to the chunk grid (if enabled) and return its signature."""

    roi_val = roi or viewport_roi_for_level(worker, source, int(level))
    chunk_shape = chunk_shape_for_level(source, int(level))
    aligned_roi = roi_val
    if worker._roi_align_chunks and chunk_shape is not None:  # type: ignore[attr-defined]
        full_h, full_w = plane_wh_for_level(source, int(level))
        aligned_roi = align_roi_to_chunk_grid(
            roi_val,
            chunk_shape,
            int(worker._roi_pad_chunks),  # type: ignore[attr-defined]
            height=full_h,
            width=full_w,
        )
    signature = roi_chunk_signature(aligned_roi, chunk_shape)
    return aligned_roi, chunk_shape, signature


def dims_signature(snapshot: RenderLedgerSnapshot) -> tuple:
    """Compute a signature for snapshot dims updates."""

    return (
        int(snapshot.ndisplay) if snapshot.ndisplay is not None else None,
        tuple(int(v) for v in snapshot.order) if snapshot.order is not None else None,
        tuple(int(v) for v in snapshot.displayed) if snapshot.displayed is not None else None,
        tuple(int(v) for v in snapshot.current_step) if snapshot.current_step is not None else None,
        int(snapshot.current_level) if snapshot.current_level is not None else None,
        tuple(str(v) for v in snapshot.axis_labels) if snapshot.axis_labels is not None else None,
    )


def apply_dims_from_snapshot(
    worker: Any,
    snapshot: RenderLedgerSnapshot,
    *,
    signature: tuple,
) -> None:
    """Apply dims metadata from ``snapshot`` back onto the worker viewer."""

    viewer = worker._viewer  # type: ignore[attr-defined]
    if viewer is None:
        return

    worker._last_dims_signature = signature  # type: ignore[attr-defined]

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "dims.apply: ndisplay=%s order=%s displayed=%s current_step=%s",
            str(snapshot.ndisplay),
            str(snapshot.order),
            str(snapshot.displayed),
            str(snapshot.current_step),
        )

    dims = viewer.dims
    ndim = int(getattr(dims, "ndim", 0) or 0)

    axis_labels_src = snapshot.axis_labels
    if axis_labels_src:
        labels = tuple(str(v) for v in axis_labels_src)
        dims.axis_labels = labels
        ndim = max(ndim, len(labels))

    order_src = snapshot.order
    if order_src:
        ndim = max(ndim, len(tuple(int(v) for v in order_src)))

    fallback_order: Optional[tuple[int, ...]] = None
    current_order = getattr(dims, "order", None)
    if current_order is not None:
        fallback_order = tuple(int(v) for v in current_order)

    if snapshot.level_shapes and snapshot.current_level is not None:
        level_shapes = snapshot.level_shapes
        level_idx = int(snapshot.current_level)
        if level_shapes and 0 <= level_idx < len(level_shapes):
            ndim = max(ndim, len(level_shapes[level_idx]))

    if ndim <= 0:
        ndim = max(len(dims.current_step), len(dims.axis_labels)) or 1
    if dims.ndim != ndim:
        dims.ndim = ndim

    if snapshot.current_step is not None:
        step_tuple = tuple(int(v) for v in snapshot.current_step)
        if len(step_tuple) < ndim:
            step_tuple = step_tuple + tuple(0 for _ in range(ndim - len(step_tuple)))
        elif len(step_tuple) > ndim:
            step_tuple = step_tuple[:ndim]
        dims.current_step = step_tuple
    if snapshot.current_level is not None:
        worker._set_current_level_index(int(snapshot.current_level))  # type: ignore[attr-defined]

    if snapshot.order is not None:
        order_values = tuple(int(v) for v in snapshot.order)
    else:
        assert fallback_order is not None, "ledger missing dims order"
        order_values = fallback_order
    assert order_values, "ledger emitted empty dims order"
    dims.order = order_values

    displayed_src = snapshot.displayed
    if displayed_src is not None:
        displayed_tuple = tuple(int(v) for v in displayed_src)
    else:
        current_displayed = getattr(dims, "displayed", None)
        assert current_displayed is not None, "ledger missing dims displayed"
        displayed_tuple = tuple(int(v) for v in current_displayed)
    assert displayed_tuple, "ledger emitted empty dims displayed"
    expected_displayed = tuple(order_values[-len(displayed_tuple):])
    assert displayed_tuple == expected_displayed, "ledger displayed mismatch order/ndisplay"

    if snapshot.ndisplay is not None:
        dims.ndisplay = max(1, int(snapshot.ndisplay))

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "dims.applied: ndim=%d order=%s displayed=%s ndisplay=%d",
            int(dims.ndim),
            str(tuple(dims.order)),
            str(tuple(dims.displayed)),
            int(dims.ndisplay),
        )


def update_z_index_from_snapshot(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Update the cached z-index if the snapshot provides axis labels."""

    if snapshot.axis_labels is None or snapshot.current_step is None:
        return
    labels = [str(label).lower() for label in snapshot.axis_labels]
    if "z" not in labels:
        return
    idx = labels.index("z")
    if idx < len(snapshot.current_step):
        worker._z_index = int(snapshot.current_step[idx])  # type: ignore[attr-defined]


def apply_slice_snapshot(
    worker: Any,
    source: Any,
    snapshot: RenderLedgerSnapshot,
) -> SliceApplyResult:
    """Apply plane metadata, camera pose, and ROI from the snapshot."""

    prev_level = int(worker._current_level_index())  # type: ignore[attr-defined]
    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]
    snapshot_level = int(snapshot.current_level) if snapshot.current_level is not None else None
    target_level = snapshot_level if snapshot_level is not None else prev_level
    if plane_state.target_ndisplay < 3:
        target_level = int(plane_state.target_level)

    step_hint: Optional[tuple[int, ...]] = None
    if plane_state.target_ndisplay < 3 and plane_state.target_step is not None:
        step_hint = tuple(int(v) for v in plane_state.target_step)
    elif snapshot.current_step is not None:
        step_hint = tuple(int(v) for v in snapshot.current_step)
    if step_hint is None:
        recorded_step = ledger_step(worker._ledger)
        if recorded_step is not None:
            step_hint = tuple(int(v) for v in recorded_step)

    was_volume = worker.viewport_state.mode is RenderMode.VOLUME  # type: ignore[attr-defined]
    stage_prev_level = target_level if was_volume else prev_level

    decision = lod.LevelDecision(
        desired_level=int(target_level),
        selected_level=int(target_level),
        reason="direct",
        timestamp=time.perf_counter(),
        oversampling={},
        downgraded=False,
    )
    applied_context = lod.build_level_context(
        decision,
        source=source,
        prev_level=stage_prev_level,
        last_step=step_hint,
    )

    if was_volume:
        worker.viewport_state.mode = RenderMode.PLANE  # type: ignore[attr-defined]
        worker._configure_camera_for_mode()
        worker._last_dims_signature = None
        if hasattr(worker, "_last_plane_pose"):
            worker._last_plane_pose = None  # type: ignore[attr-defined]

    apply_plane_metadata(worker, source, applied_context)
    worker.viewport_state.volume.downgraded = False  # type: ignore[attr-defined]
    apply_slice_camera_pose(worker, snapshot)
    return apply_slice_level(worker, source, applied_context)


def apply_slice_camera_pose(
    worker: Any,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Apply slice camera pose from the snapshot to the active view."""

    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, PanZoomCamera):
        return

    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]
    rect_tuple, center_tuple, zoom_value = assign_pose_from_snapshot(plane_state, snapshot)
    apply_pose_to_camera(
        cam,
        rect=rect_tuple,
        center=center_tuple,
        zoom=zoom_value,
    )


def apply_slice_level(
    worker: Any,
    source: Any,
    applied: lod.LevelContext,
) -> SliceApplyResult:
    """Load the slice for ``applied`` and update worker metadata."""

    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]
    layer = worker._napari_layer  # type: ignore[attr-defined]
    sy, sx = applied.scale_yx

    if layer is not None:
        layer.depiction = "plane"
        layer.rendering = "mip"
        layer.scale = (float(sy), float(sx))
        if hasattr(layer, "_set_view_slice"):
            layer._set_view_slice()  # type: ignore[misc]
        worker._ensure_plane_visual()  # type: ignore[attr-defined]

    view = worker.view
    assert view is not None, "VisPy view must be initialised for 2D apply"

    roi = viewport_roi_for_level(worker, source, int(applied.level))
    aligned_roi, chunk_shape, roi_signature = aligned_roi_signature(
        worker,
        source,
        int(applied.level),
        roi,
    )
    if worker.viewport_state.mode is not RenderMode.VOLUME:  # type: ignore[attr-defined]
        worker._emit_current_camera_pose("slice-apply")
    height_px, width_px = apply_slice_roi(
        worker,
        source,
        int(applied.level),
        aligned_roi,
        update_contrast=not worker._sticky_contrast,
        step=applied.step,
    )

    pose = plane_state.pose
    assert pose.rect is not None and pose.center is not None and pose.zoom is not None, "plane pose must be cached before slice level apply"
    apply_pose_to_camera(
        view.camera,
        rect=pose.rect,
        center=pose.center,
        zoom=pose.zoom,
    )

    worker._layer_logger.log(
        enabled=worker._log_layer_debug,
        mode="slice",
        level=applied.level,
        z_index=worker._z_index,
        shape=(int(height_px), int(width_px)),
        contrast=applied.contrast,
        downgraded=worker.viewport_state.volume.downgraded,  # type: ignore[attr-defined]
    )

    mark_slice_applied(
        plane_state,
        level=int(applied.level),
        step=applied.step,
        roi=aligned_roi,
        roi_signature=roi_signature,
    )

    runner = worker._viewport_runner
    if runner is not None:
        applied_step = (
            tuple(int(v) for v in applied.step) if applied.step is not None else None
        )
        chunk_tuple = (
            (int(chunk_shape[0]), int(chunk_shape[1]))
            if chunk_shape is not None
            else (0, 0)
        )
        runner.mark_slice_applied(
            _create_slice_task(
                level=int(applied.level),
                step=applied_step,
                roi=aligned_roi,
                chunk_shape=chunk_tuple,
                signature=roi_signature,
            )
        )

    worker._last_slice_signature = (
        int(applied.level),
        tuple(int(v) for v in applied.step),
        roi_signature,
    )

    return SliceApplyResult(
        level=int(applied.level),
        roi=roi,
        aligned_roi=aligned_roi,
        chunk_shape=chunk_shape,
        width_px=int(width_px),
        height_px=int(height_px),
    )


def apply_slice_roi(
    worker: Any,
    source: Any,
    level: int,
    roi: SliceROI,
    *,
    update_contrast: bool,
    step: Optional[Sequence[int]] = None,
) -> Tuple[int, int]:
    """Load and apply a slice for the given ROI."""

    aligned_roi, chunk_shape, roi_signature = aligned_roi_signature(worker, source, int(level), roi)

    step_source: Optional[Sequence[int]] = step
    plane_state: PlaneState = worker.viewport_state.plane  # type: ignore[attr-defined]
    if step_source is None:
        if plane_state.target_step is not None:
            step_source = plane_state.target_step
        else:
            step_source = plane_state.applied_step
    step_tuple = tuple(int(v) for v in step_source) if step_source is not None else None

    z_idx = int(worker._z_index or 0)
    slab = source.slice(int(level), z_idx, compute=True, roi=aligned_roi)

    layer = worker._napari_layer
    if layer is not None:
        view = worker.view
        assert view is not None, "VisPy view required for slice apply"
        apply_slice_layer_data(
            layer=layer,
            source=source,
            level=int(level),
            slab=slab,
            roi=aligned_roi,
            update_contrast=bool(update_contrast),
        )

    height_px = int(slab.shape[0])
    width_px = int(slab.shape[1])
    worker._data_wh = (int(width_px), int(height_px))
    worker._data_d = None

    runner = worker._viewport_runner
    if runner is not None:
        if worker.viewport_state.mode is not RenderMode.VOLUME:  # type: ignore[attr-defined]
            chunk_tuple = (
                (int(chunk_shape[0]), int(chunk_shape[1]))
                if chunk_shape is not None
                else (0, 0)
            )
            runner.mark_slice_applied(
                _create_slice_task(
                    level=int(level),
                    step=step_tuple,
                    roi=aligned_roi,
                    chunk_shape=chunk_tuple,
                    signature=roi_signature,
                )
            )
            runner.mark_level_applied(int(level))
            rect = worker._current_panzoom_rect()
            if rect is not None:
                runner.update_camera_rect(rect)

    worker._last_slice_signature = (
        int(level),
        step_tuple,
        roi_signature,
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "updated last slice signature: level=%s step=%s roi_sig=%s",
            int(level),
            step_tuple,
            roi_signature,
        )
    worker._mark_render_tick_needed()
    return height_px, width_px


__all__ = [
    "aligned_roi_signature",
    "apply_dims_from_snapshot",
    "SliceApplyResult",
    "apply_slice_snapshot",
    "apply_slice_camera_pose",
    "apply_slice_level",
    "apply_slice_roi",
    "dims_signature",
    "load_slice",
    "update_z_index_from_snapshot",
]
def _create_slice_task(
    *,
    level: int,
    step: Optional[tuple[int, ...]],
    roi: SliceROI,
    chunk_shape: tuple[int, int],
    signature: Optional[tuple[int, int, int, int]],
):
    from napari_cuda.server.runtime.viewport.runner import SliceTask

    return SliceTask(
        level=level,
        step=step,
        roi=roi,
        chunk_shape=chunk_shape,
        signature=signature,
    )
