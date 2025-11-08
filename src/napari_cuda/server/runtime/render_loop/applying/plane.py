"""Slice snapshot application helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from vispy.scene.cameras import PanZoomCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.data import (
    SliceROI,
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
    roi_chunk_signature,
)
from napari_cuda.server.data.roi import plane_wh_for_level
from napari_cuda.server.runtime.lod.context import build_level_context
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.runtime.render_loop.applying.layer_data import apply_slice_layer_data
from napari_cuda.server.runtime.render_loop.applying.plane_ops import (
    apply_pose_to_camera,
    assign_pose_from_snapshot,
    mark_slice_applied,
)
from napari_cuda.server.scene.viewport import PlaneState, RenderMode
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.utils.signatures import SignatureToken
from napari_cuda.shared.dims_spec import (
    DimsSpec,
    dims_spec_axis_labels,
    dims_spec_displayed,
    dims_spec_order,
)

from .viewer_metadata import apply_plane_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliceApplyResult:
    """Outcome of applying a slice snapshot."""

    level: int
    roi: SliceROI
    aligned_roi: SliceROI
    chunk_shape: Optional[tuple[int, int]]
    width_px: int
    height_px: int



def aligned_roi_signature(
    snapshot_iface: RenderApplyInterface,
    source: Any,
    level: int,
    roi: Optional[SliceROI] = None,
) -> tuple[SliceROI, Optional[tuple[int, int]], Optional[tuple[int, int, int, int]]]:
    """Align ``roi`` to the chunk grid (if enabled) and return its signature."""

    roi_val = roi or snapshot_iface.viewport_roi_for_level(source, int(level))
    chunk_shape = chunk_shape_for_level(source, int(level))
    aligned_roi = roi_val
    if snapshot_iface.roi_align_chunks and chunk_shape is not None:
        full_h, full_w = plane_wh_for_level(source, int(level))
        aligned_roi = align_roi_to_chunk_grid(
            roi_val,
            chunk_shape,
            int(snapshot_iface.roi_pad_chunks),
            height=full_h,
            width=full_w,
        )
    signature = roi_chunk_signature(aligned_roi, chunk_shape)
    return aligned_roi, chunk_shape, signature


def apply_dims_from_snapshot(
    snapshot_iface: RenderApplyInterface,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Apply dims metadata from ``snapshot`` back onto the worker viewer."""

    viewer = snapshot_iface.viewer
    if viewer is None:
        return


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

    dims_spec: DimsSpec | None = snapshot.dims_spec  # type: ignore[assignment]
    assert dims_spec is not None, "render snapshot missing dims_spec"

    ndim = max(ndim, int(dims_spec.ndim))
    labels = dims_spec_axis_labels(dims_spec)
    dims.axis_labels = labels

    step_tuple = dims_spec.current_step
    order_values = dims_spec_order(dims_spec)
    displayed_tuple = dims_spec_displayed(dims_spec)
    target_ndisplay = int(dims_spec.ndisplay)

    if snapshot.axis_labels is not None:
        assert tuple(str(v) for v in snapshot.axis_labels) == labels
    if snapshot.order is not None:
        assert tuple(int(v) for v in snapshot.order) == order_values
    if snapshot.displayed is not None:
        assert tuple(int(v) for v in snapshot.displayed) == displayed_tuple

    if dims_spec.level_shapes and dims_spec.current_level is not None:
        level_idx = int(dims_spec.current_level)
        level_shapes_spec = dims_spec.level_shapes
        if level_shapes_spec and 0 <= level_idx < len(level_shapes_spec):
            ndim = max(ndim, len(level_shapes_spec[level_idx]))
    if dims.ndim != ndim:
        dims.ndim = ndim

    # Do not mutate worker level caches here. Dims apply should only update
    # the napari viewer's dims model. Level switching is handled by the
    # planner/apply stage to ensure consistency with cached state.

    if len(step_tuple) < ndim:
        step_tuple = step_tuple + tuple(0 for _ in range(ndim - len(step_tuple)))
    elif len(step_tuple) > ndim:
        step_tuple = step_tuple[:ndim]
    dims.current_step = step_tuple

    assert order_values, "ledger emitted empty dims order"
    dims.order = order_values

    dims.ndisplay = max(1, target_ndisplay)

    assert displayed_tuple, "dims spec emitted empty displayed"
    computed_displayed = tuple(int(v) for v in dims.displayed)
    assert computed_displayed == displayed_tuple, "napari displayed mismatch order/ndisplay"

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "dims.applied: ndim=%d order=%s displayed=%s ndisplay=%d",
            int(dims.ndim),
            str(tuple(dims.order)),
            str(tuple(dims.displayed)),
            int(dims.ndisplay),
        )


def update_z_index_from_snapshot(
    snapshot_iface: RenderApplyInterface, snapshot: RenderLedgerSnapshot
) -> None:
    """Update the cached z-index if the snapshot provides axis labels."""

    if snapshot.axis_labels is None or snapshot.current_step is None:
        return
    labels = [str(label).lower() for label in snapshot.axis_labels]
    if "z" not in labels:
        return
    idx = labels.index("z")
    if idx < len(snapshot.current_step):
        snapshot_iface.set_z_index(int(snapshot.current_step[idx]))


def apply_slice_snapshot(
    snapshot_iface: RenderApplyInterface,
    source: Any,
    snapshot: RenderLedgerSnapshot,
) -> SliceApplyResult:
    """Apply plane metadata, camera pose, and ROI from the snapshot."""

    plane_state: PlaneState = snapshot_iface.viewport_state.plane
    dims_spec = snapshot.dims_spec
    assert dims_spec is not None, "render snapshot missing dims_spec"

    was_volume = snapshot_iface.viewport_state.mode is RenderMode.VOLUME
    level_idx = int(dims_spec.current_level)
    step_tuple = tuple(int(v) for v in dims_spec.current_step)
    applied_context = build_level_context(
        source=source,
        level=level_idx,
        step=step_tuple,
    )

    if was_volume:
        snapshot_iface.viewport_state.mode = RenderMode.PLANE
        snapshot_iface.configure_camera_for_mode()
        snapshot_iface.reset_last_plane_pose()

    apply_plane_metadata(worker, source, applied_context)
    apply_slice_camera_pose(snapshot_iface, snapshot)
    return apply_slice_level(snapshot_iface, source, applied_context)


def apply_slice_camera_pose(
    snapshot_iface: RenderApplyInterface,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Apply slice camera pose from the snapshot to the active view."""

    view = snapshot_iface.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, PanZoomCamera):
        return

    plane_state: PlaneState = snapshot_iface.viewport_state.plane
    rect_tuple, center_tuple, zoom_value = assign_pose_from_snapshot(plane_state, snapshot)
    apply_pose_to_camera(
        cam,
        rect=rect_tuple,
        center=center_tuple,
        zoom=zoom_value,
    )


def apply_slice_level(
    snapshot_iface: RenderApplyInterface,
    source: Any,
    applied: lod.LevelContext,
) -> SliceApplyResult:
    """Load the slice for ``applied`` and update worker metadata."""

    plane_state: PlaneState = snapshot_iface.viewport_state.plane
    layer = snapshot_iface.napari_layer
    sy, sx = applied.scale_yx

    if layer is not None:
        layer.depiction = "plane"
        layer.rendering = "mip"
        layer.scale = (float(sy), float(sx))
        if hasattr(layer, "_set_view_slice"):
            layer._set_view_slice()  # type: ignore[misc]
        visible_flag = bool(getattr(layer, "visible", True))
        visual = snapshot_iface.ensure_plane_visual()
        visual.visible = visible_flag  # type: ignore[attr-defined]

    view = snapshot_iface.view
    assert view is not None, "VisPy view must be initialised for 2D apply"

    roi = snapshot_iface.viewport_roi_for_level(source, int(applied.level))
    aligned_roi, chunk_shape, roi_signature = aligned_roi_signature(
        snapshot_iface,
        source,
        int(applied.level),
        roi,
    )
    if snapshot_iface.viewport_state.mode is not RenderMode.VOLUME:
        snapshot_iface.emit_current_camera_pose("slice-apply")
    height_px, width_px = apply_slice_roi(
        snapshot_iface,
        source,
        int(applied.level),
        aligned_roi,
        update_contrast=not snapshot_iface.sticky_contrast,
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

    layer_logger = snapshot_iface.layer_logger
    if layer_logger is not None:
        layer_logger.log(
            enabled=snapshot_iface.log_layer_debug,
            mode="slice",
            level=applied.level,
            z_index=snapshot_iface.z_index(),
            shape=(int(height_px), int(width_px)),
            contrast=applied.contrast,
        )

    mark_slice_applied(
        plane_state,
        level=int(applied.level),
        step=applied.step,
        roi=aligned_roi,
        roi_signature=roi_signature,
    )

    runner = snapshot_iface.viewport_runner
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

    slice_token = SignatureToken(
        (
            int(applied.level),
            tuple(int(v) for v in applied.step),
            roi_signature,
        )
    )
    snapshot_iface.set_last_slice_signature(slice_token)

    return SliceApplyResult(
        level=int(applied.level),
        roi=roi,
        aligned_roi=aligned_roi,
        chunk_shape=chunk_shape,
        width_px=int(width_px),
        height_px=int(height_px),
    )


def apply_slice_roi(
    snapshot_iface: RenderApplyInterface,
    source: Any,
    level: int,
    roi: SliceROI,
    *,
    update_contrast: bool,
    step: Optional[Sequence[int]] = None,
) -> tuple[int, int]:
    """Load and apply a slice for the given ROI."""

    aligned_roi, chunk_shape, roi_signature = aligned_roi_signature(
        snapshot_iface, source, int(level), roi
    )

    step_source: Optional[Sequence[int]] = step
    plane_state: PlaneState = snapshot_iface.viewport_state.plane
    if step_source is None:
        if plane_state.target_step is not None:
            step_source = plane_state.target_step
        else:
            step_source = plane_state.applied_step
    step_tuple = tuple(int(v) for v in step_source) if step_source is not None else None

    z_idx = int(snapshot_iface.z_index() or 0)
    slab = source.slice(int(level), z_idx, compute=True, roi=aligned_roi)

    layer = snapshot_iface.napari_layer
    if layer is not None:
        view = snapshot_iface.view
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
    snapshot_iface.set_data_shape(width_px, height_px)
    snapshot_iface.set_data_depth(None)

    runner = snapshot_iface.viewport_runner
    if runner is not None:
        if snapshot_iface.viewport_state.mode is not RenderMode.VOLUME:
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
            rect = snapshot_iface.current_panzoom_rect()
            if rect is not None:
                runner.update_camera_rect(rect)

    slice_token = SignatureToken(
        (
            int(level),
            step_tuple,
            roi_signature,
        )
    )
    snapshot_iface.set_last_slice_signature(slice_token)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "updated last slice signature: level=%s step=%s roi_sig=%s",
            int(level),
            step_tuple,
            roi_signature,
        )
    snapshot_iface.mark_render_tick_needed()
    return height_px, width_px


__all__ = [
    "SliceApplyResult",
    "aligned_roi_signature",
    "apply_dims_from_snapshot",
    "apply_slice_camera_pose",
    "apply_slice_level",
    "apply_slice_roi",
    "apply_slice_snapshot",
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
    from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask

    return SliceTask(
        level=level,
        step=step,
        roi=roi,
        chunk_shape=chunk_shape,
        signature=signature,
    )
