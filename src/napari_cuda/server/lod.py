"""Level-of-detail (multiscale) helpers.

Pure, testable functions that decide and apply multiscale level changes and
compute slice ROIs from the active view. This module centralizes logic that was
previously embedded in the worker so we can make napari the authority for level
choice while keeping our stabilizers and budgets.

Phase A: helpers only; wiring happens in the worker in a later step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple
import logging
import math

import numpy as np

from napari.components.viewer_model import ViewerModel
from napari_cuda.server.zarr_source import ZarrSceneSource
from napari_cuda.server.scene_types import SliceROI
from napari_cuda.server.roi import plane_scale_for_level, plane_wh_for_level


logger = logging.getLogger(__name__)


# ---- Types -------------------------------------------------------------------


class LevelBudgetError(RuntimeError):
    """Raised when a multiscale level exceeds configured budgets."""


@dataclass(frozen=True)
class LevelDecision:
    desired_level: Optional[int]
    selected_level: Optional[int]
    applied_level: int
    roi: Optional[SliceROI]
    reason: str
    needs_apply: bool


@dataclass(frozen=True)
class AppliedLevel:
    level: int
    step: Tuple[int, ...]
    z_index: Optional[int]
    shape: Tuple[int, ...]
    scale_yx: Tuple[float, float]
    contrast: Tuple[float, float]
    axes: str
    dtype: str


@dataclass(frozen=True)
class LevelPolicyConfig:
    threshold_in: float
    threshold_out: float
    fine_threshold: float
    hysteresis: float
    cooldown_ms: float


@dataclass(frozen=True)
class LevelPolicyInputs:
    current_level: int
    oversampling: Mapping[int, float]
    zoom_ratio: Optional[float]
    lock_level: Optional[int]
    last_switch_ts: float
    now_ts: float


@dataclass(frozen=True)
class LevelPolicyDecision:
    desired_level: int
    selected_level: int
    action: str
    should_switch: bool
    blocked_reason: Optional[str]
    cooldown_remaining_ms: float = 0.0


# ---- Stabilizer --------------------------------------------------------------


def stabilize_level(napari_level: int, prev_level: int, *, hysteresis: float = 0.0) -> int:
    """Return a stabilized level around the previous level.

    For Phase A this is a pass-through unless hysteresis>0 (then clamp single-
    step oscillations). More sophisticated policies can plug in here later.
    """
    cur = int(prev_level)
    nxt = int(napari_level)
    if hysteresis <= 0.0:
        return nxt
    # Simple clamp: require two-step intent to move away from prev_level
    if abs(nxt - cur) == 1:
        return cur
    return nxt


# ---- ROI computation ---------------------------------------------------------


def _clamp_level(value: int, *, max_level: int) -> int:
    return max(0, min(int(value), int(max_level)))


def select_level(
    config: LevelPolicyConfig,
    inputs: LevelPolicyInputs,
) -> LevelPolicyDecision:
    overs_map = {int(k): float(v) for k, v in inputs.oversampling.items() if v is not None}
    if not overs_map:
        return LevelPolicyDecision(
            desired_level=int(inputs.current_level),
            selected_level=int(inputs.current_level),
            action="no-oversampling",
            should_switch=False,
            blocked_reason="no-oversampling",
        )

    current = int(inputs.current_level)
    max_level = max(overs_map.keys())

    desired: Optional[int] = None
    action = "heuristic"
    zoom_ratio = inputs.zoom_ratio
    if zoom_ratio is not None:
        eps = 1e-3
        if zoom_ratio < 1.0 - eps and current > 0:
            desired = current - 1
            action = "zoom-in"
        elif zoom_ratio > 1.0 + eps and current < max_level:
            desired = current + 1
            action = "zoom-out"

    if desired is None:
        desired = current
        if current > 0:
            finer = current - 1
            ratio = overs_map.get(finer)
            if ratio is not None and ratio <= config.fine_threshold:
                desired = finer
        if desired == current and current < max_level:
            ratio_cur = overs_map.get(current)
            if ratio_cur is not None and ratio_cur > config.threshold_out:
                desired = current + 1
        action = "heuristic"
    else:
        desired = _clamp_level(desired, max_level=max_level)
        if desired < current:
            desired = max(desired, current - 1)
        elif desired > current:
            desired = min(desired, current + 1)

    selected = stabilize_level(int(desired), current, hysteresis=config.hysteresis)
    sel = int(selected)

    # Respect zoom hints by tightening the outgoing threshold when the user is
    # explicitly zooming in or out. This mirrors the previous inline policy in
    # ``EGLRendererWorker`` and preserves the softer 1.20 band for hint-driven
    # zoom-outs so they may trigger sooner than heuristic switching.
    thr_in = float(config.threshold_in)
    thr_out = float(config.threshold_out)
    if action == "zoom-in":
        thr_in = min(1.20, thr_in)
    elif action == "zoom-out":
        thr_out = min(1.20, thr_out)

    if sel < current:
        if overs_map.get(sel, float("inf")) > thr_in:
            sel = current
    elif sel > current:
        if overs_map.get(current, 0.0) <= thr_out:
            sel = current
    selected = sel

    reason = action
    if inputs.lock_level is not None:
        selected = int(inputs.lock_level)
        reason = "locked"

    if selected == current:
        return LevelPolicyDecision(
            desired_level=int(desired),
            selected_level=int(selected),
            action=reason,
            should_switch=False,
            blocked_reason=None,
        )

    if inputs.last_switch_ts > 0.0:
        elapsed_ms = (inputs.now_ts - inputs.last_switch_ts) * 1000.0
        if elapsed_ms < config.cooldown_ms:
            remaining = max(0.0, config.cooldown_ms - elapsed_ms)
            return LevelPolicyDecision(
                desired_level=int(desired),
                selected_level=int(selected),
                action=reason,
                should_switch=False,
                blocked_reason="cooldown",
                cooldown_remaining_ms=remaining,
            )

    return LevelPolicyDecision(
        desired_level=int(desired),
        selected_level=int(selected),
        action=reason,
        should_switch=True,
        blocked_reason=None,
    )


def compute_viewport_roi(
    *,
    viewer: ViewerModel,
    source: ZarrSceneSource,
    level: int,
    align_chunks: bool = True,
    edge_threshold: int = 4,
    prev_roi: Optional[SliceROI] = None,
    init_fullframe: bool = False,
) -> SliceROI:
    """Compute ROI in target level index space from the current viewport.

    - Uses view.scene.transform to derive world bounds, converts to index space
      using level scale, clamps to plane shape.
    - Optionally aligns to chunk boundaries and applies a small-move hysteresis
      relative to ``prev_roi``.
    - If ``init_fullframe`` is True, returns a full-frame ROI for the very
      first call to guarantee visible content.
    """
    h, w = plane_wh_for_level(source, level)
    if init_fullframe:
        return SliceROI(0, h, 0, w)

    view = getattr(viewer, "_view", None) or getattr(viewer, "_qt_window", None)
    canvas = getattr(viewer, "canvas", None)
    # Fallback to private attribute used in our adapter path
    if canvas is None and hasattr(viewer, "_canvas"):
        canvas = getattr(viewer, "_canvas")

    # We expect a vispy ViewBox with a scene graph; compute world-space bounds
    sy, sx = plane_scale_for_level(source, level)
    sy = max(1e-12, float(sy))
    sx = max(1e-12, float(sx))

    vis_view = getattr(getattr(viewer, "_view", None), "view", None)
    if vis_view is None and hasattr(viewer, "window"):
        vis_view = getattr(getattr(viewer, "window", None), "_qt_viewer", None)
    if vis_view is None:
        raise RuntimeError("No vispy view available for ROI compute")

    scene = getattr(vis_view, "scene", None)
    transform = getattr(scene, "transform", None) if scene is not None else None
    if transform is None or not hasattr(transform, "imap"):
        raise RuntimeError("No transform.imap available for ROI compute")

    width = int(getattr(vis_view, "size", (w, h))[0])
    height = int(getattr(vis_view, "size", (w, h))[1])
    corners = (
        (0.0, 0.0),
        (float(width), 0.0),
        (0.0, float(height)),
        (float(width), float(height)),
    )
    world_pts = [transform.imap((float(x), float(y), 0.0)) for x, y in corners]
    xs = [float(pt[0]) for pt in world_pts]
    ys = [float(pt[1]) for pt in world_pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    x_start = int(math.floor(min(x0, x1) / sx))
    x_stop = int(math.ceil(max(x0, x1) / sx))
    y_start = int(math.floor(min(y0, y1) / sy))
    y_stop = int(math.ceil(max(y0, y1) / sy))
    roi = SliceROI(y_start, y_stop, x_start, x_stop).clamp(h, w)

    # Align to chunk boundaries to stabilize IO and avoid sub-chunk phasing
    if align_chunks:
        arr = source.get_level(level)
        chunks = getattr(arr, "chunks", None)
        if chunks is not None:
            lower = [str(a).lower() for a in source.axes]
            y_pos = lower.index("y") if "y" in lower else max(0, len(chunks) - 2)
            x_pos = lower.index("x") if "x" in lower else max(0, len(chunks) - 1)
            cy = int(chunks[y_pos]) if 0 <= y_pos < len(chunks) else 1
            cx = int(chunks[x_pos]) if 0 <= x_pos < len(chunks) else 1
            cy = max(1, cy)
            cx = max(1, cx)
            ys = (roi.y_start // cy) * cy
            ye = ((roi.y_stop + cy - 1) // cy) * cy
            xs = (roi.x_start // cx) * cx
            xe = ((roi.x_stop + cx - 1) // cx) * cx
            roi = SliceROI(ys, ye, xs, xe).clamp(h, w)

    # Small-move hysteresis relative to previous ROI
    if prev_roi is not None and edge_threshold > 0:
        thr = int(edge_threshold)
        if (
            abs(roi.y_start - prev_roi.y_start) < thr
            and abs(roi.y_stop - prev_roi.y_stop) < thr
            and abs(roi.x_start - prev_roi.x_start) < thr
            and abs(roi.x_stop - prev_roi.x_stop) < thr
        ):
            roi = prev_roi

    if roi.is_empty():
        return SliceROI(0, h, 0, w)
    return roi


# ---- Budget checks -----------------------------------------------------------


def assert_volume_within_budget(
    source: ZarrSceneSource,
    level: int,
    *,
    max_bytes: int | None,
    max_voxels: int | None,
) -> None:
    """Raise if level volume exceeds memory/voxel budgets.

    Budgets are inclusive; 0 or None means no budget.
    """
    if not max_bytes and not max_voxels:
        return
    arr = source.get_level(level)
    dtype_sz = int(np.dtype(getattr(arr, "dtype", source.dtype)).itemsize)
    shape = tuple(int(s) for s in getattr(arr, "shape", ()))
    vox = 1
    for dim in shape:
        vox *= max(1, dim)
    bytes_est = int(vox) * int(dtype_sz)
    if max_voxels and vox > int(max_voxels):
        raise LevelBudgetError(f"volume voxels over budget: {vox} > {int(max_voxels)}")
    if max_bytes and bytes_est > int(max_bytes):
        raise LevelBudgetError(f"volume bytes over budget: {bytes_est} > {int(max_bytes)}")


def assert_slice_within_budget(
    source: ZarrSceneSource,
    level: int,
    *,
    dtype_size: int | None,
    max_bytes: int | None,
) -> None:
    """Raise if a full-frame slice at this level would exceed budget.

    Use ROI-aware estimates from SceneSource where available when integrating.
    """
    if not max_bytes:
        return
    desc = source.level_descriptors[level]
    axes = source.axes
    lower = [str(a).lower() for a in axes]
    try:
        y_pos = lower.index("y")
    except ValueError:
        y_pos = max(0, len(desc.shape) - 2)
    try:
        x_pos = lower.index("x")
    except ValueError:
        x_pos = max(0, len(desc.shape) - 1)
    h = int(desc.shape[y_pos]) if 0 <= y_pos < len(desc.shape) else int(desc.shape[-2])
    w = int(desc.shape[x_pos]) if 0 <= x_pos < len(desc.shape) else int(desc.shape[-1])
    if dtype_size is None:
        dtype_size = int(np.dtype(source.dtype).itemsize)
    bytes_est = int(h) * int(w) * int(dtype_size)
    if bytes_est > int(max_bytes):
        raise LevelBudgetError(f"slice bytes over budget: {bytes_est} > {int(max_bytes)}")


# ---- Application -------------------------------------------------------------


def _proportional_z_remap(
    *,
    prev_level: Optional[int],
    prev_shape: Optional[Sequence[int]],
    new_shape: Sequence[int],
    axes: Sequence[str],
    current_step: Optional[Sequence[int]],
    current_z: Optional[int],
) -> Optional[int]:
    lower = [str(a).lower() for a in axes]
    if "z" not in lower:
        return current_z
    try:
        zi = lower.index("z")
    except Exception:
        return current_z
    try:
        z_src = int(prev_shape[zi]) if prev_shape is not None else None
    except Exception:
        z_src = None
    try:
        z_tgt = int(new_shape[zi])
    except Exception:
        return current_z
    if z_src is None or z_src <= 1 or z_tgt <= 1:
        return 0 if z_tgt > 0 and current_z is None else current_z
    if current_z is None:
        if current_step is not None and len(current_step) > zi:
            current_z = int(current_step[zi])
        else:
            current_z = 0
    new_z = int(round(float(current_z) * float(max(0, z_tgt - 1)) / float(max(1, z_src - 1))))
    return max(0, min(int(new_z), int(z_tgt - 1)))


def set_dims_range_for_level(viewer: Optional[ViewerModel], source: ZarrSceneSource, level: int) -> None:
    if viewer is None:
        return
    desc = source.level_descriptors[level]
    shape = tuple(int(s) for s in desc.shape)
    ranges = tuple((0, max(0, s - 1), 1) for s in shape)
    try:
        viewer.dims.range = ranges
    except Exception:
        logger.debug("set_dims_range_for_level failed", exc_info=True)


def apply_level(
    *,
    source: ZarrSceneSource,
    target_level: int,
    prev_level: Optional[int],
    last_step: Optional[Sequence[int]],
    viewer: Optional[ViewerModel],
) -> AppliedLevel:
    """Apply target level to the scene source and viewer; return applied snapshot.

    - Preserves/adjusts Z proportionally across depth changes.
    - Clamps step to descriptor shape; updates viewer dims and range.
    - Computes contrast limits lazily and returns applied metadata.
    """
    axes = source.axes
    lower = [str(a).lower() for a in axes]
    desc = source.level_descriptors[target_level]

    prev_shape: Optional[Sequence[int]] = None
    if prev_level is not None and 0 <= int(prev_level) < len(source.level_descriptors):
        prev_shape = source.level_descriptors[int(prev_level)].shape

    # Current Z guess (None -> derive from last_step)
    cur_z = None
    if last_step is not None and "z" in lower:
        zi = lower.index("z")
        if len(last_step) > zi:
            try:
                cur_z = int(last_step[zi])
            except Exception:
                cur_z = None

    # Proportional remap if needed
    new_z = _proportional_z_remap(
        prev_level=prev_level,
        prev_shape=prev_shape,
        new_shape=desc.shape,
        axes=axes,
        current_step=last_step,
        current_z=cur_z,
    )

    # Build step hint and set level
    step_hint: list[int] = [0] * len(desc.shape)
    if new_z is not None and "z" in lower:
        zi = lower.index("z")
        if 0 <= zi < len(step_hint):
            step_hint[zi] = int(new_z)

    step = source.set_current_level(int(target_level), step=tuple(step_hint))

    # Sync viewer: set range first, then current_step to avoid clamping to old range
    if viewer is not None:
        try:
            set_dims_range_for_level(viewer, source, int(target_level))
            viewer.dims.current_step = tuple(int(x) for x in step)
        except Exception:
            logger.debug("apply_level: syncing viewer dims failed", exc_info=True)

    # Contrast and scale
    contrast = source.ensure_contrast(level=int(target_level))
    sy, sx = plane_scale_for_level(source, int(target_level))
    axes_str = "".join(str(a) for a in axes)
    dtype_str = str(source.dtype)

    # z-index from applied step
    z_index: Optional[int] = None
    if "z" in lower and len(step) > lower.index("z"):
        z_index = int(step[lower.index("z")])
    elif step:
        z_index = int(step[0])

    return AppliedLevel(
        level=int(target_level),
        step=tuple(int(x) for x in step),
        z_index=z_index,
        shape=tuple(int(s) for s in desc.shape),
        scale_yx=(float(sy), float(sx)),
        contrast=(float(contrast[0]), float(contrast[1])),
        axes=axes_str,
        dtype=dtype_str,
    )


__all__ = [
    "LevelBudgetError",
    "LevelDecision",
    "AppliedLevel",
    "LevelPolicyConfig",
    "LevelPolicyDecision",
    "LevelPolicyInputs",
    "stabilize_level",
    "compute_viewport_roi",
    "assert_volume_within_budget",
    "assert_slice_within_budget",
    "apply_level",
    "select_level",
    "set_dims_range_for_level",
]
