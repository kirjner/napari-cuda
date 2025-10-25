"""Volume snapshot application helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

from vispy.scene.cameras import TurntableCamera

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.scene_state_applier import SceneStateApplier

from .state_structs import RenderMode
from .volume_state import assign_pose_from_snapshot, update_level, update_scale
from .viewer_stage import apply_volume_metadata


@dataclass(frozen=True)
class VolumeApplyResult:
    """Outcome of applying a volume snapshot."""

    level: int
    downgraded: bool
    data_wh: Tuple[int, int]
    data_d: Optional[int]
    scale: Tuple[float, float, float]


def apply_volume_snapshot(
    worker: Any,
    source: Any,
    snapshot: RenderLedgerSnapshot,
) -> Optional[VolumeApplyResult]:
    """Apply volume metadata, level selection, and camera pose from the snapshot."""

    prev_level = int(worker._current_level_index())  # type: ignore[attr-defined]
    target_level = (
        int(snapshot.current_level) if snapshot.current_level is not None else prev_level
    )

    step_hint: Optional[tuple[int, ...]]
    if snapshot.current_step is not None:
        step_hint = tuple(int(v) for v in snapshot.current_step)
    else:
        recorded_step = worker._ledger_step()
        step_hint = (
            tuple(int(v) for v in recorded_step)
            if recorded_step is not None
            else None
        )

    was_volume = worker.viewport_state.mode is RenderMode.VOLUME  # type: ignore[attr-defined]
    if not was_volume:
        worker.viewport_state.mode = RenderMode.VOLUME  # type: ignore[attr-defined]
        worker._last_dims_signature = None
        if hasattr(worker, "_last_volume_pose"):
            worker._last_volume_pose = None  # type: ignore[attr-defined]
        runner = worker._viewport_runner
        if runner is not None:
            state = runner.state
            state.level_reload_required = False
            state.awaiting_level_confirm = False
            state.roi_reload_required = False
            state.pending_roi = None
            state.pending_roi_signature = None
            state.applied_roi = None
            state.applied_roi_signature = None
            state.pose_reason = None
            state.camera_pose_dirty = False
    else:
        runner = worker._viewport_runner

    requested_level = int(target_level)
    selected_level, downgraded = worker._resolve_volume_intent_level(source, requested_level)
    effective_level = int(selected_level)
    worker.viewport_state.volume.downgraded = bool(downgraded)  # type: ignore[attr-defined]

    load_needed = (not was_volume) or (int(effective_level) != prev_level)
    result: Optional[VolumeApplyResult] = None
    if load_needed:
        decision = lod.LevelDecision(
            desired_level=int(effective_level),
            selected_level=int(effective_level),
            reason="direct",
            timestamp=time.perf_counter(),
            oversampling={},
            downgraded=bool(downgraded),
        )
        context = lod.build_level_context(
            decision,
            source=source,
            prev_level=prev_level,
            last_step=step_hint,
        )
        apply_volume_metadata(worker, source, context)
        result = apply_volume_level(
            worker,
            source,
            context,
            downgraded=bool(downgraded),
        )
        if runner is not None:
            runner.mark_level_applied(int(context.level))
            rect = worker._current_panzoom_rect()
            runner.update_camera_rect(rect)
        worker._configure_camera_for_mode()

    apply_volume_camera_pose(worker, snapshot)
    return result


def apply_volume_camera_pose(
    worker: Any,
    snapshot: RenderLedgerSnapshot,
) -> None:
    """Apply volume camera pose from the snapshot to the active view."""

    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, TurntableCamera):
        return

    volume_state = worker.viewport_state.volume  # type: ignore[attr-defined]
    assign_pose_from_snapshot(volume_state, snapshot)

    center = volume_state.pose_center
    if center is not None and len(center) >= 3:
        cam.center = (
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )

    angles = volume_state.pose_angles
    if angles is not None and len(angles) >= 2:
        cam.azimuth = float(angles[0])
        cam.elevation = float(angles[1])
        if len(angles) >= 3:
            cam.roll = float(angles[2])  # type: ignore[attr-defined]

    if volume_state.pose_distance is not None:
        cam.distance = float(volume_state.pose_distance)
    if volume_state.pose_fov is not None:
        cam.fov = float(volume_state.pose_fov)


def apply_volume_level(
    worker: Any,
    source: Any,
    applied: lod.LevelContext,
    *,
    downgraded: bool,
) -> VolumeApplyResult:
    """Load the volume level for ``applied`` and update worker metadata."""

    scale_vals = list(source.level_scale(applied.level)) if hasattr(source, "level_scale") else []
    scales = [float(s) for s in scale_vals[-3:]] if scale_vals else []
    while len(scales) < 3:
        scales.insert(0, 1.0)
    scale_tuple = (float(scales[-3]), float(scales[-2]), float(scales[-1]))
    worker._volume_scale = scale_tuple

    volume = worker._get_level_volume(source, applied.level)
    cam = worker.view.camera if getattr(worker, "view", None) is not None else None
    ctx = worker._build_scene_state_context(cam)
    if ctx.volume_scale is None:
        ctx = replace(ctx, volume_scale=scale_tuple)

    data_wh, data_d = SceneStateApplier.apply_volume_layer(
        ctx,
        volume=volume,
        contrast=applied.contrast,
    )
    layer_obj = ctx.layer
    if layer_obj is not None:
        layer_obj.depiction = "volume"
        layer_obj.rendering = "attenuated_mip"
        if hasattr(layer_obj, "_set_view_slice"):
            layer_obj._set_view_slice()  # type: ignore[misc]
        clear_visual = getattr(worker, "_clear_visual", None)
        if callable(clear_visual):
            clear_visual()
    worker._data_wh = data_wh
    worker._data_d = data_d

    shape = (
        (int(data_d), int(data_wh[1]), int(data_wh[0]))
        if data_d is not None
        else (int(data_wh[1]), int(data_wh[0]))
    )

    worker._layer_logger.log(
        enabled=worker._log_layer_debug,
        mode="volume",
        level=applied.level,
        z_index=None,
        shape=shape,
        contrast=applied.contrast,
        downgraded=downgraded,
    )

    volume_state = worker.viewport_state.volume  # type: ignore[attr-defined]
    update_level(volume_state, int(applied.level), downgraded=downgraded)
    update_scale(volume_state, scale_tuple)

    return VolumeApplyResult(
        level=int(applied.level),
        downgraded=bool(downgraded),
        data_wh=(int(data_wh[0]), int(data_wh[1])),
        data_d=int(data_d) if data_d is not None else None,
        scale=scale_tuple,
    )


__all__ = [
    "VolumeApplyResult",
    "apply_volume_snapshot",
    "apply_volume_camera_pose",
    "apply_volume_level",
]
