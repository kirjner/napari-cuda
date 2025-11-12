"""Render update helpers extracted from the EGL worker."""

from __future__ import annotations

import logging
from collections.abc import Iterable, MutableMapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Optional

from napari_cuda.server.runtime.render_loop.applying import (
    apply as snapshot_apply,
    drain as snapshot_drain,
    viewport as snapshot_viewport,
)
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.scene.viewport import PlaneViewportCache, RenderMode, VolumeViewportCache
from napari_cuda.server.scene import (
    LayerVisualState,
    RenderLedgerSnapshot,
)
from napari_cuda.server.scene.blocks import ENABLE_VIEW_AXES_INDEX_BLOCKS
from napari_cuda.server.utils.signatures import snapshot_versions

from .interface import RenderPlanInterface

AppliedVersions = MutableMapping[tuple[str, str, str], int]

apply_render_snapshot = snapshot_apply.apply_render_snapshot
apply_viewport_state_snapshot = snapshot_viewport.apply_viewport_state_snapshot

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


def _plane_pose_complete(snapshot: RenderLedgerSnapshot) -> bool:
    return (
        snapshot.plane_rect is not None
        and snapshot.plane_center is not None
        and snapshot.plane_zoom is not None
        and snapshot.current_step is not None
        and snapshot.current_level is not None
    )


def _plane_cache_from_snapshot(snapshot: RenderLedgerSnapshot) -> PlaneViewportCache:
    plane_state = PlaneViewportCache()
    target_level = int(snapshot.current_level) if snapshot.current_level is not None else plane_state.target_level
    plane_state.target_level = target_level
    plane_state.applied_level = target_level
    plane_state.target_ndisplay = int(snapshot.ndisplay) if snapshot.ndisplay is not None else plane_state.target_ndisplay
    if snapshot.current_step is not None:
        step_tuple = tuple(int(v) for v in snapshot.current_step)
        plane_state.target_step = step_tuple
        plane_state.applied_step = step_tuple
    if snapshot.plane_rect is not None:
        plane_state.update_pose(rect=tuple(float(v) for v in snapshot.plane_rect))
    if snapshot.plane_center is not None and len(snapshot.plane_center) >= 2:
        plane_state.update_pose(center=(float(snapshot.plane_center[0]), float(snapshot.plane_center[1])))
    if snapshot.plane_zoom is not None:
        plane_state.update_pose(zoom=float(snapshot.plane_zoom))
    plane_state.awaiting_level_confirm = False
    plane_state.camera_pose_dirty = False
    plane_state.applied_roi = None
    plane_state.applied_roi_signature = None
    return plane_state


def _volume_pose_complete(snapshot: RenderLedgerSnapshot) -> bool:
    return (
        snapshot.volume_center is not None
        and snapshot.volume_angles is not None
        and snapshot.volume_distance is not None
        and snapshot.volume_fov is not None
        and snapshot.current_level is not None
    )


def _volume_cache_from_snapshot(snapshot: RenderLedgerSnapshot) -> VolumeViewportCache:
    volume_state = VolumeViewportCache()
    target_level = int(snapshot.current_level) if snapshot.current_level is not None else volume_state.level
    volume_state.level = target_level
    if snapshot.volume_center is not None:
        volume_state.update_pose(center=tuple(float(v) for v in snapshot.volume_center))
    if snapshot.volume_angles is not None:
        volume_state.update_pose(angles=tuple(float(v) for v in snapshot.volume_angles))
    if snapshot.volume_distance is not None:
        volume_state.update_pose(distance=float(snapshot.volume_distance))
    if snapshot.volume_fov is not None:
        volume_state.update_pose(fov=float(snapshot.volume_fov))
    return volume_state


def normalize_scene_state(state: RenderLedgerSnapshot) -> RenderLedgerSnapshot:
    """Ledger snapshots already contain canonical values."""
    return state


def record_snapshot_versions(
    applied_versions: AppliedVersions,
    state: RenderLedgerSnapshot,
) -> None:
    snapshot_versions(state).apply(applied_versions)


def extract_layer_changes(
    applied_versions: AppliedVersions,
    state: RenderLedgerSnapshot,
) -> dict[str, LayerVisualState]:
    layer_changes: dict[str, LayerVisualState] = {}
    if not state.layer_values:
        return layer_changes

    for layer_id, layer_state in state.layer_values.items():
        changed_props: list[str] = []
        if layer_state.versions:
            for prop, version in layer_state.versions.items():
                key = ("layer", str(layer_id), str(prop))
                version_value = int(version)
                previous = applied_versions.get(key)
                if previous is not None and previous == version_value:
                    continue
                applied_versions[key] = version_value
                changed_props.append(str(prop))
        else:
            changed_props = list(layer_state.keys())

        if changed_props:
            layer_changes[str(layer_id)] = layer_state.subset(changed_props)

    return layer_changes


def consume_render_snapshot(worker: EGLRendererWorker, state: RenderLedgerSnapshot) -> None:
    """Apply a render snapshot immediately."""

    normalized = normalize_scene_state(state)
    _apply_snapshot(worker, normalized)


def drain_scene_updates(worker: EGLRendererWorker) -> None:  # noqa: D401
    """Legacy hook retained for compatibility."""
    return None


def _apply_snapshot(worker: EGLRendererWorker, state: RenderLedgerSnapshot) -> None:
    tick_iface = RenderPlanInterface(worker)
    apply_iface = RenderApplyInterface(worker)

    snapshot_op_seq = int(state.op_seq) if state.op_seq is not None else 0
    if snapshot_op_seq < int(tick_iface.viewport_state.op_seq):
        logger.debug(
            "render snapshot skipped: stale op_seq snapshot=%d latest=%d",
            int(snapshot_op_seq),
            int(tick_iface.viewport_state.op_seq),
        )
        return
    tick_iface.viewport_state.op_seq = int(snapshot_op_seq)

    applied_versions = tick_iface.applied_versions
    record_snapshot_versions(applied_versions, state)
    layer_changes = extract_layer_changes(applied_versions, state)

    state_for_apply = state
    if layer_changes:
        state_for_apply = replace(state, layer_values=layer_changes)
    elif state.layer_values is not None:
        state_for_apply = replace(state, layer_values=None)

    if ENABLE_VIEW_AXES_INDEX_BLOCKS and _plane_pose_complete(state):
        plane_state = _plane_cache_from_snapshot(state)
        tick_iface.viewport_state.plane = plane_state
        if tick_iface.viewport_runner is not None:
            tick_iface.viewport_runner._plane = plane_state  # type: ignore[attr-defined]
    if ENABLE_VIEW_AXES_INDEX_BLOCKS and _volume_pose_complete(state):
        tick_iface.viewport_state.volume = _volume_cache_from_snapshot(state)

    previous_mode = tick_iface.viewport_state.mode
    mode_update = previous_mode
    dims_mode = getattr(state, "dims_mode", None)
    if isinstance(dims_mode, str):
        try:
            mode_update = RenderMode[dims_mode.upper()]
        except KeyError:
            mode_update = previous_mode
    signature_changed = True

    if signature_changed:
        apply_render_snapshot(apply_iface, state_for_apply)

    if mode_update is not None:
        tick_iface.viewport_state.mode = mode_update
    else:
        tick_iface.viewport_state.mode = previous_mode

    if tick_iface.viewport_state.mode is RenderMode.VOLUME:
        tick_iface.level_policy_suppressed = False
    elif mode_update is RenderMode.PLANE or previous_mode is RenderMode.VOLUME:
        tick_iface.level_policy_suppressed = True

    if signature_changed and tick_iface.viewport_runner is not None:
        tick_iface.viewport_runner.ingest_snapshot(state)

    drain_res = snapshot_drain.drain_render_state(worker, state_for_apply)

    if drain_res.z_index is not None:
        apply_iface.set_z_index(int(drain_res.z_index))
    if drain_res.data_wh is not None:
        apply_iface.set_data_shape(int(drain_res.data_wh[0]), int(drain_res.data_wh[1]))

    runner = tick_iface.viewport_runner
    if runner is not None and tick_iface.viewport_state.mode is RenderMode.PLANE:
        rect = tick_iface.current_panzoom_rect()
        if rect is not None:
            runner.update_camera_rect(rect)
        if tick_iface.current_level_index() == int(runner.state.target_level):
            runner.mark_level_applied(tick_iface.current_level_index())
    elif runner is not None and tick_iface.current_level_index() == int(runner.state.target_level):
        runner.mark_level_applied(tick_iface.current_level_index())

    if (
        drain_res.render_marked
        and tick_iface.viewport_state.mode is not RenderMode.VOLUME
        and not tick_iface.level_policy_suppressed
    ):
        tick_iface.evaluate_level_policy()

    if tick_iface.level_policy_suppressed:
        ledger = tick_iface.ledger
        assert ledger is not None, "ledger must be attached before rendering"
        op_kind_entry = ledger.get("scene", "main", "op_kind")
        if op_kind_entry is not None and str(op_kind_entry.value) == "dims-update":
            tick_iface.level_policy_suppressed = False


__all__ = [
    "AppliedVersions",
    "apply_viewport_state_snapshot",
    "consume_render_snapshot",
    "drain_scene_updates",
    "extract_layer_changes",
    "normalize_scene_state",
    "record_snapshot_versions",
]
