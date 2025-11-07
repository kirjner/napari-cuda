"""Render update helpers extracted from the EGL worker."""

from __future__ import annotations

import logging
from collections.abc import Iterable, MutableMapping
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Optional

from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdate
from napari_cuda.server.runtime.render_loop.applying import (
    apply as snapshot_apply,
    drain as snapshot_drain,
    viewport as snapshot_viewport,
)
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import (
    LayerVisualState,
    RenderLedgerSnapshot,
)
from napari_cuda.server.utils.signatures import snapshot_versions

from .interface import RenderPlanInterface

apply_render_snapshot = snapshot_apply.apply_render_snapshot
apply_viewport_state_snapshot = snapshot_viewport.apply_viewport_state_snapshot

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


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
    """Queue a complete scene state snapshot for the next frame."""

    normalized = normalize_scene_state(state)
    tick_iface = RenderPlanInterface(worker)
    tick_iface.render_mailbox_set_scene_state(normalized)
    tick_iface.mark_render_tick_needed()
    tick_iface.update_last_interaction_timestamp()


def drain_scene_updates(worker: EGLRendererWorker) -> None:
    tick_iface = RenderPlanInterface(worker)
    updates: RenderUpdate = tick_iface.render_mailbox_drain()
    apply_iface = RenderApplyInterface(worker)
    state = updates.scene_state
    if state is None:
        apply_viewport_state_snapshot(
            apply_iface,
            mode=updates.mode,
            plane_state=updates.plane_state,
            volume_state=updates.volume_state,
        )
        return

    snapshot_op_seq = updates.op_seq
    if snapshot_op_seq is None:
        snapshot_op_seq = int(state.op_seq) if state.op_seq is not None else 0
    if int(snapshot_op_seq) < int(tick_iface.viewport_state.op_seq):
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

    if layer_changes:
        state_for_apply = replace(state, layer_values=layer_changes)
    else:
        state_for_apply = replace(state, layer_values=None)

    if updates.plane_state is not None:
        tick_iface.viewport_state.plane = deepcopy(updates.plane_state)
        if tick_iface.viewport_runner is not None:
            tick_iface.viewport_runner._plane = tick_iface.viewport_state.plane  # type: ignore[attr-defined]
    if updates.volume_state is not None:
        tick_iface.viewport_state.volume = deepcopy(updates.volume_state)

    previous_mode = tick_iface.viewport_state.mode
    mode_update = updates.mode
    mode_changed = mode_update is not None and mode_update is not previous_mode
    signature_changed = tick_iface.render_mailbox_update_signature(state)
    if mode_changed:
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
    "apply_viewport_state_snapshot",
    "consume_render_snapshot",
    "drain_scene_updates",
    "extract_layer_changes",
    "normalize_scene_state",
    "record_snapshot_versions",
]
