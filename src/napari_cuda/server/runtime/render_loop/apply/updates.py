"""Render update helpers extracted from the EGL worker."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdate
from napari_cuda.server.runtime.render_loop.apply.render_state import apply as snapshot_apply
from napari_cuda.server.runtime.render_loop.apply.render_state import viewport as snapshot_viewport
from napari_cuda.server.viewstate import (
    RenderLedgerSnapshot,
)
from napari_cuda.server.runtime.render_loop.apply_interface import RenderApplyInterface
from napari_cuda.server.runtime.viewport import updates as viewport_updates
from napari_cuda.server.runtime.viewport.state import RenderMode

from ..plan_interface import RenderPlanInterface

apply_render_snapshot = snapshot_apply.apply_render_snapshot
apply_viewport_state_snapshot = snapshot_viewport.apply_viewport_state_snapshot

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


def normalize_scene_state(state: RenderLedgerSnapshot) -> RenderLedgerSnapshot:
    """Return a render-thread-friendly copy of *state*."""

    layer_values = None
    if state.layer_values:
        values: dict[str, dict[str, object]] = {}
        for layer_id, props in state.layer_values.items():
            if not props:
                continue
            values[str(layer_id)] = {str(key): value for key, value in props.items()}
        if values:
            layer_values = values

    layer_versions = None
    if state.layer_versions:
        versions: dict[str, dict[str, int]] = {}
        for layer_id, props in state.layer_versions.items():
            if not props:
                continue
            mapped: dict[str, int] = {}
            for key, version in props.items():
                mapped[str(key)] = int(version)
            if mapped:
                versions[str(layer_id)] = mapped
        if versions:
            layer_versions = versions

    return RenderLedgerSnapshot(
        plane_center=(
            tuple(float(c) for c in state.plane_center)
            if getattr(state, "plane_center", None) is not None
            else None
        ),
        plane_zoom=(
            float(state.plane_zoom) if getattr(state, "plane_zoom", None) is not None else None
        ),
        plane_rect=(
            tuple(float(v) for v in state.plane_rect)
            if getattr(state, "plane_rect", None) is not None
            else None
        ),
        volume_center=(
            tuple(float(c) for c in state.volume_center)
            if getattr(state, "volume_center", None) is not None
            else None
        ),
        volume_angles=(
            tuple(float(a) for a in state.volume_angles)
            if getattr(state, "volume_angles", None) is not None
            else None
        ),
        volume_distance=(
            float(state.volume_distance) if getattr(state, "volume_distance", None) is not None else None
        ),
        volume_fov=(
            float(state.volume_fov) if getattr(state, "volume_fov", None) is not None else None
        ),
        current_step=(
            tuple(int(s) for s in state.current_step)
            if state.current_step is not None
            else None
        ),
        dims_version=(int(state.dims_version) if state.dims_version is not None else None),
        ndisplay=(int(state.ndisplay) if state.ndisplay is not None else None),
        view_version=(int(state.view_version) if state.view_version is not None else None),
        displayed=(
            tuple(int(idx) for idx in state.displayed)
            if state.displayed is not None
            else None
        ),
        order=(
            tuple(int(idx) for idx in state.order)
            if state.order is not None
            else None
        ),
        axis_labels=(
            tuple(str(label) for label in state.axis_labels)
            if state.axis_labels is not None
            else None
        ),
        level_shapes=(
            tuple(tuple(int(dim) for dim in shape) for shape in state.level_shapes)
            if state.level_shapes is not None
            else None
        ),
        current_level=(int(state.current_level) if state.current_level is not None else None),
        multiscale_level_version=(
            int(state.multiscale_level_version)
            if state.multiscale_level_version is not None
            else None
        ),
        volume_mode=(str(state.volume_mode) if state.volume_mode is not None else None),
        volume_colormap=(
            str(state.volume_colormap) if state.volume_colormap is not None else None
        ),
        volume_clim=(
            tuple(float(v) for v in state.volume_clim)
            if state.volume_clim is not None
            else None
        ),
        volume_opacity=(
            float(state.volume_opacity) if state.volume_opacity is not None else None
        ),
        volume_sample_step=(
            float(state.volume_sample_step)
            if state.volume_sample_step is not None
            else None
        ),
        layer_values=layer_values,
        layer_versions=layer_versions,
    )


AppliedVersions = MutableMapping[tuple[str, str, str], int]


def record_snapshot_versions(
    applied_versions: AppliedVersions,
    state: RenderLedgerSnapshot,
) -> None:
    if state.dims_version is not None:
        applied_versions[("dims", "main", "current_step")] = int(state.dims_version)
    if state.view_version is not None:
        applied_versions[("view", "main", "ndisplay")] = int(state.view_version)
    if state.multiscale_level_version is not None:
        applied_versions[("multiscale", "main", "level")] = int(state.multiscale_level_version)
    if state.camera_versions:
        for key, version in state.camera_versions.items():
            scope = "camera"
            attr = str(key)
            if "." in attr:
                prefix, remainder = attr.split(".", 1)
                if prefix == "plane":
                    scope = "camera_plane"
                    attr = remainder
                elif prefix == "volume":
                    scope = "camera_volume"
                    attr = remainder
                elif prefix == "legacy":
                    scope = "camera"
                    attr = remainder
            applied_versions[(scope, "main", attr)] = int(version)


def extract_layer_changes(
    applied_versions: AppliedVersions,
    state: RenderLedgerSnapshot,
) -> dict[str, dict[str, Any]]:
    layer_changes: dict[str, dict[str, Any]] = {}
    layer_versions = state.layer_versions or {}
    if not state.layer_values:
        return layer_changes

    for raw_layer_id, props in state.layer_values.items():
        if not props:
            continue
        layer_id = str(raw_layer_id)
        version_map = layer_versions.get(layer_id)
        if version_map is None and raw_layer_id in layer_versions:
            version_map = layer_versions[raw_layer_id]
        for raw_prop, value in props.items():
            prop = str(raw_prop)
            version_value = None
            if version_map is not None and prop in version_map:
                version_value = int(version_map[prop])
                key = ("layer", layer_id, prop)
                previous = applied_versions.get(key)
                if previous is not None and previous == version_value:
                    continue
                applied_versions[key] = version_value
            layer_changes.setdefault(layer_id, {})[prop] = value
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
        layer_version_subset: dict[str, dict[str, int]] = {}
        original_versions = state.layer_versions or {}
        for layer_id, props in layer_changes.items():
            version_map = original_versions.get(layer_id, {})
            subset: dict[str, int] = {}
            for prop in props:
                if prop in version_map:
                    subset[prop] = int(version_map[prop])
            if subset:
                layer_version_subset[layer_id] = subset
        state_for_apply = replace(
            state,
            layer_values=layer_changes,
            layer_versions=layer_version_subset or None,
        )
    else:
        state_for_apply = replace(state, layer_values=None, layer_versions=None)

    if updates.plane_state is not None:
        tick_iface.viewport_state.plane = deepcopy(updates.plane_state)
        if tick_iface.viewport_runner is not None:
            tick_iface.viewport_runner._plane = tick_iface.viewport_state.plane  # type: ignore[attr-defined]
    if updates.volume_state is not None:
        tick_iface.viewport_state.volume = deepcopy(updates.volume_state)

    previous_mode = tick_iface.viewport_state.mode
    signature_changed = tick_iface.render_mailbox_update_signature(state)
    if signature_changed:
        apply_render_snapshot(apply_iface, state_for_apply)

    if updates.mode is not None:
        tick_iface.viewport_state.mode = updates.mode
    else:
        tick_iface.viewport_state.mode = previous_mode

    if tick_iface.viewport_state.mode is RenderMode.VOLUME:
        tick_iface.level_policy_suppressed = False
    elif updates.mode is RenderMode.PLANE or previous_mode is RenderMode.VOLUME:
        tick_iface.level_policy_suppressed = True

    if signature_changed and tick_iface.viewport_runner is not None:
        tick_iface.viewport_runner.ingest_snapshot(state)

    drain_res = viewport_updates.drain_render_state(worker, state_for_apply)

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
    "ledger_axis_labels",
    "ledger_displayed",
    "ledger_level",
    "ledger_level_shapes",
    "ledger_ndisplay",
    "ledger_order",
    "ledger_step",
    "normalize_scene_state",
    "record_snapshot_versions",
]
