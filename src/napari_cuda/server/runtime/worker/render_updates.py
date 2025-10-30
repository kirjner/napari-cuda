"""Render update helpers extracted from the EGL worker."""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import MutableMapping

from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdate
from napari_cuda.server.runtime.core.snapshot_build import RenderLedgerSnapshot
from napari_cuda.server.runtime.worker.snapshots import apply_render_snapshot
from napari_cuda.server.runtime.viewport import RenderMode, PlaneState, VolumeState
from napari_cuda.server.runtime.viewport import updates as viewport_updates
from napari_cuda.server.runtime.worker.napari_viewer.viewport_state import (
    _apply_viewport_state_snapshot as apply_viewport_state_snapshot,
)

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


def consume_render_snapshot(worker: "EGLRendererWorker", state: RenderLedgerSnapshot) -> None:
    """Queue a complete scene state snapshot for the next frame."""

    normalized = normalize_scene_state(state)
    worker._render_mailbox.set_scene_state(normalized)  # noqa: SLF001
    worker._mark_render_tick_needed()  # noqa: SLF001
    worker._last_interaction_ts = time.perf_counter()  # noqa: SLF001


def drain_scene_updates(worker: "EGLRendererWorker") -> None:
    updates: RenderUpdate = worker._render_mailbox.drain()  # noqa: SLF001
    state = updates.scene_state
    if state is None:
        apply_viewport_state_snapshot(
            worker,
            mode=updates.mode,
            plane_state=updates.plane_state,
            volume_state=updates.volume_state,
        )
        return

    snapshot_op_seq = updates.op_seq
    if snapshot_op_seq is None:
        snapshot_op_seq = int(state.op_seq) if state.op_seq is not None else 0
    if int(snapshot_op_seq) < int(worker.viewport_state.op_seq):
        logger.debug(
            "render snapshot skipped: stale op_seq snapshot=%d latest=%d",
            int(snapshot_op_seq),
            int(worker.viewport_state.op_seq),
        )
        return
    worker.viewport_state.op_seq = int(snapshot_op_seq)

    applied_versions = worker._applied_versions  # noqa: SLF001
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
        worker._viewport_state.plane = deepcopy(updates.plane_state)  # noqa: SLF001
        if worker._viewport_runner is not None:  # noqa: SLF001
            worker._viewport_runner._plane = worker._viewport_state.plane  # type: ignore[attr-defined]  # noqa: SLF001
    if updates.volume_state is not None:
        worker._viewport_state.volume = deepcopy(updates.volume_state)  # noqa: SLF001

    previous_mode = worker._viewport_state.mode  # noqa: SLF001
    signature_changed = worker._render_mailbox.update_state_signature(state)  # noqa: SLF001
    if signature_changed:
        apply_render_snapshot(worker, state_for_apply)

    if updates.mode is not None:
        worker._viewport_state.mode = updates.mode  # noqa: SLF001
    else:
        worker._viewport_state.mode = previous_mode  # noqa: SLF001

    if worker._viewport_state.mode is RenderMode.VOLUME:  # noqa: SLF001
        worker._level_policy_suppressed = False  # noqa: SLF001
    elif updates.mode is RenderMode.PLANE or previous_mode is RenderMode.VOLUME:
        worker._level_policy_suppressed = True  # noqa: SLF001

    if signature_changed and worker._viewport_runner is not None:  # noqa: SLF001
        worker._viewport_runner.ingest_snapshot(state)  # noqa: SLF001

    drain_res = viewport_updates.drain_render_state(worker, state_for_apply)

    if drain_res.z_index is not None:
        worker._z_index = int(drain_res.z_index)  # noqa: SLF001
    if drain_res.data_wh is not None:
        worker._data_wh = (int(drain_res.data_wh[0]), int(drain_res.data_wh[1]))  # noqa: SLF001

    if worker._viewport_runner is not None and worker._viewport_state.mode is RenderMode.PLANE:  # noqa: SLF001
        rect = worker._current_panzoom_rect()  # noqa: SLF001
        if rect is not None:
            worker._viewport_runner.update_camera_rect(rect)  # noqa: SLF001
        if int(worker._current_level_index()) == int(worker._viewport_runner.state.target_level):  # noqa: SLF001
            worker._viewport_runner.mark_level_applied(worker._current_level_index())  # noqa: SLF001
    elif worker._viewport_runner is not None and int(worker._current_level_index()) == int(worker._viewport_runner.state.target_level):  # noqa: SLF001
        worker._viewport_runner.mark_level_applied(worker._current_level_index())  # noqa: SLF001

    if (
        drain_res.render_marked
        and worker._viewport_state.mode is not RenderMode.VOLUME  # noqa: SLF001
        and not worker._level_policy_suppressed  # noqa: SLF001
    ):
        worker._evaluate_level_policy()  # noqa: SLF001

    if worker._level_policy_suppressed:  # noqa: SLF001
        ledger = worker._ledger  # noqa: SLF001
        assert ledger is not None, "ledger must be attached before rendering"
        op_kind_entry = ledger.get("scene", "main", "op_kind")
        if op_kind_entry is not None and str(op_kind_entry.value) == "dims-update":
            worker._level_policy_suppressed = False  # noqa: SLF001


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
