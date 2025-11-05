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


def _normalize_layer_state(state: LayerVisualState) -> LayerVisualState:
    def _bool_or_none(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
        return None

    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _contrast_limits(value: Any) -> Optional[tuple[float, float]]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            lo = _float_or_none(value[0])
            hi = _float_or_none(value[1])
            if lo is None or hi is None:
                return None
            if hi < lo:
                lo, hi = hi, lo
            return (lo, hi)
        return None

    metadata = dict(state.metadata) if state.metadata else {}
    thumbnail = dict(state.thumbnail) if isinstance(state.thumbnail, Mapping) else None
    extra = {str(k): v for k, v in state.extra.items()} if state.extra else {}
    versions = {str(k): int(v) for k, v in state.versions.items()} if state.versions else {}

    return LayerVisualState(
        layer_id=str(state.layer_id),
        visible=_bool_or_none(state.visible),
        opacity=_float_or_none(state.opacity),
        blending=str(state.blending) if state.blending is not None else None,
        interpolation=str(state.interpolation) if state.interpolation is not None else None,
        colormap=str(state.colormap) if state.colormap is not None else None,
        gamma=_float_or_none(state.gamma),
        contrast_limits=_contrast_limits(state.contrast_limits),
        depiction=str(state.depiction) if state.depiction is not None else None,
        rendering=str(state.rendering) if state.rendering is not None else None,
        attenuation=_float_or_none(state.attenuation),
        iso_threshold=_float_or_none(state.iso_threshold),
        metadata=metadata,
        thumbnail=thumbnail,
        extra=extra,
        versions=versions,
    )


def normalize_scene_state(state: RenderLedgerSnapshot) -> RenderLedgerSnapshot:
    """Return a render-thread-friendly copy of *state*."""

    layer_states = None
    if state.layer_values:
        values: dict[str, LayerVisualState] = {}
        for layer_id, layer_state in state.layer_values.items():
            if not isinstance(layer_state, LayerVisualState):
                continue
            normalized = _normalize_layer_state(layer_state)
            if normalized.keys() or normalized.versions:
                values[str(layer_id)] = normalized
        if values:
            layer_states = values

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
        margin_left=(
            tuple(float(m) for m in state.margin_left)
            if getattr(state, "margin_left", None) is not None
            else None
        ),
        margin_right=(
            tuple(float(m) for m in state.margin_right)
            if getattr(state, "margin_right", None) is not None
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
        layer_values=layer_states,
        axes=state.axes,
    )


AppliedVersions = MutableMapping[tuple[str, str, str], int]


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
