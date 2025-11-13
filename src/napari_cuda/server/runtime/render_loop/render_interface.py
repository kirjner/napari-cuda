"""Unified runtime interface for render loop operations."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional

from napari_cuda.server.data import SliceROI
from napari_cuda.server.runtime.lod.context import build_level_context
from napari_cuda.server.runtime.render_loop.applying.dims import apply_dims_step
from napari_cuda.server.runtime.render_loop.applying.plane import (
    aligned_roi_signature,
    apply_dims_from_snapshot,
    apply_slice_camera_pose,
    apply_slice_level,
    update_z_index_from_snapshot,
)
from napari_cuda.server.runtime.render_loop.applying.viewer_metadata import (
    apply_plane_metadata,
    apply_volume_metadata,
)
from napari_cuda.server.runtime.render_loop.applying.volume import (
    apply_volume_camera_pose,
    apply_volume_level,
    apply_volume_visual_params,
)
from napari_cuda.server.runtime.render_loop.applying.layers import (
    apply_layer_block_updates,
    apply_layer_visual_updates,
)
from napari_cuda.server.scene import LayerVisualState, RenderLedgerSnapshot
from napari_cuda.server.scene.blocks import ENABLE_VIEW_AXES_INDEX_BLOCKS, LayerBlock
from napari_cuda.server.scene.layer_block_diff import (
    LayerBlockDelta,
    compute_layer_block_deltas,
)
from napari_cuda.server.scene.models import SceneBlockSnapshot
from napari_cuda.server.scene.viewport import (
    PlaneViewportCache,
    RenderMode,
    VolumeViewportCache,
    ViewportState,
)
from napari_cuda.server.utils.signatures import SignatureToken, snapshot_versions
from napari_cuda.shared.dims_spec import dims_spec_remap_step_for_level

if TYPE_CHECKING:
    from napari_cuda.server.runtime.render_loop.planning.viewport_planner import ViewportPlanner
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)

AppliedVersions = MutableMapping[tuple[str, str, str], int]


@dataclass(slots=True)
class RenderInterface:
    """Expose worker state for both planning and apply helpers."""

    worker: "EGLRendererWorker"
    _slice_signature: Optional[SignatureToken] = None

    # ------------------------------------------------------------------
    # Viewport + runner state
    @property
    def viewport_state(self) -> "ViewportState":
        return self.worker.viewport_state

    @property
    def viewport_runner(self) -> Optional["ViewportPlanner"]:
        return getattr(self.worker, "_viewport_runner", None)

    def current_level_index(self) -> int:
        return int(self.worker._current_level_index())  # type: ignore[attr-defined]

    def set_current_level_index(self, level: int) -> None:
        self.worker._set_current_level_index(int(level))  # type: ignore[attr-defined]

    @property
    def level_policy_suppressed(self) -> bool:
        return bool(getattr(self.worker, "_level_policy_suppressed", False))  # type: ignore[attr-defined]

    @level_policy_suppressed.setter
    def level_policy_suppressed(self, value: bool) -> None:
        self.worker._level_policy_suppressed = bool(value)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Scene sources
    def ensure_scene_source(self):
        return self.worker._ensure_scene_source()  # type: ignore[attr-defined]

    def get_scene_source(self):
        return getattr(self.worker, "_scene_source", None)  # type: ignore[attr-defined]

    def set_scene_source(self, source: Any) -> None:
        self.worker._scene_source = source  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Camera + animation helpers
    def record_zoom_hint(self, value: float) -> None:
        recorder = getattr(self.worker, "_record_zoom_hint", None)  # type: ignore[attr-defined]
        if recorder is not None:
            recorder(float(value))

    def current_panzoom_rect(self):
        return self.worker._current_panzoom_rect()  # type: ignore[attr-defined]

    def emit_current_camera_pose(self, reason: str) -> None:
        self.worker._emit_current_camera_pose(reason)  # type: ignore[attr-defined]

    def bump_camera_sequences(self, last_seq: int) -> None:
        max_seq = max(int(getattr(self.worker, "_max_camera_command_seq", 0)), int(last_seq))
        self.worker._max_camera_command_seq = max_seq  # type: ignore[attr-defined]
        pose_seq = max(int(getattr(self.worker, "_pose_seq", 0)), max_seq)
        self.worker._pose_seq = pose_seq  # type: ignore[attr-defined]

    def camera_queue_pop_all(self):
        queue = getattr(self.worker, "_camera_queue", None)  # type: ignore[attr-defined]
        return queue.pop_all() if queue is not None else []

    def camera_canvas_size(self) -> tuple[int, int]:
        canvas = getattr(self.worker, "canvas", None)
        if canvas is not None and getattr(canvas, "size", None) is not None:
            width, height = canvas.size
            return int(width), int(height)
        return (
            int(getattr(self.worker, "width", 0)),
            int(getattr(self.worker, "height", 0)),
        )

    def camera_debug_flags(self) -> tuple[bool, bool, bool, bool]:
        return (
            bool(getattr(self.worker, "_debug_zoom_drift", False)),  # type: ignore[attr-defined]
            bool(getattr(self.worker, "_debug_pan", False)),  # type: ignore[attr-defined]
            bool(getattr(self.worker, "_debug_orbit", False)),  # type: ignore[attr-defined]
            bool(getattr(self.worker, "_debug_reset", False)),  # type: ignore[attr-defined]
        )

    def reset_camera_callback(self):
        reset = getattr(self.worker, "_apply_camera_reset", None)  # type: ignore[attr-defined]
        assert reset is not None, "_apply_camera_reset must be initialised"
        return reset

    def configure_camera_for_mode(self) -> None:
        self.worker._configure_camera_for_mode()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Viewer / visuals / layers
    @property
    def viewer(self):
        return getattr(self.worker, "_viewer", None)

    @property
    def view(self):
        return self.worker.view

    def ensure_plane_visual(self):
        return self.worker._ensure_plane_visual()  # type: ignore[attr-defined]

    def ensure_volume_visual(self):
        return self.worker._ensure_volume_visual()  # type: ignore[attr-defined]

    @property
    def plane_visual_handle(self):
        return getattr(self.worker, "_plane_visual_handle", None)

    @property
    def volume_visual_handle(self):
        return getattr(self.worker, "_volume_visual_handle", None)

    @property
    def napari_layer(self):
        return getattr(self.worker, "_napari_layer", None)

    def set_napari_layer(self, layer: Any) -> None:
        self.worker._napari_layer = layer  # type: ignore[attr-defined]

    @property
    def layer_logger(self):
        return getattr(self.worker, "_layer_logger", None)

    @property
    def log_layer_debug(self) -> bool:
        return bool(getattr(self.worker, "_log_layer_debug", False))

    @property
    def roi_align_chunks(self) -> bool:
        return bool(getattr(self.worker, "_roi_align_chunks", False))

    @property
    def roi_pad_chunks(self) -> int:
        return int(getattr(self.worker, "_roi_pad_chunks", 0))

    @property
    def sticky_contrast(self) -> bool:
        return bool(getattr(self.worker, "_sticky_contrast", False))

    # ------------------------------------------------------------------
    # Snapshot bookkeeping
    def last_slice_signature(self) -> Optional[SignatureToken]:
        return self._slice_signature

    def set_last_slice_signature(self, signature: Optional[SignatureToken]) -> None:
        self._slice_signature = signature

    def reset_last_plane_pose(self) -> None:
        self.worker._last_plane_pose = None  # type: ignore[attr-defined]

    def z_index(self) -> Optional[int]:
        return getattr(self.worker, "_z_index", None)

    def set_z_index(self, value: int) -> None:
        self.worker._z_index = int(value)  # type: ignore[attr-defined]

    def set_data_shape(self, width_px: int, height_px: int) -> None:
        self.worker._data_wh = (int(width_px), int(height_px))  # type: ignore[attr-defined]

    def set_data_depth(self, depth: Optional[int]) -> None:
        self.worker._data_d = None if depth is None else int(depth)  # type: ignore[attr-defined]

    def set_volume_scale(self, scale: Sequence[float]) -> None:
        self.worker._volume_scale = tuple(float(v) for v in scale)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Runtime helpers / ROI policy
    def viewport_roi_for_level(
        self,
        source: Any,
        level: int,
        *,
        quiet: bool = False,
        for_policy: bool = False,
    ) -> SliceROI:
        from napari_cuda.server.runtime.lod.slice_loader import viewport_roi_for_lod

        return viewport_roi_for_lod(
            self.worker,
            source,
            int(level),
            quiet=quiet,
            for_policy=for_policy,
            reason="policy-roi" if for_policy else "roi-request",
        )

    def resolve_volume_intent_level(self, source: Any, requested_level: int):
        from napari_cuda.server.runtime.lod import level_policy

        return level_policy.resolve_volume_intent_level(
            self.worker,
            source,
            int(requested_level),
        )

    def load_volume(self, source: Any, level: int):
        from napari_cuda.server.runtime.lod import level_policy

        return level_policy.load_volume(self.worker, source, int(level))

    def ledger_step(self):
        from napari_cuda.server.runtime.render_loop.plan.ledger_access import step as ledger_step

        ledger = self.worker._ledger
        return ledger_step(ledger)

    def set_dims_range_for_level(self, source: Any, level: int) -> None:
        setter = getattr(self.worker, "_set_dims_range_for_level", None)  # type: ignore[attr-defined]
        if setter is not None:
            setter(source, int(level))

    def update_level_metadata(self, descriptor: Any, context: Any) -> None:
        updater = getattr(self.worker, "_update_level_metadata", None)  # type: ignore[attr-defined]
        if updater is not None:
            updater(descriptor, context)

    def mark_level_applied(self, level: int) -> None:
        runner = self.viewport_runner
        if runner is not None:
            runner.mark_level_applied(int(level))

    # ------------------------------------------------------------------
    # Worker bookkeeping
    def mark_render_tick_needed(self) -> None:
        self.worker._mark_render_tick_needed()  # type: ignore[attr-defined]

    def mark_render_tick_complete(self) -> None:
        self.worker._mark_render_tick_complete()  # type: ignore[attr-defined]

    def mark_render_loop_started(self) -> None:
        self.worker._mark_render_loop_started()  # type: ignore[attr-defined]

    def record_user_interaction(self) -> None:
        self.worker._user_interaction_seen = True  # type: ignore[attr-defined]
        self.worker._last_interaction_ts = time.perf_counter()  # type: ignore[attr-defined]

    def update_last_interaction_timestamp(self) -> None:
        self.worker._last_interaction_ts = time.perf_counter()  # type: ignore[attr-defined]

    def evaluate_level_policy(self) -> None:
        self.worker._evaluate_level_policy()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Resources / diagnostics
    @property
    def canvas(self):
        return getattr(self.worker, "canvas", None)

    @property
    def resources(self):
        return getattr(self.worker, "_resources", None)  # type: ignore[attr-defined]

    @property
    def applied_versions(self):
        return getattr(self.worker, "_applied_versions", None)  # type: ignore[attr-defined]

    @property
    def ledger(self):
        return getattr(self.worker, "_ledger", None)  # type: ignore[attr-defined]

    @property
    def animate(self) -> bool:
        return bool(getattr(self.worker, "_animate", False))  # type: ignore[attr-defined]

    @property
    def animate_dps(self) -> float:
        return float(getattr(self.worker, "_animate_dps", 0.0))  # type: ignore[attr-defined]

    @property
    def anim_start(self) -> float:
        return float(getattr(self.worker, "_anim_start", 0.0))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Snapshot apply entry point
    def apply_scene_blocks(self, snapshot: RenderLedgerSnapshot) -> None:
        viewport_state = self.viewport_state
        snapshot_op_seq = int(snapshot.op_seq) if snapshot.op_seq is not None else 0
        current_op_seq = int(getattr(viewport_state, "op_seq", 0))
        if snapshot_op_seq < current_op_seq:
            logger.debug(
                "render snapshot skipped: stale op_seq snapshot=%d latest=%d",
                snapshot_op_seq,
                current_op_seq,
            )
            return
        viewport_state.op_seq = snapshot_op_seq

        applied_versions = self.applied_versions
        layer_changes: dict[str, LayerVisualState] = {}
        layer_block_updates: dict[str, LayerBlockDelta] = {}
        if applied_versions is not None:
            _record_snapshot_versions(applied_versions, snapshot)

        blocks_snapshot = snapshot.block_snapshot if ENABLE_VIEW_AXES_INDEX_BLOCKS else None
        use_layer_blocks = ENABLE_VIEW_AXES_INDEX_BLOCKS and blocks_snapshot is not None
        if use_layer_blocks:
            layer_block_updates = compute_layer_block_deltas(
                applied_versions,
                blocks_snapshot.layers,
            )
        elif applied_versions is not None:
            layer_changes = _extract_layer_changes(applied_versions, snapshot)

        state_for_apply = snapshot
        if layer_changes:
            state_for_apply = replace(snapshot, layer_values=layer_changes)
        elif snapshot.layer_values is not None and not use_layer_blocks:
            state_for_apply = replace(snapshot, layer_values=None)

        plane_state = _resolve_plane_state(snapshot, blocks_snapshot)
        if plane_state is not None:
            _hydrate_plane_pose_from_snapshot(plane_state, snapshot)
            viewport_state.plane = plane_state
            runner = self.viewport_runner
            if runner is not None:
                runner._plane = plane_state  # type: ignore[attr-defined]

        volume_state = _resolve_volume_state(snapshot, blocks_snapshot)
        if volume_state is not None:
            _hydrate_volume_pose_from_snapshot(volume_state, snapshot)
            viewport_state.volume = volume_state

        previous_mode = viewport_state.mode
        mode_update = _resolve_mode(previous_mode, getattr(snapshot, "dims_mode", None))

        _apply_render_snapshot(self, state_for_apply, blocks_snapshot)

        viewport_state.mode = mode_update or previous_mode
        if viewport_state.mode is RenderMode.VOLUME:
            self.level_policy_suppressed = False
        elif mode_update is RenderMode.PLANE or previous_mode is RenderMode.VOLUME:
            self.level_policy_suppressed = True

        runner = self.viewport_runner
        if runner is not None:
            runner.ingest_snapshot(snapshot)

        render_marked = False
        if viewport_state.mode is not RenderMode.VOLUME and snapshot.current_step is not None:
            z_index, marked = apply_dims_step(self.worker, snapshot.current_step)
            if z_index is not None:
                self.set_z_index(int(z_index))
            render_marked = render_marked or marked
        elif viewport_state.mode is RenderMode.VOLUME:
            apply_volume_visual_params(self.worker, snapshot)

        if layer_block_updates:
            if apply_layer_block_updates(self.worker, layer_block_updates):
                render_marked = True
        elif state_for_apply.layer_values and apply_layer_visual_updates(self.worker, state_for_apply.layer_values):
            render_marked = True

        runner = self.viewport_runner
        if runner is not None and viewport_state.mode is RenderMode.PLANE:
            rect = self.current_panzoom_rect()
            if rect is not None:
                runner.update_camera_rect(rect)
            if self.current_level_index() == int(runner.state.target_level):
                runner.mark_level_applied(self.current_level_index())
        elif runner is not None and self.current_level_index() == int(runner.state.target_level):
            runner.mark_level_applied(self.current_level_index())

        if render_marked and viewport_state.mode is not RenderMode.VOLUME and not self.level_policy_suppressed:
            self.evaluate_level_policy()

        if self.level_policy_suppressed:
            ledger = self.ledger
            assert ledger is not None, "ledger must be attached before rendering"
            op_kind_entry = ledger.get("scene", "main", "op_kind")
            if op_kind_entry is not None and str(op_kind_entry.value) == "dims-update":
                self.level_policy_suppressed = False


__all__ = ["RenderInterface"]


# ---------------------------------------------------------------------------
# Internal helpers


def _resolve_mode(previous_mode: RenderMode, dims_mode: Any) -> RenderMode | None:
    if isinstance(dims_mode, str):
        try:
            return RenderMode[dims_mode.upper()]
        except KeyError:
            return previous_mode
    return previous_mode


def _resolve_plane_state(
    snapshot: RenderLedgerSnapshot,
    blocks: SceneBlockSnapshot | None,
) -> PlaneViewportCache | None:
    plane_state = None
    if ENABLE_VIEW_AXES_INDEX_BLOCKS:
        if blocks is not None:
            plane_state = _plane_cache_from_blocks(blocks)
        if plane_state is None and _plane_pose_complete(snapshot):
            plane_state = _plane_cache_from_snapshot(snapshot)
    elif _plane_pose_complete(snapshot):
        plane_state = _plane_cache_from_snapshot(snapshot)
    return plane_state


def _resolve_volume_state(
    snapshot: RenderLedgerSnapshot,
    blocks: SceneBlockSnapshot | None,
) -> VolumeViewportCache | None:
    volume_state = None
    if ENABLE_VIEW_AXES_INDEX_BLOCKS:
        if blocks is not None:
            volume_state = _volume_cache_from_blocks(blocks)
        if volume_state is None and _volume_pose_complete(snapshot):
            volume_state = _volume_cache_from_snapshot(snapshot)
    elif _volume_pose_complete(snapshot):
        volume_state = _volume_cache_from_snapshot(snapshot)
    return volume_state


def _apply_render_snapshot(
    snapshot_iface: RenderInterface,
    snapshot: RenderLedgerSnapshot,
    blocks: SceneBlockSnapshot | None = None,
) -> None:
    viewer = snapshot_iface.viewer
    assert viewer is not None, "RenderTxn requires an active viewer"

    snapshot_ops = _resolve_snapshot_ops(snapshot_iface, snapshot, blocks)

    with _suspend_fit_callbacks(viewer):
        if logger.isEnabledFor(logging.INFO):
            logger.info("snapshot.apply.begin: suppress fit; applying dims")
        apply_dims_from_snapshot(snapshot_iface, snapshot)
        _apply_snapshot_ops(snapshot_iface, snapshot, snapshot_ops)
        if logger.isEnabledFor(logging.INFO):
            logger.info("snapshot.apply.end: dims applied; resuming fit callbacks")


@contextmanager
def _suspend_fit_callbacks(viewer: Any):
    nd_event = viewer.dims.events.ndisplay
    order_event = viewer.dims.events.order

    nd_event.disconnect(viewer.fit_to_view)
    order_event.disconnect(viewer.fit_to_view)
    try:
        yield
    finally:
        nd_event.connect(viewer.fit_to_view)
        order_event.connect(viewer.fit_to_view)


def _resolve_snapshot_ops(
    snapshot_iface: RenderInterface,
    snapshot: RenderLedgerSnapshot,
    blocks: SceneBlockSnapshot | None = None,
) -> dict[str, Any]:
    if snapshot.ndisplay is not None:
        nd = int(snapshot.ndisplay)
    elif blocks is not None:
        nd = int(blocks.view.ndim)
    else:
        nd = 2
    target_volume = nd >= 3

    source = snapshot_iface.ensure_scene_source()
    current_level = snapshot_iface.current_level_index()
    if snapshot.current_level is not None:
        snapshot_level = int(snapshot.current_level)
    elif blocks is not None and blocks.lod is not None:
        snapshot_level = int(blocks.lod.level)
    else:
        snapshot_level = int(current_level)

    dims_spec = snapshot.dims_spec
    assert dims_spec is not None, "render snapshot missing dims_spec"

    snapshot_step = dims_spec.current_step
    if snapshot_step is None and blocks is not None and blocks.index is not None:
        snapshot_step = tuple(int(v) for v in blocks.index.value)

    was_volume = snapshot_iface.viewport_state.mode is RenderMode.VOLUME
    ledger_snapshot_step = snapshot_iface.ledger_step()

    ops: dict[str, Any] = {
        "source": source,
        "target_volume": target_volume,
        "was_volume": was_volume,
        "plane": None,
        "volume": None,
    }

    if target_volume:
        load_needed = (snapshot_level != current_level) or (not was_volume)
        step_hint = snapshot_step
        if step_hint is None and ledger_snapshot_step is not None:
            step_hint = tuple(int(v) for v in ledger_snapshot_step)

        spec = snapshot.dims_spec
        applied_context = None
        target_level = snapshot_level
        if not was_volume:
            resolved = snapshot_iface.resolve_volume_intent_level(source, snapshot_level)
            if resolved is not None:
                target_level = int(resolved)
        if load_needed:
            assert step_hint is not None, "volume load requires explicit step"
            base_step = tuple(int(v) for v in step_hint)
            if spec is not None and spec.current_level is not None:
                curr_level = int(spec.current_level)
                if curr_level != int(target_level):
                    base_step = dims_spec_remap_step_for_level(
                        spec,
                        step=base_step,
                        prev_level=curr_level,
                        next_level=int(target_level),
                    )
            applied_context = build_level_context(
                source=source,
                level=target_level,
                step=base_step,
            )

        ops["volume"] = {
            "entering_volume": not was_volume,
            "load_needed": load_needed,
            "applied_context": applied_context,
            "snapshot_level": int(target_level),
            "step_hint": step_hint,
            "current_level": current_level,
        }
        return ops

    step_hint = snapshot_step
    if step_hint is None and ledger_snapshot_step is not None:
        step_hint = tuple(int(v) for v in ledger_snapshot_step)
    assert step_hint is not None, "plane apply requires explicit step"
    spec = snapshot.dims_spec
    base_step = tuple(int(v) for v in step_hint)
    if spec is not None and spec.current_level is not None:
        curr_level = int(spec.current_level)
        if curr_level != int(snapshot_level):
            base_step = dims_spec_remap_step_for_level(
                spec,
                step=base_step,
                prev_level=curr_level,
                next_level=int(snapshot_level),
            )
    applied_context = build_level_context(
        source=source,
        level=int(snapshot_level),
        step=base_step,
    )
    step_tuple = tuple(int(v) for v in applied_context.step)
    level_int = int(applied_context.level)
    roi_current = snapshot_iface.viewport_roi_for_level(source, level_int)
    aligned_roi, chunk_shape, roi_signature = aligned_roi_signature(
        snapshot_iface,
        source,
        level_int,
        roi_current,
    )
    slice_token = SignatureToken((level_int, step_tuple, roi_signature))
    last_slice_token = snapshot_iface.last_slice_signature()
    level_changed = snapshot_level != current_level
    skip_slice = (
        not level_changed
        and last_slice_token is not None
        and last_slice_token.value == slice_token.value
    )
    chunk_tuple = (
        (int(chunk_shape[0]), int(chunk_shape[1]))
        if chunk_shape is not None
        else (0, 0)
    )
    ops["plane"] = {
        "applied_context": applied_context,
        "aligned_roi": aligned_roi,
        "chunk_shape": chunk_shape,
        "slice_payload": {
            "level": level_int,
            "step": step_tuple,
            "roi": aligned_roi,
            "chunk_shape": chunk_tuple,
            "signature": roi_signature,
        },
        "skip_slice": skip_slice,
        "level_int": level_int,
        "level_changed": level_changed,
        "was_volume": was_volume,
        "slice_token": slice_token,
    }
    return ops


def _apply_snapshot_ops(
    snapshot_iface: RenderInterface,
    snapshot: RenderLedgerSnapshot,
    ops: dict[str, Any],
) -> None:
    source = ops["source"]

    if ops["target_volume"]:
        volume_ops = ops["volume"]
        assert volume_ops is not None
        entering_volume = volume_ops["entering_volume"]
        if entering_volume:
            snapshot_iface.viewport_state.mode = RenderMode.VOLUME
            snapshot_iface.ensure_volume_visual()

        runner = snapshot_iface.viewport_runner
        if runner is not None and entering_volume:
            runner.reset_for_volume()

        if volume_ops["load_needed"]:
            applied_context = volume_ops["applied_context"]
            assert applied_context is not None
            apply_volume_metadata(snapshot_iface, source, applied_context)
            apply_volume_level(
                snapshot_iface,
                source,
                applied_context,
            )
            snapshot_iface.mark_level_applied(int(applied_context.level))
            if (
                runner is not None
                and snapshot_iface.viewport_state.mode is RenderMode.PLANE
            ):
                rect = snapshot_iface.current_panzoom_rect()
                if rect is not None:
                    runner.update_camera_rect(rect)
            snapshot_iface.configure_camera_for_mode()
        elif entering_volume:
            snapshot_iface.configure_camera_for_mode()

        apply_volume_camera_pose(snapshot_iface, snapshot)
        return

    plane_ops = ops["plane"]
    assert plane_ops is not None
    slice_token: Optional[SignatureToken] = plane_ops.get("slice_token")

    if plane_ops["was_volume"]:
        snapshot_iface.viewport_state.mode = RenderMode.PLANE
        snapshot_iface.ensure_plane_visual()
        snapshot_iface.configure_camera_for_mode()

    apply_plane_metadata(snapshot_iface, source, plane_ops["applied_context"])
    apply_slice_camera_pose(snapshot_iface, snapshot)

    if plane_ops["skip_slice"]:
        runner = snapshot_iface.viewport_runner
        if runner is not None:
            from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask

            slice_task = SliceTask(**plane_ops["slice_payload"])
            runner.mark_level_applied(slice_task.level)
            runner.mark_slice_applied(slice_task)
        if slice_token is not None:
            snapshot_iface.set_last_slice_signature(slice_token)
        update_z_index_from_snapshot(snapshot_iface, snapshot)
        return

    apply_slice_level(snapshot_iface, source, plane_ops["applied_context"])
    update_z_index_from_snapshot(snapshot_iface, snapshot)

    if TYPE_CHECKING:
        from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask


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


def _plane_cache_from_blocks(blocks: SceneBlockSnapshot) -> PlaneViewportCache | None:
    restore = blocks.plane_restore
    if restore is None:
        return None
    pose = restore.pose
    if pose.rect is None or pose.center is None or pose.zoom is None:
        return None
    plane_state = PlaneViewportCache()
    level_value = int(restore.level)
    step_value = tuple(int(v) for v in restore.index)
    plane_state.target_level = level_value
    plane_state.applied_level = level_value
    plane_state.target_step = step_value
    plane_state.applied_step = step_value
    plane_state.target_ndisplay = 2
    plane_state.snapshot_level = level_value
    plane_state.update_pose(
        rect=tuple(float(v) for v in pose.rect) if pose.rect is not None else None,
        center=tuple(float(v) for v in pose.center) if pose.center is not None else None,
        zoom=float(pose.zoom) if pose.zoom is not None else None,
    )
    plane_state.awaiting_level_confirm = False
    plane_state.camera_pose_dirty = False
    plane_state.applied_roi = None
    plane_state.applied_roi_signature = None
    return plane_state


def _volume_cache_from_blocks(blocks: SceneBlockSnapshot) -> VolumeViewportCache | None:
    restore = blocks.volume_restore
    if restore is None:
        return None
    pose = restore.pose
    if pose.center is None or pose.angles is None or pose.distance is None or pose.fov is None:
        return None
    volume_state = VolumeViewportCache()
    level_value = int(restore.level)
    volume_state.level = level_value
    if pose.center is not None:
        volume_state.update_pose(center=tuple(float(v) for v in pose.center))
    if pose.angles is not None:
        volume_state.update_pose(angles=tuple(float(v) for v in pose.angles))
    if pose.distance is not None:
        volume_state.update_pose(distance=float(pose.distance))
    if pose.fov is not None:
        volume_state.update_pose(fov=float(pose.fov))
    return volume_state


def _hydrate_plane_pose_from_snapshot(
    plane_state: PlaneViewportCache,
    state: RenderLedgerSnapshot,
) -> None:
    pose = plane_state.pose
    if pose.rect is None and state.plane_rect is not None:
        plane_state.update_pose(rect=tuple(float(v) for v in state.plane_rect))
    if pose.center is None and state.plane_center is not None:
        plane_state.update_pose(center=tuple(float(v) for v in state.plane_center))
    if pose.zoom is None and state.plane_zoom is not None:
        plane_state.update_pose(zoom=float(state.plane_zoom))


def _hydrate_volume_pose_from_snapshot(
    volume_state: VolumeViewportCache,
    state: RenderLedgerSnapshot,
) -> None:
    pose = volume_state.pose
    if pose.center is None and state.volume_center is not None:
        volume_state.update_pose(center=tuple(float(v) for v in state.volume_center))
    if pose.angles is None and state.volume_angles is not None:
        volume_state.update_pose(angles=tuple(float(v) for v in state.volume_angles))
    if pose.distance is None and state.volume_distance is not None:
        volume_state.update_pose(distance=float(state.volume_distance))
    if pose.fov is None and state.volume_fov is not None:
        volume_state.update_pose(fov=float(state.volume_fov))


def _record_snapshot_versions(
    applied_versions: AppliedVersions,
    state: RenderLedgerSnapshot,
) -> None:
    snapshot_versions(state).apply(applied_versions)


def _extract_layer_changes(
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
