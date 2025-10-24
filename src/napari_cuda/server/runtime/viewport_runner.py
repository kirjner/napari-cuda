"""ViewportRunner orchestrates per-frame level/ROI decisions on the worker.

The runner keeps the authoritative view of:
* what the controller asked us to render (`target_*` fields),
* what we have actually applied (`applied_*`), and
* whether a level/ROI reload must happen on the next render tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.roi_math import (
    chunk_shape_for_level,
    roi_chunk_signature,
)
from napari_cuda.server.runtime.scene_types import SliceROI


@dataclass(frozen=True)
class ViewportIntent:
    level_change: bool
    roi_change: bool
    pose_reason: Optional[str]
    zoom_hint: Optional[float] = None


@dataclass
class ViewportRunnerState:
    # Controller intent (from the latest snapshot)
    target_level: int = 0
    target_ndisplay: int = 2
    target_step: Optional[Tuple[int, ...]] = None
    applied_step: Optional[Tuple[int, ...]] = None
    awaiting_level_confirm: bool = False
    snapshot_level: Optional[int] = None

    # Camera pose snapshot we base ROI decisions on
    camera_rect: Optional[Tuple[float, float, float, float]] = None

    # Applied state on the worker
    applied_level: int = 0
    applied_roi: Optional[SliceROI] = None
    applied_roi_signature: Optional[Tuple[int, int, int, int]] = None

    # Pending work captured during plan_tick
    pending_roi: Optional[SliceROI] = None
    pending_roi_signature: Optional[Tuple[int, int, int, int]] = None

    # Reload flags / logging
    level_reload_required: bool = False
    roi_reload_required: bool = False
    pose_reason: Optional[str] = None

    # Latest zoom factor from camera deltas (consumed once per plan)
    zoom_hint: Optional[float] = None
    camera_pose_dirty: bool = False


class ViewportRunner:
    """Local decision surface for level and ROI application."""

    def __init__(self) -> None:
        self._state = ViewportRunnerState()
        self._snapshot_pending: Optional[RenderLedgerSnapshot] = None

    @property
    def state(self) -> ViewportRunnerState:
        return self._state

    def ingest_snapshot(self, snapshot: RenderLedgerSnapshot) -> None:
        """Record controller intent for the upcoming plan."""

        self._snapshot_pending = snapshot
        state = self._state

        snapshot_level: Optional[int] = None
        if snapshot.current_level is not None:
            snapshot_level = int(snapshot.current_level)
            state.snapshot_level = snapshot_level
            if state.level_reload_required:
                state.awaiting_level_confirm = snapshot_level != state.target_level
            else:
                state.target_level = snapshot_level
                if state.applied_level != snapshot_level:
                    state.level_reload_required = True
                    state.awaiting_level_confirm = False
                    state.pose_reason = "level-reload"
                else:
                    state.level_reload_required = False
                    state.awaiting_level_confirm = False
        else:
            state.level_reload_required = state.applied_level != state.target_level
            state.awaiting_level_confirm = state.level_reload_required

        target_level = state.target_level
        target_ndisplay = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
        target_step = (
            tuple(int(v) for v in snapshot.current_step)
            if snapshot.current_step is not None
            else None
        )

        state.target_level = target_level
        state.target_ndisplay = target_ndisplay
        state.target_step = target_step

        if target_ndisplay >= 3:
            # Volume snapshots do not require ROI tracking.
            state.roi_reload_required = False
            state.pending_roi = None
            state.pending_roi_signature = None
            state.applied_roi = None
            state.applied_roi_signature = None
            state.applied_step = None

        if target_level == state.applied_level and not state.level_reload_required:
            state.awaiting_level_confirm = False

        if (
            target_step is not None
            and target_step != state.applied_step
            and not state.awaiting_level_confirm
        ):
            state.roi_reload_required = True
            state.pending_roi = None
            state.pending_roi_signature = None

        if snapshot.rect is not None:
            state.camera_rect = tuple(float(v) for v in snapshot.rect)

    def ingest_camera_deltas(self, commands: Sequence[Any]) -> None:
        """Fold camera deltas into cached info (zoom hints only)."""

        if not commands:
            return

        state = self._state
        dirty = state.camera_pose_dirty
        for command in commands:
            if getattr(command, "kind", None) != "zoom":
                continue
            factor = getattr(command, "factor", None)
            if factor is None:
                continue
            factor = float(factor)
            if factor > 0.0:
                state.zoom_hint = factor
                dirty = True
        state.camera_pose_dirty = dirty

    def request_level(self, level: int) -> bool:
        level = int(level)
        state = self._state
        if state.target_level == level:
            if state.level_reload_required:
                return False
            if state.applied_level == level:
                return False
        state.target_level = level
        state.level_reload_required = True
        state.awaiting_level_confirm = True
        state.pose_reason = "level-reload"
        if state.target_ndisplay >= 3:
            level_change = False
            state.awaiting_level_confirm = False
        return True

    def update_camera_rect(self, rect: Optional[Tuple[float, float, float, float]]) -> None:
        """Record the current camera rect (called after apply)."""

        self._state.camera_rect = tuple(float(v) for v in rect) if rect is not None else None

    def plan_tick(
        self,
        *,
        source: Any,
        roi_resolver: Callable[[int, Tuple[float, float, float, float]], SliceROI],
    ) -> ViewportIntent:
        """Decide what needs to change during the next render tick."""

        state = self._state
        pose_reason = state.pose_reason
        intent_zoom_hint = state.zoom_hint
        state.zoom_hint = None

        snapshot = self._snapshot_pending
        if snapshot is not None:
            self._snapshot_pending = None
            if snapshot.rect is not None:
                state.camera_rect = tuple(float(v) for v in snapshot.rect)

        level_change = bool(state.level_reload_required and not state.awaiting_level_confirm)
        roi_change = False

        if state.target_ndisplay >= 3:
            state.level_reload_required = False
            state.awaiting_level_confirm = False
            state.roi_reload_required = False
            state.pending_roi = None
            state.pending_roi_signature = None
            pose_reason_out = pose_reason
            if pose_reason_out is None and level_change:
                pose_reason_out = "level-reload"
            if pose_reason_out is None and state.camera_pose_dirty:
                pose_reason_out = "camera-delta"
                state.camera_pose_dirty = False
            if state.applied_level != state.target_level:
                state.applied_level = state.target_level
            if state.pose_reason == "level-reload":
                state.pose_reason = None
            return ViewportIntent(level_change, False, pose_reason_out, intent_zoom_hint)

        if level_change:
            state.roi_reload_required = False
            state.pending_roi = None
            state.pending_roi_signature = None

        if state.awaiting_level_confirm:
            return ViewportIntent(False, False, pose_reason, intent_zoom_hint)

        rect = state.camera_rect
        if rect is not None:
            chunk_shape = chunk_shape_for_level(source, state.target_level)
            target_roi = roi_resolver(state.target_level, rect)
            signature = roi_chunk_signature(target_roi, chunk_shape)

            step_mismatch = state.target_step != state.applied_step
            if state.applied_roi_signature != signature or step_mismatch or state.roi_reload_required:
                state.roi_reload_required = True
                state.pending_roi = target_roi
                state.pending_roi_signature = signature
                roi_change = True
                if pose_reason is None:
                    pose_reason = "roi-reload"

        pose_reason_out = pose_reason
        camera_dirty = state.camera_pose_dirty
        if pose_reason_out is None and camera_dirty:
            pose_reason_out = "camera-delta"
            camera_dirty = False
        state.camera_pose_dirty = camera_dirty

        return ViewportIntent(level_change, roi_change, pose_reason_out, intent_zoom_hint)

    def mark_level_applied(self, level: int) -> None:
        """Clear level reload state once the worker finishes applying it."""

        state = self._state
        state.applied_level = int(level)
        state.level_reload_required = False
        state.awaiting_level_confirm = False
        state.target_level = state.applied_level
        if state.pose_reason == "level-reload":
            state.pose_reason = None

    def mark_roi_applied(
        self,
        roi: SliceROI,
        *,
        chunk_shape: Optional[Tuple[int, int]],
    ) -> None:
        """Clear ROI reload state once the worker finishes applying it."""

        state = self._state
        state.applied_roi = roi
        state.applied_roi_signature = roi_chunk_signature(roi, chunk_shape)
        state.roi_reload_required = False
        state.pending_roi = None
        state.pending_roi_signature = None
        state.applied_step = state.target_step
        if state.pose_reason == "roi-reload":
            state.pose_reason = None
