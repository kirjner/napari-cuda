"""ViewportPlanner orchestrates per-frame level/ROI decisions on the worker.

The runner keeps the authoritative view of:
* what the controller asked us to render (`target_*` fields),
* what we have actually applied (`applied_*`), and
* whether a level/ROI reload must happen on the next render tick.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from napari_cuda.server.data import (
    SliceROI,
    chunk_shape_for_level,
    roi_chunk_signature,
)
import napari_cuda.server.data.lod as lod
from napari_cuda.server.scene import (
    RenderLedgerSnapshot,
)

from napari_cuda.server.scene.viewport import PlaneResult, PlaneState, PoseEvent
from napari_cuda.server.runtime.lod.context import build_level_context


@dataclass(frozen=True)
class SliceTask:
    """Aligned slice data the worker should apply."""

    level: int
    step: Optional[tuple[int, ...]]
    roi: SliceROI
    chunk_shape: tuple[int, int]
    signature: tuple[int, int, int, int]


@dataclass(frozen=True)
class ViewportOps:
    """Work orders for the next render tick."""

    level_change: bool
    slice_task: Optional[SliceTask]
    pose_event: Optional[PoseEvent]
    zoom_hint: Optional[float] = None
    level_context: Optional[lod.LevelContext] = None


class ViewportPlanner:
    """Local decision surface for level and ROI application."""

    def __init__(self, plane_state: Optional[PlaneState] = None) -> None:
        self._plane = plane_state if plane_state is not None else PlaneState()
        self._snapshot_pending: Optional[RenderLedgerSnapshot] = None
        self._level_reload_required: bool = False
        self._slice_reload_required: bool = False
        self._pending_slice: Optional[SliceTask] = None
        self._pending_pose_event: Optional[PoseEvent] = None

    @property
    def state(self) -> PlaneState:
        return self._plane

    def ingest_snapshot(self, snapshot: RenderLedgerSnapshot) -> None:
        """Record controller intent for the upcoming plan."""

        self._snapshot_pending = snapshot
        state = self._plane
        request = state.request
        applied = state.applied

        snapshot_level: Optional[int] = None
        if snapshot.current_level is not None:
            snapshot_level = int(snapshot.current_level)
            request.snapshot_level = snapshot_level
            if self._level_reload_required:
                request.awaiting_level_confirm = snapshot_level != request.level
            else:
                request.level = snapshot_level
                if applied.level != snapshot_level:
                    self._level_reload_required = True
                    request.awaiting_level_confirm = False
                    if self._pending_pose_event is None:
                        self._pending_pose_event = PoseEvent.LEVEL_RELOAD
                else:
                    self._level_reload_required = False
                    request.awaiting_level_confirm = False
        else:
            self._level_reload_required = applied.level != request.level
            request.awaiting_level_confirm = self._level_reload_required

        target_level = request.level
        target_ndisplay = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
        target_step = (
            tuple(int(v) for v in snapshot.current_step)
            if snapshot.current_step is not None
            else None
        )

        request.level = target_level
        request.ndisplay = target_ndisplay
        request.step = target_step

        if target_ndisplay >= 3:
            # Volume snapshots do not require ROI tracking.
            self._slice_reload_required = False
            self._pending_slice = None
            state.applied = PlaneResult(level=applied.level, step=None, roi_signature=None)
            request.awaiting_level_confirm = False
        else:
            if snapshot.plane_rect is not None:
                state.update_pose(rect=tuple(float(v) for v in snapshot.plane_rect))
            if snapshot.plane_center is not None and len(snapshot.plane_center) >= 2:
                state.update_pose(center=(float(snapshot.plane_center[0]), float(snapshot.plane_center[1])))
            if snapshot.plane_zoom is not None:
                state.update_pose(zoom=float(snapshot.plane_zoom))

        if target_level == applied.level and not self._level_reload_required:
            request.awaiting_level_confirm = False

        if (
            target_step is not None
            and target_step != applied.step
            and not request.awaiting_level_confirm
        ):
            self._slice_reload_required = True
            self._pending_slice = None

    def ingest_camera_deltas(self, commands: Sequence[Any]) -> None:
        """Fold camera deltas into cached info (zoom hints only)."""

        if not commands:
            return
        state = self._plane
        dirty = state.camera_dirty
        for command in commands:
            kind = getattr(command, "kind", None)
            if kind is None:
                continue
            dirty = True
            if kind != "zoom":
                continue
            factor = getattr(command, "factor", None)
            if factor is None:
                continue
            factor = float(factor)
            if factor > 0.0:
                state.zoom_hint = factor
        state.camera_dirty = dirty

    def request_level(self, level: int) -> bool:
        level = int(level)
        state = self._plane
        request = state.request
        applied = state.applied
        if request.level == level:
            if self._level_reload_required:
                return False
            if applied.level == level:
                return False
        request.level = level
        self._level_reload_required = True
        request.awaiting_level_confirm = True
        self._pending_pose_event = PoseEvent.LEVEL_RELOAD
        if request.ndisplay >= 3:
            request.awaiting_level_confirm = False
        return True

    def update_camera_rect(self, rect: Optional[tuple[float, float, float, float]]) -> None:
        """Record the current camera rect (called after apply)."""

        if rect is None:
            return
        self._plane.update_pose(rect=tuple(float(v) for v in rect))

    def plan_tick(
        self,
        *,
        source: Any,
        roi_resolver: Callable[[int, tuple[float, float, float, float]], SliceROI],
    ) -> ViewportOps:
        """Decide what needs to change during the next render tick."""

        state = self._plane
        intent_state = state.request
        applied = state.applied

        pose_event = self._pending_pose_event
        self._pending_pose_event = None

        intent_zoom_hint = state.zoom_hint
        state.zoom_hint = None

        snapshot = self._snapshot_pending
        if snapshot is not None and intent_state.ndisplay < 3:
            self._snapshot_pending = None
            if snapshot.plane_rect is not None:
                state.update_pose(rect=tuple(float(v) for v in snapshot.plane_rect))
            if snapshot.plane_center is not None and len(snapshot.plane_center) >= 2:
                state.update_pose(center=(float(snapshot.plane_center[0]), float(snapshot.plane_center[1])))
            if snapshot.plane_zoom is not None:
                state.update_pose(zoom=float(snapshot.plane_zoom))

        level_change_ready = bool(self._level_reload_required and not intent_state.awaiting_level_confirm)
        slice_out: Optional[SliceTask] = self._pending_slice if self._slice_reload_required else None

        level_context_out: Optional[lod.LevelContext] = None

        if intent_state.ndisplay >= 3:
            self._level_reload_required = False
            intent_state.awaiting_level_confirm = False
            self._slice_reload_required = False
            self._pending_slice = None
            if applied.level != intent_state.level:
                applied.level = intent_state.level
            if pose_event is None and level_change_ready:
                pose_event = PoseEvent.LEVEL_RELOAD
            if pose_event is None and state.camera_dirty:
                pose_event = PoseEvent.CAMERA_DELTA
                state.camera_dirty = False
            return ViewportOps(level_change_ready, None, pose_event, intent_zoom_hint, level_context_out)

        if level_change_ready:
            self._slice_reload_required = False
            self._pending_slice = None
            if pose_event is None:
                pose_event = PoseEvent.LEVEL_RELOAD

        if intent_state.awaiting_level_confirm:
            if pose_event is None and state.camera_dirty:
                pose_event = PoseEvent.CAMERA_DELTA
                state.camera_dirty = False
            return ViewportOps(False, None, pose_event, intent_zoom_hint, level_context_out)

        rect = state.pose.rect
        if rect is not None:
            chunk_shape_src = chunk_shape_for_level(source, intent_state.level)
            chunk_shape = (int(chunk_shape_src[0]), int(chunk_shape_src[1]))
            target_roi = roi_resolver(intent_state.level, rect)
            signature = roi_chunk_signature(target_roi, chunk_shape)

            step_mismatch = intent_state.step != applied.step
            if (
                signature != applied.roi_signature
                or step_mismatch
                or self._slice_reload_required
            ):
                slice_candidate = SliceTask(
                    level=intent_state.level,
                    step=intent_state.step,
                    roi=target_roi,
                    chunk_shape=chunk_shape,
                    signature=signature,
                )
                self._pending_slice = slice_candidate
                self._slice_reload_required = True
                slice_out = slice_candidate
                if pose_event is None:
                    pose_event = PoseEvent.ROI_RELOAD
            elif slice_out is not None:
                slice_out = self._pending_slice

        if pose_event is None and state.camera_dirty:
            pose_event = PoseEvent.CAMERA_DELTA
            state.camera_dirty = False

        if level_change_ready:
            assert intent_state.step is not None, "level_change requires explicit step"
            level_context_out = build_level_context(
                source=source,
                level=int(intent_state.level),
                step=intent_state.step,
            )

        return ViewportOps(level_change_ready, slice_out, pose_event, intent_zoom_hint, level_context_out)

    def mark_level_applied(self, level: int) -> None:
        """Clear level reload state once the worker finishes applying it."""

        state = self._plane
        request = state.request
        applied = state.applied
        applied.level = int(level)
        self._level_reload_required = False
        request.awaiting_level_confirm = False
        request.level = applied.level

    def mark_slice_applied(self, slice_task: SliceTask) -> None:
        """Clear ROI reload state once the worker finishes applying it."""

        state = self._plane
        applied = state.applied
        applied.step = slice_task.step
        applied.roi_signature = slice_task.signature
        self._slice_reload_required = False
        self._pending_slice = None
        state.applied_roi = slice_task.roi

    def reset_for_volume(self) -> None:
        """Reset plane-specific intent when switching into volume mode."""

        state = self._plane
        request = state.request
        applied = state.applied
        self._level_reload_required = False
        self._slice_reload_required = False
        self._pending_slice = None
        self._pending_pose_event = None
        request.awaiting_level_confirm = False
        applied.step = None
        applied.roi_signature = None
        state.applied_roi = None
        state.zoom_hint = None
        state.camera_dirty = False
