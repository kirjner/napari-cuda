"""Operational interface for render tick helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


@dataclass(slots=True)
class RenderPlanInterface:
    """Bridge tick planners to the worker without exposing private state."""

    worker: EGLRendererWorker

    # Viewport / runner ---------------------------------------------------
    @property
    def viewport_runner(self):
        return getattr(self.worker, "_viewport_runner", None)

    @property
    def viewport_state(self):
        return self.worker.viewport_state

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

    # Scene sources -------------------------------------------------------
    def ensure_scene_source(self):
        return self.worker._ensure_scene_source()  # type: ignore[attr-defined]

    def get_scene_source(self):
        return getattr(self.worker, "_scene_source", None)  # type: ignore[attr-defined]

    def set_scene_source(self, source: Any) -> None:
        self.worker._scene_source = source  # type: ignore[attr-defined]

    # Zoom hint helpers ---------------------------------------------------
    def record_zoom_hint(self, value: float) -> None:
        recorder = getattr(self.worker, "_record_zoom_hint", None)  # type: ignore[attr-defined]
        if recorder is not None:
            recorder(float(value))

    # Camera helpers ------------------------------------------------------
    def current_panzoom_rect(self):
        return self.worker._current_panzoom_rect()  # type: ignore[attr-defined]

    def emit_camera_pose(self, reason: str) -> None:
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

    def reset_camera_callback(self) -> Callable[[Any], None]:
        reset = getattr(self.worker, "_apply_camera_reset", None)  # type: ignore[attr-defined]
        assert reset is not None, "_apply_camera_reset must be initialised"
        return reset

    # Timing / bookkeeping -----------------------------------------------
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

    # Render resources ----------------------------------------------------
    @property
    def canvas(self):
        return getattr(self.worker, "canvas", None)

    @property
    def view(self):
        return getattr(self.worker, "view", None)

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


__all__ = ["RenderPlanInterface"]
