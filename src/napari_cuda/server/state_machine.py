"""Scene state coordination helpers for the EGL renderer worker.

This module centralises the bookkeeping previously sprinkled through
``egl_worker`` so that the worker can delegate state queuing, signature
tracking, and zoom intent handling to small, testable components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional
import threading
import time


@dataclass(frozen=True)
class ServerSceneState:
    """Minimal scene state snapshot applied atomically per frame."""

    center: Optional[tuple[float, float, float]] = None
    zoom: Optional[float] = None
    angles: Optional[tuple[float, float, float]] = None
    current_step: Optional[tuple[int, ...]] = None
    volume_mode: Optional[str] = None
    volume_colormap: Optional[str] = None
    volume_clim: Optional[tuple[float, float]] = None
    volume_opacity: Optional[float] = None
    volume_sample_step: Optional[float] = None


@dataclass(frozen=True)
class CameraCommand:
    """Queued camera command consumed by the render thread."""

    kind: Literal["zoom", "pan", "orbit", "reset"]
    factor: Optional[float] = None
    anchor_px: Optional[tuple[float, float]] = None
    dx_px: float = 0.0
    dy_px: float = 0.0
    d_az_deg: float = 0.0
    d_el_deg: float = 0.0


@dataclass(frozen=True)
class MultiscaleLevelRequest:
    level: int
    path: Optional[str] = None


@dataclass(frozen=True)
class SceneUpdateBundle:
    display_mode: Optional[int] = None
    multiscale: Optional[MultiscaleLevelRequest] = None
    scene_state: Optional[ServerSceneState] = None


@dataclass(frozen=True)
class ZoomIntent:
    ratio: float
    timestamp: float


class SceneStateCoordinator:
    """Thread-safe coordinator for pending scene updates and zoom intents."""

    def __init__(self, *, time_fn: Callable[[], float] = time.perf_counter) -> None:
        self._lock = threading.Lock()
        self._pending_display_mode: Optional[int] = None
        self._pending_multiscale: Optional[MultiscaleLevelRequest] = None
        self._pending_scene_state: Optional[ServerSceneState] = None
        self._zoom_intent: Optional[ZoomIntent] = None
        self._last_signature: Optional[tuple] = None
        self._time_fn = time_fn

    def queue_display_mode(self, ndisplay: int) -> None:
        with self._lock:
            self._pending_display_mode = int(ndisplay)

    def queue_multiscale_level(self, level: int, path: Optional[str]) -> None:
        with self._lock:
            self._pending_multiscale = MultiscaleLevelRequest(int(level), str(path) if path else None)

    def queue_scene_state(self, state: ServerSceneState) -> None:
        with self._lock:
            self._pending_scene_state = state

    def drain_pending_updates(self) -> SceneUpdateBundle:
        with self._lock:
            updates = SceneUpdateBundle(
                display_mode=self._pending_display_mode,
                multiscale=self._pending_multiscale,
                scene_state=self._pending_scene_state,
            )
            self._pending_display_mode = None
            self._pending_multiscale = None
            self._pending_scene_state = None
        return updates

    def record_zoom_intent(self, ratio: float, *, timestamp: Optional[float] = None) -> None:
        if ratio <= 0.0:
            raise ValueError("zoom ratio must be positive")
        ts = self._time_fn() if timestamp is None else float(timestamp)
        with self._lock:
            self._zoom_intent = ZoomIntent(float(ratio), ts)

    def consume_zoom_intent(self, max_age: float) -> Optional[ZoomIntent]:
        now = self._time_fn()
        with self._lock:
            zoom = self._zoom_intent
            if zoom is None:
                return None
            age = now - float(zoom.timestamp)
            if age > float(max_age):
                self._zoom_intent = None
                return None
            self._zoom_intent = None
            return zoom

    def update_state_signature(self, state: ServerSceneState) -> bool:
        signature = self._build_signature(state)
        with self._lock:
            if signature == self._last_signature:
                return False
            self._last_signature = signature
            return True

    @staticmethod
    def _build_signature(state: ServerSceneState) -> tuple:
        center = tuple(float(c) for c in state.center) if state.center is not None else None
        zoom = float(state.zoom) if state.zoom is not None else None
        angles = tuple(float(a) for a in state.angles) if state.angles is not None else None
        current_step = (
            tuple(int(s) for s in state.current_step)
            if state.current_step is not None
            else None
        )
        return (center, zoom, angles, current_step)


__all__ = [
    "CameraCommand",
    "MultiscaleLevelRequest",
    "SceneUpdateBundle",
    "SceneStateCoordinator",
    "ServerSceneState",
    "ZoomIntent",
]
