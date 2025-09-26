"""ServerScene queue helpers shared between the headless server and worker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Deque, Literal, Optional

from napari_cuda.server.scene_state import ServerSceneState


# `ServerSceneState` moves to `napari_cuda.server.scene_state` to break the
# circular dependency between the worker and state queue. Import lazily below.


@dataclass(frozen=True)
class ServerSceneCommand:
    """Queued camera command consumed by the render thread."""

    kind: Literal["zoom", "pan", "orbit", "reset"]
    factor: Optional[float] = None
    anchor_px: Optional[tuple[float, float]] = None
    dx_px: float = 0.0
    dy_px: float = 0.0
    d_az_deg: float = 0.0
    d_el_deg: float = 0.0


@dataclass(frozen=True)
class ServerSceneLevelRequest:
    level: int
    path: Optional[str] = None


@dataclass(frozen=True)
class PendingServerSceneUpdate:
    display_mode: Optional[int] = None
    multiscale: Optional[ServerSceneLevelRequest] = None
    scene_state: Optional[ServerSceneState] = None


@dataclass(frozen=True)
class ServerSceneZoomIntent:
    ratio: float
    timestamp: float


class ServerSceneQueue:
    """Thread-safe queue for pending scene updates and zoom intents."""

    def __init__(self, *, time_fn: Callable[[], float] = time.perf_counter) -> None:
        self._lock = threading.Lock()
        self._pending_display_mode: Optional[int] = None
        self._pending_multiscale: Optional[ServerSceneLevelRequest] = None
        self._pending_scene_state: Optional[ServerSceneState] = None
        self._zoom_intent: Optional[ServerSceneZoomIntent] = None
        self._last_signature: Optional[tuple] = None
        self._time_fn = time_fn

    def queue_display_mode(self, ndisplay: int) -> None:
        with self._lock:
            self._pending_display_mode = int(ndisplay)

    def queue_multiscale_level(self, level: int, path: Optional[str]) -> None:
        with self._lock:
            self._pending_multiscale = ServerSceneLevelRequest(int(level), str(path) if path else None)

    def queue_scene_state(self, state: ServerSceneState) -> None:
        with self._lock:
            self._pending_scene_state = state

    def drain_pending_updates(self) -> PendingServerSceneUpdate:
        with self._lock:
            updates = PendingServerSceneUpdate(
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
            self._zoom_intent = ServerSceneZoomIntent(float(ratio), ts)

    def consume_zoom_intent(self, max_age: float) -> Optional[ServerSceneZoomIntent]:
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


@dataclass(frozen=True)
class WorkerSceneNotification:
    """Worker-thread message consumed by the control loop."""

    kind: Literal["dims_update", "meta_refresh"]
    step: Optional[tuple[int, ...]] = None
    last_client_id: Optional[str] = None
    ack: bool = False
    intent_seq: Optional[int] = None


class WorkerSceneNotificationQueue:
    """Thread-safe FIFO for worker→control notifications."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: Deque[WorkerSceneNotification] = deque()

    def push(self, notification: WorkerSceneNotification) -> None:
        with self._lock:
            self._items.append(notification)

    def drain(self) -> list[WorkerSceneNotification]:
        with self._lock:
            if not self._items:
                return []
            items = list(self._items)
            self._items.clear()
            return items


__all__ = [
    "ServerSceneCommand",
    "ServerSceneLevelRequest",
    "PendingServerSceneUpdate",
    "ServerSceneQueue",
    "ServerSceneZoomIntent",
    "WorkerSceneNotification",
    "WorkerSceneNotificationQueue",
]
