
"""Render-thread command queue for coalescing scene updates before each frame."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot


@dataclass(frozen=True)
class RenderLevelRequest:
    """Pending multiscale level switch request."""

    level: int
    path: Optional[str] = None


@dataclass(frozen=True)
class RenderDelta:
    """Delta enqueued by the control thread for the render worker."""

    multiscale: Optional[RenderLevelRequest] = None
    scene_state: Optional[RenderLedgerSnapshot] = None


@dataclass(frozen=True)
class PendingRenderUpdate:
    """Coalesced updates drained by the render worker."""

    multiscale: Optional[RenderLevelRequest] = None
    scene_state: Optional[RenderLedgerSnapshot] = None


@dataclass(frozen=True)
class RenderZoomHint:
    """Latest zoom hint emitted by the control loop."""

    ratio: float
    timestamp: float


class RenderUpdateQueue:
    """Thread-safe queue that coalesces render updates for the worker."""

    def __init__(self, *, time_fn: Callable[[], float] = time.perf_counter) -> None:
        self._time_fn = time_fn
        self._multiscale: Optional[RenderLevelRequest] = None
        self._scene_state: Optional[RenderLedgerSnapshot] = None
        self._zoom_hint: Optional[RenderZoomHint] = None
        self._last_signature: Optional[tuple] = None
        self._lock = threading.Lock()

    def enqueue(self, delta: RenderDelta) -> None:
        with self._lock:
            if delta.multiscale is not None:
                self._multiscale = RenderLevelRequest(
                    int(delta.multiscale.level),
                    str(delta.multiscale.path) if delta.multiscale.path else None,
                )
            if delta.scene_state is not None:
                self._scene_state = delta.scene_state

    def enqueue_multiscale(self, level: int, path: Optional[str]) -> None:
        self.enqueue(RenderDelta(multiscale=RenderLevelRequest(int(level), str(path) if path else None)))

    def enqueue_scene_state(self, state: RenderLedgerSnapshot) -> None:
        self.enqueue(RenderDelta(scene_state=state))

    def drain(self) -> PendingRenderUpdate:
        with self._lock:
            drained = PendingRenderUpdate(
                multiscale=self._multiscale,
                scene_state=self._scene_state,
            )
            self._multiscale = None
            self._scene_state = None
        return drained

    def record_zoom_hint(self, ratio: float, *, timestamp: Optional[float] = None) -> None:
        if ratio <= 0.0:
            raise ValueError("zoom ratio must be positive")
        ts = self._time_fn() if timestamp is None else float(timestamp)
        with self._lock:
            self._zoom_hint = RenderZoomHint(float(ratio), ts)

    def consume_zoom_hint(self, max_age: float) -> Optional[RenderZoomHint]:
        now = self._time_fn()
        with self._lock:
            zoom = self._zoom_hint
            if zoom is None:
                return None
            age = now - float(zoom.timestamp)
            if age > float(max_age):
                self._zoom_hint = None
                return None
            self._zoom_hint = None
            return zoom

    def update_state_signature(self, state: RenderLedgerSnapshot) -> bool:
        signature = self._build_signature(state)
        with self._lock:
            if signature == self._last_signature:
                return False
            self._last_signature = signature
            return True

    @staticmethod
    def _build_signature(state: RenderLedgerSnapshot) -> tuple:
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
    "RenderDelta",
    "RenderLevelRequest",
    "RenderUpdateQueue",
    "RenderZoomHint",
    "PendingRenderUpdate",
]
