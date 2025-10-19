"""Render-thread mailbox for coalescing scene updates before each frame."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable, Optional

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderLevelRequest:
    """Pending multiscale level switch request."""

    level: int
    path: Optional[str] = None


@dataclass(frozen=True)
class RenderUpdate:
    """Latest-wins state drained by the render worker."""

    multiscale: Optional[RenderLevelRequest]
    scene_state: Optional[RenderLedgerSnapshot]


@dataclass(frozen=True)
class RenderZoomHint:
    """Latest zoom hint emitted by the control loop."""

    ratio: float
    timestamp: float


class RenderUpdateMailbox:
    """Thread-safe mailbox that coalesces render updates for the worker."""

    def __init__(self, *, time_fn: Callable[[], float] = time.perf_counter) -> None:
        self._time_fn = time_fn
        self._multiscale: Optional[RenderLevelRequest] = None
        self._scene_state: Optional[RenderLedgerSnapshot] = None
        self._zoom_hint: Optional[RenderZoomHint] = None
        self._camera_ops: list = []
        self._last_signature: Optional[tuple] = None
        self._lock = threading.Lock()

    def set_multiscale_target(self, level: int, path: Optional[str]) -> None:
        with self._lock:
            self._multiscale = RenderLevelRequest(
                int(level),
                str(path) if path else None,
            )

    def set_scene_state(self, state: RenderLedgerSnapshot) -> None:
        with self._lock:
            self._scene_state = state

    def drain(self) -> RenderUpdate:
        with self._lock:
            drained = RenderUpdate(
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

    # ---- Camera ops ---------------------------------------------------------
    def append_camera_ops(self, ops) -> None:
        """Append a batch of camera ops for the render loop to consume.

        Type of ``ops`` is intentionally untyped here to avoid a hard import
        on the camera command class; we treat them as opaque objects and hand
        them to the existing camera controller on the render thread.
        """
        if not ops:
            return
        with self._lock:
            self._camera_ops.extend(list(ops))

    def drain_camera_ops(self):
        with self._lock:
            if not self._camera_ops:
                return []
            drained = list(self._camera_ops)
            self._camera_ops.clear()
            return drained

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
        distance = float(state.distance) if state.distance is not None else None
        fov = float(state.fov) if state.fov is not None else None
        current_step = (
            tuple(int(s) for s in state.current_step)
            if state.current_step is not None
            else None
        )
        # Include display mode axes in the signature so 2D/3D toggles and
        # displayed/order changes are not treated as no-ops. Previously, only
        # camera pose and step were considered, which caused us to skip
        # applying dims on ndisplay changes and prevented camera class switches
        # (PanZoom <-> Turntable) from taking effect.
        ndisplay = int(state.ndisplay) if state.ndisplay is not None else None
        displayed = (
            tuple(int(i) for i in state.displayed)
            if state.displayed is not None
            else None
        )
        order = (
            tuple(int(i) for i in state.order)
            if state.order is not None
            else None
        )
        return (center, zoom, angles, distance, fov, current_step, ndisplay, displayed, order)


__all__ = [
    "RenderLevelRequest",
    "RenderUpdate",
    "RenderUpdateMailbox",
    "RenderZoomHint",
]
