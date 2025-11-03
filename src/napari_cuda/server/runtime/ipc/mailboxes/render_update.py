"""Render-thread mailbox for coalescing scene updates before each frame."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from napari_cuda.server.scene import (
    LayerVisualState,
    PlaneState,
    RenderLedgerSnapshot,
    RenderMode,
    RenderUpdate,
    VolumeState,
)
from napari_cuda.server.utils.signatures import scene_content_signature_tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderZoomHint:
    """Latest zoom hint emitted by the control loop."""

    ratio: float
    timestamp: float


class RenderUpdateMailbox:
    """Thread-safe mailbox that coalesces render updates for the worker."""

    def __init__(self, *, time_fn: Callable[[], float] = time.perf_counter) -> None:
        self._time_fn = time_fn
        self._scene_state: Optional[RenderLedgerSnapshot] = None
        self._scene_op_seq: Optional[int] = None
        self._mode: Optional[RenderMode] = None
        self._plane_state: Optional[PlaneState] = None
        self._volume_state: Optional[VolumeState] = None
        self._zoom_hint: Optional[RenderZoomHint] = None
        self._camera_ops: list = []
        self._last_signature: Optional[tuple] = None
        self._lock = threading.Lock()

    def set_scene_state(self, state: RenderLedgerSnapshot) -> None:
        op_seq = int(state.op_seq) if state.op_seq is not None else 0
        with self._lock:
            if self._scene_op_seq is not None:
                current_op = int(self._scene_op_seq)
                if op_seq < current_op:
                    return
                if op_seq == current_op:
                    signature = self._build_signature(state)
                    if signature == self._last_signature:
                        return
            self._scene_state = state
            self._scene_op_seq = op_seq

    def set_viewport_state(
        self,
        *,
        mode: Optional[RenderMode],
        plane_state: Optional[PlaneState],
        volume_state: Optional[VolumeState],
    ) -> None:
        with self._lock:
            self._mode = mode
            self._plane_state = deepcopy(plane_state) if plane_state is not None else None
            self._volume_state = deepcopy(volume_state) if volume_state is not None else None

    def drain(self) -> RenderUpdate:
        with self._lock:
            op_seq = self._scene_op_seq
            drained = RenderUpdate(
                scene_state=self._scene_state,
                mode=self._mode,
                plane_state=self._plane_state,
                volume_state=self._volume_state,
                op_seq=op_seq,
            )
            self._scene_state = None
            self._mode = None
            self._plane_state = None
            self._volume_state = None
            self._scene_op_seq = op_seq
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
        # Content-only signature tuple (op_seq-free), unified across server
        # Note: dataset_id not tracked at this point; pass None.
        return scene_content_signature_tuple(state, dataset_id=None)


__all__ = [
    "RenderUpdate",
    "RenderUpdateMailbox",
    "RenderZoomHint",
]
