
"""Worker→control notification helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional

from napari_cuda.protocol.messages import NotifyDimsPayload


@dataclass(frozen=True)
class WorkerSceneNotification:
    """Notification emitted by the render worker to the control loop."""

    kind: Literal["dims_snapshot"]
    seq: int
    payload: NotifyDimsPayload
    timestamp: Optional[float] = None


class WorkerSceneNotificationQueue:
    """Thread-safe FIFO for worker→control notifications."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._items: Deque[WorkerSceneNotification] = deque()

    def push(self, notification: WorkerSceneNotification) -> None:
        with self._lock:
            self._items.append(notification)

    def discard(self, predicate) -> None:
        with self._lock:
            if not self._items:
                return
            self._items = deque(item for item in self._items if not predicate(item))

    def discard_kind(self, kind: Literal["dims_snapshot"]) -> None:
        self.discard(lambda note: note.kind == kind)

    def drain(self) -> list[WorkerSceneNotification]:
        with self._lock:
            if not self._items:
                return []
            items = list(self._items)
            self._items.clear()
            return items


__all__ = [
    "WorkerSceneNotification",
    "WorkerSceneNotificationQueue",
]
