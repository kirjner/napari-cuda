
"""Worker→control notification helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Mapping, Optional


@dataclass(frozen=True)
class WorkerSceneNotification:
    """Notification emitted by the render worker to the control loop."""

    kind: Literal["dims_update", "scene_level"]
    step: Optional[tuple[int, ...]] = None
    level: Optional[Mapping[str, object]] = None
    meta: Optional[Mapping[str, object]] = None


class WorkerSceneNotificationQueue:
    """Thread-safe FIFO for worker→control notifications."""

    def __init__(self) -> None:
        import threading

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
    "WorkerSceneNotification",
    "WorkerSceneNotificationQueue",
]
