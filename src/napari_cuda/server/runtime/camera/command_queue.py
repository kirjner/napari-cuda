"""Thread-safe FIFO queue for camera delta commands."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List
import threading

from napari_cuda.server.scene import CameraDeltaCommand


class CameraCommandQueue:
    """Queue that stores camera delta commands until the worker drains them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._commands: Deque[CameraDeltaCommand] = deque()

    def extend(self, commands: Iterable[CameraDeltaCommand]) -> None:
        with self._lock:
            for command in commands:
                self._commands.append(command)

    def append(self, command: CameraDeltaCommand) -> None:
        with self._lock:
            self._commands.append(command)

    def pop_all(self) -> List[CameraDeltaCommand]:
        with self._lock:
            if not self._commands:
                return []
            drained = list(self._commands)
            self._commands.clear()
            return drained

    def __len__(self) -> int:
        with self._lock:
            return len(self._commands)


__all__ = ["CameraCommandQueue"]
