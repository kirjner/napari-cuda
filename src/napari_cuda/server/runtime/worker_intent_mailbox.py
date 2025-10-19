"""Thread-safe mailbox for worker-to-controller intent messages."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List
import threading

from .intents import LevelSwitchIntent


class WorkerIntentMailbox:
    """FIFO mailbox that buffers intents raised by the render worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._level_switches: Deque[LevelSwitchIntent] = deque()

    # -- Level switch intents -------------------------------------------------

    def enqueue_level_switch(self, intent: LevelSwitchIntent) -> None:
        """Append ``intent`` to the mailbox."""

        with self._lock:
            self._level_switches.append(intent)

    def drain_level_switches(self) -> List[LevelSwitchIntent]:
        """Drain and return pending level switch intents."""

        with self._lock:
            if not self._level_switches:
                return []
            drained = list(self._level_switches)
            self._level_switches.clear()
            return drained

    def iter_level_switches(self) -> Iterable[LevelSwitchIntent]:
        """Snapshot the current queue contents without draining."""

        with self._lock:
            snapshot = list(self._level_switches)
        return iter(snapshot)


__all__ = ["WorkerIntentMailbox"]
