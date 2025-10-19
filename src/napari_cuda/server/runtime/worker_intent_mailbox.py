"""Thread-safe mailbox for worker-to-controller intent messages."""

from __future__ import annotations

from typing import Optional
import threading

from .intents import LevelSwitchIntent


class WorkerIntentMailbox:
    """Latest-wins mailbox for worker level-switch intents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_intent: Optional[LevelSwitchIntent] = None

    # -- Level switch intents -------------------------------------------------

    def enqueue_level_switch(self, intent: LevelSwitchIntent) -> None:
        """Store the latest level switch intent."""

        with self._lock:
            self._latest_intent = intent

    def pop_level_switch(self) -> Optional[LevelSwitchIntent]:
        """Pop the most recent level switch intent, if any."""

        with self._lock:
            intent = self._latest_intent
            self._latest_intent = None
            return intent


__all__ = ["WorkerIntentMailbox"]
