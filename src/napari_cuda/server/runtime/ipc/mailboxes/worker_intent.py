"""Thread-safe mailbox for worker-to-controller intent messages."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..messages.level_switch import LevelSwitchIntent


@dataclass
class ThumbnailCapture:
    """Worker → control thumbnail candidate.

    The payload carries only the raw pixel array and target layer id.
    Dedupe is performed on the control loop using inputs-only content tokens.
    """
    layer_id: str
    array: np.ndarray


class WorkerIntentMailbox:
    """Latest-wins mailbox for worker intents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_level_switch: Optional[LevelSwitchIntent] = None
        self._latest_thumbnail_capture: Optional[ThumbnailCapture] = None

    # -- Level switch intents -------------------------------------------------

    def enqueue_level_switch(self, intent: LevelSwitchIntent) -> None:
        """Store the latest level switch intent."""

        with self._lock:
            self._latest_level_switch = intent

    def pop_level_switch(self) -> Optional[LevelSwitchIntent]:
        """Pop the most recent level switch intent, if any."""

        with self._lock:
            intent = self._latest_level_switch
            self._latest_level_switch = None
            return intent

    # -- Thumbnail payloads ---------------------------------------------------

    def enqueue_thumbnail_capture(self, payload: ThumbnailCapture) -> None:
        """Store the latest thumbnail capture payload."""

        with self._lock:
            self._latest_thumbnail_capture = payload

    def pop_thumbnail_capture(self) -> Optional[ThumbnailCapture]:
        """Pop the most recent thumbnail capture payload, if any."""

        with self._lock:
            payload = self._latest_thumbnail_capture
            self._latest_thumbnail_capture = None
            return payload


__all__ = ["ThumbnailCapture", "WorkerIntentMailbox"]
