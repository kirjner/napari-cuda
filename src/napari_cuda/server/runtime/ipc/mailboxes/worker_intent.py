"""Thread-safe mailbox for worker-to-controller intent messages."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..messages.level_switch import LevelSwitchIntent


RenderSignaturePayload = Tuple[int, int, Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, int], ...]]


@dataclass
class ThumbnailIntent:
    layer_id: str
    signature: RenderSignaturePayload
    array: np.ndarray


class WorkerIntentMailbox:
    """Latest-wins mailbox for worker intents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_intent: Optional[LevelSwitchIntent] = None
        self._latest_thumbnail: Optional[ThumbnailIntent] = None

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

    # -- Thumbnail payloads ---------------------------------------------------

    def enqueue_thumbnail(self, payload: ThumbnailIntent) -> None:
        """Store the latest thumbnail payload."""

        with self._lock:
            self._latest_thumbnail = payload

    def pop_thumbnail(self) -> Optional[ThumbnailIntent]:
        """Pop the most recent thumbnail payload, if any."""

        with self._lock:
            payload = self._latest_thumbnail
            self._latest_thumbnail = None
            return payload


__all__ = ["ThumbnailIntent", "WorkerIntentMailbox"]
