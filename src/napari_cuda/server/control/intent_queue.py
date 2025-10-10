"""Thread-safe primitives for managing reducer intents and worker confirmations."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional

from asyncio import QueueEmpty, QueueFull

from napari_cuda.server.control.state_models import WorkerStateUpdateConfirmation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerIntent:
    """Structured payload describing a reducer-issued action awaiting confirmation."""

    intent_id: str
    scope: str
    origin: str
    payload: Dict[str, object]
    timestamp: float
    metadata: Optional[Dict[str, object]] = None

    def with_metadata(self, **updates: object) -> "ServerIntent":
        merged = dict(self.metadata or {})
        merged.update(updates)
        return ServerIntent(
            intent_id=self.intent_id,
            scope=self.scope,
            origin=self.origin,
            payload=self.payload,
            timestamp=self.timestamp,
            metadata=merged or None,
        )


class ReducerIntentQueue:
    """FIFO queue that preserves reducer intents until they are confirmed or discarded."""

    def __init__(self, *, maxsize: int = 32) -> None:
        self._maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._items: Deque[ServerIntent] = deque()
        self._index: Dict[str, ServerIntent] = {}

    def push(self, intent: ServerIntent) -> Optional[ServerIntent]:
        """Append an intent, evicting the oldest if the queue is full.

        Returns the evicted intent when an overflow occurs.
        """
        with self._lock:
            if intent.intent_id in self._index:
                logger.debug("intent enqueue dedupe: id=%s scope=%s", intent.intent_id, intent.scope)
                self._remove_locked(intent.intent_id)
            evicted: Optional[ServerIntent] = None
            if len(self._items) >= self._maxsize:
                evicted = self._items.popleft()
                if evicted is not None:
                    self._index.pop(evicted.intent_id, None)
                    logger.debug(
                        "intent queue overflow: evicted id=%s scope=%s origin=%s",
                        evicted.intent_id,
                        evicted.scope,
                        evicted.origin,
                    )
            self._items.append(intent)
            self._index[intent.intent_id] = intent
            return evicted

    def peek(self) -> Optional[ServerIntent]:
        with self._lock:
            return self._items[0] if self._items else None

    def get(self, intent_id: str) -> Optional[ServerIntent]:
        with self._lock:
            return self._index.get(intent_id)

    def pop(self, intent_id: str) -> Optional[ServerIntent]:
        with self._lock:
            return self._remove_locked(intent_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._index.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def items(self) -> Iterable[ServerIntent]:
        with self._lock:
            return tuple(self._items)

    def _remove_locked(self, intent_id: str) -> Optional[ServerIntent]:
        intent = self._index.pop(intent_id, None)
        if intent is None:
            return None
        for idx, candidate in enumerate(self._items):
            if candidate.intent_id == intent_id:
                del self._items[idx]
                break
        return intent


class WorkerConfirmationQueue:
    """Asyncio queue wrapper with bounded size for worker confirmations."""

    def __init__(self, loop: asyncio.AbstractEventLoop, *, maxsize: int = 32) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[WorkerStateUpdateConfirmation] = asyncio.Queue(maxsize=max(1, int(maxsize)))

    def push(self, confirmation: WorkerStateUpdateConfirmation) -> None:
        def _enqueue() -> None:
            try:
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except QueueEmpty:
                        pass
                self._queue.put_nowait(confirmation)
            except QueueFull:
                logger.debug("worker confirmation queue overflow; dropping confirmation")

        self._loop.call_soon_threadsafe(_enqueue)

    async def pull(self) -> WorkerStateUpdateConfirmation:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


def make_server_intent(
    *,
    intent_id: str,
    scope: str,
    origin: str,
    payload: Dict[str, object],
    metadata: Optional[Dict[str, object]] = None,
    timestamp: Optional[float] = None,
) -> ServerIntent:
    """Helper for constructing intents with consistent timestamp defaults."""

    ts = time.time() if timestamp is None else float(timestamp)
    return ServerIntent(
        intent_id=str(intent_id),
        scope=str(scope),
        origin=str(origin),
        payload=dict(payload),
        metadata=dict(metadata) if metadata is not None else None,
        timestamp=ts,
    )
