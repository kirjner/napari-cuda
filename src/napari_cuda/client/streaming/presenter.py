from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from napari_cuda.client.streaming.types import (
    ReadyFrame,
    Source,
    SubmittedFrame,
    TimestampMode,
)

logger = logging.getLogger(__name__)


class ClockSync:
    """Computes due times based on timestamp mode and offset."""

    def __init__(self, mode: TimestampMode, offset: Optional[float] = None):
        self.mode = mode
        self.offset = offset

    def set_mode(self, mode: TimestampMode) -> None:
        self.mode = mode

    def set_offset(self, offset: Optional[float]) -> None:
        self.offset = offset

    def compute_due(self, arrival_ts: float, server_ts: Optional[float], latency_s: float) -> float:
        if self.mode == TimestampMode.ARRIVAL or server_ts is None:
            return float(arrival_ts) + float(latency_s)
        off = self.offset
        if off is None or abs(off) > 5.0:
            return float(arrival_ts) + float(latency_s)
        return float(server_ts) + float(off) + float(latency_s)


@dataclass
class _BufItem:
    due_ts: float
    payload: object
    release_cb: Optional[object]
    server_ts: Optional[float]
    arrival_ts: float


class FixedLatencyPresenter:
    """Per-source fixed-latency presenter with simple source mux compatibility.

    - Maintains per-source buffers of frames with computed due times.
    - Drops oldest frames beyond `buffer_limit` per source.
    - Returns the latest due frame at `pop_due()` for the active source.
    - If nothing is due and the earliest future frame is far (>200ms), it can
      return a preview of the latest frame to avoid visible stalls.
    """

    def __init__(
        self,
        latency_s: float,
        buffer_limit: int,
        ts_mode: TimestampMode,
        preview_guard_s: float = 1.0/60.0,
    ) -> None:
        self.latency_s = float(max(0.0, latency_s))
        self.buffer_limit = max(1, int(buffer_limit))
        self.clock = ClockSync(ts_mode)
        # If nothing is due yet in SERVER mode, allow previewing when the
        # earliest due time is more than this guard ahead of now. Keep small
        # (about one frame at 60 Hz) to avoid visible stalls.
        self._preview_guard_s = max(0.0, float(preview_guard_s))
        self._buf: Dict[Source, List[_BufItem]] = {Source.VT: [], Source.PYAV: []}
        self._submit_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        self._out_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}

    def set_latency(self, latency_s: float) -> None:
        self.latency_s = float(max(0.0, latency_s))

    def set_buffer_limit(self, n: int) -> None:
        self.buffer_limit = max(1, int(n))

    def set_mode(self, mode: TimestampMode) -> None:
        self.clock.set_mode(mode)

    def set_offset(self, offset: Optional[float]) -> None:
        self.clock.set_offset(offset)

    def submit(self, sf: SubmittedFrame) -> None:
        due = self.clock.compute_due(sf.arrival_ts, sf.server_ts, self.latency_s)
        items = self._buf[sf.source]
        items.append(
            _BufItem(
                due_ts=due,
                payload=sf.payload,
                release_cb=sf.release_cb,
                server_ts=sf.server_ts,
                arrival_ts=sf.arrival_ts,
            )
        )
        self._submit_count[sf.source] += 1
        # Trim to buffer limit, keep most recent by due time
        if len(items) > self.buffer_limit:
            items.sort(key=lambda it: it.due_ts)
            drop = items[:-self.buffer_limit]
            self._release_many(drop)
            del items[:-self.buffer_limit]

    def clear(self, source: Optional[Source] = None) -> None:
        if source is None:
            for src in list(self._buf.keys()):
                self._release_many(self._buf[src])
                self._buf[src].clear()
            return
        self._release_many(self._buf[source])
        self._buf[source].clear()

    def _release_many(self, items: List[_BufItem]) -> None:
        for it in items:
            cb = it.release_cb
            if cb is not None:
                try:
                    # Stored as object to avoid type checker on Optional[Callable]
                    cb = cb  # type: ignore[assignment]
                    cb(it.payload)  # type: ignore[misc]
                except Exception:
                    logger.debug("release_cb failed during buffer clear", exc_info=True)

    def pop_due(self, now: Optional[float], active: Source) -> Optional[ReadyFrame]:
        n = float(now if now is not None else time.time())
        items = self._buf[active]
        if not items:
            return None
        # Sort by due time ascending
        items.sort(key=lambda it: it.due_ts)
        due_now = [it for it in items if it.due_ts <= n]
        if due_now:
            sel = due_now[-1]
            # Consume all <= now (they are late); release all but the selected one
            for it in due_now[:-1]:
                cb = it.release_cb
                if cb is not None:
                    try:
                        cb(it.payload)  # type: ignore[misc]
                    except Exception:
                        logger.debug("release_cb failed for dropped due frame", exc_info=True)
            # Remove consumed
            remain: List[_BufItem] = [it for it in items if it.due_ts > n]
            self._buf[active] = remain
            self._out_count[active] += 1
            return ReadyFrame(source=active, due_ts=sel.due_ts, payload=sel.payload, release_cb=sel.release_cb, preview=False)
        # Nothing due yet
        if self.clock.mode == TimestampMode.ARRIVAL:
            # In ARRIVAL mode, favor smoothness: present and consume the latest frame
            sel = items[-1]
            # Drop all older frames
            for it in items[:-1]:
                cb = it.release_cb
                if cb is not None:
                    try:
                        cb(it.payload)  # type: ignore[misc]
                    except Exception:
                        logger.debug("release_cb failed for dropped future frame", exc_info=True)
            self._buf[active] = []
            self._out_count[active] += 1
            return ReadyFrame(source=active, due_ts=sel.due_ts, payload=sel.payload, release_cb=sel.release_cb, preview=False)
        # In SERVER mode, allow a short preview window to avoid visible stalls
        earliest_due = items[0].due_ts
        if earliest_due - n > self._preview_guard_s:
            sel = items[-1]
            return ReadyFrame(source=active, due_ts=sel.due_ts, payload=sel.payload, release_cb=sel.release_cb, preview=True)
        return None

    def stats(self) -> Dict[str, object]:
        return {
            "submit": {s.value: self._submit_count[s] for s in (Source.VT, Source.PYAV)},
            "out": {s.value: self._out_count[s] for s in (Source.VT, Source.PYAV)},
            "buf": {s.value: len(self._buf[s]) for s in (Source.VT, Source.PYAV)},
            "latency_ms": int(round(self.latency_s * 1000.0)),
            "mode": self.clock.mode.value,
        }

    # Adaptive offset helper: compute median of (arrival - server_ts)
    def relearn_offset(self, source: Source) -> Optional[float]:
        items = self._buf.get(source) or []
        diffs: List[float] = []
        for it in items[-20:]:  # last N samples
            if it.server_ts is not None:
                diffs.append(float(it.arrival_ts) - float(it.server_ts))
        if not diffs:
            return None
        diffs.sort()
        mid = len(diffs) // 2
        med = diffs[mid] if len(diffs) % 2 == 1 else 0.5 * (diffs[mid - 1] + diffs[mid])
        self.set_offset(float(med))
        return float(med)


class SourceMux:
    """Tracks the active presentation source to avoid cross-talk."""

    def __init__(self, initial: Source) -> None:
        self._active = initial

    @property
    def active(self) -> Source:
        return self._active

    def set_active(self, src: Source) -> None:
        self._active = src
