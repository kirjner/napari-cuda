from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
import threading
import math

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
    - In ARRIVAL mode, return the newest item without consuming it ("sticky").
      This avoids empty-buffer stalls when draw and decode phases misalign.
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
        # Deque per source for O(1) append/pop from ends
        self._buf: Dict[Source, Deque[_BufItem]] = {Source.VT: deque(), Source.PYAV: deque()}
        self._submit_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        # Count frames actually consumed (removed from buffer)
        self._out_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        # Count preview returns in ARRIVAL mode (not consumed)
        self._preview_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        self._lock = threading.Lock()

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
        to_release: List[_BufItem] = []
        with self._lock:
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
            # Trim to buffer limit from the left (oldest first)
            while len(items) > self.buffer_limit:
                to_release.append(items.popleft())
        for it in to_release:
            cb = it.release_cb
            if cb is not None:
                try:
                    cb(it.payload)  # type: ignore[misc]
                except Exception:
                    logger.debug("release_cb failed during buffer trim", exc_info=True)

    def clear(self, source: Optional[Source] = None) -> None:
        if source is None:
            with self._lock:
                to_rel = {src: list(self._buf[src]) for src in list(self._buf.keys())}
                for src in list(self._buf.keys()):
                    self._buf[src].clear()
            for items in to_rel.values():
                self._release_many(items)
            return
        with self._lock:
            items = list(self._buf[source])
            self._buf[source].clear()
        self._release_many(items)

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
        # Work on a single selected item and a list of items to release outside the lock
        to_release: List[_BufItem] = []
        sel: Optional[_BufItem] = None
        is_preview = False
        with self._lock:
            items = self._buf[active]
            if not items:
                return None
            if self.clock.mode == TimestampMode.ARRIVAL:
                # Sticky latest: trim older items, then preview the newest without consuming
                while len(items) > 1:
                    to_release.append(items.popleft())
                sel = items[-1]
                is_preview = True
                # Do not increment out_count for previews; track separately
                self._preview_count[active] += 1
            else:
                last_due: Optional[_BufItem] = None
                while items and items[0].due_ts <= n:
                    if last_due is not None:
                        to_release.append(last_due)
                    last_due = items.popleft()
                if last_due is not None:
                    sel = last_due
                    self._out_count[active] += 1
                else:
                    # In SERVER mode, avoid previewing future frames to prevent
                    # out-of-order visual churn. Let the renderer keep the last
                    # displayed frame until a frame becomes due.
                    sel = None
        # Release outside the lock
        if to_release:
            self._release_many(to_release)
        if sel is None:
            return None
        return ReadyFrame(source=active, due_ts=sel.due_ts, payload=sel.payload, release_cb=sel.release_cb, preview=is_preview)

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "submit": {s.value: self._submit_count[s] for s in (Source.VT, Source.PYAV)},
                "out": {s.value: self._out_count[s] for s in (Source.VT, Source.PYAV)},
                "preview": {s.value: self._preview_count[s] for s in (Source.VT, Source.PYAV)},
                "buf": {s.value: len(self._buf[s]) for s in (Source.VT, Source.PYAV)},
                "latency_ms": int(round(self.latency_s * 1000.0)),
                "mode": self.clock.mode.value,
            }

    # Adaptive offset helper: compute median of (arrival - server_ts)
    def relearn_offset(self, source: Source) -> Optional[float]:
        with self._lock:
            items = list(self._buf.get(source) or [])
        diffs: List[float] = []
        for it in items[-20:]:  # last N samples
            # Guard against None/NaN values from decoder timestamps
            if it.server_ts is not None and math.isfinite(it.arrival_ts) and math.isfinite(it.server_ts):
                diffs.append(float(it.arrival_ts) - float(it.server_ts))
        if not diffs:
            return None
        diffs.sort()
        mid = len(diffs) // 2
        med = diffs[mid] if len(diffs) % 2 == 1 else 0.5 * (diffs[mid - 1] + diffs[mid])
        # Reject unreasonable or non-finite offsets; keep SERVER mode from latching junk
        if not math.isfinite(med) or abs(float(med)) > 5.0:
            return None
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
