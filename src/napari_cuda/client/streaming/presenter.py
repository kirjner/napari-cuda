from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
import threading
import math

from napari_cuda.client.streaming.types import ReadyFrame, Source, SubmittedFrame

logger = logging.getLogger(__name__)


class ClockSync:
    """Computes due times based on server timestamp + offset, with fallback."""

    def __init__(self, offset: Optional[float] = None):
        self.offset = offset

    def set_offset(self, offset: Optional[float]) -> None:
        self.offset = offset

    def compute_due(self, arrival_ts: float, server_ts: Optional[float], latency_s: float) -> float:
        # Prefer server timestamp + offset; fallback to arrival only if server_ts/offset unavailable.
        off = self.offset
        if server_ts is None or off is None:
            return float(arrival_ts) + float(latency_s)
        if not math.isfinite(float(off)):
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
    - Always consume newest due; if nothing is due and the earliest future
      frame is sufficiently far, return a preview of the latest frame to avoid
      visible stalls (without consuming it).
    """

    def __init__(
        self,
        latency_s: float,
        buffer_limit: int,
        preview_guard_s: float = 0.0,
        *,
        unified: bool = False,
    ) -> None:
        self.latency_s = float(max(0.0, latency_s))
        self.buffer_limit = max(1, int(buffer_limit))
        self.clock = ClockSync()
        # Force unified scheduler behavior for jitter robustness.
        self._unified = True
        # Single preview guard (seconds)
        self._preview_guard_s = max(0.0, float(preview_guard_s))
        # Early-consume tolerance: treat frames within this window before due as due
        # to avoid preview churn from timer quantization and minor scheduling jitter.
        # Allow env override (ms) via NAPARI_CUDA_EARLY_CONSUME_MS.
        try:
            import os as _os
            e_ms = _os.getenv('NAPARI_CUDA_EARLY_CONSUME_MS')
            if e_ms is not None and e_ms.strip() != '':
                self._early_consume_s = max(0.0, float(e_ms) / 1000.0)
            else:
                self._early_consume_s = min(0.004, max(0.0, float(preview_guard_s))) if preview_guard_s else 0.003
        except Exception:
            self._early_consume_s = 0.003
        # Deque per source for O(1) append/pop from ends
        self._buf: Dict[Source, Deque[_BufItem]] = {Source.VT: deque(), Source.PYAV: deque()}
        self._submit_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        # Count frames actually consumed (removed from buffer)
        self._out_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        # Count preview returns (not consumed)
        self._preview_count: Dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        self._lock = threading.Lock()
        # Offset PLL (bounded) state for SERVER mode
        self._pll_offset: Optional[float] = None
        self._pll_last_update: float = 0.0
        # Diagnostics: count how due times are computed
        self._due_server: int = 0
        self._due_arrival: int = 0

    def set_latency(self, latency_s: float) -> None:
        self.latency_s = float(max(0.0, latency_s))

    def set_buffer_limit(self, n: int) -> None:
        self.buffer_limit = max(1, int(n))

    # set_mode removed; server timestamping is always preferred.

    def set_offset(self, offset: Optional[float]) -> None:
        self.clock.set_offset(offset)
        # Seed PLL with explicit offset if provided
        self._pll_offset = float(offset) if (offset is not None and math.isfinite(float(offset))) else None

    def submit(self, sf: SubmittedFrame) -> None:
        # Seed offset immediately on first good sample to avoid startup churn
        try:
            if self.clock.offset is None and sf.server_ts is not None and math.isfinite(float(sf.server_ts)) and math.isfinite(float(sf.arrival_ts)):
                seed = float(sf.arrival_ts) - float(sf.server_ts)
                self.clock.set_offset(seed)
                self._pll_offset = float(seed)
                self._pll_last_update = float(sf.arrival_ts)
        except Exception:
            logger.debug("submit: offset seeding failed", exc_info=True)
        due = self.clock.compute_due(sf.arrival_ts, sf.server_ts, self.latency_s)
        to_release: List[_BufItem] = []
        with self._lock:
            items = self._buf[sf.source]
            # Insert by due_ts to avoid head-of-line blocking when arrivals are
            # out-of-order (e.g., with simulated or real network jitter). The
            # buffer sizes are small (on the order of tens of frames), so an
            # O(n) insert keeps things simple and predictable.
            new_item = _BufItem(
                due_ts=due,
                payload=sf.payload,
                release_cb=sf.release_cb,
                server_ts=sf.server_ts,
                arrival_ts=sf.arrival_ts,
            )
            if not items:
                items.append(new_item)
            else:
                # Walk from the right for amortized efficiency when due_ts is
                # typically non-decreasing, but robust when out-of-order.
                pos = len(items)
                while pos > 0 and items[pos - 1].due_ts > due:
                    pos -= 1
                if pos == len(items):
                    items.append(new_item)
                else:
                    items.insert(pos, new_item)
            self._submit_count[sf.source] += 1
            # Diagnostics: track which path was used (server vs arrival)
            try:
                off = self.clock.offset
                used_server = (sf.server_ts is not None) and (off is not None) and math.isfinite(float(off))
                if used_server:
                    self._due_server += 1
                else:
                    self._due_arrival += 1
            except Exception:
                logger.debug("submit: due path count failed", exc_info=True)
            # Trim to buffer limit from the left (oldest first)
            while len(items) > self.buffer_limit:
                to_release.append(items.popleft())
        # Update PLL estimator opportunistically when unified
        try:
            if self._unified:
                self._maybe_update_pll(sf.arrival_ts, sf.server_ts)
        except Exception:
            logger.debug("submit: PLL update failed", exc_info=True)
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
        # Use monotonic clock for scheduling decisions
        n = float(now if now is not None else time.perf_counter())
        # Work on a single selected item and a list of items to release outside the lock
        to_release: List[_BufItem] = []
        sel: Optional[_BufItem] = None
        is_preview = False
        with self._lock:
            items = self._buf[active]
            if not items:
                return None
            # Unified: consume newest due (allow small early-consume epsilon);
            # else preview latest if earliest future is sufficiently far
            last_due: Optional[_BufItem] = None
            # Consume frames that are due or within early-consume window
            early_now = n + float(self._early_consume_s or 0.0)
            while items and items[0].due_ts <= early_now:
                if last_due is not None:
                    to_release.append(last_due)
                last_due = items.popleft()
            if last_due is not None:
                sel = last_due
                self._out_count[active] += 1
            else:
                earliest = items[0]
                if (earliest.due_ts - n) > self._preview_guard_s:
                    sel = items[-1]
                    is_preview = True
                    self._preview_count[active] += 1
                else:
                    sel = None
        # Release outside the lock
        if to_release:
            self._release_many(to_release)
        if sel is None:
            return None
        return ReadyFrame(source=active, due_ts=sel.due_ts, payload=sel.payload, release_cb=sel.release_cb, preview=is_preview)

    def stats(self) -> Dict[str, object]:
        # Use monotonic clock to compare against due_ts values
        now = time.perf_counter()
        with self._lock:
            # Compute next_due (ms) per source and simple fill metrics
            next_due: Dict[str, Optional[int]] = {}
            fill: Dict[str, int] = {}
            for s in (Source.VT, Source.PYAV):
                items = self._buf[s]
                fill[s.value] = len(items)
                if not items:
                    next_due[s.value] = None
                else:
                    # Clamp negative to zero if already due
                    delta_ms = int(round(max(0.0, items[0].due_ts - now) * 1000.0))
                    next_due[s.value] = delta_ms
            return {
                "submit": {s.value: self._submit_count[s] for s in (Source.VT, Source.PYAV)},
                "out": {s.value: self._out_count[s] for s in (Source.VT, Source.PYAV)},
                "preview": {s.value: self._preview_count[s] for s in (Source.VT, Source.PYAV)},
                "buf": {s.value: len(self._buf[s]) for s in (Source.VT, Source.PYAV)},
                "latency_ms": int(round(self.latency_s * 1000.0)),
                "server_timestamp": True,
                "next_due_ms": next_due,
                "fill": fill,
                "due_path": {"server": int(self._due_server), "arrival": int(self._due_arrival)},
            }

    def peek_next_due(self, active: Source) -> Optional[float]:
        """Return the earliest due_ts for the active source, or None.

        Callers should schedule a wake for max(0, due_ts - now).
        """
        with self._lock:
            items = self._buf.get(active)
            if not items:
                return None
            return float(items[0].due_ts)

    # Adaptive offset helper: compute median of (arrival - server_ts)
    def relearn_offset(self, source: Source) -> Optional[float]:
        with self._lock:
            items = list(self._buf.get(source) or [])
        diffs: List[float] = []
        considered = items[-20:]  # last N samples
        total = len(considered)
        for it in considered:
            # Guard against None/NaN values from decoder timestamps
            if it.server_ts is not None and math.isfinite(it.arrival_ts) and math.isfinite(it.server_ts):
                diffs.append(float(it.arrival_ts) - float(it.server_ts))
        if not diffs:
            try:
                logger.debug(
                    "relearn_offset: insufficient samples (total=%d finite=%d)",
                    total,
                    0,
                )
            except Exception:
                pass
            return None
        diffs.sort()
        mid = len(diffs) // 2
        med = diffs[mid] if len(diffs) % 2 == 1 else 0.5 * (diffs[mid - 1] + diffs[mid])
        # Reject only non-finite values; accept large offsets (epoch alignment)
        if not math.isfinite(med):
            try:
                logger.debug(
                    "relearn_offset: rejected median (val=%s, samples=%d)",
                    med,
                    len(diffs),
                )
            except Exception:
                pass
            return None
        self.set_offset(float(med))
        return float(med)

    def _maybe_update_pll(self, arrival_ts: float, server_ts: Optional[float]) -> None:
        """Bounded PLL update toward median offset target.

        - Samples (arrival - server) to estimate offset when server_ts is available.
        - Moves internal PLL slowly toward the median with a rate limit (ms/s).
        - Writes back to clock.offset so compute_due uses the smoothed value.
        """
        try:
            if server_ts is None or not math.isfinite(float(server_ts)):
                return
            # Estimate target as simple median over recent samples via relearn_offset
            target = self.relearn_offset(Source.VT)
            if target is None or not math.isfinite(float(target)):
                return
            now = float(arrival_ts)
            if self._pll_offset is None:
                self._pll_offset = float(target)
                self.clock.set_offset(self._pll_offset)
                self._pll_last_update = now
                return
            dt = max(1e-3, now - float(self._pll_last_update or now))
            # Limit change rate to ±0.5 ms/s toward target
            max_rate = 0.0005  # seconds per second
            err = float(target) - float(self._pll_offset)
            step = max(-max_rate * dt, min(max_rate * dt, err))
            new_off = float(self._pll_offset) + step
            # Write back
            self._pll_offset = new_off
            self.clock.set_offset(new_off)
            self._pll_last_update = now
        except Exception:
            logger.debug("PLL update failed", exc_info=True)


class SourceMux:
    """Tracks the active presentation source to avoid cross-talk."""

    def __init__(self, initial: Source) -> None:
        self._active = initial

    @property
    def active(self) -> Source:
        return self._active

    def set_active(self, src: Source) -> None:
        self._active = src
