from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .types import ReadyFrame, Source, SubmittedFrame


class ClockSync:
    def __init__(self, offset: Optional[float] = None) -> None:
        self.offset = offset

    def set_offset(self, offset: Optional[float]) -> None:
        self.offset = offset

    def compute_due(self, arrival_ts: float, server_ts: Optional[float], latency_s: float) -> float:
        off = self.offset
        if server_ts is None or off is None or not math.isfinite(float(off)):
            return float(arrival_ts) + float(latency_s)
        return float(server_ts) + float(off) + float(latency_s)


@dataclass
class _Item:
    due_ts: float
    payload: object
    release_cb: Optional[object]
    server_ts: Optional[float]
    arrival_ts: float
    seq: Optional[int]


class MinimalPresenter:
    """Minimal, deterministic presenter.

    - Computes a single due_ts per submitted frame (server_ts + offset + latency).
    - Keeps per-source buffers sorted by due_ts.
    - On pop, presents at most one frame: the oldest due item (<= now).
    - Enforces strict forward progress using seq and server_ts monotonicity.
    - Exposes only the small API required by the coordinator.
    """

    def __init__(self, latency_s: float, buffer_limit: int, preview_guard_s: float = 0.0) -> None:
        self.latency_s = float(max(0.0, latency_s))
        # Minimal safety cap to avoid unbounded buffers
        self.buffer_limit = max(1, int(buffer_limit))
        self.clock = ClockSync()
        self._log = logging.getLogger(__name__)
        try:
            import os as _os
            self._step_log = (_os.getenv('NAPARI_CUDA_PRESENTER_STEP_LOG', '0') or '0').lower() in ('1','true','yes','on')
        except Exception:
            self._step_log = False
        # Per-source buffers
        self._buf: dict[Source, deque[_Item]] = {Source.VT: deque(), Source.PYAV: deque()}
        # Sequence maps (for tick-locked selection)
        self._seq_map: dict[Source, dict[int, _Item]] = {Source.VT: {}, Source.PYAV: {}}
        # Last presented markers
        self._last_seq: dict[Source, Optional[int]] = {Source.VT: None, Source.PYAV: None}
        self._last_ts: dict[Source, Optional[float]] = {Source.VT: None, Source.PYAV: None}
        # Time->seq mapping base and period
        self._base_seq: Optional[int] = None
        self._base_ts: Optional[float] = None  # server wall at base_seq
        self._mono0: Optional[float] = None
        self._fps: float = 60.0
        self._period: float = 1.0 / 60.0
        # Stats
        self._submit_count: dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        self._out_count: dict[Source, int] = {Source.VT: 0, Source.PYAV: 0}
        # Epsilon for timestamp monotonicity (seconds)
        self._ts_eps = 0.00025

    # Coordinator API
    def set_latency(self, latency_s: float) -> None:
        self.latency_s = float(max(0.0, latency_s))

    def set_fps_hint(self, _fps: float) -> None:
        try:
            f = float(_fps)
            if f >= 1.0:
                self._fps = f
                self._period = 1.0 / f
        except Exception:
            pass

    def set_offset(self, offset: Optional[float]) -> None:
        self.clock.set_offset(offset)

    def submit(self, sf: SubmittedFrame) -> None:
        due = self.clock.compute_due(sf.arrival_ts, sf.server_ts, self.latency_s)
        item = _Item(
            due_ts=float(due),
            payload=sf.payload,
            release_cb=sf.release_cb,
            server_ts=sf.server_ts,
            arrival_ts=sf.arrival_ts,
            seq=getattr(sf, 'seq', None),
        )
        dq = self._buf[sf.source]
        # Seed offset and time->seq base opportunistically
        if self.clock.offset is None and sf.server_ts is not None and math.isfinite(float(sf.server_ts)):
            try:
                self.clock.set_offset(float(sf.arrival_ts) - float(sf.server_ts))
                if self._mono0 is None:
                    self._mono0 = float(time.perf_counter())
            except Exception:
                pass
        if self._base_seq is None and item.seq is not None and sf.server_ts is not None and math.isfinite(float(sf.server_ts)):
            try:
                self._base_seq = int(item.seq)
                self._base_ts = float(sf.server_ts)
                if self._mono0 is None:
                    self._mono0 = float(time.perf_counter())
            except Exception:
                pass
        # Insert by due_ts (stable; buffers are small)
        if not dq:
            dq.append(item)
        else:
            pos = len(dq)
            while pos > 0 and dq[pos - 1].due_ts > item.due_ts:
                pos -= 1
            if pos == len(dq):
                dq.append(item)
            else:
                dq.insert(pos, item)
        # Index by seq for tick-locked selection
        if item.seq is not None:
            try:
                self._seq_map[sf.source][int(item.seq)] = item
            except Exception:
                pass
        # Trim from left (oldest) if over limit
        while len(dq) > self.buffer_limit:
            old = dq.popleft()
            cb = old.release_cb
            if cb is not None:
                try:
                    cb(old.payload)  # type: ignore[misc]
                except Exception:
                    pass
        # Also enforce a soft limit on seq map
        try:
            m = self._seq_map[sf.source]
            if len(m) > (self.buffer_limit * 2):
                # Drop far-behind entries
                last = self._last_seq.get(sf.source)
                if last is not None:
                    to_drop = [k for k in m.keys() if k < int(last) - 2]
                    for k in to_drop:
                        m.pop(k, None)
        except Exception:
            pass
        self._submit_count[sf.source] += 1

    def clear(self, source: Optional[Source] = None) -> None:
        if source is None:
            targets = list(self._buf.keys())
        else:
            targets = [source]
        for s in targets:
            items = list(self._buf[s])
            self._buf[s].clear()
            self._last_seq[s] = None
            self._last_ts[s] = None
            for it in items:
                cb = it.release_cb
                if cb is not None:
                    try:
                        cb(it.payload)  # type: ignore[misc]
                    except Exception:
                        pass

    def pop_due(self, now: Optional[float], active: Source) -> Optional[ReadyFrame]:
        n = float(now if now is not None else time.perf_counter())
        # Tick-locked path when we have mapping information
        try:
            if (self._base_seq is not None) and (self._base_ts is not None) and (self._mono0 is not None) and (self.clock.offset is not None):
                server_target = n - float(self.clock.offset) - float(self.latency_s)
                delta = (server_target - float(self._base_ts)) / max(1e-6, self._period)
                seq_target = int(math.floor(delta)) + int(self._base_seq)
                last_seq = self._last_seq.get(active)
                if last_seq is not None:
                    seq_target = max(seq_target, int(last_seq) + 1)
                m = self._seq_map[active]
                it = m.pop(seq_target, None)
                if it is None:
                    return None
                # Remove selected from due-sorted deque as well
                dq2 = self._buf[active]
                for idx, itm in enumerate(dq2):
                    if itm is it:
                        dq2.remove(itm)
                        break
                # Monotonic guards
                if last_seq is not None and it.seq is not None and int(it.seq) <= int(last_seq):
                    cb = it.release_cb
                    if cb is not None:
                        try:
                            cb(it.payload)  # type: ignore[misc]
                        except Exception:
                            pass
                    return None
                last_ts = self._last_ts.get(active)
                if last_ts is not None and it.server_ts is not None and (float(it.server_ts) < float(last_ts) - self._ts_eps):
                    cb = it.release_cb
                    if cb is not None:
                        try:
                            cb(it.payload)  # type: ignore[misc]
                        except Exception:
                            pass
                    return None
                if it.server_ts is not None:
                    self._last_ts[active] = float(it.server_ts)
                if it.seq is not None:
                    try:
                        step = None
                        if last_seq is not None:
                            try:
                                step = int(it.seq) - int(last_seq)
                            except Exception:
                                step = None
                        self._last_seq[active] = int(it.seq)
                        if self._step_log and step is not None:
                            try:
                                late_ms = (n - float(it.due_ts)) * 1000.0
                                self._log.info("PRES_STEP src=%s seq=%d step=%d late_ms=%.3f", active.value, int(it.seq), int(step), float(late_ms))
                            except Exception:
                                pass
                    except Exception:
                        pass
                self._out_count[active] += 1
                return ReadyFrame(source=active, due_ts=it.due_ts, payload=it.payload, release_cb=it.release_cb, preview=False, server_ts=it.server_ts, seq=it.seq)
        except Exception:
            pass
        # Fallback: due_ts-driven oldest-due selection
        dq = self._buf[active]
        if not dq or dq[0].due_ts > n:
            return None
        while dq and dq[0].due_ts <= n:
            it = dq.popleft()
            last_seq = self._last_seq.get(active)
            if last_seq is not None and it.seq is not None and int(it.seq) <= int(last_seq):
                cb = it.release_cb
                if cb is not None:
                    try:
                        cb(it.payload)  # type: ignore[misc]
                    except Exception:
                        pass
                continue
            last_ts = self._last_ts.get(active)
            if last_ts is not None and it.server_ts is not None and (float(it.server_ts) < float(last_ts) - self._ts_eps):
                cb = it.release_cb
                if cb is not None:
                    try:
                        cb(it.payload)  # type: ignore[misc]
                    except Exception:
                        pass
                continue
            if it.server_ts is not None:
                self._last_ts[active] = float(it.server_ts)
            if it.seq is not None:
                try:
                    self._last_seq[active] = int(it.seq)
                except Exception:
                    pass
            self._out_count[active] += 1
            return ReadyFrame(source=active, due_ts=it.due_ts, payload=it.payload, release_cb=it.release_cb, preview=False, server_ts=it.server_ts, seq=it.seq)
        return None

    def peek_next_due(self, active: Source) -> Optional[float]:
        dq = self._buf[active]
        if not dq:
            return None
        return float(dq[0].due_ts)

    def stats(self) -> dict[str, object]:  # type: ignore[override]
        now = time.perf_counter()
        next_due = {}
        fill = {}
        for s in (Source.VT, Source.PYAV):
            dq = self._buf[s]
            fill[s.value] = len(dq)
            if dq:
                next_due[s.value] = int(round(max(0.0, dq[0].due_ts - now) * 1000.0))
            else:
                next_due[s.value] = None
        return {
            "submit": {s.value: self._submit_count[s] for s in (Source.VT, Source.PYAV)},
            "out": {s.value: self._out_count[s] for s in (Source.VT, Source.PYAV)},
            "preview": {s.value: 0 for s in (Source.VT, Source.PYAV)},
            "buf": {s.value: fill[s.value] for s in (Source.VT, Source.PYAV)},
            "latency_ms": int(round(self.latency_s * 1000.0)),
            "server_timestamp": True,
            "next_due_ms": next_due,
            "fill": fill,
            "due_path": {"server": 1, "arrival": 0},
        }

    # Compatibility: allow coordinator to relearn offset from buffered samples
    def compute_offset_median(self, source: Source) -> Optional[float]:
        dq = list(self._buf.get(source) or [])
        diffs: list[float] = []
        for it in dq[-20:]:  # recent items
            if it.server_ts is not None and math.isfinite(float(it.server_ts)):
                diffs.append(float(it.arrival_ts) - float(it.server_ts))
        if not diffs:
            return None
        diffs.sort()
        mid = len(diffs) // 2
        if len(diffs) % 2 == 1:
            med = diffs[mid]
        else:
            med = 0.5 * (diffs[mid - 1] + diffs[mid])
        return float(med) if math.isfinite(float(med)) else None

    def relearn_offset(self, source: Source) -> Optional[float]:
        med = self.compute_offset_median(source)
        if med is None:
            return None
        self.set_offset(float(med))
        return float(med)
