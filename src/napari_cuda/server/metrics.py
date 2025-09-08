"""
Lightweight JSON metrics for napari-cuda.

This module provides a small, dependency-free metrics aggregator that supports:
- counters (monotonic totals)
- gauges (latest value)
- histograms (rolling window with basic stats and percentiles)

All timings are expected in milliseconds by convention (e.g., *_ms).
The aggregator exposes a `snapshot()` that returns a JSON‑ready dict.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass
class _Hist:
    window: int
    values: Deque[float] = field(default_factory=deque)
    last: float = 0.0
    total: float = 0.0
    count: int = 0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    def observe(self, v: float) -> None:
        self.last = float(v)
        # manage rolling window
        if len(self.values) == self.window:
            # drop oldest from rolling totals
            try:
                dropped = self.values.popleft()
                self.total -= dropped
            except Exception:
                # if popleft fails (shouldn't), reset window
                self.values.clear()
                self.total = 0.0
        self.values.append(self.last)
        self.total += self.last
        # track min/max over all-time and within-window conservatively
        if self.last < self.min_v:
            self.min_v = self.last
        if self.last > self.max_v:
            self.max_v = self.last
        self.count += 1

    def stats(self) -> Dict[str, float]:
        n = len(self.values)
        if n == 0:
            return {
                "last_ms": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }
        mean = (self.total / n) if n else 0.0
        # Full stats: compute percentiles from current window snapshot
        arr: List[float] = sorted(self.values)
        def q(p: float) -> float:
            if n == 0:
                return 0.0
            idx = min(max(int(round(p * (n - 1))), 0), n - 1)
            return arr[idx]
        return {
            "last_ms": self.last,
            "mean_ms": mean,
            "p50_ms": q(0.50),
            "p90_ms": q(0.90),
            "p99_ms": q(0.99),
            "min_ms": min(arr[0], self.min_v if self.min_v != float("inf") else arr[0]),
            "max_ms": max(arr[-1], self.max_v if self.max_v != float("-inf") else arr[-1]),
        }


class Metrics:
    """Small metrics aggregator with JSON snapshot."""

    def __init__(self) -> None:
        self._window = max(16, _env_int("NAPARI_CUDA_METRICS_WINDOW", 512))
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._hists: Dict[str, _Hist] = {}
        # for derived fps
        self._last_frames: float = 0.0
        self._last_ts: float = time.time()
        self._last_fps: float = 0.0

    # Public API (kept compatible with existing call sites)
    def inc(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + float(value)

    def set(self, name: str, value: float) -> None:
        self._gauges[name] = float(value)

    def observe_ms(self, name: str, value_ms: float, buckets: Tuple[float, ...] | None = None) -> None:  # buckets ignored
        h = self._hists.get(name)
        if h is None:
            h = _Hist(window=self._window, values=deque(maxlen=self._window))
            self._hists[name] = h
        h.observe(float(value_ms))

    # Snapshot for HTTP JSON endpoint
    def snapshot(self) -> Dict[str, object]:
        now = time.time()
        frames = self._counters.get("napari_cuda_frames_total", 0.0)
        dt = max(1e-3, now - self._last_ts)
        df = max(0.0, frames - self._last_frames)
        fps = df / dt if df >= 0 else 0.0
        # smooth slightly: carry last when gaps are long
        self._last_fps = fps
        self._last_frames = frames
        self._last_ts = now

        gauges = {k: float(v) for k, v in self._gauges.items()}
        counters = {k: int(v) if float(v).is_integer() else float(v) for k, v in self._counters.items()}
        hist = {k: v.stats() for k, v in self._hists.items()}

        return {
            "version": "v1",
            "ts": now,
            "gauges": gauges,
            "counters": counters,
            "histograms": hist,
            "derived": {
                "fps": self._last_fps,
            },
        }
