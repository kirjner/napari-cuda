from __future__ import annotations

"""
Client-side lightweight metrics collector mirroring server Metrics.

- Minimal, dependency-free; disabled unless explicitly enabled.
- API: inc(name), set(name, value), observe_ms(name, value_ms), snapshot().
- Optional CSV writer helper for periodic dumps driven by callers.
"""

import csv
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


def _truthy(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    return s in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


@dataclass
class _Hist:
    window: int
    values: deque[float] = field(default_factory=deque)
    last: float = 0.0
    total: float = 0.0
    count: int = 0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    def observe(self, v: float) -> None:
        self.last = float(v)
        if len(self.values) == self.values.maxlen:
            try:
                dropped = self.values.popleft()
                self.total -= dropped
            except Exception:
                self.values.clear()
                self.total = 0.0
        self.values.append(self.last)
        self.total += self.last
        if self.last < self.min_v:
            self.min_v = self.last
        if self.last > self.max_v:
            self.max_v = self.last
        self.count += 1

    def stats(self) -> dict[str, float]:
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
        mean = (self.total / n)
        arr: list[float] = sorted(self.values)
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


class ClientMetrics:
    """Small client metrics aggregator with optional CSV output.

    Enabled via env NAPARI_CUDA_CLIENT_METRICS=1 or constructor flag.
    """

    def __init__(self, enabled: Optional[bool] = None, *, window: int = 512) -> None:
        if enabled is None:
            enabled = _truthy(os.getenv("NAPARI_CUDA_CLIENT_METRICS", "0"))
        self.enabled = bool(enabled)
        # Allow environment override for metrics window length
        try:
            win_env = os.getenv("NAPARI_CUDA_CLIENT_METRICS_WINDOW")
            if win_env is not None and win_env.strip() != "":
                window = int(float(win_env))
        except Exception:
            pass
        self._window = max(16, int(window))
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._hists: dict[str, _Hist] = {}
        # CSV writer state
        self._csv_path: Optional[str] = _env_str("NAPARI_CUDA_CLIENT_METRICS_OUT", None)
        self._csv_lock = threading.Lock()
        self._csv_header_written = False
        # Derived fps from client perspective (presented frames may be tracked elsewhere)
        self._last_frames: float = 0.0
        self._last_ts: float = time.time()
        self._last_fps: float = 0.0

    # Public API
    def inc(self, name: str, value: float = 1.0) -> None:
        if not self.enabled:
            return
        self._counters[name] = self._counters.get(name, 0.0) + float(value)

    def set(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self._gauges[name] = float(value)

    def observe_ms(self, name: str, value_ms: float) -> None:
        if not self.enabled:
            return
        h = self._hists.get(name)
        if h is None:
            h = _Hist(window=self._window, values=deque(maxlen=self._window))
            self._hists[name] = h
        h.observe(float(value_ms))

    def snapshot(self) -> dict[str, object]:
        if not self.enabled:
            return {"version": "v1", "ts": time.time(), "gauges": {}, "counters": {}, "histograms": {}, "derived": {}}
        now = time.time()
        frames = self._counters.get("napari_cuda_client_presented_total", 0.0)
        dt = max(1e-3, now - self._last_ts)
        df = max(0.0, frames - self._last_frames)
        fps = df / dt if df >= 0 else 0.0
        self._last_fps = fps
        self._last_frames = frames
        self._last_ts = now
        gauges = {k: float(v) for k, v in self._gauges.items()}
        counters = {k: int(v) if float(v).is_integer() else float(v) for k, v in self._counters.items()}
        hist = {k: v.stats() for k, v in self._hists.items()}
        return {"version": "v1", "ts": now, "gauges": gauges, "counters": counters, "histograms": hist, "derived": {"fps": self._last_fps}}

    # CSV writer: caller drives cadence (e.g., 1 Hz) by calling dump_csv_row()
    def dump_csv_row(self) -> None:
        if not self.enabled:
            return
        path = self._csv_path
        if not path:
            return
        snap = self.snapshot()
        row: dict[str, object] = {"ts": snap.get("ts", time.time())}
        # Flatten a few common fields for convenience
        for k, v in (snap.get("gauges", {}) or {}).items():
            row[f"g:{k}"] = v
        for k, v in (snap.get("counters", {}) or {}).items():
            row[f"c:{k}"] = v
        for k, stats in (snap.get("histograms", {}) or {}).items():
            if not isinstance(stats, dict):
                continue
            for kk, vv in stats.items():
                row[f"h:{k}:{kk}"] = vv
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass
        with self._csv_lock:
            try:
                write_header = not (os.path.exists(path) and os.path.getsize(path) > 0) and not self._csv_header_written
            except Exception:
                write_header = not self._csv_header_written
            try:
                with open(path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        w.writeheader()
                        self._csv_header_written = True
                    w.writerow(row)
            except Exception:
                # Best-effort only
                pass
