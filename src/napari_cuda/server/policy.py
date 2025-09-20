"""Policy surface for multiscale level selection (function-first scaffolding)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Callable, Deque, Dict, Mapping, Sequence, Tuple

from .zarr_source import LevelDescriptor


@dataclass
class LevelMetricsWindow:
    window: int = 32
    times_ms: Dict[int, Deque[float]] = field(default_factory=dict)
    chunks: Dict[int, Deque[int]] = field(default_factory=dict)
    bytes_est: Dict[int, Deque[int]] = field(default_factory=dict)
    oversampling: Dict[int, Deque[float]] = field(default_factory=dict)

    def _append(self, store: Dict[int, Deque], level: int, value) -> None:
        dq = store.get(level)
        if dq is None:
            dq = deque(maxlen=self.window)
            store[level] = dq
        dq.append(value)

    def observe_time(self, level: int, value_ms: float) -> None:
        self._append(self.times_ms, level, float(value_ms))

    def observe_chunks(self, level: int, value: int) -> None:
        self._append(self.chunks, level, int(value))

    def observe_bytes(self, level: int, value: int) -> None:
        self._append(self.bytes_est, level, int(value))

    def observe_oversampling(self, level: int, value: float) -> None:
        self._append(self.oversampling, level, float(value))

    def mean_time_ms(self, level: int) -> float:
        dq = self.times_ms.get(level)
        if not dq:
            return 0.0
        return float(sum(dq) / len(dq))

    def last_time_ms(self, level: int) -> float:
        dq = self.times_ms.get(level)
        if not dq:
            return 0.0
        return float(dq[-1])

    def samples(self, level: int) -> int:
        dq = self.times_ms.get(level)
        return int(len(dq)) if dq else 0

    def mean_bytes(self, level: int) -> float:
        dq = self.bytes_est.get(level)
        if not dq:
            return 0.0
        return float(sum(dq) / len(dq))

    def last_bytes(self, level: int) -> float:
        dq = self.bytes_est.get(level)
        if not dq:
            return 0.0
        return float(dq[-1])

    def mean_chunks(self, level: int) -> float:
        dq = self.chunks.get(level)
        if not dq:
            return 0.0
        return float(sum(dq) / len(dq))

    def last_chunks(self, level: int) -> float:
        dq = self.chunks.get(level)
        if not dq:
            return 0.0
        return float(dq[-1])

    def latest_oversampling(self, level: int) -> float:
        dq = self.oversampling.get(level)
        if not dq:
            return 0.0
        return float(dq[-1])

    def snapshot(self) -> Dict[int, Dict[str, float]]:
        levels: set[int] = set()
        levels.update(self.times_ms.keys())
        levels.update(self.bytes_est.keys())
        levels.update(self.chunks.keys())
        levels.update(self.oversampling.keys())
        summary: Dict[int, Dict[str, float]] = {}
        for lvl in sorted(levels):
            summary[lvl] = {
                'mean_time_ms': self.mean_time_ms(lvl),
                'last_time_ms': self.last_time_ms(lvl),
                'mean_bytes': self.mean_bytes(lvl),
                'last_bytes': self.last_bytes(lvl),
                'mean_chunks': self.mean_chunks(lvl),
                'last_chunks': self.last_chunks(lvl),
                'latest_oversampling': self.latest_oversampling(lvl),
                'samples': float(self.samples(lvl)),
            }
        return summary

    def stats_for_level(self, level: int) -> Dict[str, float]:
        return {
            'mean_time_ms': self.mean_time_ms(level),
            'last_time_ms': self.last_time_ms(level),
            'mean_bytes': self.mean_bytes(level),
            'last_bytes': self.last_bytes(level),
            'mean_chunks': self.mean_chunks(level),
            'last_chunks': self.last_chunks(level),
            'latest_oversampling': self.latest_oversampling(level),
            'samples': float(self.samples(level)),
        }


@dataclass(frozen=True)
class PrimedLevelMetrics:
    latency_ms: float
    bytes_est: int
    chunks: int
    oversampling: float
    viewport_px: Tuple[int, int]
    timestamp_ms: float
    roi: Tuple[int, int, int, int] | None = None

    def is_applicable(self, viewport_px: Tuple[int, int], oversampling: float, tolerance: float = 0.25) -> bool:
        if viewport_px != self.viewport_px:
            return False
        if not math.isfinite(oversampling) or oversampling <= 0.0:
            return True
        diff = abs(self.oversampling - oversampling)
        margin = max(tolerance, oversampling * tolerance)
        return diff <= margin

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            'latency_ms': float(self.latency_ms),
            'bytes_est': int(self.bytes_est),
            'chunks': int(self.chunks),
            'oversampling': float(self.oversampling),
            'viewport_px': (int(self.viewport_px[0]), int(self.viewport_px[1])),
            'timestamp_ms': float(self.timestamp_ms),
        }
        if self.roi is not None:
            data['roi'] = tuple(int(v) for v in self.roi)
        return data


@dataclass(frozen=True)
class LevelSelectionContext:
    levels: Sequence[LevelDescriptor]
    viewport_px: tuple[int, int]
    physical_scale: tuple[float, float, float]
    oversampling: float
    current_level: int
    idle_ms: float
    use_volume: bool
    ms_state: object
    metrics: LevelMetricsWindow
    intent_level: int | None = None
    level_oversampling: Mapping[int, float] = field(default_factory=dict)
    primed_metrics: Mapping[int, PrimedLevelMetrics] | None = None
    estimated_bytes: Mapping[int, int] = field(default_factory=dict)


PolicyFunc = Callable[[LevelSelectionContext], int | None]


def select_latency_aware(ctx: LevelSelectionContext) -> int | None:
    budget_ms = 18.0 if not ctx.use_volume else 28.0
    overs_map = ctx.level_oversampling or {}
    primed_map = ctx.primed_metrics or {}
    est_map = ctx.estimated_bytes or {}

    def _primed(level: int) -> PrimedLevelMetrics | None:
        entry = primed_map.get(level)
        if isinstance(entry, PrimedLevelMetrics):
            overs = float(overs_map.get(level, float('nan')))
            if entry.is_applicable(ctx.viewport_px, overs):
                return entry
        return None

    candidates = sorted({int(k) for k in overs_map.keys()} | {int(k) for k in primed_map.keys()})

    def _time_for(level: int) -> float:
        t = ctx.metrics.mean_time_ms(level)
        if t > 0.0:
            return t
        t = ctx.metrics.last_time_ms(level)
        if t > 0.0:
            return t
        primed = _primed(level)
        return float(primed.latency_ms) if primed else 0.0

    def _bytes_for(level: int) -> float:
        est = est_map.get(level)
        if isinstance(est, (int, float)) and est > 0:
            return float(est)
        b = ctx.metrics.mean_bytes(level)
        if b > 0.0:
            return b
        b = ctx.metrics.last_bytes(level)
        if b > 0.0:
            return b
        primed = _primed(level)
        return float(primed.bytes_est) if primed else 0.0

    # Try to find finest level within budget based on recorded timings (live or primed)
    measured: list[tuple[int, float]] = []
    for level in candidates:
        latency = _time_for(level)
        if latency > 0.0:
            measured.append((level, latency))
            if latency <= budget_ms:
                return level

    # Build a reference throughput (ms per byte) from the finest measured level, else current level
    ref_time = 0.0
    ref_bytes = 0.0
    if measured:
        finest_level, finest_time = measured[0]
        ref_time = finest_time
        ref_bytes = _bytes_for(finest_level)
    else:
        ref_level = ctx.current_level
        ref_time = _time_for(ref_level)
        ref_bytes = _bytes_for(ref_level)

    predictions: list[tuple[float, int]] = []
    if ref_time > 0.0 and ref_bytes > 0.0:
        for level in sorted(candidates):
            if _time_for(level) > 0.0:
                continue
            bytes_est = _bytes_for(level)
            if bytes_est <= 0.0:
                continue
            predicted = ref_time * (bytes_est / ref_bytes)
            if predicted <= budget_ms:
                return level
            predictions.append((predicted, level))
        for predicted, level in sorted(predictions):
            # Already sorted by predicted time; pick the most promising fallback
            return level

    if candidates:
        return max(candidates)
    # Fall back to intent/current level when we lack timing data
    return ctx.intent_level if ctx.intent_level is not None else ctx.current_level


def resolve_policy(name: str) -> PolicyFunc:
    return select_latency_aware
