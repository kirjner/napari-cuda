"""Policy surface for multiscale level selection (function-first scaffolding)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, Mapping, Sequence

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


PolicyFunc = Callable[[LevelSelectionContext], int | None]

POLICIES: Dict[str, PolicyFunc] = {}


def register_policy(name: str) -> Callable[[PolicyFunc], PolicyFunc]:
    def decorator(func: PolicyFunc) -> PolicyFunc:
        POLICIES[name] = func
        return func

    return decorator


@register_policy("budget")
def select_budget_only(ctx: LevelSelectionContext) -> int | None:
    return ctx.intent_level if ctx.intent_level is not None else ctx.current_level


# Manual alias for backward compatibility
POLICIES["fixed"] = select_budget_only


@register_policy("screen_pixel")
def select_screen_pixel(ctx: LevelSelectionContext) -> int | None:
    intent = ctx.intent_level if ctx.intent_level is not None else ctx.current_level
    overs_map = ctx.level_oversampling or {}
    if not overs_map:
        return intent
    target = 1.5
    best_level = intent
    for level, overs in sorted(overs_map.items()):
        if overs <= target:
            best_level = level
            break
    else:
        best_level = max(overs_map.keys())
    return best_level


@register_policy("latency")
def select_latency_aware(ctx: LevelSelectionContext) -> int | None:
    budget_ms = 18.0 if not ctx.use_volume else 28.0
    overs_map = ctx.level_oversampling or {}
    candidates = sorted(overs_map.keys()) if overs_map else []
    # Try to find finest level within budget based on recorded timings
    measured: list[tuple[int, float]] = []
    for level in candidates:
        mean_ms = ctx.metrics.mean_time_ms(level)
        if mean_ms > 0.0:
            measured.append((level, mean_ms))
            if mean_ms <= budget_ms:
                return level

    # Build a reference throughput (ms per byte) from the finest measured level, else current level
    ref_time = 0.0
    ref_bytes = 0.0
    if measured:
        finest_level, finest_time = measured[0]
        ref_time = finest_time
        ref_bytes = ctx.metrics.mean_bytes(finest_level) or ctx.metrics.last_bytes(finest_level)
    else:
        ref_level = ctx.current_level
        ref_time = ctx.metrics.mean_time_ms(ref_level) or ctx.metrics.last_time_ms(ref_level)
        ref_bytes = ctx.metrics.mean_bytes(ref_level) or ctx.metrics.last_bytes(ref_level)

    predictions: list[tuple[float, int]] = []
    if ref_time > 0.0 and ref_bytes > 0.0:
        for level in candidates:
            if ctx.metrics.mean_time_ms(level) > 0.0:
                continue
            bytes_est = ctx.metrics.mean_bytes(level) or ctx.metrics.last_bytes(level)
            if bytes_est <= 0.0:
                continue
            predicted = ref_time * (bytes_est / ref_bytes)
            predictions.append((predicted, level))
        for predicted, level in sorted(predictions):
            if predicted <= budget_ms:
                return level

    if candidates:
        return max(candidates)
    # Fall back to oversampling heuristic when we lack timing data
    return select_screen_pixel(ctx)


@register_policy("coarse_first")
def select_coarse_first(ctx: LevelSelectionContext) -> int | None:
    overs_map = ctx.level_oversampling or {}
    if not overs_map:
        return ctx.intent_level if ctx.intent_level is not None else ctx.current_level
    cap = 2.0
    chosen = max(overs_map.keys())
    for level, overs in sorted(overs_map.items()):
        if overs <= cap:
            chosen = level
            break
    return chosen


def available_policies() -> Iterable[str]:
    return POLICIES.keys()


def resolve_policy(name: str) -> PolicyFunc:
    if name in POLICIES:
        return POLICIES[name]
    default = POLICIES.get("budget")
    return default if default is not None else select_budget_only
