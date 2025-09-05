"""
Prometheus-backed metrics for napari-cuda (MVP-friendly).

Provides a tiny wrapper with inc/set/observe_ms and a per-server registry.
Uses milliseconds for timing metrics to match existing names (e.g., *_ms).
"""

from __future__ import annotations

from typing import Dict, Tuple

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._hists: Dict[str, Histogram] = {}

    def _get_counter(self, name: str) -> Counter:
        c = self._counters.get(name)
        if c is None:
            c = Counter(name, name, registry=self.registry)
            self._counters[name] = c
        return c

    def _get_gauge(self, name: str) -> Gauge:
        g = self._gauges.get(name)
        if g is None:
            g = Gauge(name, name, registry=self.registry)
            self._gauges[name] = g
        return g

    def _get_hist(self, name: str, buckets: Tuple[float, ...]) -> Histogram:
        h = self._hists.get(name)
        if h is None:
            # buckets are in milliseconds by convention here
            h = Histogram(name, name, buckets=buckets, registry=self.registry)
            self._hists[name] = h
        return h

    def inc(self, name: str, value: float = 1.0) -> None:
        self._get_counter(name).inc(value)

    def set(self, name: str, value: float) -> None:
        self._get_gauge(name).set(value)

    def observe_ms(self, name: str, value_ms: float, buckets: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)) -> None:
        self._get_hist(name, buckets).observe(value_ms)

