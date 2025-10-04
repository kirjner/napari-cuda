"""Telemetry helpers for the client streaming loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from qtpy import QtCore

from napari_cuda.client.rendering.metrics import ClientMetrics


@dataclass(frozen=True)
class TelemetryConfig:
    """Resolved logging/metrics behaviour for the client loop."""

    stats_level: Optional[int]
    metrics_enabled: bool
    metrics_interval_ms: int


def build_telemetry_config(
    *,
    stats_mode: str,
    metrics_enabled: bool,
    metrics_interval_ms: int,
) -> TelemetryConfig:
    """Translate raw env flags into a TelemetryConfig."""

    mode = (stats_mode or "").strip().lower()
    if mode in {"1", "true", "yes", "on", "info"}:
        stats_level: Optional[int] = logging.INFO
    elif mode in {"debug", "dbg"}:
        stats_level = logging.DEBUG
    else:
        stats_level = None

    interval = max(100, int(metrics_interval_ms))

    return TelemetryConfig(
        stats_level=stats_level,
        metrics_enabled=bool(metrics_enabled),
        metrics_interval_ms=interval,
    )


def create_metrics(config: TelemetryConfig) -> ClientMetrics:
    """Instantiate the metrics facade respecting enablement."""

    return ClientMetrics(enabled=config.metrics_enabled)


def start_stats_timer(
    parent,
    *,
    stats_level: Optional[int],
    callback,
    logger: logging.Logger,
) -> Optional[QtCore.QTimer]:
    """Start the presenter stats timer when enabled."""

    if stats_level is None:
        return None
    try:
        timer = QtCore.QTimer(parent)
        timer.setTimerType(QtCore.Qt.PreciseTimer)
        timer.setInterval(1000)
        timer.timeout.connect(callback)
        timer.start()
        return timer
    except Exception:
        logger.debug("Failed to start stats timer", exc_info=True)
        return None


def start_metrics_timer(
    parent,
    *,
    config: TelemetryConfig,
    metrics: ClientMetrics,
    logger: logging.Logger,
) -> Optional[QtCore.QTimer]:
    """Start the metrics timer if metrics collection is enabled."""

    if not config.metrics_enabled:
        return None
    try:
        timer = QtCore.QTimer(parent)
        timer.setTimerType(QtCore.Qt.PreciseTimer)
        timer.setInterval(config.metrics_interval_ms)
        timer.timeout.connect(metrics.dump_csv_row)  # type: ignore[arg-type]
        timer.start()
        return timer
    except Exception:
        logger.debug("Failed to start client metrics timer", exc_info=True)
        return None
