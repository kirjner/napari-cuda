"""Server metrics helpers (optional dashboard bootstrap)."""

from __future__ import annotations

from .metrics_core import Metrics


def start_metrics_dashboard(
    host: str,
    port: int,
    metrics: Metrics,
    refresh_ms: int,
):
    """Start the Dash metrics dashboard and return the runner handle."""

    from .dash_dashboard import start_dash_dashboard  # type: ignore

    return start_dash_dashboard(host, int(port), metrics, int(refresh_ms))


def stop_metrics_dashboard(_runner) -> None:
    """Stop the metrics dashboard (no-op for current daemon thread)."""

    return
