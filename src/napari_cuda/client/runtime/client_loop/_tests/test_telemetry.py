"""Telemetry helper tests."""

from __future__ import annotations

import logging

from qtpy import QtCore

from napari_cuda.client.runtime.client_loop.telemetry import (
    TelemetryConfig,
    build_telemetry_config,
    create_metrics,
    start_metrics_timer,
    start_stats_timer,
)


def test_build_telemetry_config_modes():
    cfg_info = build_telemetry_config(stats_mode="info", metrics_enabled=True, metrics_interval_ms=250)
    assert cfg_info.stats_level == logging.INFO
    assert cfg_info.metrics_enabled is True
    assert cfg_info.metrics_interval_ms == 250

    cfg_dbg = build_telemetry_config(stats_mode="dbg", metrics_enabled=False, metrics_interval_ms=10)
    assert cfg_dbg.stats_level == logging.DEBUG
    assert cfg_dbg.metrics_enabled is False
    # clamped to >=100
    assert cfg_dbg.metrics_interval_ms == 100

    cfg_off = build_telemetry_config(stats_mode="", metrics_enabled=True, metrics_interval_ms=1000)
    assert cfg_off.stats_level is None


def test_create_metrics_tracks_enablement():
    cfg = TelemetryConfig(stats_level=None, metrics_enabled=False, metrics_interval_ms=500)
    metrics = create_metrics(cfg)
    assert metrics.enabled is False


def test_start_timers():
    parent = QtCore.QObject()
    cfg = TelemetryConfig(stats_level=logging.INFO, metrics_enabled=True, metrics_interval_ms=200)
    metrics = create_metrics(cfg)

    stats_timer = start_stats_timer(parent, stats_level=cfg.stats_level, callback=lambda: None, logger=logging.getLogger(__name__))
    assert stats_timer is not None

    metrics_timer = start_metrics_timer(parent, config=cfg, metrics=metrics, logger=logging.getLogger(__name__))
    assert metrics_timer is not None


def test_metrics_timer_skipped_when_disabled():
    parent = QtCore.QObject()
    cfg = TelemetryConfig(stats_level=None, metrics_enabled=False, metrics_interval_ms=200)
    metrics = create_metrics(cfg)

    timer = start_metrics_timer(parent, config=cfg, metrics=metrics, logger=logging.getLogger(__name__))
    assert timer is None
