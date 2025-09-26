"""Tests for ``client_loop_config`` helper."""

from __future__ import annotations

import os

import pytest

from napari_cuda.client.streaming.client_loop.client_loop_config import (
    ClientLoopConfig,
    load_client_loop_config,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure relevant env vars are unset before each test."""

    keys = [
        "NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MS",
        "NAPARI_CUDA_CLIENT_STARTUP_WARMUP_WINDOW_S",
        "NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MARGIN_MS",
        "NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MAX_MS",
        "NAPARI_CUDA_SERVER_TS_BIAS_MS",
        "NAPARI_CUDA_WAKE_FUDGE_MS",
        "NAPARI_CUDA_CLIENT_METRICS",
        "NAPARI_CUDA_CLIENT_METRICS_INTERVAL",
        "NAPARI_CUDA_USE_DISPLAY_LOOP",
        "NAPARI_CUDA_VT_STATS",
        "NAPARI_CUDA_CLIENT_DRAW_WATCHDOG_MS",
        "NAPARI_CUDA_CLIENT_EVENTLOOP_STALL_MS",
        "NAPARI_CUDA_CLIENT_EVENTLOOP_SAMPLE_MS",
        "NAPARI_CUDA_ZARR_Z",
        "NAPARI_CUDA_ZARR_Z_MIN",
        "NAPARI_CUDA_ZARR_Z_MAX",
        "NAPARI_CUDA_WHEEL_Z_STEP",
        "NAPARI_CUDA_DIMS_SET_RATE",
        "NAPARI_CUDA_CAMERA_SET_RATE",
        "NAPARI_CUDA_SETTINGS_SET_RATE",
        "NAPARI_CUDA_ORBIT_DEG_PER_PX_X",
        "NAPARI_CUDA_ORBIT_DEG_PER_PX_Y",
        "NAPARI_CUDA_INPUT_MAX_RATE",
        "NAPARI_CUDA_RESIZE_DEBOUNCE_MS",
        "NAPARI_CUDA_INPUT_LOG",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield


def test_load_client_loop_config_defaults():
    cfg = load_client_loop_config()
    assert isinstance(cfg, ClientLoopConfig)
    assert cfg.warmup_ms_override is None
    assert cfg.warmup_window_s == pytest.approx(0.75)
    assert cfg.server_bias_s == pytest.approx(0.0)
    assert cfg.wake_fudge_ms == pytest.approx(1.0)
    assert cfg.metrics_enabled is False
    assert cfg.metrics_interval_ms == 1000
    assert cfg.use_display_loop is False
    assert cfg.watchdog_ms == 0
    assert cfg.evloop_sample_ms == 100
    assert cfg.wheel_step == 1
    assert cfg.dims_rate_hz == pytest.approx(60.0)
    assert cfg.orbit_deg_per_px_x == pytest.approx(0.2)
    assert cfg.input_max_rate_hz == pytest.approx(120.0)
    assert cfg.resize_debounce_ms == 80
    assert cfg.input_log is False


def test_load_client_loop_config_reads_env(monkeypatch):
    monkeypatch.setenv("NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MS", "12.5")
    monkeypatch.setenv("NAPARI_CUDA_CLIENT_METRICS", "1")
    monkeypatch.setenv("NAPARI_CUDA_CLIENT_METRICS_INTERVAL", "0.5")
    monkeypatch.setenv("NAPARI_CUDA_USE_DISPLAY_LOOP", "true")
    monkeypatch.setenv("NAPARI_CUDA_VT_STATS", "debug")
    monkeypatch.setenv("NAPARI_CUDA_RESIZE_DEBOUNCE_MS", "120")
    monkeypatch.setenv("NAPARI_CUDA_INPUT_LOG", "on")
    monkeypatch.setenv("NAPARI_CUDA_WHEEL_Z_STEP", "3")
    monkeypatch.setenv("NAPARI_CUDA_SERVER_TS_BIAS_MS", "4.2")

    cfg = load_client_loop_config()

    assert cfg.warmup_ms_override == pytest.approx(12.5)
    assert cfg.metrics_enabled is True
    # 0.5s -> 500ms, clamped to >=100
    assert cfg.metrics_interval_ms == 500
    assert cfg.use_display_loop is True
    assert cfg.vt_stats_mode == "debug"
    assert cfg.resize_debounce_ms == 120
    assert cfg.input_log is True
    assert cfg.wheel_step == 3
    # 4.2 ms -> 0.0042 s bias
    assert cfg.server_bias_s == pytest.approx(0.0042)


def test_invalid_numeric_values_fall_back(monkeypatch):
    monkeypatch.setenv("NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MS", "oops")
    monkeypatch.setenv("NAPARI_CUDA_DIMS_SET_RATE", "-1")
    monkeypatch.setenv("NAPARI_CUDA_WHEEL_Z_STEP", "0")

    cfg = load_client_loop_config()

    # Invalid float -> None
    assert cfg.warmup_ms_override is None
    # Negative rate still flows through env_float; ensure denominator stays safe
    assert cfg.dims_rate_hz == pytest.approx(-1.0)
    # Wheel step must stay >=1
    assert cfg.wheel_step == 1
