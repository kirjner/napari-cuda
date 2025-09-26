"""Tests for the preset registry and context integration."""

from __future__ import annotations

import pytest

from napari_cuda.server.config import (
    EncoderRuntime,
    ServerConfig,
    _apply_dataclass_overrides,
    load_server_ctx,
)
from napari_cuda.server.presets import resolve_preset


def test_resolve_preset_is_case_insensitive():
    assert resolve_preset("p1") is not None
    assert resolve_preset("P1") is not None
    assert resolve_preset(" P3 \n") is not None
    assert resolve_preset("does-not-exist") is None


def test_load_server_ctx_applies_nvenc_preset_overrides():
    env = {
        "NAPARI_CUDA_PRESET": "P1",
        # Runtime JSON intentionally selects a different preset; registry override should win.
        "NAPARI_CUDA_ENCODER_CONFIG": '{"runtime": {"preset": "P5"}}',
    }
    ctx = load_server_ctx(env)
    assert ctx.encoder_runtime.preset == "P1"


def test_latency_preset_overrides_profile_and_encode_fields():
    env = {"NAPARI_CUDA_PRESET": "latency"}
    ctx = load_server_ctx(env)
    assert ctx.cfg.profile == "latency"
    assert ctx.cfg.encode.bitrate == 10_000_000
    assert ctx.encoder_runtime.preset == "P3"
    assert ctx.encoder_runtime.bframes == 0
    assert ctx.encoder_runtime.lookahead == 0


def test_quality_preset_sets_higher_quality_runtime():
    env = {"NAPARI_CUDA_PRESET": "quality"}
    ctx = load_server_ctx(env)
    assert ctx.cfg.profile == "quality"
    assert ctx.cfg.encode.bitrate == 35_000_000
    assert ctx.encoder_runtime.preset == "P5"
    assert ctx.encoder_runtime.lookahead == 16
    assert ctx.encoder_runtime.aq == 1
    assert ctx.encoder_runtime.temporalaq == 1
    assert ctx.encoder_runtime.bframes == 2


def test_apply_dataclass_overrides_rejects_unknown_keys():
    cfg = ServerConfig()
    with pytest.raises(KeyError):
        _apply_dataclass_overrides(cfg, {"nope": 1}, "ServerConfig")

    runtime = EncoderRuntime()
    updated = _apply_dataclass_overrides(runtime, {"preset": "P2"}, "EncoderRuntime")
    assert updated.preset == "P2"
