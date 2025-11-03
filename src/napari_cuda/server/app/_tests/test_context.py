from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from napari_cuda.server.app import context


def test_capture_env_returns_copy() -> None:
    source = {"A": "1", "B": "2"}
    captured = context.capture_env(source)
    assert captured == source
    captured["A"] = "changed"
    assert source["A"] == "1"


def test_resolve_env_path_normalizes(tmp_path) -> None:
    raw = tmp_path / ".." / tmp_path.name
    resolved = context.resolve_env_path(str(raw))
    assert resolved == str(tmp_path.resolve())


def test_resolve_data_roots_env_override(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    env = {"NAPARI_CUDA_DATA_ROOT": str(data_root)}

    resolved_root, browse_root = context.resolve_data_roots(env)

    assert resolved_root == str(data_root.resolve())
    assert browse_root == data_root.resolve()


def test_resolve_data_roots_defaults_log(caplog, tmp_path) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("napari_cuda.test.context")

    resolved_root, browse_root = context.resolve_data_roots({}, logger=logger, cwd=tmp_path)

    assert resolved_root is None
    assert browse_root == tmp_path.resolve()
    assert "NAPARI_CUDA_DATA_ROOT not set" in caplog.text


def test_configure_bitstream_policy_handles_errors(monkeypatch, caplog) -> None:
    ctx = SimpleNamespace(bitstream="payload")

    def boom(bitstream: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(context, "configure_bitstream", boom)

    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger("napari_cuda.test.context")
    context.configure_bitstream_policy(ctx, logger=logger)

    assert "Bitstream configuration failed" in caplog.text


@pytest.mark.parametrize(
    "codec,expected_id",
    [
        ("h264", 1),
        ("hevc", 2),
        ("av1", 3),
        ("unknown", 1),
    ],
)
def test_resolve_encode_config_variants(codec, expected_id) -> None:
    encode_cfg = SimpleNamespace(codec=codec, fps=30, bitrate=5_000_000, keyint=60)

    codec_name, cfg = context.resolve_encode_config(encode_cfg, fallback_fps=25)

    assert codec_name in {"h264", "hevc", "av1"}
    assert cfg.codec == expected_id
    assert cfg.fps == 30
    assert cfg.bitrate == 5_000_000
    assert cfg.keyint == 60


def test_resolve_encode_config_none_uses_defaults() -> None:
    codec_name, cfg = context.resolve_encode_config(None, fallback_fps=50)

    assert codec_name == "h264"
    assert cfg.codec == 1
    assert cfg.fps == 50
    assert cfg.bitrate == 12_000_000
    assert cfg.keyint == 120


def test_resolve_volume_caps_uses_cfg_and_hw_limits() -> None:
    ctx = SimpleNamespace(cfg=SimpleNamespace(max_volume_bytes=1024, max_volume_voxels=2048))
    hw_limits = SimpleNamespace(volume_max_bytes=4096, volume_max_voxels=8192)

    cfg_bytes, cfg_voxels, hw_bytes, hw_voxels = context.resolve_volume_caps(ctx, hw_limits)

    assert cfg_bytes == 1024
    assert cfg_voxels == 2048
    assert hw_bytes == 4096
    assert hw_voxels == 8192
