"""Helpers for resolving the server runtime context and configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from napari_cuda.server.app.config import load_server_ctx
from napari_cuda.server.config import ServerCtx
from napari_cuda.server.engine.api import configure_bitstream

EncodeConfigMap = Tuple[str, "EncodeConfig"]


@dataclass
class EncodeConfig:
    fps: int = 60
    codec: int = 1  # 1=h264, 2=hevc, 3=av1
    bitrate: int = 12_000_000
    keyint: int = 120


def capture_env(source: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Copy environment variables so tests can inject overrides."""

    if source is None:
        from os import environ

        return dict(environ)
    return dict(source)


def resolve_server_ctx(env: Mapping[str, str]) -> ServerCtx:
    """Load ServerCtx from environment, falling back to defaults on failure."""

    try:
        return load_server_ctx(env)
    except Exception:
        return load_server_ctx({})  # type: ignore[arg-type]


def resolve_env_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return str(Path(value).expanduser().resolve())


def resolve_data_roots(
    ctx_env: Mapping[str, str],
    *,
    logger: Optional[logging.Logger] = None,
    cwd: Optional[Path] = None,
) -> tuple[Optional[str], Path]:
    """Resolve the configured dataset roots.

    Returns a tuple of (data_root, browse_root). data_root is the canonical
    string path from `NAPARI_CUDA_DATA_ROOT` if set, otherwise None. browse_root
    is the path we expose for directory listings.
    """

    env_root = resolve_env_path(ctx_env.get("NAPARI_CUDA_DATA_ROOT"))
    if env_root:
        return env_root, Path(env_root)

    base = (cwd or Path.cwd()).resolve()
    if logger is not None:
        logger.info("NAPARI_CUDA_DATA_ROOT not set; defaulting browse root to %s", base)
    return None, base


def configure_bitstream_policy(ctx: ServerCtx, *, logger: Optional[logging.Logger] = None) -> None:
    """Apply encoder bitstream policy with defensive logging."""

    try:
        configure_bitstream(ctx.bitstream)
    except Exception:
        if logger is not None:
            logger.debug("Bitstream configuration failed", exc_info=True)


def resolve_volume_caps(ctx: ServerCtx, hw_limits: object) -> tuple[int, int, int, int]:
    """Return volume size limits from config and hardware probes."""

    cfg_limits = ctx.cfg
    cfg_bytes = int(max(0, getattr(cfg_limits, "max_volume_bytes", 0)))
    cfg_voxels = int(max(0, getattr(cfg_limits, "max_volume_voxels", 0)))
    hw_bytes = int(getattr(hw_limits, "volume_max_bytes", 0))
    hw_voxels = int(getattr(hw_limits, "volume_max_voxels", 0))
    return cfg_bytes, cfg_voxels, hw_bytes, hw_voxels


def resolve_encode_config(encode_cfg: object, *, fallback_fps: int) -> EncodeConfigMap:
    """Derive encode configuration overrides from ServerConfig."""

    codec_map = {"h264": 1, "hevc": 2, "av1": 3}
    if encode_cfg is None:
        return "h264", EncodeConfig(fps=fallback_fps)

    codec_name_raw = getattr(encode_cfg, "codec", "h264")
    codec_name = str(codec_name_raw).lower()
    codec_name = codec_name if codec_name in codec_map else "h264"
    config = EncodeConfig(
        fps=int(getattr(encode_cfg, "fps", fallback_fps)),
        codec=codec_map.get(codec_name, 1),
        bitrate=int(getattr(encode_cfg, "bitrate", 12_000_000)),
        keyint=int(getattr(encode_cfg, "keyint", 120)),
    )
    return codec_name, config


__all__ = [
    "EncodeConfig",
    "capture_env",
    "configure_bitstream_policy",
    "resolve_data_roots",
    "resolve_encode_config",
    "resolve_env_path",
    "resolve_server_ctx",
    "resolve_volume_caps",
]
