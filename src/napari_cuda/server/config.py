"""Centralized server configuration stubs.

This module introduces typed configuration objects and a minimal loader to
consolidate environment parsing. It does NOT change behavior yet; existing
modules can continue using their current env reads. When we wire this in, the
coordinator will call `load_server_config()` once and pass the config down.

Usage plan (PR1/PR2):
- PR1: import this module in the headless server but do not change behavior;
  just log the resolved config for observability.
- PR2: replace scattered os.getenv calls with fields from `ServerConfig` and
  `EncodeCfg` passed through a `ServerCtx`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional
import os


# ---- Helpers -----------------------------------------------------------------

def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    v = env.get(name)
    if v is None:
        return bool(default)
    v = v.strip().lower()
    return v not in ("0", "", "false", "no", "off")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    v = env.get(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    v = env.get(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_str(env: Mapping[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    v = env.get(name)
    if v is None:
        return default
    v = v.strip()
    return v if v != "" else default


# ---- Types -------------------------------------------------------------------

@dataclass(frozen=True)
class EncodeCfg:
    """Encoder parameters for the streaming pipeline.

    Note: `codec` is a symbolic string for readability; existing code may use
    numeric mapping (1=h264, 2=hevc, 3=av1). We provide helpers to translate
    when needed during integration.
    """

    fps: int = 60
    codec: str = "h264"  # "h264" | "hevc" | "av1"
    bitrate: int = 10_000_000
    keyint: int = 120


def _profile_defaults(name: str) -> EncodeCfg:
    p = (name or "").strip().lower()
    if p == "quality":
        return EncodeCfg(
            fps=60,
            codec="h264",
            bitrate=35_000_000,
            keyint=120,
        )
    # Default to latency profile
    return EncodeCfg(fps=60, codec="h264", bitrate=10_000_000, keyint=120)


def _codec_to_int(codec: str) -> int:
    c = (codec or "").strip().lower()
    if c == "h264":
        return 1
    if c == "hevc":
        return 2
    if c == "av1":
        return 3
    return 1


@dataclass(frozen=True)
class ServerConfig:
    """Top‑level server configuration.

    This consolidates runtime flags and provides defaults aligned with the
    refactor plan to reduce branching and env sprawl.
    """

    # Network
    host: str = "0.0.0.0"
    state_port: int = 8081
    pixel_port: int = 8082

    # Data (Zarr)
    zarr_path: Optional[str] = None
    zarr_level: Optional[str] = None
    zarr_axes: Optional[str] = None
    zarr_z: Optional[int] = None

    # Render
    width: int = 1920
    height: int = 1080
    use_volume: bool = False
    ndisplay: int = 2  # 2|3 intent; worker may clamp

    # Animation (dev only)
    animate: bool = False
    animate_dps: float = 30.0

    # Queueing
    frame_queue: int = 1

    # Encoder profile
    profile: str = "latency"  # "latency" | "quality"
    encode: EncodeCfg = EncodeCfg()

    # Behavior toggles (kept minimal)
    idr_on_reset: bool = True
    packer: str = "cython"  # "cython" | "python"


def load_server_config(env: Optional[Mapping[str, str]] = None) -> ServerConfig:
    """Load server configuration from environment (no side effects).

    Environment keys consulted (subset; see plan for reductions):
    - NAPARI_CUDA_HOST, NAPARI_CUDA_STATE_PORT, NAPARI_CUDA_PIXEL_PORT
    - NAPARI_CUDA_ZARR_PATH, NAPARI_CUDA_ZARR_LEVEL, NAPARI_CUDA_ZARR_AXES, NAPARI_CUDA_ZARR_Z
    - NAPARI_CUDA_WIDTH, NAPARI_CUDA_HEIGHT, NAPARI_CUDA_USE_VOLUME, NAPARI_CUDA_NDISPLAY
    - NAPARI_CUDA_ANIMATE, NAPARI_CUDA_TURNTABLE_DPS
    - NAPARI_CUDA_FRAME_QUEUE
    - NAPARI_CUDA_PROFILE (latency|quality)
    - NAPARI_CUDA_FPS, NAPARI_CUDA_BITRATE, NAPARI_CUDA_KEYINT, NAPARI_CUDA_CODEC
    - NAPARI_CUDA_IDR_ON_RESET
    - NAPARI_CUDA_PACKER (cython|python)
    """

    env = env or os.environ

    host = _env_str(env, "NAPARI_CUDA_HOST", "0.0.0.0") or "0.0.0.0"
    state_port = _env_int(env, "NAPARI_CUDA_STATE_PORT", 8081)
    pixel_port = _env_int(env, "NAPARI_CUDA_PIXEL_PORT", 8082)

    zarr_path = _env_str(env, "NAPARI_CUDA_ZARR_PATH")
    zarr_level = _env_str(env, "NAPARI_CUDA_ZARR_LEVEL")
    zarr_axes = _env_str(env, "NAPARI_CUDA_ZARR_AXES")
    zarr_z = None
    _zz = _env_str(env, "NAPARI_CUDA_ZARR_Z")
    if _zz is not None:
        try:
            zarr_z = int(_zz)
        except Exception:
            zarr_z = None

    width = _env_int(env, "NAPARI_CUDA_WIDTH", 1920)
    height = _env_int(env, "NAPARI_CUDA_HEIGHT", 1080)
    use_volume = _env_bool(env, "NAPARI_CUDA_USE_VOLUME", False)
    ndisplay = _env_int(env, "NAPARI_CUDA_NDISPLAY", 2)
    animate = _env_bool(env, "NAPARI_CUDA_ANIMATE", False)
    animate_dps = _env_float(env, "NAPARI_CUDA_TURNTABLE_DPS", 30.0)
    frame_queue = max(1, _env_int(env, "NAPARI_CUDA_FRAME_QUEUE", 1))

    profile = (_env_str(env, "NAPARI_CUDA_PROFILE", "latency") or "latency").lower()
    enc = _profile_defaults(profile)
    # Allow basic overrides
    fps = _env_int(env, "NAPARI_CUDA_FPS", enc.fps)
    bitrate = _env_int(env, "NAPARI_CUDA_BITRATE", enc.bitrate)
    keyint = _env_int(env, "NAPARI_CUDA_KEYINT", enc.keyint)
    codec = _env_str(env, "NAPARI_CUDA_CODEC", enc.codec) or enc.codec
    enc = EncodeCfg(fps=fps, codec=codec, bitrate=bitrate, keyint=keyint)

    idr_on_reset = _env_bool(env, "NAPARI_CUDA_IDR_ON_RESET", True)
    packer = (_env_str(env, "NAPARI_CUDA_PACKER", "cython") or "cython").lower()

    return ServerConfig(
        host=host,
        state_port=state_port,
        pixel_port=pixel_port,
        zarr_path=zarr_path,
        zarr_level=zarr_level,
        zarr_axes=zarr_axes,
        zarr_z=zarr_z,
        width=width,
        height=height,
        use_volume=use_volume,
        ndisplay=ndisplay,
        animate=animate,
        animate_dps=animate_dps,
        frame_queue=frame_queue,
        profile=profile,
        encode=enc,
        idr_on_reset=idr_on_reset,
        packer=packer,
    )


@dataclass(frozen=True)
class ServerCtx:
    """Resolved server runtime context.

    Wraps the structured `ServerConfig` and collects additional operational
    toggles that were previously read ad-hoc from the environment. Built once
    at startup and passed around (observe-only at first, then authoritative).
    """

    cfg: ServerConfig

    # Queuing / dumps / watchdogs
    frame_queue: int = 1
    dump_bitstream: int = 0
    dump_dir: str = "benchmarks/bitstreams"
    kf_watchdog_cooldown_s: float = 2.0

    # Logging toggles
    log_camera_info: bool = False
    log_camera_debug: bool = False
    log_state_traces: bool = False
    log_volume_info: bool = False
    log_dims_info: bool = False
    debug: bool = False
    log_sends_env: bool = False

    # Metrics UI
    metrics_port: int = 8083
    metrics_refresh_ms: int = 1000

    # Policy event path
    policy_event_path: str = "tmp/policy_events.jsonl"


def load_server_ctx(env: Optional[Mapping[str, str]] = None) -> ServerCtx:
    """Build a `ServerCtx` by reading environment once.

    Note: This does not mutate the process environment and is side-effect free.
    """
    env = env or os.environ
    cfg = load_server_config(env)

    frame_queue = max(1, _env_int(env, "NAPARI_CUDA_FRAME_QUEUE", 1))
    dump_bitstream = max(0, _env_int(env, "NAPARI_CUDA_DUMP_BITSTREAM", 0))
    dump_dir = _env_str(env, "NAPARI_CUDA_DUMP_DIR", "benchmarks/bitstreams") or "benchmarks/bitstreams"

    _cool = _env_str(env, "NAPARI_CUDA_KF_WATCHDOG_COOLDOWN")
    try:
        kf_watchdog_cooldown_s = float(_cool) if _cool is not None and _cool != "" else 2.0
    except Exception:
        kf_watchdog_cooldown_s = 2.0

    log_camera_info = _env_bool(env, "NAPARI_CUDA_LOG_CAMERA_INFO", False)
    log_camera_debug = _env_bool(env, "NAPARI_CUDA_LOG_CAMERA_DEBUG", False)
    log_state_traces = _env_bool(env, "NAPARI_CUDA_LOG_STATE_TRACES", False)
    log_volume_info = _env_bool(env, "NAPARI_CUDA_LOG_VOLUME_INFO", False)
    log_dims_info = _env_bool(env, "NAPARI_CUDA_LOG_DIMS_INFO", False)
    debug = _env_bool(env, "NAPARI_CUDA_DEBUG", False)
    log_sends_env = _env_bool(env, "NAPARI_CUDA_LOG_SENDS", False)

    metrics_port = _env_int(env, "NAPARI_CUDA_METRICS_PORT", 8083)
    metrics_refresh_ms = _env_int(env, "NAPARI_CUDA_METRICS_REFRESH_MS", 1000)

    policy_event_path = _env_str(env, "NAPARI_CUDA_POLICY_EVENT_PATH", "tmp/policy_events.jsonl") or "tmp/policy_events.jsonl"

    return ServerCtx(
        cfg=cfg,
        frame_queue=frame_queue,
        dump_bitstream=dump_bitstream,
        dump_dir=dump_dir,
        kf_watchdog_cooldown_s=kf_watchdog_cooldown_s,
        log_camera_info=log_camera_info,
        log_camera_debug=log_camera_debug,
        log_state_traces=log_state_traces,
        log_volume_info=log_volume_info,
        log_dims_info=log_dims_info,
        debug=debug,
        log_sends_env=log_sends_env,
        metrics_port=metrics_port,
        metrics_refresh_ms=metrics_refresh_ms,
        policy_event_path=policy_event_path,
    )


__all__ = [
    "EncodeCfg",
    "ServerConfig",
    "ServerCtx",
    "load_server_config",
    "load_server_ctx",
]
