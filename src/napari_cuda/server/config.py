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

from dataclasses import dataclass, field
from typing import Mapping, Optional
import json
import logging
import os

from napari_cuda.server.logging_policy import DebugPolicy, load_debug_policy


logger = logging.getLogger(__name__)


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


def _cfg_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        if val in {"0", "false", "no", "off", ""}:
            return False
    return bool(default)


def _cfg_int(value: object, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return int(default)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except Exception:
            return int(default)
    return int(default)


def _cfg_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return float(default)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return float(default)
    return float(default)


def _cfg_str(value: object, default: str) -> str:
    if value is None:
        return default
    return str(value).strip() or default


def _cfg_optional_int(value: object, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return default
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return default
        try:
            return int(stripped)
        except Exception:
            return default
    return default


def _load_json_config(env: Mapping[str, str], name: str) -> dict[str, object]:
    raw = env.get(name)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        logger.warning("Failed to parse %s; ignoring", name, exc_info=True)
        return {}
    if isinstance(data, dict):
        return data
    logger.warning("%s must be a JSON object; ignoring", name)
    return {}


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
    max_slice_bytes: int = 0
    max_volume_bytes: int = 0
    max_volume_voxels: int = 0

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


@dataclass(frozen=True)
class EncoderRuntime:
    """Advanced NVENC runtime tuning overrides.

    These values capture the various environment-based knobs that historically
    lived inside `rendering.encoder`. They are now parsed once during context
    construction so the encoder no longer touches `os.environ`.
    """

    input_format: str = "YUV444"
    rc_mode: str = "cbr"
    preset: str = "P3"
    max_bitrate: Optional[int] = None
    lookahead: int = 0
    aq: int = 0
    temporalaq: int = 0
    enable_non_ref_p: bool = False
    bframes: int = 0
    idr_period: int = 600


@dataclass(frozen=True)
class BitstreamRuntime:
    """Bitstream packer toggles resolved from environment."""

    build_cython: bool = True
    disable_fast_pack: bool = False
    allow_py_fallback: bool = False


def load_server_config(env: Optional[Mapping[str, str]] = None) -> ServerConfig:
    """Load server configuration from environment (no side effects).

    Environment keys consulted (subset; see plan for reductions):
    - NAPARI_CUDA_HOST, NAPARI_CUDA_STATE_PORT, NAPARI_CUDA_PIXEL_PORT
    - NAPARI_CUDA_ZARR_PATH, NAPARI_CUDA_ZARR_LEVEL, NAPARI_CUDA_ZARR_AXES, NAPARI_CUDA_ZARR_Z
    - NAPARI_CUDA_WIDTH, NAPARI_CUDA_HEIGHT, NAPARI_CUDA_USE_VOLUME, NAPARI_CUDA_NDISPLAY
    - NAPARI_CUDA_ANIMATE, NAPARI_CUDA_TURNTABLE_DPS
    - NAPARI_CUDA_FRAME_QUEUE
    - NAPARI_CUDA_PROFILE (latency|quality)
    - NAPARI_CUDA_ENCODER_CONFIG (JSON encode/runtime/bitstream overrides)
    - NAPARI_CUDA_POLICY_CONFIG (JSON policy thresholds and hysteresis)
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
    encoder_cfg = _load_json_config(env, "NAPARI_CUDA_ENCODER_CONFIG")
    encode_section = encoder_cfg.get("encode") if isinstance(encoder_cfg, dict) else None
    if not isinstance(encode_section, dict):
        encode_section = encoder_cfg
    fps = _cfg_int(encode_section.get("fps") if encode_section else None, enc.fps)
    bitrate = _cfg_int(encode_section.get("bitrate") if encode_section else None, enc.bitrate)
    keyint = _cfg_int(encode_section.get("keyint") if encode_section else None, enc.keyint)
    codec = _cfg_str(encode_section.get("codec") if encode_section else enc.codec, enc.codec)
    enc = EncodeCfg(fps=fps, codec=codec.lower(), bitrate=bitrate, keyint=keyint)

    max_slice_bytes = max(0, _env_int(env, "NAPARI_CUDA_MAX_SLICE_BYTES", 0))
    max_volume_bytes = max(0, _env_int(env, "NAPARI_CUDA_MAX_VOLUME_BYTES", 0))
    max_volume_voxels = max(0, _env_int(env, "NAPARI_CUDA_MAX_VOLUME_VOXELS", 0))

    idr_on_reset = _cfg_bool(encoder_cfg.get("idr_on_reset") if encoder_cfg else None, True)
    packer = _cfg_str(encoder_cfg.get("packer") if encoder_cfg else "cython", "cython").lower()

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
        max_slice_bytes=max_slice_bytes,
        max_volume_bytes=max_volume_bytes,
        max_volume_voxels=max_volume_voxels,
        animate=animate,
        animate_dps=animate_dps,
        frame_queue=frame_queue,
        profile=profile,
        encode=enc,
        idr_on_reset=idr_on_reset,
        packer=packer,
    )


@dataclass(frozen=True)
class LevelPolicySettings:
    """Level selection policy configuration resolved from environment."""

    threshold_in: float = 1.05
    threshold_out: float = 1.35
    hysteresis: float = 0.0
    fine_threshold: float = 1.05
    cooldown_ms: float = 150.0
    log_policy_eval: bool = False
    preserve_view_on_switch: bool = True
    sticky_contrast: bool = True
    oversampling_thresholds: Optional[Mapping[int, float]] = None
    oversampling_hysteresis: float = 0.1


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

    # Debug/logging policy
    debug_policy: DebugPolicy = field(default_factory=lambda: load_debug_policy({}))

    # Encoder runtime overrides (NVENC tuning)
    encoder_runtime: EncoderRuntime = field(default_factory=EncoderRuntime)

    # Bitstream/packer toggles
    bitstream: BitstreamRuntime = field(default_factory=BitstreamRuntime)

    # Metrics UI
    metrics_port: int = 8083
    metrics_refresh_ms: int = 1000

    # Policy event path
    policy_event_path: str = "tmp/policy_events.jsonl"

    # Level policy settings
    policy: LevelPolicySettings = LevelPolicySettings()


def load_server_ctx(env: Optional[Mapping[str, str]] = None) -> ServerCtx:
    """Build a `ServerCtx` by reading environment once.

    Note: This does not mutate the process environment and is side-effect free.
    """
    env = env or os.environ
    cfg = load_server_config(env)
    encoder_cfg = _load_json_config(env, "NAPARI_CUDA_ENCODER_CONFIG")

    frame_queue = max(1, int(getattr(cfg, "frame_queue", 1)))
    dump_bitstream = max(0, _cfg_int(encoder_cfg.get("dump_bitstream") if encoder_cfg else None, 0))
    dump_dir = _cfg_str(encoder_cfg.get("dump_dir") if encoder_cfg else "benchmarks/bitstreams", "benchmarks/bitstreams")

    _cool = _env_str(env, "NAPARI_CUDA_KF_WATCHDOG_COOLDOWN")
    try:
        kf_watchdog_cooldown_s = float(_cool) if _cool is not None and _cool != "" else 2.0
    except Exception:
        kf_watchdog_cooldown_s = 2.0

    debug_policy = load_debug_policy(env)

    # Encoder runtime overrides
    runtime_cfg = encoder_cfg.get("runtime") if isinstance(encoder_cfg, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    input_fmt = _cfg_str(runtime_cfg.get("input_format"), "YUV444").upper()
    rc_mode = _cfg_str(runtime_cfg.get("rc_mode"), "cbr").lower()
    preset = _cfg_str(runtime_cfg.get("preset"), "P3")
    max_bitrate_val = _cfg_optional_int(runtime_cfg.get("max_bitrate"), None)
    lookahead = max(0, _cfg_int(runtime_cfg.get("lookahead"), 0))
    aq = max(0, _cfg_int(runtime_cfg.get("aq"), 0))
    temporalaq = max(0, _cfg_int(runtime_cfg.get("temporalaq"), 0))
    nonrefp = _cfg_bool(runtime_cfg.get("non_ref_p"), False)
    bframes = max(0, _cfg_int(runtime_cfg.get("bframes"), 0))
    idr_period = max(1, _cfg_int(runtime_cfg.get("idr_period"), 600))
    encoder_runtime = EncoderRuntime(
        input_format=input_fmt or "YUV444",
        rc_mode=rc_mode or "cbr",
        preset=preset or "P3",
        max_bitrate=max_bitrate_val if (max_bitrate_val or 0) > 0 else None,
        lookahead=lookahead,
        aq=aq,
        temporalaq=temporalaq,
        enable_non_ref_p=bool(nonrefp),
        bframes=bframes,
        idr_period=int(idr_period),
    )

    bitstream_cfg = encoder_cfg.get("bitstream") if isinstance(encoder_cfg, dict) else {}
    if not isinstance(bitstream_cfg, dict):
        bitstream_cfg = {}
    bitstream_runtime = BitstreamRuntime(
        build_cython=_cfg_bool(bitstream_cfg.get("build_cython"), True),
        disable_fast_pack=_cfg_bool(bitstream_cfg.get("disable_fast_pack"), False),
        allow_py_fallback=_cfg_bool(bitstream_cfg.get("allow_py_fallback"), False),
    )

    metrics_port = _env_int(env, "NAPARI_CUDA_METRICS_PORT", 8083)
    metrics_refresh_ms = _env_int(env, "NAPARI_CUDA_METRICS_REFRESH_MS", 1000)

    policy_event_path = _env_str(env, "NAPARI_CUDA_POLICY_EVENT_PATH", "tmp/policy_events.jsonl") or "tmp/policy_events.jsonl"

    policy_cfg = _load_json_config(env, "NAPARI_CUDA_POLICY_CONFIG")
    policy_threshold_in = _cfg_float(policy_cfg.get("threshold_in") if policy_cfg else None, 1.05)
    policy_threshold_out = _cfg_float(policy_cfg.get("threshold_out") if policy_cfg else None, 1.35)
    policy_hysteresis = _cfg_float(policy_cfg.get("hysteresis") if policy_cfg else None, 0.0)
    fine_default = max(policy_threshold_in, 1.05)
    policy_fine_threshold = _cfg_float(policy_cfg.get("fine_threshold") if policy_cfg else None, fine_default)
    policy_fine_threshold = max(policy_threshold_in, policy_fine_threshold)
    policy_cooldown_ms = _cfg_float(policy_cfg.get("cooldown_ms") if policy_cfg else None, 150.0)
    policy_log_eval = debug_policy.logging.log_policy_eval
    policy_preserve_view = _cfg_bool(policy_cfg.get("preserve_view_on_switch") if policy_cfg else None, True)
    policy_sticky_contrast = _cfg_bool(policy_cfg.get("sticky_contrast") if policy_cfg else None, True)

    overs_cfg = policy_cfg.get("oversampling") if isinstance(policy_cfg, dict) else {}
    if not isinstance(overs_cfg, dict):
        overs_cfg = {}
    oversampling_thresholds_raw = overs_cfg.get("thresholds") if isinstance(overs_cfg, dict) else None
    oversampling_thresholds = None
    if isinstance(oversampling_thresholds_raw, Mapping):
        oversampling_thresholds = {
            int(k): float(v)
            for k, v in oversampling_thresholds_raw.items()
            if v is not None
        }
    oversampling_hysteresis = _cfg_float(overs_cfg.get("hysteresis") if overs_cfg else None, 0.1)

    policy_settings = LevelPolicySettings(
        threshold_in=float(policy_threshold_in),
        threshold_out=float(policy_threshold_out),
        hysteresis=float(policy_hysteresis),
        fine_threshold=float(policy_fine_threshold),
        cooldown_ms=float(policy_cooldown_ms),
        log_policy_eval=bool(policy_log_eval),
        preserve_view_on_switch=bool(policy_preserve_view),
        sticky_contrast=bool(policy_sticky_contrast),
        oversampling_thresholds=oversampling_thresholds,
        oversampling_hysteresis=float(oversampling_hysteresis),
    )

    return ServerCtx(
        cfg=cfg,
        frame_queue=frame_queue,
        dump_bitstream=dump_bitstream,
        dump_dir=dump_dir,
        kf_watchdog_cooldown_s=kf_watchdog_cooldown_s,
        debug_policy=debug_policy,
        metrics_port=metrics_port,
        metrics_refresh_ms=metrics_refresh_ms,
        policy_event_path=policy_event_path,
        policy=policy_settings,
        encoder_runtime=encoder_runtime,
        bitstream=bitstream_runtime,
    )


__all__ = [
    "EncodeCfg",
    "LevelPolicySettings",
    "ServerConfig",
    "ServerCtx",
    "EncoderRuntime",
    "BitstreamRuntime",
    "load_server_config",
    "load_server_ctx",
]
