"""Configuration dataclasses shared across the server package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from napari_cuda.server.config.logging_policy import DebugPolicy, load_debug_policy


@dataclass(frozen=True)
class EncodeCfg:
    """Encoder parameters for the streaming pipeline."""

    fps: int = 60
    codec: str = "h264"  # "h264" | "hevc" | "av1"
    bitrate: int = 12_000_000
    keyint: int = 120


@dataclass(frozen=True)
class EncoderRuntime:
    """Advanced NVENC runtime tuning overrides."""

    input_format: str = "NV12"
    rc_mode: str = "cbr"
    preset: str = "P3"
    max_bitrate: Optional[int] = 20_000_000
    lookahead: int = 0
    aq: int = 1
    temporalaq: int = 1
    enable_non_ref_p: bool = False
    bframes: int = 0
    idr_period: int = 600


@dataclass(frozen=True)
class BitstreamRuntime:
    """Bitstream packer toggles resolved from environment."""

    build_cython: bool = True
    disable_fast_pack: bool = False
    allow_py_fallback: bool = False


@dataclass(frozen=True)
class ServerConfig:
    """Top-level server configuration values."""

    host: str = "0.0.0.0"
    state_port: int = 8081
    pixel_port: int = 8082
    zarr_path: Optional[str] = None
    zarr_level: Optional[str] = None
    zarr_axes: Optional[str] = None
    zarr_z: Optional[int] = None
    width: int = 1920
    height: int = 1080
    use_volume: bool = False
    ndisplay: int = 2
    max_slice_bytes: int = 0
    max_volume_bytes: int = 0
    max_volume_voxels: int = 0
    animate: bool = False
    animate_dps: float = 30.0
    frame_queue: int = 1
    encode: EncodeCfg = field(default_factory=EncodeCfg)
    idr_on_reset: bool = True
    packer: str = "cython"


@dataclass(frozen=True)
class LevelPolicySettings:
    """Level selection policy configuration."""

    threshold_in: float = 1.05
    threshold_out: float = 1.35
    hysteresis: float = 0.0
    fine_threshold: float = 1.05
    cooldown_ms: float = 150.0
    log_policy_eval: bool = False
    sticky_contrast: bool = True
    oversampling_thresholds: Optional[Mapping[int, float]] = None
    oversampling_hysteresis: float = 0.1


@dataclass(frozen=True)
class ServerCtx:
    """Resolved server runtime context shared across subsystems."""

    cfg: ServerConfig
    frame_queue: int = 1
    dump_bitstream: int = 0
    dump_dir: str = "benchmarks/bitstreams"
    kf_watchdog_cooldown_s: float = 2.0
    debug_policy: DebugPolicy = field(default_factory=lambda: load_debug_policy({}))
    encoder_runtime: EncoderRuntime = field(default_factory=EncoderRuntime)
    bitstream: BitstreamRuntime = field(default_factory=BitstreamRuntime)
    metrics_port: int = 8083
    metrics_refresh_ms: int = 1000
    policy: LevelPolicySettings = field(default_factory=LevelPolicySettings)
