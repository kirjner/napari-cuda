from __future__ import annotations

"""Centralised debug/logging policy for the napari-cuda server stack.

This module materialises immutable dataclasses that capture every logging and
debug toggle currently consumed by the server, worker, and rendering layers.
All env var parsing happens here so the rest of the codebase can depend on a
structured policy rather than scattered `os.getenv`/`env_bool` calls.
"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional


def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    try:
        return bool(int(raw))
    except Exception:
        return default


def _env_int(env: Mapping[str, str], name: str, default: int = 0) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw, 10)
    except Exception:
        return default


def _env_float(env: Mapping[str, str], name: str, default: float = 0.0) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_str(env: Mapping[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    raw = env.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


@dataclass(frozen=True)
class LoggingToggles:
    """Worker/server logging flags."""

    log_camera_info: bool = False
    log_camera_debug: bool = False
    log_state_traces: bool = False
    log_volume_info: bool = False
    log_dims_info: bool = False
    log_policy_eval: bool = False
    log_sends_env: bool = False
    log_layer_debug: bool = False
    log_roi_anchor: bool = False


@dataclass(frozen=True)
class EncoderLogging:
    """Encoder/bitstream specific logging flags."""

    log_keyframes: bool = False
    log_encoder_settings: bool = True
    log_nals: bool = False
    log_sps: bool = False


@dataclass(frozen=True)
class DumpControls:
    """Frame dump / diagnostics configuration."""

    enabled: bool = False
    frames_budget: int = 0
    output_dir: str = "logs/napari_cuda_frames"
    flip_cuda_for_view: bool = False
    raw_budget: int = 0


@dataclass(frozen=True)
class WorkerDebug:
    """Worker-only debug knobs."""

    debug_pan: bool = False
    debug_orbit: bool = False
    debug_reset: bool = False
    debug_zoom_drift: bool = False
    debug_bg_overlay: bool = False
    debug_overlay: bool = False
    orbit_el_min: float = -85.0
    orbit_el_max: float = 85.0
    lock_level: Optional[int] = None
    roi_edge_threshold: int = 4
    roi_align_chunks: bool = False
    roi_ensure_contains_viewport: bool = True
    auto_reset_on_black: bool = True
    force_tight_pitch: bool = False
    layer_interpolation: str = "bilinear"


@dataclass(frozen=True)
class DebugPolicy:
    """Composite debug/logging policy for the server runtime."""

    enabled: bool
    logging: LoggingToggles
    encoder: EncoderLogging
    dumps: DumpControls
    worker: WorkerDebug


def load_debug_policy(env: Optional[Mapping[str, str]] = None) -> DebugPolicy:
    """Read debug/logging flags from the provided environment mapping."""

    if env is None:
        env = os.environ

    # General logging
    logging = LoggingToggles(
        log_camera_info=_env_bool(env, "NAPARI_CUDA_LOG_CAMERA_INFO", False),
        log_camera_debug=_env_bool(env, "NAPARI_CUDA_LOG_CAMERA_DEBUG", False),
        log_state_traces=_env_bool(env, "NAPARI_CUDA_LOG_STATE_TRACES", False),
        log_volume_info=_env_bool(env, "NAPARI_CUDA_LOG_VOLUME_INFO", False),
        log_dims_info=_env_bool(env, "NAPARI_CUDA_LOG_DIMS_INFO", False),
        log_policy_eval=_env_bool(env, "NAPARI_CUDA_LOG_POLICY_EVAL", False),
        log_sends_env=_env_bool(env, "NAPARI_CUDA_LOG_SENDS", False),
        log_layer_debug=_env_bool(env, "NAPARI_CUDA_LAYER_DEBUG", False),
        log_roi_anchor=_env_bool(env, "NAPARI_CUDA_LOG_ROI_ANCHOR", False),
    )

    encoder = EncoderLogging(
        log_keyframes=_env_bool(env, "NAPARI_CUDA_LOG_KEYFRAMES", False),
        log_encoder_settings=_env_bool(env, "NAPARI_CUDA_LOG_ENCODER_SETTINGS", True),
        log_nals=_env_bool(env, "NAPARI_CUDA_LOG_NALS", False),
        log_sps=_env_bool(env, "NAPARI_CUDA_LOG_SPS", False),
    )

    # Dump controls piggy-back on the master debug switch by default
    debug_enabled = _env_bool(env, "NAPARI_CUDA_DEBUG", False)
    dump_frames = _env_int(env, "NAPARI_CUDA_DEBUG_FRAMES", 3)
    dump_dir = _env_str(env, "NAPARI_CUDA_DUMP_DIR", "logs/napari_cuda_frames") or "logs/napari_cuda_frames"
    dumps = DumpControls(
        enabled=debug_enabled and dump_frames > 0,
        frames_budget=max(0, dump_frames),
        output_dir=dump_dir,
        flip_cuda_for_view=_env_bool(env, "NAPARI_CUDA_DEBUG_FLIP_CUDA", False),
        raw_budget=max(0, _env_int(env, "NAPARI_CUDA_DUMP_RAW", 0)),
    )

    orbit_el_min = _env_float(env, "NAPARI_CUDA_ORBIT_ELEV_MIN", -85.0)
    orbit_el_max = _env_float(env, "NAPARI_CUDA_ORBIT_ELEV_MAX", 85.0)
    lock_level_raw = _env_str(env, "NAPARI_CUDA_LOCK_LEVEL")
    try:
        lock_level = int(lock_level_raw) if lock_level_raw not in (None, "") else None
    except Exception:
        lock_level = None

    worker = WorkerDebug(
        debug_pan=_env_bool(env, "NAPARI_CUDA_DEBUG_PAN", False),
        debug_orbit=_env_bool(env, "NAPARI_CUDA_DEBUG_ORBIT", False),
        debug_reset=_env_bool(env, "NAPARI_CUDA_DEBUG_RESET", False),
        debug_zoom_drift=_env_bool(env, "NAPARI_CUDA_DEBUG_ZOOM_DRIFT", False),
        debug_bg_overlay=_env_bool(env, "NAPARI_CUDA_DEBUG_BG", False),
        debug_overlay=_env_bool(env, "NAPARI_CUDA_DEBUG_OVERLAY", False),
        orbit_el_min=float(orbit_el_min),
        orbit_el_max=float(orbit_el_max),
        lock_level=lock_level,
        roi_edge_threshold=_env_int(env, "NAPARI_CUDA_ROI_EDGE_THRESHOLD", 4),
        roi_align_chunks=_env_bool(env, "NAPARI_CUDA_ROI_ALIGN_CHUNKS", False),
        roi_ensure_contains_viewport=_env_bool(env, "NAPARI_CUDA_ROI_ENSURE_CONTAINS_VIEWPORT", True),
        auto_reset_on_black=_env_bool(env, "NAPARI_CUDA_AUTO_RESET_ON_BLACK", True),
        force_tight_pitch=_env_bool(env, "NAPARI_CUDA_FORCE_TIGHT_PITCH", False),
        layer_interpolation=(
            (_env_str(env, "NAPARI_CUDA_INTERP", "bilinear") or "bilinear").strip().lower()
        ),
    )

    return DebugPolicy(
        enabled=debug_enabled,
        logging=logging,
        encoder=encoder,
        dumps=dumps,
        worker=worker,
    )


__all__ = [
    "DebugPolicy",
    "DumpControls",
    "EncoderLogging",
    "LoggingToggles",
    "WorkerDebug",
    "load_debug_policy",
]
