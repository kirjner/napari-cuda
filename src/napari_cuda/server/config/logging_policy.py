from __future__ import annotations

"""Central debug/logging policy plumbing for the napari-cuda server."""

import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Optional

from napari.layers.image._image_constants import (
    Interpolation as NapariInterpolation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoggingToggles:
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
    log_keyframes: bool = False
    log_encoder_settings: bool = True
    log_nals: bool = False
    log_sps: bool = False


@dataclass(frozen=True)
class DumpControls:
    enabled: bool = False
    frames_budget: int = 0
    output_dir: str = "logs/napari_cuda_frames"
    flip_cuda_for_view: bool = False
    raw_budget: int = 0


@dataclass(frozen=True)
class WorkerDebug:
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
    force_tight_pitch: bool = False
    layer_interpolation: str = NapariInterpolation.LINEAR.value


@dataclass(frozen=True)
class DebugPolicy:
    enabled: bool
    logging: LoggingToggles
    encoder: EncoderLogging
    dumps: DumpControls
    worker: WorkerDebug


_LOG_FLAG_MAP: dict[str, Iterable[str]] = {
    "camera": ("log_camera_info", "log_camera_debug"),
    "state": ("log_state_traces",),
    "volume": ("log_volume_info",),
    "dims": ("log_dims_info",),
    "policy": ("log_policy_eval",),
    "sends": ("log_sends_env",),
    "layer": ("log_layer_debug",),
    "roi": ("log_roi_anchor",),
}

_ENCODER_FLAG_MAP: dict[str, Iterable[str]] = {
    "encoder-keyframes": ("log_keyframes",),
    "encoder-nals": ("log_nals",),
    "encoder-sps": ("log_sps",),
    "encoder-settings": ("log_encoder_settings",),
}

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in _TRUTHY:
            return True
        if val in _FALSY:
            return False
    return default


def _coerce_int(value: object, default: int = 0) -> int:
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


def _coerce_float(value: object, default: float = 0.0) -> float:
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


def _split_flags(raw: object) -> set[str]:
    result: set[str] = set()
    items: Iterable[object]
    if raw is None:
        return result
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, Iterable):
        items = raw
    else:
        return result
    for item in items:
        token = str(item).strip().lower()
        if token:
            result.add(token)
    return result


def _load_debug_config(env: Mapping[str, str]) -> tuple[bool, dict[str, object]]:
    raw = env.get("NAPARI_CUDA_DEBUG")
    if raw is None:
        return False, {}
    raw_str = raw.strip()
    if raw_str.lower() in _FALSY:
        return False, {}
    if raw_str.lower() in _TRUTHY:
        return True, {}
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, dict):
            enabled = _coerce_bool(parsed.get("enabled", True), True)
            return enabled, parsed
        if isinstance(parsed, (list, tuple)):
            return True, {"flags": parsed}
    except Exception:
        logger.debug("Failed to parse NAPARI_CUDA_DEBUG JSON; treating as flag list", exc_info=True)
    return True, {"flags": raw_str}


def load_debug_policy(env: Optional[Mapping[str, str]] = None) -> DebugPolicy:
    env = env or os.environ
    enabled, cfg = _load_debug_config(env)

    flag_source = cfg.get("flags") if isinstance(cfg, dict) else []
    flags = _split_flags(flag_source)

    log_kwargs = {field: False for field in LoggingToggles.__annotations__.keys()}
    for flag, attrs in _LOG_FLAG_MAP.items():
        if flag in flags:
            for attr in attrs:
                log_kwargs[attr] = True

    enc_kwargs = {field: getattr(EncoderLogging, field, None) for field in EncoderLogging.__annotations__.keys()}
    enc_defaults = EncoderLogging()
    enc_kwargs = {name: getattr(enc_defaults, name) for name in EncoderLogging.__annotations__.keys()}
    for flag, attrs in _ENCODER_FLAG_MAP.items():
        if flag in flags:
            for attr in attrs:
                enc_kwargs[attr] = True

    dumps_cfg = cfg.get("dumps") if isinstance(cfg, dict) else {}
    default_frames = 3
    if isinstance(dumps_cfg, dict):
        frames = _coerce_int(dumps_cfg.get("frames", default_frames), default_frames)
        raw_budget = _coerce_int(dumps_cfg.get("raw_frames", 0), 0)
        output_dir = str(dumps_cfg.get("dir", "logs/napari_cuda_frames") or "logs/napari_cuda_frames")
        flip = _coerce_bool(dumps_cfg.get("flip"), False)
    else:
        frames = default_frames
        raw_budget = 0
        output_dir = "logs/napari_cuda_frames"
        flip = False
    dumps = DumpControls(
        enabled=enabled and frames > 0,
        frames_budget=max(0, frames),
        output_dir=output_dir,
        flip_cuda_for_view=flip,
        raw_budget=max(0, raw_budget),
    )

    worker_cfg = cfg.get("worker") if isinstance(cfg, dict) else {}
    worker_defaults = WorkerDebug()
    worker_kwargs = worker_defaults.__dict__.copy()
    if isinstance(worker_cfg, dict):
        if "lock_level" in worker_cfg:
            worker_kwargs["lock_level"] = int(worker_cfg["lock_level"]) if worker_cfg["lock_level"] is not None else None
        if "roi_edge_threshold" in worker_cfg:
            worker_kwargs["roi_edge_threshold"] = max(0, _coerce_int(worker_cfg["roi_edge_threshold"], worker_defaults.roi_edge_threshold))
        if "force_tight_pitch" in worker_cfg:
            worker_kwargs["force_tight_pitch"] = _coerce_bool(worker_cfg["force_tight_pitch"], worker_defaults.force_tight_pitch)
        if "layer_interpolation" in worker_cfg:
            raw = worker_cfg["layer_interpolation"]
            worker_kwargs["layer_interpolation"] = NapariInterpolation(str(raw).strip().lower()).value
        if "orbit_el_min" in worker_cfg:
            worker_kwargs["orbit_el_min"] = _coerce_float(worker_cfg["orbit_el_min"], worker_defaults.orbit_el_min)
        if "orbit_el_max" in worker_cfg:
            worker_kwargs["orbit_el_max"] = _coerce_float(worker_cfg["orbit_el_max"], worker_defaults.orbit_el_max)

    logging_toggles = LoggingToggles(**log_kwargs)
    encoder_logging = EncoderLogging(**enc_kwargs)
    worker_debug = WorkerDebug(**worker_kwargs)

    return DebugPolicy(
        enabled=enabled,
        logging=logging_toggles,
        encoder=encoder_logging,
        dumps=dumps,
        worker=worker_debug,
    )


__all__ = [
    "DebugPolicy",
    "DumpControls",
    "EncoderLogging",
    "LoggingToggles",
    "WorkerDebug",
    "load_debug_policy",
]
