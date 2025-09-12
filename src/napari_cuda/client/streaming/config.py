from __future__ import annotations

"""
Tiny helpers for streaming configuration and parsing helpers.

Phase 0 of the scheduler refactor introduces a small, runtime `ClientConfig`
object that centralizes a few high‑level knobs and feature flags. This file
keeps behavior unchanged — we only read existing envs and expose a structured
config for downstream components to consume gradually.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
import os


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v not in (None, "") else int(default)
    except Exception:
        return int(default)


@dataclass
class ClientConfig:
    """Minimal client configuration used by the streaming components.

    Phase 0 keeps behavior unchanged; this serves as a stable place to hang
    future options while we deprecate scattered envs.
    """

    mode: str = "arrival"  # "arrival" | "server"
    base_latency_ms: float = 0.0
    buffer_limit: int = 3
    draw_fps: float = 60.0

    # Feature flags (disabled by default until later phases)
    unified_scheduler: bool = False
    draw_coalesce: bool = False
    next_due_wake: bool = False

    # Preview guards (kept for clarity; identical behavior to today)
    server_preview_guard_ms: float = 0.0
    arrival_preview_guard_ms: float = 16.0

    # Compatibility helper to ease logging/inspection
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_env(default_latency_ms: float = 0.0, default_buffer_limit: int = 3, default_draw_fps: float = 60.0) -> "ClientConfig":
        # Timestamp mode
        mode = (_env_str("NAPARI_CUDA_CLIENT_VT_TS_MODE", "arrival") or "arrival").lower()
        # Latency: prefer explicit VT latency env; else provided defaults
        base_latency_ms = _env_float("NAPARI_CUDA_CLIENT_VT_LATENCY_MS", default_latency_ms)
        # Buffer limit: mirror existing env; coordinator/Presenter still own the logic
        buffer_limit = _env_int("NAPARI_CUDA_CLIENT_VT_BUFFER", default_buffer_limit)
        # Draw FPS: prefer client display env; else fallback to smoke fps
        draw_fps = _env_float("NAPARI_CUDA_CLIENT_DISPLAY_FPS", default_draw_fps)
        if draw_fps == default_draw_fps:
            draw_fps = _env_float("NAPARI_CUDA_SMOKE_FPS", draw_fps)

        # Feature flags (default off)
        unified = _env_bool("NAPARI_CUDA_UNIFIED_SCHEDULER", False)
        coalesce = _env_bool("NAPARI_CUDA_DRAW_COALESCE", False)
        wake = _env_bool("NAPARI_CUDA_NEXT_DUE_WAKE", False)

        # Preview guards (retain current behavior)
        serv_guard = _env_float("NAPARI_CUDA_SERVER_PREVIEW_GUARD_MS", 0.0)
        arr_guard = _env_float("NAPARI_CUDA_ARRIVAL_PREVIEW_GUARD_MS", 16.0)

        return ClientConfig(
            mode=mode,
            base_latency_ms=base_latency_ms,
            buffer_limit=buffer_limit,
            draw_fps=draw_fps,
            unified_scheduler=unified,
            draw_coalesce=coalesce,
            next_due_wake=wake,
            server_preview_guard_ms=serv_guard,
            arrival_preview_guard_ms=arr_guard,
        )


def extract_video_config(data: Dict[str, Any]) -> Tuple[int, int, float, str, Optional[str]]:
    """Return (width, height, fps, stream_format, avcc_b64_or_None).

    - stream_format is 'avcc' or 'annexb'
    - Missing/invalid fields fall back to safe defaults
    """
    try:
        width = int(data.get('width') or 0)
    except Exception:
        width = 0
    try:
        height = int(data.get('height') or 0)
    except Exception:
        height = 0
    try:
        fps = float(data.get('fps') or 0.0)
    except Exception:
        fps = 0.0
    fmt = (str(data.get('format') or '')).lower() or 'avcc'
    stream_format = 'annexb' if fmt.startswith('annex') else 'avcc'
    avcc_b64 = data.get('data') if isinstance(data.get('data'), str) else None
    return width, height, fps, stream_format, avcc_b64


def nal_length_size_from_avcc(avcc: bytes) -> int:
    """Return NAL length size from avcC (1..4). Defaults to 4 on error."""
    try:
        if len(avcc) >= 5:
            n = int((avcc[4] & 0x03) + 1)
            if n in (1, 2, 3, 4):
                return n
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).debug("nal_length_size_from_avcc failed", exc_info=True)
    return 4
