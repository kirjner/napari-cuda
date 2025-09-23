"""Environment-derived configuration for the client stream loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from napari_cuda.utils.env import env_bool, env_float, env_str


def _optional_int(name: str) -> Optional[int]:
    raw = env_str(name, None)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _optional_float(name: str) -> Optional[float]:
    raw = env_str(name, None)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except Exception:
        return None


@dataclass(frozen=True)
class ClientLoopConfig:
    """Resolved environment toggles for ``ClientStreamLoop``."""

    warmup_ms_override: Optional[float]
    warmup_window_s: float
    warmup_margin_ms: float
    warmup_max_ms: float
    server_bias_s: float
    wake_fudge_ms: float
    metrics_enabled: bool
    metrics_interval_ms: int
    use_display_loop: bool
    vt_stats_mode: str
    watchdog_ms: int
    evloop_stall_ms: int
    evloop_sample_ms: int
    dims_z: Optional[int]
    dims_z_min: Optional[int]
    dims_z_max: Optional[int]
    wheel_step: int
    dims_rate_hz: float
    camera_rate_hz: float
    settings_rate_hz: float
    orbit_deg_per_px_x: float
    orbit_deg_per_px_y: float
    input_max_rate_hz: float
    resize_debounce_ms: int
    input_log: bool
    smoke_source: str
    smoke_width: int
    smoke_height: int
    smoke_preset: str
    smoke_fps: float
    smoke_mode: str
    smoke_preencode: bool
    smoke_pre_frames: int
    smoke_pre_mb: int
    smoke_pre_path: Optional[str]
    zoom_base: float


def load_client_loop_config() -> ClientLoopConfig:
    """Resolve environment variables into a ``ClientLoopConfig`` instance."""

    warmup_ms_override = _optional_float("NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MS")
    warmup_window_s = float(env_float("NAPARI_CUDA_CLIENT_STARTUP_WARMUP_WINDOW_S", 0.75))
    warmup_margin_ms = float(env_float("NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MARGIN_MS", 2.0))
    warmup_max_ms = float(env_float("NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MAX_MS", 24.0))

    server_bias_s = float(env_float("NAPARI_CUDA_SERVER_TS_BIAS_MS", 0.0)) / 1000.0
    wake_fudge_ms = float(env_float("NAPARI_CUDA_WAKE_FUDGE_MS", 1.0))

    metrics_enabled = env_bool("NAPARI_CUDA_CLIENT_METRICS", False)
    metrics_interval_s = float(env_float("NAPARI_CUDA_CLIENT_METRICS_INTERVAL", 1.0))
    metrics_interval_ms = max(100, int(round(1000.0 * max(0.1, metrics_interval_s))))

    use_display_loop = env_bool("NAPARI_CUDA_USE_DISPLAY_LOOP", False)
    vt_stats_mode = (env_str("NAPARI_CUDA_VT_STATS", "") or "").lower()

    watchdog_ms = max(0, int(env_float("NAPARI_CUDA_CLIENT_DRAW_WATCHDOG_MS", 0.0)))
    evloop_stall_ms = max(0, int(env_float("NAPARI_CUDA_CLIENT_EVENTLOOP_STALL_MS", 0.0)))
    evloop_sample_ms = max(50, int(env_float("NAPARI_CUDA_CLIENT_EVENTLOOP_SAMPLE_MS", 100.0)))

    dims_z = _optional_int("NAPARI_CUDA_ZARR_Z")
    dims_z_min = _optional_int("NAPARI_CUDA_ZARR_Z_MIN")
    dims_z_max = _optional_int("NAPARI_CUDA_ZARR_Z_MAX")

    wheel_step_raw = _optional_int("NAPARI_CUDA_WHEEL_Z_STEP")
    wheel_step = wheel_step_raw if wheel_step_raw and wheel_step_raw > 0 else 1

    dims_rate_hz = float(env_float("NAPARI_CUDA_DIMS_SET_RATE", 60.0))
    camera_rate_hz = float(env_float("NAPARI_CUDA_CAMERA_SET_RATE", 60.0))
    settings_rate_hz = float(env_float("NAPARI_CUDA_SETTINGS_SET_RATE", 60.0))

    orbit_deg_per_px_x = float(env_float("NAPARI_CUDA_ORBIT_DEG_PER_PX_X", 0.2))
    orbit_deg_per_px_y = float(env_float("NAPARI_CUDA_ORBIT_DEG_PER_PX_Y", 0.2))

    input_max_rate_hz = float(env_float("NAPARI_CUDA_INPUT_MAX_RATE", 120.0))
    resize_debounce_ms = max(0, int(env_float("NAPARI_CUDA_RESIZE_DEBOUNCE_MS", 80.0)))
    input_log = env_bool("NAPARI_CUDA_INPUT_LOG", False)

    smoke_source = (env_str("NAPARI_CUDA_SMOKE_SOURCE", "vt") or "vt").lower()
    smoke_preset = (env_str("NAPARI_CUDA_SMOKE_PRESET", "") or "").strip().lower()
    smoke_mode = (env_str("NAPARI_CUDA_SMOKE_MODE", "checker") or "checker").lower()
    smoke_preencode = env_bool("NAPARI_CUDA_SMOKE_PREENCODE", False)
    smoke_pre_frames = int(env_float("NAPARI_CUDA_SMOKE_PRE_FRAMES", 180.0))
    smoke_pre_mb = max(0, int(env_float("NAPARI_CUDA_SMOKE_PRE_MB", 0.0)))
    smoke_pre_path = env_str("NAPARI_CUDA_SMOKE_PRE_PATH", None)

    smoke_width = int(env_float("NAPARI_CUDA_SMOKE_W", 1280.0))
    smoke_height = int(env_float("NAPARI_CUDA_SMOKE_H", 720.0))
    smoke_fps = float(env_float("NAPARI_CUDA_SMOKE_FPS", 60.0))
    zoom_base = float(env_float("NAPARI_CUDA_ZOOM_BASE", 1.1))

    return ClientLoopConfig(
        warmup_ms_override=warmup_ms_override,
        warmup_window_s=warmup_window_s,
        warmup_margin_ms=warmup_margin_ms,
        warmup_max_ms=warmup_max_ms,
        server_bias_s=server_bias_s,
        wake_fudge_ms=wake_fudge_ms,
        metrics_enabled=metrics_enabled,
        metrics_interval_ms=metrics_interval_ms,
        use_display_loop=use_display_loop,
        vt_stats_mode=vt_stats_mode,
        watchdog_ms=watchdog_ms,
        evloop_stall_ms=evloop_stall_ms,
        evloop_sample_ms=evloop_sample_ms,
        dims_z=dims_z,
        dims_z_min=dims_z_min,
        dims_z_max=dims_z_max,
        wheel_step=wheel_step,
        dims_rate_hz=dims_rate_hz,
        camera_rate_hz=camera_rate_hz,
        settings_rate_hz=settings_rate_hz,
        orbit_deg_per_px_x=orbit_deg_per_px_x,
        orbit_deg_per_px_y=orbit_deg_per_px_y,
        input_max_rate_hz=input_max_rate_hz,
        resize_debounce_ms=resize_debounce_ms,
        input_log=input_log,
        smoke_source=smoke_source,
        smoke_width=smoke_width,
        smoke_height=smoke_height,
        smoke_preset=smoke_preset,
        smoke_fps=smoke_fps,
        smoke_mode=smoke_mode,
        smoke_preencode=smoke_preencode,
        smoke_pre_frames=smoke_pre_frames,
        smoke_pre_mb=smoke_pre_mb,
        smoke_pre_path=smoke_pre_path,
        zoom_base=zoom_base,
    )
