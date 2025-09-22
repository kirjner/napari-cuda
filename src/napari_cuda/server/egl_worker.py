"""
EGLRendererWorker - Headless VisPy renderer with EGL + CUDA interop + NVENC.

This worker owns an EGL OpenGL context and a CUDA context on a single thread.
It renders a minimal VisPy scene (Image/Volume), captures the composited frame
into an FBO-attached texture (GPU-only blit), maps it to CUDA, copies to a
linear CuPy buffer, and encodes via NVENC (PyNvVideoCodec).

Intended for server-side headless rendering and benchmarking.
"""

from __future__ import annotations

import os
import time
import logging

from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Mapping, Sequence, Tuple
import threading
import math

import numpy as np
try:
    import dask.array as da  # type: ignore
except Exception as _da_err:
    da = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning("dask.array not available; OME-Zarr features disabled: %s", _da_err)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

from vispy import scene  # type: ignore

from napari.components.viewer_model import ViewerModel
from napari._vispy.layers.image import VispyImageLayer

import pycuda.driver as cuda  # type: ignore
# Bitstream packing happens in the server layer
from .patterns import make_rgba_image
from .debug_tools import DebugConfig, DebugDumper
from .hw_limits import get_hw_limits
from napari_cuda.utils.env import env_bool
from .zarr_source import ZarrSceneSource
from .scene_types import SliceROI
from napari_cuda.server.rendering.adapter_scene import AdapterScene
from napari_cuda.server import camera_ops as camops
from . import policy as level_policy
from napari_cuda.server.lod import apply_level
from napari_cuda.server.config import ServerCtx
from napari_cuda.server.rendering.gl_capture import GLCapture
from napari_cuda.server.rendering.cuda_interop import CudaInterop
from napari_cuda.server.rendering.egl_context import EglContext
from napari_cuda.server.rendering.encoder import Encoder
from napari_cuda.server.rendering.frame_pipeline import FramePipeline
from napari_cuda.server.roi_applier import SliceUpdatePlanner
from napari_cuda.server.roi import (
    compute_viewport_roi,
    plane_scale_for_level,
    plane_wh_for_level,
)
from napari_cuda.server.lod import LevelPolicyConfig, LevelPolicyInputs, select_level
from napari_cuda.server.scene_state_applier import (
    SceneStateApplier,
    SceneStateApplyContext,
)
from napari_cuda.server.camera_controller import (
    CameraCommandOutcome,
    CameraDebugFlags,
    apply_camera_commands,
)
from napari_cuda.server.state_machine import (
    CameraCommand,
    SceneUpdateBundle,
    SceneStateCoordinator,
    ServerSceneState,
)

logger = logging.getLogger(__name__)

## Camera ops now live in napari_cuda.server.camera_ops as free functions.

class _LevelBudgetError(RuntimeError):
    """Raised when a multiscale level exceeds memory/voxel budgets."""
    pass


@dataclass
class FrameTimings:
    render_ms: float
    blit_gpu_ns: Optional[int]
    blit_cpu_ms: float
    map_ms: float
    copy_ms: float
    convert_ms: float
    encode_ms: float
    pack_ms: float
    total_ms: float
    packet_bytes: Optional[int]
    # Wall-clock timestamp (seconds) at frame capture start (server clock)
    capture_wall_ts: float = 0.0


class EGLRendererWorker:
    """Headless VisPy renderer using EGL with CUDA interop and NVENC."""

    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False, fps: int = 60,
                 volume_depth: int = 64, volume_dtype: str = "float32", volume_relative_step: Optional[float] = None,
                 animate: bool = False, animate_dps: float = 30.0,
                 zarr_path: Optional[str] = None, zarr_level: Optional[str] = None,
                 zarr_axes: Optional[str] = None, zarr_z: Optional[int] = None,
                 scene_refresh_cb: Optional[Callable[[], None]] = None,
                 policy_name: Optional[str] = None,
                 ctx: Optional[ServerCtx] = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.use_volume = bool(use_volume)
        self.fps = int(fps)
        self.volume_depth = int(volume_depth)
        self.volume_dtype = str(volume_dtype)
        self.volume_relative_step = volume_relative_step
        self._ctx: Optional[ServerCtx] = ctx
        # Optional simple turntable animation
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception as e:
            logger.debug("Invalid animate_dps=%r; using default 30.0: %s", animate_dps, e)
            self._animate_dps = 30.0
        self._anim_start = time.perf_counter()

        self.cuda_ctx: Optional[cuda.Context] = None

        self.canvas: Optional[scene.SceneCanvas] = None
        self.view = None
        self._visual = None
        self._egl = EglContext(self.width, self.height)

        self._viewer: Optional[ViewerModel] = None
        self._napari_layer = None
        self._scene_source: Optional[ZarrSceneSource] = None
        self._active_ms_level: int = 0
        self._level_downgraded: bool = False
        self._scene_refresh_cb = scene_refresh_cb
        self._zoom_accumulator: float = 0.0
        self._last_zoom_hint_ts: float = 0.0
        self._zoom_hint_hold_s: float = 0.35

        self._gl_capture = GLCapture(self.width, self.height)
        self._cuda = CudaInterop(self.width, self.height)
        self._frame_pipeline = FramePipeline(
            gl_capture=self._gl_capture,
            cuda=self._cuda,
            width=self.width,
            height=self.height,
            debug=None,
        )

        self._encoder: Optional[Encoder] = None

        # Track current data extents in world units (W,H)
        self._data_wh: tuple[int, int] = (int(width), int(height))
        # Gate pixel streaming until orientation/camera are settled and a non-black frame is observed
        self._orientation_ready: bool = False

        # Encoder access synchronization (reset vs encode across threads)
        self._enc_lock = threading.Lock()

        # Atomic state application
        self._state_lock = threading.Lock()
        self._scene_state_coordinator = SceneStateCoordinator()
        # Debug gating for ensure_scene_source logging
        self._last_ensure_log: Optional[tuple[int, Optional[str]]] = None
        self._last_ensure_log_ts: float = 0.0
        self._ensure_log_interval_s: float = 1.0
        self._render_tick_required: bool = False
        self._render_loop_started: bool = False
        # Priming retired: removed

        # Encoding / instrumentation state
        # Zoom drift debug flag (INFO-level logs on zoom_at)
        try:
            self._debug_zoom_drift = bool(int(os.getenv('NAPARI_CUDA_DEBUG_ZOOM_DRIFT', '0') or '0'))
        except Exception:
            self._debug_zoom_drift = False
        # Pan debug flag (INFO-level logs on pan_px)
        try:
            self._debug_pan = bool(int(os.getenv('NAPARI_CUDA_DEBUG_PAN', '0') or '0'))
        except Exception:
            self._debug_pan = False
        # Reset debug flag (INFO-level logs on camera.reset)
        try:
            self._debug_reset = bool(int(os.getenv('NAPARI_CUDA_DEBUG_RESET', '0') or '0'))
        except Exception:
            self._debug_reset = False
        # Orbit debug flag (INFO-level logs on orbit)
        try:
            self._debug_orbit = bool(int(os.getenv('NAPARI_CUDA_DEBUG_ORBIT', '0') or '0'))
        except Exception:
            self._debug_orbit = False
        # Orbit elevation clamp
        try:
            self._orbit_el_min = float(os.getenv('NAPARI_CUDA_ORBIT_ELEV_MIN', '-85.0') or '-85.0')
        except Exception:
            self._orbit_el_min = -85.0
        try:
            self._orbit_el_max = float(os.getenv('NAPARI_CUDA_ORBIT_ELEV_MAX', '85.0') or '85.0')
        except Exception:
            self._orbit_el_max = 85.0

        self._policy_func = level_policy.resolve_policy('oversampling')
        self._policy_name = 'oversampling'
        self._last_policy_decision: Dict[str, object] = {}
        self._last_interaction_ts = time.perf_counter()
        # Legacy policy logging state removed
        self._decision_seq: int = 0
        self._log_layer_debug = env_bool('NAPARI_CUDA_LAYER_DEBUG', False)
        # Focused logging for ROI anchor/center reconciliation without all layer debug
        try:
            self._log_roi_anchor = env_bool('NAPARI_CUDA_LOG_ROI_ANCHOR', False)
        except Exception:
            self._log_roi_anchor = False
        # Focused selector logging without enabling full layer debug
        try:
            self._log_policy_eval = env_bool('NAPARI_CUDA_LOG_POLICY_EVAL', False)
        except Exception:
            self._log_policy_eval = False
        # Legacy zoom-policy toggles removed; selection uses viewport oversampling
        # Optional: lock multiscale level (int index) to keep client/server in sync
        try:
            _lock = os.getenv('NAPARI_CUDA_LOCK_LEVEL')
            self._lock_level: Optional[int] = int(_lock) if (_lock is not None and _lock != '') else None
        except Exception:
            self._lock_level = None
        try:
            self._level_threshold_in = float(os.getenv('NAPARI_CUDA_LEVEL_THRESHOLD_IN', '1.05') or '1.05')
        except Exception:
            self._level_threshold_in = 1.05
        try:
            self._level_threshold_out = float(os.getenv('NAPARI_CUDA_LEVEL_THRESHOLD_OUT', '1.35') or '1.35')
        except Exception:
            self._level_threshold_out = 1.35
        try:
            self._level_hysteresis = float(os.getenv('NAPARI_CUDA_LEVEL_HYST', '0.0') or '0.0')
        except Exception:
            self._level_hysteresis = 0.0
        self._level_fine_threshold = max(self._level_threshold_in, 1.05)
        # Preserve current camera view on level switches (avoid resetting zoom)
        try:
            self._preserve_view_on_switch = env_bool('NAPARI_CUDA_PRESERVE_VIEW', True)
        except Exception:
            self._preserve_view_on_switch = True
        # Keep contrast limits stable across pans/level switches unless disabled
        try:
            self._sticky_contrast = env_bool('NAPARI_CUDA_STICKY_CONTRAST', True)
        except Exception:
            self._sticky_contrast = True
        # Last known dims.current_step from intents; used to preserve Z across
        # multiscale level switches regardless of viewer timing.
        self._last_step: Optional[tuple[int, ...]] = None
        # Coalesce selection to run once after a render when camera changed
        self.policy_eval_requested: bool = False
        # Policy init happens naturally; no deferral needed
        # One-shot safety: reset camera if initial frames are black
        try:
            self._auto_reset_on_black = env_bool('NAPARI_CUDA_AUTO_RESET_ON_BLACK', True)
        except Exception:
            self._auto_reset_on_black = True
        self._frame_pipeline.configure_auto_reset(self._auto_reset_on_black)
        self._black_reset_done: bool = False
        # Min time between level switches to avoid shimmering during pan
        try:
            self._level_switch_cooldown_ms = float(os.getenv('NAPARI_CUDA_LEVEL_SWITCH_COOLDOWN_MS', '150') or '150')
        except Exception:
            self._level_switch_cooldown_ms = 150.0
        self._last_level_switch_ts: float = 0.0
        # Initial ROI bypass removed; we now guarantee viewport containment
        # Cache the last ROI computed per level keyed by a transform signature.
        # Value: (transform_signature, roi)
        self._roi_cache: Dict[int, tuple[Optional[tuple[float, ...]], SliceROI]] = {}
        # Throttle ROI logging by tracking the most recent (roi, ts) we logged per level
        self._roi_log_state: Dict[int, tuple[SliceROI, float]] = {}
        self._roi_log_interval = 0.25  # seconds
        # Minimum ROI edge movement (pixels) to accept a new ROI and avoid shimmering
        try:
            self._roi_edge_threshold = int(os.getenv('NAPARI_CUDA_ROI_EDGE_THRESHOLD', '4') or '4')
        except Exception:
            self._roi_edge_threshold = 4
        # Optionally align ROI to level chunk/grid boundaries to stabilize sampling
        try:
            self._roi_align_chunks = env_bool('NAPARI_CUDA_ROI_ALIGN_CHUNKS', False)
        except Exception:
            self._roi_align_chunks = False
        # Ensure the render ROI always contains the current viewport to avoid black edges while panning
        try:
            self._roi_ensure_contains_viewport = env_bool('NAPARI_CUDA_ROI_ENSURE_CONTAINS_VIEWPORT', True)
        except Exception:
            self._roi_ensure_contains_viewport = True
        # Explicit default: do not force encoder IDR on Z-change unless enabled later
        self._idr_on_z: bool = False
        # No full-frame horizon on level switches; keep ROI consistent
        # Last applied ROI for the active level (for placement/translate)
        self._last_roi: Optional[tuple[int, SliceROI]] = None

        if policy_name:
            try:
                self.set_policy(policy_name)
            except Exception:
                logger.exception("policy init set failed; continuing with default")
        try:
            self._slice_max_bytes = int(os.getenv('NAPARI_CUDA_MAX_SLICE_BYTES', '0') or '0')
        except Exception:
            self._slice_max_bytes = 0
        try:
            self._volume_max_bytes = int(os.getenv('NAPARI_CUDA_MAX_VOLUME_BYTES', '0') or '0')
        except Exception:
            self._volume_max_bytes = 0
        try:
            self._volume_max_voxels = int(os.getenv('NAPARI_CUDA_MAX_VOLUME_VOXELS', '0') or '0')
        except Exception:
            self._volume_max_voxels = 0
        self._last_layer_debug: Optional[tuple[str, int, int]] = None

        # Ensure partial initialization is cleaned up if any step fails
        try:
            # Zarr/NGFF dataset configuration (optional); prefer explicit args from server
            self._zarr_path = zarr_path
            self._zarr_level = zarr_level
            self._zarr_axes = (zarr_axes or 'zyx')
            self._zarr_init_z = (zarr_z if (zarr_z is not None and int(zarr_z) >= 0) else None)

            self._zarr_shape: Optional[tuple[int, ...]] = None
            self._zarr_dtype: Optional[str] = None
            self._z_index: Optional[int] = None
            self._zarr_clim: Optional[tuple[float, float]] = None
            self._hw_limits = get_hw_limits()

            # Initialize CUDA first (independent of GL)
            self._init_cuda()
            # Create VisPy SceneCanvas (EGL backend) which makes a GL context current
            self._init_vispy_scene()
            # Adopt the current EGL context created by VisPy for capture
            self._init_egl()
            self._init_capture()
            self._init_cuda_interop()
            self._init_encoder()
        except Exception as e:
            # Best-effort cleanup of resources allocated so far
            logger.warning("Initialization failed; attempting cleanup: %s", e, exc_info=True)
            try:
                self.cleanup()
            except Exception as e2:
                logger.debug("Cleanup after failed init also failed: %s", e2)
            raise
        # Bitstream parameter cache is maintained by the server (for avcC/hvcC)
        # Debug configuration (single place)
        self._debug = DebugDumper(DebugConfig.from_env())
        if self._debug.cfg.enabled:
            self._debug.log_env_once()
            self._debug.ensure_out_dir()
        self._frame_pipeline.set_debug(self._debug)
        # Log format/size once for clarity
        logger.info(
            "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s, zarr=%s",
            self.width, self.height, self._frame_pipeline.enc_input_format, self.fps, self._animate,
            bool(self._zarr_path),
        )

    def _frame_volume_camera(self, w: int, h: int, d: int) -> None:
        """Choose stable initial center and distance for TurntableCamera.

        Center at the volume centroid and set distance so the full height fits
        in view (adds a small margin). This avoids dead pan response before the
        first zoom and prevents the initial zoom from overshooting.
        """
        cam = self.view.camera
        if not isinstance(cam, scene.cameras.TurntableCamera):
            return
        cam.center = (float(w) * 0.5, float(h) * 0.5, float(d) * 0.5)  # type: ignore[attr-defined]
        fov_deg = float(getattr(cam, 'fov', 60.0) or 60.0)
        fov_rad = math.radians(max(1e-3, min(179.0, fov_deg)))
        dist = (0.5 * float(h)) / max(1e-6, math.tan(0.5 * fov_rad))
        cam.distance = float(dist * 1.1)  # type: ignore[attr-defined]

    def _init_adapter_scene(self, source: Optional[ZarrSceneSource]) -> None:
        adapter = AdapterScene(self)
        canvas, view, viewer = adapter.init(source)
        # Mirror attributes for call sites expecting them
        self.canvas = canvas
        self.view = view
        self._viewer = viewer
        # Ensure the camera frames current data extents on first init
        try:
            if hasattr(self, 'view') and hasattr(self.view, 'camera'):
                self._apply_camera_reset(self.view.camera)
        except Exception:
            logger.debug("initial camera reset after adapter init failed", exc_info=True)

    def _set_dims_range_for_level(self, source: ZarrSceneSource, level: int) -> None:
        """Set napari dims.range to match the chosen level shape.

        This avoids Z slider reflecting base-resolution depth when rendering
        a coarser level.
        """
        if self._viewer is None:
            return
        descriptor = source.level_descriptors[level]
        shape = tuple(int(s) for s in descriptor.shape)
        ranges = tuple((0, max(0, s - 1), 1) for s in shape)
        self._viewer.dims.range = ranges

    def _set_level_with_budget(self, desired_level: int, *, reason: str) -> None:
        source = self._ensure_scene_source()
        levels_count = len(source.level_descriptors)
        if levels_count == 0:
            return
        desired_level = max(0, min(int(desired_level), levels_count - 1))

        candidates: List[int]
        if self.use_volume:
            candidates = list(range(desired_level, levels_count))
        else:
            # Allow graceful downgrade for 2D slices when over slice budget
            candidates = list(range(desired_level, levels_count))

        # Capture current viewport center in world space to reconcile ROI after switch
        # No center-anchored correction: placement uses layer.translate; keep ROI simple

        for idx, level in enumerate(candidates):
            try:
                prev_level = int(self._active_ms_level)
                # Enforce budgets prior to any data compute
                if self.use_volume:
                    self._volume_budget_allows(source, level)
                else:
                    self._slice_budget_allows(source, level)
                start = time.perf_counter()
                # Clear any stale ROI cache for target level
                cache = getattr(self, '_roi_cache', None)
                if isinstance(cache, dict):
                    cache.pop(int(level), None)
                roi_log = getattr(self, '_roi_log_state', None)
                if isinstance(roi_log, dict):
                    roi_log.pop(int(level), None)
                self._apply_level_internal(source, level, prev_level=prev_level)
                self._level_downgraded = (level != desired_level)
                if self._level_downgraded:
                    logger.info(
                        "level downgrade: requested=%d active=%d reason=%s",
                        desired_level,
                        level,
                        reason,
                    )
                else:
                    self._level_downgraded = False
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._log_ms_switch(
                    previous=prev_level,
                    applied=int(self._active_ms_level),
                    source=source,
                    elapsed_ms=elapsed_ms,
                    reason=reason,
                )
                self._mark_render_tick_needed()
                return
            except _LevelBudgetError as exc:
                if self._log_layer_debug:
                    logger.info(
                        "budget reject: mode=%s level=%d reason=%s",
                        'volume' if self.use_volume else 'slice',
                        int(level),
                        str(exc),
                    )
                if idx == len(candidates) - 1:
                    raise
                logger.debug("level %d rejected by budget: %s", level, exc)
                continue

        raise RuntimeError("Unable to select multiscale level within budget")

    # Legacy zoom-delta policy path removed; selection uses oversampling + stabilizer

    # Legacy _apply_zoom_based_level_switch removed

    def _init_egl(self) -> None:
        self._egl.ensure()

    def _init_cuda(self) -> None:
        cuda.init()
        dev = cuda.Device(0)
        ctx = dev.retain_primary_context()
        ctx.push()
        self.cuda_ctx = ctx

    def _create_scene_source(self) -> Optional[ZarrSceneSource]:
        if self._zarr_path is None:
            return None
        if da is None:
            raise RuntimeError("ZarrSceneSource requires dask.array to be available")
        axes_override = tuple(self._zarr_axes) if isinstance(self._zarr_axes, str) else None
        source = ZarrSceneSource(
            self._zarr_path,
            preferred_level=self._zarr_level,
            axis_override=axes_override,
        )
        return source

    def _ensure_scene_source(self) -> ZarrSceneSource:
        """Return a valid ZarrSceneSource and sync metadata.

        Raises a clear runtime error if Zarr is not configured or dask is missing.
        """
        if not self._zarr_path:
            raise RuntimeError("No OME-Zarr path configured for scene source")
        if da is None:
            raise RuntimeError("ZarrSceneSource requires dask.array to be available")
        source = self._scene_source
        if source is None:
            created = self._create_scene_source()
            if created is None:
                raise RuntimeError("Failed to create ZarrSceneSource")
            source = created
            self._scene_source = source

        target = source.current_level
        if self._zarr_level:
            target = source.level_index_for_path(self._zarr_level)
        if self._log_layer_debug:
            # Log only on change (no periodic spam) and at DEBUG level
            key = (int(source.current_level), str(self._zarr_level) if self._zarr_level else None)
            if self._last_ensure_log != key:
                logger.debug(
                    "ensure_source: current=%d target=%d path=%s",
                    int(source.current_level), int(target), self._zarr_level,
                )
                self._last_ensure_log = key
                self._last_ensure_log_ts = time.perf_counter()

        with self._state_lock:
            step = source.set_current_level(target, step=source.current_step)

        descriptor = source.level_descriptors[source.current_level]
        self._active_ms_level = int(source.current_level)
        self._zarr_level = descriptor.path or None
        self._zarr_axes = ''.join(source.axes)
        self._zarr_shape = descriptor.shape
        self._zarr_dtype = str(source.dtype)

        axes_lower = [str(ax).lower() for ax in source.axes]
        if step:
            if 'z' in axes_lower:
                z_pos = axes_lower.index('z')
                self._z_index = int(step[z_pos])
            else:
                self._z_index = int(step[0])

        return source

    def _init_vispy_scene(self) -> None:
        """Adapter-only scene initialization (legacy path removed)."""
        source = self._create_scene_source()
        if source is not None:
            self._scene_source = source
            try:
                self._zarr_shape = source.level_shape(0)
                self._zarr_dtype = str(source.dtype)
                self._active_ms_level = source.current_level
                descriptor = source.level_descriptors[source.current_level]
                self._zarr_level = descriptor.path or None
            except Exception:
                logger.debug("scene source metadata bootstrap failed", exc_info=True)

        try:
            self._init_adapter_scene(source)
        except Exception:
            logger.exception("Adapter scene initialization failed (legacy path removed)")
            # Fail fast: adapter is mandatory
            raise
        
    # --- Multiscale: request switch (thread-safe) ---
    def request_multiscale_level(self, level: int, path: Optional[str] = None) -> None:
        """Queue a multiscale level switch; applied on render thread next frame."""
        if getattr(self, '_lock_level', None) is not None:
            if self._log_layer_debug and logger.isEnabledFor(logging.INFO):
                logger.info("request_multiscale_level ignored due to lock_level=%s", str(self._lock_level))
            return
        if self._log_layer_debug and logger.isEnabledFor(logging.INFO):
            logger.info(
                "request multiscale level: level=%d path=%s",
                int(level),
                str(path) if path else "<default>",
            )
        self._scene_state_coordinator.queue_multiscale_level(int(level), str(path) if path else None)
        self._last_interaction_ts = time.perf_counter()
        self._mark_render_tick_needed()

    def _notify_scene_refresh(self) -> None:
        cb = self._scene_refresh_cb
        if cb is None:
            return
        # Try to provide the most authoritative step we have
        step_hint = None
        try:
            src = getattr(self, '_scene_source', None)
            if src is not None:
                st = getattr(src, 'current_step', None)
                if st is not None:
                    step_hint = tuple(int(x) for x in st)
        except Exception:
            step_hint = None
        if step_hint is None:
            try:
                if self._viewer is not None:
                    step_hint = tuple(int(x) for x in self._viewer.dims.current_step)
            except Exception:
                step_hint = None
        if step_hint is None and self._z_index is not None:
            step_hint = (int(self._z_index),)
        # Pass step hint directly to server callback if supported
        try:
            cb(step_hint)  # type: ignore[misc]
        except TypeError:
            cb()
        except Exception:
            logger.debug("scene refresh callback failed", exc_info=True)

    def _mark_render_tick_needed(self) -> None:
        self._render_tick_required = True

    def set_policy(self, name: str) -> None:
        new_name = str(name or '').strip().lower()
        # Accept only simplified oversampling policy (and close synonyms)
        allowed = {'oversampling', 'thresholds', 'ratio'}
        if new_name not in allowed:
            raise ValueError(f"Unsupported policy: {new_name}")
        self._policy_name = new_name
        # Map all aliases to the same selector
        self._policy_func = level_policy.resolve_policy(new_name)
        self._last_policy_decision = {}
        # No history maintenance; snapshot excludes history
        if self._log_layer_debug:
            logger.info("policy set: name=%s", new_name)

    def _format_level_roi(self, source: ZarrSceneSource, level: int) -> str:
        if self.use_volume:
            return "volume"
        roi = self._viewport_roi_for_level(source, level)
        if roi.is_empty():
            return "full"
        return f"y={roi.y_start}:{roi.y_stop} x={roi.x_start}:{roi.x_stop}"

    def _log_ms_switch(self, *, previous: int, applied: int, source: ZarrSceneSource, elapsed_ms: float, reason: str) -> None:
        # Suppress no-op logs when level doesn't change
        if int(previous) == int(applied):
            return
        roi_desc = self._format_level_roi(source, applied)
        logger.info(
            "ms.switch: level=%d->%d roi=%s reason=%s elapsed=%.2fms",
            int(previous),
            int(applied),
            roi_desc,
            reason,
            float(elapsed_ms),
        )

    # Priming retired: method removed


    def _physical_scale_for_level(self, source: ZarrSceneSource, level: int) -> tuple[float, float, float]:
        try:
            scale = source.level_scale(level)
        except Exception:
            return (1.0, 1.0, 1.0)
        values = [float(s) for s in scale]
        if not values:
            return (1.0, 1.0, 1.0)
        if len(values) >= 3:
            return (values[-3], values[-2], values[-1])
        if len(values) == 2:
            return (1.0, values[0], values[1])
        # single element; reuse across axes
        return (values[0], values[0], values[0])

    def _oversampling_for_level(self, source: ZarrSceneSource, level: int) -> float:
        try:
            # For policy calculations, ignore full-frame horizon and ROI hysteresis
            roi = self._viewport_roi_for_level(source, level, quiet=True, for_policy=True)
            if roi.is_empty():
                h, w = plane_wh_for_level(source, level)
            else:
                h, w = roi.height, roi.width
        except Exception:
            h, w = self.height, self.width
        vh = max(1, int(self.height))
        vw = max(1, int(self.width))
        return float(max(h / vh, w / vw))

    def _ensure_panzoom_camera(self, *, reason: str) -> Optional[scene.cameras.PanZoomCamera]:
        if self.use_volume:
            return None
        view = getattr(self, "view", None)
        if view is None:
            return None
        cam = getattr(view, "camera", None)
        if isinstance(cam, scene.cameras.PanZoomCamera):
            return cam
        if self._log_layer_debug and cam is not None:
            logger.info(
                "ensure panzoom camera: reason=%s current=%s",
                reason,
                cam.__class__.__name__,
            )
        try:
            panzoom = scene.cameras.PanZoomCamera(aspect=1.0)
            view.camera = panzoom
            data_wh = getattr(self, "_data_wh", None)
            if data_wh:
                try:
                    w, h = data_wh
                    panzoom.set_range(x=(0, float(w)), y=(0, float(h)))
                except Exception:
                    logger.debug("ensure_panzoom: data_wh set_range failed", exc_info=True)
            else:
                try:
                    panzoom.set_range(x=(0, float(self.width)), y=(0, float(self.height)))
                except Exception:
                    logger.debug("ensure_panzoom: canvas set_range failed", exc_info=True)
            return panzoom
        except Exception:
            logger.debug("ensure_panzoom_camera failed", exc_info=True)
            return None

    def _viewport_debug_snapshot(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "canvas_size": (int(self.width), int(self.height)),
            "data_wh": tuple(int(v) for v in self._data_wh) if self._data_wh else None,
            "data_depth": int(self._data_d) if getattr(self, "_data_d", None) is not None else None,
        }
        view = getattr(self, "view", None)
        if view is not None:
            info["view_class"] = view.__class__.__name__
            try:
                cam = getattr(view, "camera", None)
                if cam is not None:
                    cam_info: Dict[str, Any] = {"type": cam.__class__.__name__}
                    rect = getattr(cam, "rect", None)
                    if rect is not None:
                        try:
                            cam_info["rect"] = tuple(float(v) for v in rect)
                        except Exception:
                            cam_info["rect"] = str(rect)
                    center = getattr(cam, "center", None)
                    if center is not None:
                        try:
                            cam_info["center"] = tuple(float(v) for v in center)
                        except Exception:
                            cam_info["center"] = str(center)
                    if hasattr(cam, "zoom"):
                        try:
                            cam_info["zoom"] = float(cam.zoom)  # type: ignore[arg-type]
                        except Exception:
                            cam_info["zoom"] = str(cam.zoom)
                    if hasattr(cam, "scale"):
                        try:
                            cam_info["scale"] = tuple(float(v) for v in cam.scale)
                        except Exception:
                            cam_info["scale"] = str(cam.scale)
                    if hasattr(cam, "_viewbox"):
                        try:
                            vb = cam._viewbox
                            cam_info["viewbox_size"] = tuple(float(v) for v in getattr(vb, "size", ()) or ())
                        except Exception:
                            cam_info["viewbox_size"] = "unavailable"
                    info["camera"] = cam_info
            except Exception:
                info["camera"] = "error"
            try:
                transform = getattr(view, "scene", None)
                if transform is not None and hasattr(transform, "transform"):
                    xform = transform.transform
                    if hasattr(xform, "matrix"):
                        mat = getattr(xform, "matrix")
                        try:
                            info["transform_matrix"] = tuple(float(v) for v in mat.ravel())
                        except Exception:
                            info["transform_matrix"] = str(mat)
            except Exception:
                info["transform"] = "error"
        else:
            info["view"] = None
        return info

    def _log_roi_fallback(
        self,
        *,
        level: int,
        reason: str,
        plane_h: int,
        plane_w: int,
        scale: tuple[float, float],
    ) -> None:
        if not self._log_layer_debug or not logger.isEnabledFor(logging.INFO):
            return
        snapshot = self._viewport_debug_snapshot()
        logger.info(
            "viewport ROI fallback: level=%d reason=%s dims=%dx%d scale=(%.6f,%.6f) snapshot=%s",
            level,
            reason,
            plane_h,
            plane_w,
            scale[0],
            scale[1],
            snapshot,
        )

    def _viewport_world_bounds(self) -> Optional[tuple[float, float, float, float]]:
        view = getattr(self, "view", None)
        if view is None or not hasattr(view, "camera"):
            return None
        try:
            transform = view.scene.transform
            corners = (
                (0.0, 0.0),
                (float(self.width), 0.0),
                (0.0, float(self.height)),
                (float(self.width), float(self.height)),
            )
            world_pts = [transform.imap((float(x), float(y), 0.0)) for x, y in corners]
            xs = [float(pt[0]) for pt in world_pts]
            ys = [float(pt[1]) for pt in world_pts]
            return (min(xs), max(xs), min(ys), max(ys))
        except Exception:
            if self._log_layer_debug:
                logger.debug("viewport_world_bounds failed", exc_info=True)
            return None

    # Note: we previously computed a world-space viewport center for a one-shot
    # ROI reconciliation. With layer.translate placement, this is no longer used.

    def _viewport_roi_for_level(self, source: ZarrSceneSource, level: int, *, quiet: bool = False, for_policy: bool = False) -> SliceROI:
        view = getattr(self, "view", None)
        plane_h, plane_w = plane_wh_for_level(source, level)
        scale = plane_scale_for_level(source, level)

        if view is None or not hasattr(view, "camera"):
            if not quiet:
                self._log_roi_fallback(level=level, reason="no-view", plane_h=plane_h, plane_w=plane_w, scale=scale)
            return SliceROI(0, plane_h, 0, plane_w)

        cam = view.camera
        if not isinstance(cam, scene.cameras.PanZoomCamera):
            ensured = self._ensure_panzoom_camera(reason="roi-request")
            cam = ensured or cam
        if not isinstance(cam, scene.cameras.PanZoomCamera):
            if not quiet:
                self._log_roi_fallback(level=level, reason="no-panzoom", plane_h=plane_h, plane_w=plane_w, scale=scale)
            return SliceROI(0, plane_h, 0, plane_w)

        signature: Optional[tuple[float, ...]] = None
        scene_graph = getattr(view, "scene", None)
        transform = getattr(scene_graph, "transform", None)
        if transform is not None and hasattr(transform, "matrix"):
            try:
                signature = tuple(float(v) for v in transform.matrix.ravel())
            except Exception:
                signature = None

        cached = self._roi_cache.get(int(level))
        if cached is not None and signature is not None and cached[0] == signature:
            return cached[1]

        align_chunks = (not for_policy) and bool(getattr(self, "_roi_align_chunks", True))
        ensure_contains = (not for_policy) and bool(getattr(self, "_roi_ensure_contains_viewport", True))
        edge_threshold = int(getattr(self, "_roi_edge_threshold", 4))
        chunk_pad = int(getattr(self, "_roi_pad_chunks", 1))

        prev_logged: Optional[SliceROI] = None
        if not for_policy:
            roi_log = getattr(self, "_roi_log_state", None)
            if isinstance(roi_log, dict):
                logged = roi_log.get(int(level))
                if logged is not None:
                    prev_logged = logged[0]

        try:
            result = compute_viewport_roi(
                view=view,
                canvas_size=(int(self.width), int(self.height)),
                source=source,
                level=int(level),
                align_chunks=align_chunks,
                chunk_pad=chunk_pad,
                ensure_contains_viewport=ensure_contains,
                edge_threshold=edge_threshold,
                prev_roi=prev_logged,
                for_policy=for_policy,
                transform_signature=signature,
            )
        except Exception:
            if not quiet and self._log_layer_debug:
                logger.exception(
                    "viewport ROI computation failed; returning full frame (level=%d dims=%dx%d)",
                    level,
                    plane_h,
                    plane_w,
                )
            if not quiet:
                self._log_roi_fallback(level=level, reason="compute-failed", plane_h=plane_h, plane_w=plane_w, scale=scale)
            return SliceROI(0, plane_h, 0, plane_w)

        roi = result.roi
        if roi.is_empty():
            if not quiet:
                self._log_roi_fallback(level=level, reason="empty", plane_h=plane_h, plane_w=plane_w, scale=scale)
            return SliceROI(0, plane_h, 0, plane_w)

        if result.transform_signature is not None:
            self._roi_cache[int(level)] = (result.transform_signature, roi)

        if not for_policy:
            roi_log = getattr(self, "_roi_log_state", None)
            if isinstance(roi_log, dict):
                roi_log[int(level)] = (roi, time.perf_counter())

        return roi

    def _estimate_volume_io(self, source: ZarrSceneSource, level: int) -> tuple[int, int]:
        chunk_count = 0
        bytes_est = 0
        try:
            arr = source.get_level(level)
        except Exception:
            return chunk_count, bytes_est

        try:
            dtype_size = int(np.dtype(getattr(arr, 'dtype', source.dtype)).itemsize)
        except Exception:
            dtype_size = int(np.dtype(source.dtype).itemsize) if hasattr(source, 'dtype') else 0

        try:
            shape = tuple(int(s) for s in getattr(arr, 'shape', ()))
            voxels = 1
            for dim in shape:
                voxels *= max(1, dim)
            bytes_est = int(voxels) * int(dtype_size)
        except Exception:
            bytes_est = 0

        chunks_attr = getattr(arr, 'chunks', None)
        if chunks_attr is None:
            return chunk_count, bytes_est

        try:
            chunk_count = 1
            for axis_chunks in list(chunks_attr):
                chunk_count *= max(1, len(axis_chunks))
        except Exception:
            chunk_count = 0
        return int(chunk_count), int(bytes_est)
    def _record_policy_decision(
        self,
        *,
        policy: str,
        intent_level: Optional[int],
        selected_level: Optional[int],
        desired_level: int,
        applied_level: int,
        reason: str,
        idle_ms: float,
        oversampling: Mapping[int, float] | None,
        from_level: Optional[int] = None,
    ) -> None:
        overs = {int(k): float(v) for k, v in (oversampling or {}).items()}
        self._decision_seq += 1
        self._last_policy_decision = {
            'timestamp_ms': time.time() * 1000.0,
            'seq': int(self._decision_seq),
            'policy': str(policy),
            'intent_level': int(intent_level) if intent_level is not None else None,
            'selected_level': int(selected_level) if selected_level is not None else None,
            'desired_level': int(desired_level),
            'applied_level': int(applied_level),
            'from_level': int(from_level) if from_level is not None else None,
            'reason': str(reason),
            'idle_ms': float(idle_ms),
            'oversampling': overs,
            'downgraded': bool(self._level_downgraded),
        }
        # No history maintenance; events are the source of truth

    def policy_metrics_snapshot(self) -> Dict[str, object]:
        # Minimal, stable snapshot for external consumers
        return {
            'last_decision': dict(self._last_policy_decision),
            'policy': self._policy_name,
            'active_level': int(self._active_ms_level),
            'level_downgraded': bool(self._level_downgraded),
        }

    def _load_slice(
        self,
        source: ZarrSceneSource,
        level: int,
        z_idx: int,
    ) -> np.ndarray:
        roi = self._viewport_roi_for_level(source, level)
        # Remember ROI used for this level so we can place the slab in world coords
        try:
            self._last_roi = (int(level), roi)
        except Exception:
            self._last_roi = None
        slab = source.slice(level, z_idx, compute=True, roi=roi)
        if not isinstance(slab, np.ndarray):
            slab = np.asarray(slab, dtype=np.float32)
        return slab

    def _load_volume(
        self,
        source: ZarrSceneSource,
        level: int,
    ) -> np.ndarray:
        volume = source.level_volume(level, compute=True)
        if not isinstance(volume, np.ndarray):
            volume = np.asarray(volume, dtype=np.float32)
        return volume

    def _build_policy_context(
        self,
        source: ZarrSceneSource,
        *,
        intent_level: Optional[int],
    ) -> level_policy.LevelSelectionContext:
        levels = tuple(source.level_descriptors)
        overs_map: Dict[int, float] = {}
        for descriptor in levels:
            try:
                lvl = int(descriptor.index)
            except Exception:
                lvl = int(levels.index(descriptor))
            try:
                overs_map[lvl] = float(self._oversampling_for_level(source, lvl))
            except Exception:
                overs_map[lvl] = float('nan')
        return level_policy.LevelSelectionContext(
            levels=levels,
            current_level=int(self._active_ms_level),
            intent_level=int(intent_level) if intent_level is not None else None,
            level_oversampling=overs_map,
            thresholds=None,
        )

    def _estimate_level_bytes(self, source: ZarrSceneSource, level: int) -> tuple[int, int]:
        descriptor = source.level_descriptors[level]
        voxels = 1
        for dim in descriptor.shape:
            voxels *= max(1, int(dim))
        dtype_size = np.dtype(source.dtype).itemsize
        return int(voxels), int(voxels * dtype_size)

    def _volume_budget_allows(self, source: ZarrSceneSource, level: int) -> None:
        voxels, bytes_est = self._estimate_level_bytes(source, level)
        limit_bytes = self._volume_max_bytes or self._hw_limits.volume_max_bytes
        limit_voxels = self._volume_max_voxels or self._hw_limits.volume_max_voxels
        if limit_voxels and voxels > limit_voxels:
            msg = f"voxels={voxels} exceeds cap={limit_voxels}"
            if self._log_layer_debug:
                logger.info("budget check (volume): level=%d voxels=%d bytes=%d -> REJECT: %s", level, voxels, bytes_est, msg)
            raise _LevelBudgetError(msg)
        if limit_bytes and bytes_est > limit_bytes:
            msg = f"bytes={bytes_est} exceeds cap={limit_bytes}"
            if self._log_layer_debug:
                logger.info("budget check (volume): level=%d voxels=%d bytes=%d -> REJECT: %s", level, voxels, bytes_est, msg)
            raise _LevelBudgetError(msg)
        if self._log_layer_debug:
            logger.info("budget check (volume): level=%d voxels=%d bytes=%d -> OK", level, voxels, bytes_est)

    def _slice_budget_allows(self, source: ZarrSceneSource, level: int) -> None:
        """Enforce optional max-bytes budget for a single 2D slice.

        Uses YX plane dimensions and the source dtype size. Only applies when
        NAPARI_CUDA_MAX_SLICE_BYTES is non-zero.
        """
        limit_bytes = int(self._slice_max_bytes or 0)
        if limit_bytes <= 0:
            return
        h, w = plane_wh_for_level(source, level)
        dtype_size = int(np.dtype(source.dtype).itemsize)
        bytes_est = int(h) * int(w) * dtype_size
        if bytes_est > limit_bytes:
            msg = f"slice bytes={bytes_est} exceeds cap={limit_bytes} at level={level}"
            if self._log_layer_debug:
                logger.info(
                    "budget check (slice): level=%d h=%d w=%d dtype=%s bytes=%d -> REJECT: %s",
                    level, h, w, str(source.dtype), bytes_est, msg,
                )
            raise _LevelBudgetError(msg)
        if self._log_layer_debug:
            logger.info(
                "budget check (slice): level=%d h=%d w=%d dtype=%s bytes=%d -> OK",
                level, h, w, str(source.dtype), bytes_est,
            )

    def _get_level_volume(
        self,
        source: ZarrSceneSource,
        level: int,
    ) -> np.ndarray:
        self._volume_budget_allows(source, level)
        return self._load_volume(source, level)

    def _log_layer_assignment(
        self,
        mode: str,
        level: int,
        z_index: Optional[int],
        shape: tuple[int, ...],
        contrast: tuple[float, float],
    ) -> None:
        if not self._log_layer_debug or not logger.isEnabledFor(logging.INFO):
            return
        key = (mode, level, int(z_index) if z_index is not None else -1)
        if key == self._last_layer_debug:
            return
        self._last_layer_debug = key
        logger.info(
            "layer assign: mode=%s level=%d z=%s shape=%s contrast=(%.4f, %.4f) downgraded=%s",
            mode,
            level,
            z_index if z_index is not None else 'na',
            'x'.join(str(int(x)) for x in shape),
            float(contrast[0]),
            float(contrast[1]),
            self._level_downgraded,
        )

    def _apply_level_internal(
        self,
        source: ZarrSceneSource,
        level: int,
        *,
        prev_level: Optional[int] = None,
    ) -> None:
        applied = apply_level(
            source=source,
            target_level=int(level),
            prev_level=prev_level,
            last_step=self._last_step,
            viewer=self._viewer,
        )

        descriptor = source.level_descriptors[int(level)]
        self._active_ms_level = applied.level
        self._last_step = applied.step
        self._z_index = applied.z_index
        self._zarr_level = descriptor.path or None
        self._zarr_shape = descriptor.shape
        self._zarr_axes = applied.axes
        self._zarr_dtype = applied.dtype
        self._zarr_clim = applied.contrast

        layer = self._napari_layer
        if self.use_volume:
            volume = self._get_level_volume(source, applied.level)
            d, h, w = int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])
            self._data_wh = (w, h)
            self._data_d = d
            if layer is not None:
                layer.data = volume
                layer.contrast_limits = [float(applied.contrast[0]), float(applied.contrast[1])]
            self._log_layer_assignment('volume', applied.level, None, (d, h, w), applied.contrast)
        else:
            sy, sx = applied.scale_yx
            if layer is not None:
                try:
                    layer.scale = (sy, sx)
                except Exception:
                    logger.debug("apply_level: setting 2D layer scale pre-slab failed", exc_info=True)
            z_idx = int(self._z_index or 0)
            slab = self._load_slice(source, applied.level, z_idx)
            roi_for_layer = None
            try:
                if self._last_roi is not None and int(self._last_roi[0]) == int(applied.level):
                    roi_for_layer = self._last_roi[1]
            except Exception:
                roi_for_layer = None
            if layer is not None:
                cam = getattr(self.view, 'camera', None)
                ctx = self._build_scene_state_context(cam)
                SceneStateApplier.apply_slice_to_layer(
                    ctx,
                    source=source,
                    slab=slab,
                    roi=roi_for_layer,
                    update_contrast=not self._sticky_contrast,
                )
            h, w = int(slab.shape[0]), int(slab.shape[1])
            self._data_wh = (w, h)
            self._data_d = None
            if not self._preserve_view_on_switch and self.view is not None and hasattr(self.view, 'camera'):
                try:
                    world_w = float(w) * float(max(1e-12, sx))
                    world_h = float(h) * float(max(1e-12, sy))
                    self.view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
                except Exception:
                    logger.debug("apply_level: camera set_range failed", exc_info=True)
            self._log_layer_assignment('slice', applied.level, self._z_index, (h, w), applied.contrast)

        self._notify_scene_refresh()

    def _perform_level_switch(
        self,
        *,
        target_level: int,
        reason: str,
        intent_level: Optional[int],
        selected_level: Optional[int],
        source: Optional[ZarrSceneSource] = None,
    ) -> None:
        """Switch level immediately and record policy decision."""
        if not self._zarr_path:
            return
        if getattr(self, '_lock_level', None) is not None:
            if self._log_layer_debug and logger.isEnabledFor(logging.INFO):
                logger.info("perform_level_switch ignored due to lock_level=%s", str(self._lock_level))
            return
        if source is None:
            source = self._ensure_scene_source()
        target_level = int(target_level)
        ctx = self._build_policy_context(source, intent_level=target_level)
        prev = int(self._active_ms_level)
        self._set_level_with_budget(target_level, reason=reason)
        if reason in {"zoom-in", "zoom-out"}:
            self._last_zoom_hint_ts = time.perf_counter()
        idle_ms = max(0.0, (time.perf_counter() - self._last_interaction_ts) * 1000.0)
        self._record_policy_decision(
            policy=self._policy_name,
            intent_level=intent_level if intent_level is not None else ctx.intent_level,
            selected_level=selected_level if selected_level is not None else target_level,
            desired_level=target_level,
            applied_level=int(self._active_ms_level),
            reason=reason,
            idle_ms=idle_ms,
            oversampling=ctx.level_oversampling,
            from_level=prev,
        )

    def _apply_multiscale_switch(self, level: int, path: Optional[str]) -> None:
        if not self._zarr_path:
            return
        source = self._ensure_scene_source()
        target = level
        if path:
            target = source.level_index_for_path(path)
        self._perform_level_switch(
            target_level=int(target),
            reason="intent",
            intent_level=int(target),
            selected_level=int(target),
            source=source,
        )

    def _clear_visual(self) -> None:
        if self._visual is not None and hasattr(self._visual, 'parent'):
            self._visual.parent = None  # type: ignore[attr-defined]
        self._visual = None

    def _apply_ndisplay_switch(self, ndisplay: int) -> None:
        """Apply 2D/3D toggle on the render thread."""
        assert self.view is not None and hasattr(self.view, 'scene'), "VisPy view must be initialized"
        # Adapter-only: do not rebuild visuals; just update viewer dims
        target = 3 if int(ndisplay) >= 3 else 2
        self.use_volume = bool(target == 3)
        if self._viewer is not None:
            try:
                self._viewer.dims.ndisplay = target
            except Exception:
                logger.debug("ndisplay update via viewer failed", exc_info=True)
        logger.info("ndisplay switch: %s", "3D" if target == 3 else "2D")
        # Evaluate policy once after display mode changes
        try:
            self._evaluate_level_policy()
        except Exception:
            logger.debug("ndisplay selection failed", exc_info=True)

    def _init_capture(self) -> None:
        self._gl_capture.ensure()

    def _init_cuda_interop(self) -> None:
        """Init CUDA-GL interop and allocate destination via one-time DLPack bridge."""
        tex = self._gl_capture.texture_id
        if tex is None:
            raise RuntimeError("Capture texture is not available for CUDA interop")
        self._cuda.initialize(tex)

    def _init_encoder(self) -> None:
        with self._enc_lock:
            if self._encoder is None:
                self._encoder = Encoder(self.width, self.height, fps_hint=int(self.fps))
            encoder = self._encoder
            encoder.set_fps_hint(int(self.fps))
            encoder.setup(self._ctx)
            self._frame_pipeline.set_enc_input_format(encoder.input_format)

    def reset_encoder(self) -> None:
        with self._enc_lock:
            encoder = self._encoder
            if encoder is None:
                return
            encoder.set_fps_hint(int(self.fps))
            encoder.reset(self._ctx)
            self._enc_input_fmt = encoder.input_format

    def force_idr(self) -> None:
        logger.debug("Requesting encoder force IDR")
        with self._enc_lock:
            encoder = self._encoder
            if encoder is None:
                return
            encoder.force_idr()

    def _request_encoder_idr(self) -> None:
        with self._enc_lock:
            encoder = self._encoder
            if encoder is None:
                return
            encoder.request_idr()

    def render_frame(self, azimuth_deg: Optional[float] = None) -> None:
        if azimuth_deg is not None and hasattr(self.view, "camera"):
            self.view.camera.azimuth = float(azimuth_deg)
        self.drain_scene_updates()
        t0 = time.perf_counter()
        self.canvas.render()
        self._render_tick_required = False
        self._render_loop_started = True

    def apply_state(self, state: ServerSceneState) -> None:
        """Queue a complete scene state snapshot for the next frame."""
        self._last_interaction_ts = time.perf_counter()
        normalized = ServerSceneState(
            center=tuple(state.center) if state.center is not None else None,
            zoom=float(state.zoom) if state.zoom is not None else None,
            angles=tuple(state.angles) if state.angles is not None else None,
            current_step=tuple(state.current_step) if state.current_step is not None else None,
            volume_mode=str(getattr(state, 'volume_mode', None)) if getattr(state, 'volume_mode', None) is not None else None,
            volume_colormap=str(getattr(state, 'volume_colormap', None)) if getattr(state, 'volume_colormap', None) is not None else None,
            volume_clim=tuple(getattr(state, 'volume_clim')) if getattr(state, 'volume_clim', None) is not None else None,
            volume_opacity=float(getattr(state, 'volume_opacity')) if getattr(state, 'volume_opacity', None) is not None else None,
            volume_sample_step=float(getattr(state, 'volume_sample_step')) if getattr(state, 'volume_sample_step', None) is not None else None,
        )
        self._scene_state_coordinator.queue_scene_state(normalized)

    def process_camera_commands(self, commands: Sequence[CameraCommand]) -> None:
        if not commands:
            return
        # Interaction observed (kept for future metrics)
        self._user_interaction_seen = True
        logger.debug("worker processing %d camera command(s)", len(commands))
        cam = self.view.camera
        if cam is None:
            for cmd in commands:
                if cmd.kind == 'zoom' and cmd.factor is not None and cmd.factor > 0.0:
                    self._scene_state_coordinator.record_zoom_intent(float(cmd.factor))
            return

        canvas_wh: tuple[int, int]
        if self.canvas is not None and hasattr(self.canvas, 'size'):
            canvas_wh = (int(self.canvas.size[0]), int(self.canvas.size[1]))
        else:
            canvas_wh = (self.width, self.height)

        debug_flags = CameraDebugFlags(
            zoom=self._debug_zoom_drift,
            pan=self._debug_pan,
            orbit=self._debug_orbit,
            reset=self._debug_reset,
        )
        outcome: CameraCommandOutcome = apply_camera_commands(
            commands,
            camera=cam,
            view=self.view,
            canvas_size=canvas_wh,
            reset_camera=self._apply_camera_reset,
            debug_flags=debug_flags,
        )

        if outcome.zoom_intent is not None:
            now_ts = time.perf_counter()
            if (now_ts - float(self._last_zoom_hint_ts)) >= float(self._zoom_hint_hold_s):
                zoom_ratio = float(outcome.zoom_intent)
                if zoom_ratio > 1.0:
                    # Napari emits factors >1 for zoom-in; selector expects <1.0 to mean zoom-in.
                    zoom_ratio = 1.0 / zoom_ratio
                self._scene_state_coordinator.record_zoom_intent(zoom_ratio)
                self._last_zoom_hint_ts = now_ts

        self._last_interaction_ts = time.perf_counter()

        if outcome.camera_changed:
            self._mark_render_tick_needed()
        if outcome.policy_triggered:
            self.policy_eval_requested = True

    def _apply_camera_reset(self, cam) -> None:
        if cam is None or not hasattr(cam, 'set_range'):
            return
        w, h = getattr(self, '_data_wh', (self.width, self.height))
        d = getattr(self, '_data_d', None)
        # Convert pixel dims to world extents using current level scale
        sx = sy = 1.0
        try:
            if self._scene_source is not None and not self.use_volume:
                sy, sx = plane_scale_for_level(self._scene_source, int(self._active_ms_level))
        except Exception:
            logger.debug('camera reset: level scale lookup failed', exc_info=True)
        if d is not None:
            # 3D: set full volume range
            cam.set_range(x=(0, int(w)), y=(0, int(h)), z=(0, int(d)))
            self._frame_volume_camera(int(w), int(h), int(d))
        else:
            # 2D: set full world extents (shape * scale)
            world_w = float(w) * float(max(1e-12, sx))
            world_h = float(h) * float(max(1e-12, sy))
            cam.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))

    def _build_scene_state_context(self, cam) -> SceneStateApplyContext:
        return SceneStateApplyContext(
            use_volume=bool(self.use_volume),
            viewer=self._viewer,
            camera=cam,
            visual=self._visual,
            layer=self._napari_layer,
            scene_source=self._scene_source,
            active_ms_level=int(self._active_ms_level),
            z_index=self._z_index,
            last_roi=self._last_roi,
            preserve_view_on_switch=self._preserve_view_on_switch,
            sticky_contrast=self._sticky_contrast,
            idr_on_z=self._idr_on_z,
            data_wh=self._data_wh,
            state_lock=self._state_lock,
            ensure_scene_source=self._ensure_scene_source,
            plane_scale_for_level=plane_scale_for_level,
            load_slice=self._load_slice,
            notify_scene_refresh=self._notify_scene_refresh,
            mark_render_tick_needed=self._mark_render_tick_needed,
            request_encoder_idr=self._request_encoder_idr,
        )

    def drain_scene_updates(self) -> None:
        updates: SceneUpdateBundle = self._scene_state_coordinator.drain_pending_updates()

        if updates.display_mode is not None:
            self._apply_ndisplay_switch(updates.display_mode)

        if updates.multiscale is not None:
            lvl = int(updates.multiscale.level)
            pth = updates.multiscale.path
            self._apply_multiscale_switch(lvl, pth)

        state = updates.scene_state
        if state is None:
            return

        cam = getattr(self.view, 'camera', None)

        # Delegate dims/Z handling to the applier in 2D mode
        if not bool(self.use_volume) and state.current_step is not None:
            ctx = self._build_scene_state_context(cam)
            res = SceneStateApplier.apply_dims_and_slice(ctx, current_step=state.current_step)
            if res.z_index is not None:
                self._z_index = int(res.z_index)
            if res.data_wh is not None:
                self._data_wh = (int(res.data_wh[0]), int(res.data_wh[1]))
            if res.last_step is not None:
                self._last_step = tuple(int(x) for x in res.last_step)

        # Volume visual parameters in 3D mode
        if bool(self.use_volume):
            ctx = self._build_scene_state_context(cam)
            SceneStateApplier.apply_volume_params(
                ctx,
                mode=getattr(state, 'volume_mode', None),
                colormap=getattr(state, 'volume_colormap', None),
                clim=getattr(state, 'volume_clim', None),
                opacity=getattr(state, 'volume_opacity', None),
                sample_step=getattr(state, 'volume_sample_step', None),
            )

        if cam is None:
            self._evaluate_level_policy()
            return

        # Legacy absolute camera fields
        if state.center is not None and hasattr(cam, 'center'):
            cam.center = state.center  # type: ignore[attr-defined]
        if state.zoom is not None and hasattr(cam, 'zoom'):
            cam.zoom = state.zoom  # type: ignore[attr-defined]
        if state.angles is not None and hasattr(cam, 'angles'):
            cam.angles = state.angles  # type: ignore[attr-defined]

        # Evaluate policy only if the absolute state actually changed
        if self._scene_state_coordinator.update_state_signature(state):
            self._evaluate_level_policy()

    # --- Public coalesced toggles ------------------------------------------------
    def request_ndisplay(self, ndisplay: int) -> None:
        """Queue a 2D/3D view switch to apply on the render thread."""
        self._scene_state_coordinator.queue_display_mode(3 if int(ndisplay) >= 3 else 2)

    def _capture_blit_gpu_ns(self) -> Optional[int]:
        return self._gl_capture.blit_with_timing()


    def capture_and_encode_packet(self) -> tuple[FrameTimings, Optional[bytes], int, int]:
        """Same as capture_and_encode, but also returns the packet and flags.

        Flags bit 0x01 indicates keyframe (IDR/CRA).
        """
        # Debug env is logged once at init by DebugDumper
        # Capture wall and monotonic timestamps at frame start
        wall_ts = time.time()
        t0 = time.perf_counter()
        render_ms = self.render_tick()
        # Blit timing (GPU + CPU)
        t_b0 = time.perf_counter()
        blit_gpu_ns = self._frame_pipeline.capture_blit_gpu_ns()
        t_b1 = time.perf_counter()
        blit_cpu_ms = (t_b1 - t_b0) * 1000.0
        debug_cb = None
        if hasattr(self, "_debug") and self._debug.cfg.enabled and self._debug.cfg.frames_remaining > 0:
            def _cb(tex_id: int, w: int, h: int, frame) -> None:
                self._debug.dump_triplet(tex_id, w, h, frame)  # type: ignore[attr-defined]

            debug_cb = _cb
        map_ms, copy_ms = self._frame_pipeline.map_and_copy_to_torch(debug_cb)

        reset_cb: Optional[Callable[[], None]] = None
        if self._auto_reset_on_black and hasattr(self, 'view') and getattr(self.view, 'camera', None) is not None:
            def _reset_camera() -> None:
                cam = getattr(self.view, 'camera', None)
                if cam is not None:
                    self._apply_camera_reset(cam)

            reset_cb = _reset_camera

        dst, convert_ms = self._frame_pipeline.convert_for_encoder(reset_camera=reset_cb)
        self._orientation_ready = self._frame_pipeline.orientation_ready
        self._black_reset_done = self._frame_pipeline.black_reset_done
        with self._enc_lock:
            encoder = self._encoder
        if encoder is None or not encoder.is_ready:
            pkt_obj = None
            encode_ms = 0.0
            pack_ms = 0.0
        else:
            pkt_obj, timings = encoder.encode(dst)
            encode_ms = timings.encode_ms
            pack_ms = timings.pack_ms
        total_ms = render_ms + blit_cpu_ms + map_ms + copy_ms + convert_ms + encode_ms + pack_ms
        pkt_bytes = None
        timings = FrameTimings(
            render_ms,
            blit_gpu_ns,
            blit_cpu_ms,
            map_ms,
            copy_ms,
            convert_ms,
            encode_ms,
            pack_ms,
            total_ms,
            pkt_bytes,
            wall_ts,
        )
        flags = 0
        # Use the worker's frame index as the capture-indexed sequence number
        seq = int(encoder.frame_index) if encoder is not None else 0
        return timings, pkt_obj, flags, seq

    # ---- C5 helpers (pure refactor; no behavior change) ---------------------
    def render_tick(self) -> float:
        t_r0 = time.perf_counter()
        # Optional animation
        if self._animate and hasattr(self.view, "camera"):
            t = time.perf_counter() - self._anim_start
            cam = self.view.camera
            if isinstance(cam, scene.cameras.TurntableCamera):
                try:
                    cam.azimuth = (self._animate_dps * t) % 360.0
                except Exception as e:
                    logger.debug("Animate(3D) failed: %s", e)
            else:
                try:
                    cx = self.width * 0.5
                    cy = self.height * 0.5
                    pan_ax = self.width * 0.05
                    pan_ay = self.height * 0.05
                    ox = pan_ax * math.sin(0.6 * t)
                    oy = pan_ay * math.cos(0.4 * t)
                    s = 1.0 + 0.08 * math.sin(0.8 * t)
                    half_w = (self.width * 0.5) / max(1e-6, s)
                    half_h = (self.height * 0.5) / max(1e-6, s)
                    x_rng = (cx + ox - half_w, cx + ox + half_w)
                    y_rng = (cy + oy - half_h, cy + oy + half_h)
                    cam.set_range(x=x_rng, y=y_rng)
                except Exception as e:
                    logger.debug("Animate(2D) failed: %s", e)
        self.drain_scene_updates()
        # If in 2D slice mode, update the slab when panning moves the viewport
        # outside the last ROI or beyond the hysteresis threshold.
        if not self.use_volume and self._scene_source is not None and self._napari_layer is not None:
            source = self._scene_source
            level = int(self._active_ms_level)
            roi = self._viewport_roi_for_level(source, level)
            planner = SliceUpdatePlanner(int(self._roi_edge_threshold))
            decision = planner.evaluate(level=level, roi=roi, last_roi=self._last_roi)
            if decision.refresh:
                z_idx = int(self._z_index or 0)
                slab = self._load_slice(source, level, z_idx)
                cam = getattr(self.view, 'camera', None)
                ctx = self._build_scene_state_context(cam)
                SceneStateApplier.apply_slice_to_layer(
                    ctx,
                    source=source,
                    slab=slab,
                    roi=roi,
                    update_contrast=False,
                )
                self._last_roi = decision.new_last_roi
        self.canvas.render()
        if self.policy_eval_requested:
            self.policy_eval_requested = False
            self._evaluate_level_policy()
        self._render_tick_required = False
        self._render_loop_started = True
        t_r1 = time.perf_counter()
        return (t_r1 - t_r0) * 1000.0

    # ---- C6 selection (napari-anchored) -------------------------------------
    def _evaluate_level_policy(self) -> None:
        """Evaluate multiscale policy inputs and perform a level switch if needed."""
        if not self._zarr_path:
            return
        try:
            source = self._ensure_scene_source()
        except Exception:
            logger.debug("ensure_scene_source failed in selection", exc_info=True)
            return

        level_indices = list(range(len(source.level_descriptors)))
        if not level_indices:
            return

        overs_map: Dict[int, float] = {}
        for lvl in level_indices:
            try:
                overs_map[int(lvl)] = float(self._oversampling_for_level(source, int(lvl)))
            except Exception:
                continue
        if not overs_map:
            return

        current = int(self._active_ms_level)
        zoom_hint = self._scene_state_coordinator.consume_zoom_intent(max_age=0.5)
        zoom_ratio = float(zoom_hint.ratio) if zoom_hint is not None else None

        now_ts = time.perf_counter()
        config = LevelPolicyConfig(
            threshold_in=float(self._level_threshold_in),
            threshold_out=float(self._level_threshold_out),
            fine_threshold=float(self._level_fine_threshold),
            hysteresis=float(self._level_hysteresis),
            cooldown_ms=float(self._level_switch_cooldown_ms),
        )
        inputs = LevelPolicyInputs(
            current_level=current,
            oversampling=overs_map,
            zoom_ratio=zoom_ratio,
            lock_level=self._lock_level,
            last_switch_ts=float(self._last_level_switch_ts),
            now_ts=now_ts,
        )
        decision = select_level(config, inputs)

        if not decision.should_switch:
            if decision.blocked_reason == 'cooldown':
                if self._log_policy_eval and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "lod.cooldown: level=%d target=%d remaining=%.1fms",
                        int(current),
                        int(decision.selected_level),
                        decision.cooldown_remaining_ms,
                    )
                return
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "lod.hold: level=%d desired=%d selected=%d overs=%s",
                    int(current),
                    int(decision.desired_level),
                    int(decision.selected_level),
                    '{' + ', '.join(f"{k}:{overs_map[k]:.2f}" for k in sorted(overs_map)) + '}',
                )
            return

        if int(decision.selected_level) < current:
            logger.info(
                "lod.zoom_in: current=%d -> selected=%d overs=%.3f reason=%s",
                current,
                int(decision.selected_level),
                overs_map.get(int(decision.selected_level), float('nan')),
                decision.action,
            )
        elif int(decision.selected_level) > current:
            logger.info(
                "lod.zoom_out: current=%d -> selected=%d overs=%.3f reason=%s",
                current,
                int(decision.selected_level),
                overs_map.get(int(decision.selected_level), float('nan')),
                decision.action,
            )

        prev = current
        try:
            self._set_level_with_budget(int(decision.selected_level), reason=decision.action)
        except _LevelBudgetError as exc:
            logger.info(
                "ms.switch: hold=%d (budget reject %s)",
                int(current),
                str(exc),
            )
            return
        except Exception:
            logger.exception(
                "ms.switch: level apply failed (reason=%s)",
                decision.action,
            )
            raise

        self._last_level_switch_ts = now_ts
        if decision.action in {"zoom-in", "zoom-out"}:
            self._last_zoom_hint_ts = now_ts

        logger.info(
            "ms.switch: %d -> %d (reason=%s) overs=%s",
            int(prev),
            int(self._active_ms_level),
            decision.action,
            '{' + ', '.join(f"{k}:{overs_map[k]:.2f}" for k in sorted(overs_map)) + '}',
        )

    # (packer is now provided by bitstream.py)

    # Removed legacy _torch_from_cupy helper (unused)

    def cleanup(self) -> None:
        try:
            self._cuda.cleanup()
        except Exception:
            logger.debug("Cleanup: CUDA interop cleanup failed", exc_info=True)
        try:
            self._gl_capture.cleanup()
        except Exception:
            logger.debug("Cleanup: GL capture cleanup failed", exc_info=True)

        with self._enc_lock:
            encoder = self._encoder
            if encoder is not None:
                encoder.shutdown()

        try:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
        except Exception:
            logger.debug("Cleanup: CUDA context pop/detach failed", exc_info=True)
        self.cuda_ctx = None

        try:
            if self.canvas is not None:
                self.canvas.close()
        except Exception:
            logger.debug("Cleanup: canvas close failed", exc_info=True)
        self.canvas = None
        self._viewer = None
        self._napari_layer = None

        self._egl.cleanup()

    def viewer_model(self) -> Optional[ViewerModel]:
        """Expose the napari ``ViewerModel`` when adapter mode is active."""
        return self._viewer

    # Adapter is always used; legacy path removed

    # --- Debug helpers -------------------------------------------------------------
    def _log_zoom_drift(self, zf: float, anc: tuple[float, float], center_world: tuple[float, float], cw: int, ch: int) -> None:
        """Instrument anchored zoom to quantify pixel-space drift.

        Computes pre/post canvas TL mapping error of the intended world center
        to the requested canvas anchor and logs one concise INFO line.
        Also applies the zoom to the camera.
        """
        try:
            cam = self.view.camera
            tr = self.view.transform * self.view.scene.transform
            ax_px = float(anc[0])
            ay_tl = float(ch) - float(anc[1])
            # pre-zoom mapping
            pre_map = tr.map([float(center_world[0]), float(center_world[1]), 0, 1])
            pre_x = float(pre_map[0]); pre_y = float(pre_map[1])
            pre_dx = pre_x - ax_px; pre_dy = pre_y - ay_tl
            # apply zoom
            cam.zoom(float(zf), center=center_world)  # type: ignore[call-arg]
            # post-zoom mapping
            tr2 = self.view.transform * self.view.scene.transform
            post_map = tr2.map([float(center_world[0]), float(center_world[1]), 0, 1])
            post_x = float(post_map[0]); post_y = float(post_map[1])
            post_dx = post_x - ax_px; post_dy = post_y - ay_tl
            rect = getattr(cam, 'rect', None)
            rect_tuple = None
            if rect is not None:
                rect_tuple = (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))
            logger.info(
                "zoom_drift: f=%.4f ancTL=(%.1f,%.1f) world=(%.3f,%.3f) preTL=(%.1f,%.1f) err_pre=(%.2f,%.2f) postTL=(%.1f,%.1f) err_post=(%.2f,%.2f) cam.rect=%s canvas=%dx%d",
                float(zf), float(ax_px), float(ay_tl), float(center_world[0]), float(center_world[1]),
                pre_x, pre_y, pre_dx, pre_dy, post_x, post_y, post_dx, post_dy, str(rect_tuple), int(cw), int(ch)
            )
        except Exception:
            logger.debug("zoom_drift instrumentation failed", exc_info=True)

    def _log_pan_mapping(self, dx_px: float, dy_px: float, cw: int, ch: int) -> None:
        """Instrument pixel-space pan mapping to world delta.

        Logs canvas center, requested pixel delta, mapped world delta, and camera center after applying pan.
        Also applies the pan to the camera.
        """
        try:
            cam = self.view.camera
            # Compute world delta using transform at canvas center
            tr = self.view.transform * self.view.scene.transform
            cx_px = float(cw) * 0.5
            cy_px = float(ch) * 0.5
            p0 = tr.imap((cx_px, cy_px))
            p1 = tr.imap((cx_px + dx_px, cy_px + dy_px))
            dwx = float(p1[0] - p0[0])
            dwy = float(p1[1] - p0[1])
            c0 = getattr(cam, 'center', None)
            if isinstance(c0, (tuple, list)) and len(c0) >= 2:
                cam.center = (float(c0[0]) - dwx, float(c0[1]) - dwy)  # type: ignore[attr-defined]
            else:
                cam.center = (-dwx, -dwy)  # type: ignore[attr-defined]
            c1 = getattr(cam, 'center', None)
            logger.info(
                "pan_map: dpx=(%.2f,%.2f) world=(%.4f,%.4f) center_before=%s center_after=%s canvas=%dx%d",
                float(dx_px), float(dy_px), dwx, dwy, str(c0), str(c1), int(cw), int(ch)
            )
        except Exception:
            logger.debug("pan instrumentation failed", exc_info=True)
