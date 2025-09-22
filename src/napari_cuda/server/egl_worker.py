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
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
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
from napari_cuda.server.lod import stabilize_level
from napari_cuda.server.config import ServerCtx
from napari_cuda.server.rendering.gl_capture import GLCapture
from napari_cuda.server.rendering.cuda_interop import CudaInterop
from napari_cuda.server.rendering.egl_context import EglContext
from napari_cuda.server.rendering.encoder import Encoder
from napari_cuda.server.state_machine import (
    CameraCommand,
    PendingSceneUpdates,
    SceneStateMachine,
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

        self._gl_capture = GLCapture(self.width, self.height)
        self._cuda = CudaInterop(self.width, self.height)

        self._encoder: Optional[Encoder] = None
        self._enc_input_fmt: str = "YUV444"

        # Track current data extents in world units (W,H)
        self._data_wh: tuple[int, int] = (int(width), int(height))
        # Gate pixel streaming until orientation/camera are settled and a non-black frame is observed
        self._orientation_ready: bool = False

        # Encoder access synchronization (reset vs encode across threads)
        self._enc_lock = threading.Lock()

        # Atomic state application
        self._state_lock = threading.Lock()
        self._scene_state_machine = SceneStateMachine()
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
        self._eval_after_render: bool = False
        # Policy init happens naturally; no deferral needed
        # One-shot safety: reset camera if initial frames are black
        try:
            self._auto_reset_on_black = env_bool('NAPARI_CUDA_AUTO_RESET_ON_BLACK', True)
        except Exception:
            self._auto_reset_on_black = True
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
        # Log format/size once for clarity
        logger.info(
            "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s, zarr=%s",
            self.width, self.height, getattr(self, '_enc_input_fmt', 'unknown'), self.fps, self._animate,
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
        self._scene_state_machine.queue_multiscale_level(int(level), str(path) if path else None)
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


    def _plane_wh_for_level(self, source: ZarrSceneSource, level: int) -> tuple[int, int]:
        descriptor = source.level_descriptors[level]
        axes = source.axes
        axes_lower = [str(ax).lower() for ax in axes]
        try:
            y_pos = axes_lower.index('y')
        except ValueError:
            y_pos = max(0, len(descriptor.shape) - 2)
        try:
            x_pos = axes_lower.index('x')
        except ValueError:
            x_pos = max(0, len(descriptor.shape) - 1)
        h = int(descriptor.shape[y_pos]) if 0 <= y_pos < len(descriptor.shape) else int(descriptor.shape[-2])
        w = int(descriptor.shape[x_pos]) if 0 <= x_pos < len(descriptor.shape) else int(descriptor.shape[-1])
        return h, w

    def _plane_scale_for_level(self, source: ZarrSceneSource, level: int) -> tuple[float, float]:
        axes = source.axes
        axes_lower = [str(ax).lower() for ax in axes]
        scale = source.level_scale(level)
        try:
            y_pos = axes_lower.index('y')
        except ValueError:
            y_pos = max(0, len(scale) - 2)
        try:
            x_pos = axes_lower.index('x')
        except ValueError:
            x_pos = max(0, len(scale) - 1)
        sy = float(scale[y_pos]) if 0 <= y_pos < len(scale) else float(scale[-2])
        sx = float(scale[x_pos]) if 0 <= x_pos < len(scale) else float(scale[-1])
        return sy, sx

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
                h, w = self._plane_wh_for_level(source, level)
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
        # Attempt to reuse cached ROI when the view transform has not changed
        view = getattr(self, "view", None)
        xform_sig: Optional[tuple[float, ...]] = None
        if view is not None and hasattr(view, "scene") and hasattr(view.scene, "transform"):
            try:
                xform = view.scene.transform
                if hasattr(xform, "matrix"):
                    mat = getattr(xform, "matrix")
                    # Flatten a small signature; convert to float tuple to make hashable and stable
                    xform_sig = tuple(float(v) for v in mat.ravel())
            except Exception:
                xform_sig = None

        cached = self._roi_cache.get(int(level))
        if cached is not None and cached[0] is not None and xform_sig is not None and cached[0] == xform_sig:
            # Cached ROI is valid for this transform signature
            return cached[1]
        # During the first few frames after a level switch, force full frame for active level.
        # However, policy/oversampling computation should ignore this to avoid oscillation.
        # No fullframe overrides; compute ROI from viewport and guarantee containment
        h, w = self._plane_wh_for_level(source, level)
        sy, sx = self._plane_scale_for_level(source, level)
        if self.view is None or not hasattr(self.view, 'camera'):
            self._log_roi_fallback(level=level, reason="no-view", plane_h=h, plane_w=w, scale=(sy, sx))
            return SliceROI(0, h, 0, w)
        cam = self.view.camera
        if not isinstance(cam, scene.cameras.PanZoomCamera):
            ensured = self._ensure_panzoom_camera(reason="roi-request")
            cam = ensured or cam
        if not isinstance(cam, scene.cameras.PanZoomCamera):
            self._log_roi_fallback(level=level, reason="no-panzoom", plane_h=h, plane_w=w, scale=(sy, sx))
            return SliceROI(0, h, 0, w)
        sx_world = max(1e-12, float(sx))
        sy_world = max(1e-12, float(sy))
        try:
            bounds = self._viewport_world_bounds()
            if bounds is None:
                self._log_roi_fallback(
                    level=level,
                    reason="no-bounds",
                    plane_h=h,
                    plane_w=w,
                    scale=(sy_world, sx_world),
                )
                return SliceROI(0, h, 0, w)
            x0, x1, y0, y1 = bounds
            x_start = int(math.floor(min(x0, x1) / sx_world))
            x_stop = int(math.ceil(max(x0, x1) / sx_world))
            y_start = int(math.floor(min(y0, y1) / sy_world))
            y_stop = int(math.ceil(max(y0, y1) / sy_world))
            # Keep a copy of viewport index bounds so we can guarantee containment later
            vp_x_start, vp_x_stop = x_start, x_stop
            vp_y_start, vp_y_stop = y_start, y_stop
            roi = SliceROI(y_start, y_stop, x_start, x_stop).clamp(h, w)
            if (self._log_layer_debug or getattr(self, '_log_roi_anchor', False)) and logger.isEnabledFor(logging.INFO):
                # Log initial world bounds and resulting index ROI
                cx_i = 0.5 * (roi.x_start + roi.x_stop)
                cy_i = 0.5 * (roi.y_start + roi.y_stop)
                logger.info(
                    "roi.initial: level=%d world=(x=[%.2f,%.2f], y=[%.2f,%.2f]) scale=(%.6f,%.6f) index=(y=%d:%d x=%d:%d) center_i=(%.2f,%.2f)",
                    int(level), float(x0), float(x1), float(y0), float(y1), float(sy_world), float(sx_world),
                    int(roi.y_start), int(roi.y_stop), int(roi.x_start), int(roi.x_stop),
                    float(cx_i), float(cy_i)
                )
            # Align ROI to chunk boundaries to avoid sub-chunk phasing artifacts when panning
            if (not for_policy) and getattr(self, '_roi_align_chunks', True):
                try:
                    arr = source.get_level(level)
                    chunks = getattr(arr, 'chunks', None)
                    axes = source.axes
                    axes_lower = [str(ax).lower() for ax in axes]
                    if chunks is not None:
                        if 'y' in axes_lower:
                            y_pos = axes_lower.index('y')
                        else:
                            y_pos = max(0, len(chunks) - 2)
                        if 'x' in axes_lower:
                            x_pos = axes_lower.index('x')
                        else:
                            x_pos = max(0, len(chunks) - 1)
                        cy = int(chunks[y_pos]) if 0 <= y_pos < len(chunks) else 1
                        cx = int(chunks[x_pos]) if 0 <= x_pos < len(chunks) else 1
                        cy = max(1, cy)
                        cx = max(1, cx)
                        # Snap start to floor(chunk) and stop to ceil(chunk), with small symmetric padding
                        ys = (roi.y_start // cy) * cy
                        ye = ((roi.y_stop + cy - 1) // cy) * cy
                        xs = (roi.x_start // cx) * cx
                        xe = ((roi.x_stop + cx - 1) // cx) * cx
                        pad_chunks = int(getattr(self, '_roi_pad_chunks', 1))
                        ys = max(0, ys - pad_chunks * cy)
                        ye = min(h, ye + pad_chunks * cy)
                        xs = max(0, xs - pad_chunks * cx)
                        xe = min(w, xe + pad_chunks * cx)
                        pre = roi
                        roi = SliceROI(ys, ye, xs, xe).clamp(h, w)
                        if (self._log_layer_debug or getattr(self, '_log_roi_anchor', False)) and logger.isEnabledFor(logging.INFO):
                            logger.info(
                                "roi.aligned: level=%d chunks=(y=%d,x=%d) pre=(y=%d:%d x=%d:%d) -> aligned=(y=%d:%d x=%d:%d)",
                                int(level), int(cy), int(cx),
                                int(pre.y_start), int(pre.y_stop), int(pre.x_start), int(pre.x_stop),
                                int(roi.y_start), int(roi.y_stop), int(roi.x_start), int(roi.x_stop)
                            )
                except Exception:
                    logger.debug('roi chunk alignment failed', exc_info=True)
            # Apply small-move hysteresis only for rendering, not for policy decisions
            if not for_policy:
                roi_log = getattr(self, '_roi_log_state', None)
                if isinstance(roi_log, dict):
                    prev_logged = roi_log.get(int(level))
                    if prev_logged is not None:
                        prev_roi, _ = prev_logged
                        thr = int(getattr(self, '_roi_edge_threshold', 4))
                        # Only accept prev_roi when it still covers the current viewport
                        prev_covers_view = (
                            int(prev_roi.y_start) <= int(vp_y_start) and int(prev_roi.y_stop) >= int(vp_y_stop) and
                            int(prev_roi.x_start) <= int(vp_x_start) and int(prev_roi.x_stop) >= int(vp_x_stop)
                        )
                        if (
                            abs(roi.y_start - prev_roi.y_start) < thr and
                            abs(roi.y_stop  - prev_roi.y_stop)  < thr and
                            abs(roi.x_start - prev_roi.x_start) < thr and
                            abs(roi.x_stop  - prev_roi.x_stop)  < thr and
                            prev_covers_view
                        ):
                            roi = prev_roi

            # Ensure ROI fully contains the current viewport to avoid black edges during pan
            if not for_policy and getattr(self, '_roi_ensure_contains_viewport', True):
                try:
                    ys = min(int(roi.y_start), int(vp_y_start))
                    ye = max(int(roi.y_stop),  int(vp_y_stop))
                    xs = min(int(roi.x_start), int(vp_x_start))
                    xe = max(int(roi.x_stop),  int(vp_x_stop))
                    # If chunk alignment is enabled, expand to chunk boundaries (outward only)
                    if getattr(self, '_roi_align_chunks', True):
                        try:
                            arr = source.get_level(level)
                            chunks = getattr(arr, 'chunks', None)
                            if chunks is not None:
                                axes = source.axes
                                axes_lower = [str(ax).lower() for ax in axes]
                                y_pos = axes_lower.index('y') if 'y' in axes_lower else max(0, len(chunks) - 2)
                                x_pos = axes_lower.index('x') if 'x' in axes_lower else max(0, len(chunks) - 1)
                                cy = int(chunks[y_pos]) if 0 <= y_pos < len(chunks) else 1
                                cx = int(chunks[x_pos]) if 0 <= x_pos < len(chunks) else 1
                                cy = max(1, cy)
                                cx = max(1, cx)
                                ys = (ys // cy) * cy
                                ye = ((ye + cy - 1) // cy) * cy
                                xs = (xs // cx) * cx
                                xe = ((xe + cx - 1) // cx) * cx
                        except Exception:
                            logger.debug('roi ensure-contains: chunk align failed', exc_info=True)
                    pre = roi
                    roi = SliceROI(ys, ye, xs, xe).clamp(h, w)
                    if (self._log_layer_debug or getattr(self, '_log_roi_anchor', False)) and logger.isEnabledFor(logging.INFO):
                        logger.info(
                            'roi.ensure_viewport: level=%d pre=(y=%d:%d x=%d:%d) vp=(y=%d:%d x=%d:%d) -> final=(y=%d:%d x=%d:%d)',
                            int(level), int(pre.y_start), int(pre.y_stop), int(pre.x_start), int(pre.x_stop),
                            int(vp_y_start), int(vp_y_stop), int(vp_x_start), int(vp_x_stop),
                            int(roi.y_start), int(roi.y_stop), int(roi.x_start), int(roi.x_stop)
                        )
                except Exception:
                    logger.debug('roi ensure-contains failed', exc_info=True)

            # Center-anchored reconciliation removed; placement uses layer.translate now
            # (ROI verbose logging removed)
            if roi.is_empty():
                self._log_roi_fallback(level=level, reason="empty", plane_h=h, plane_w=w, scale=(sy_world, sx_world))
                return SliceROI(0, h, 0, w)
            # Update cache against the current transform signature
            if xform_sig is not None:
                self._roi_cache[int(level)] = (xform_sig, roi)
            return roi
        except Exception:
            if self._log_layer_debug:
                logger.exception(
                    "viewport ROI computation failed; returning full frame (level=%d dims=%dx%d)",
                    level,
                    h,
                    w,
                )
            return SliceROI(0, h, 0, w)

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
        h, w = self._plane_wh_for_level(source, level)
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
        descriptor = source.level_descriptors[level]
        axes = source.axes
        axes_lower = [str(ax).lower() for ax in axes]

        # Preserve current Z (and other indices) across level switches.
        step_hint: Optional[Sequence[int]] = None
        # 1) Use last dims step received from intents, if any
        if self._last_step is not None:
            step_hint = tuple(int(x) for x in self._last_step)
        # 2) Else fall back to viewer's dims if available
        if step_hint is None and self._viewer is not None:
            try:
                step_hint = tuple(int(x) for x in self._viewer.dims.current_step)
            except Exception:
                step_hint = None
        # 3) Else use source current step
        if step_hint is None:
            step_hint = source.current_step

        # Proportional Z remap only when switching between different levels (depth may change)
        try:
            if prev_level is not None and int(prev_level) != int(level) and 'z' in axes_lower:
                zi = axes_lower.index('z')
                try:
                    prev_desc = source.level_descriptors[int(prev_level)]
                    z_src = int(prev_desc.shape[zi]) if 0 <= zi < len(prev_desc.shape) else int(prev_desc.shape[0])
                except Exception:
                    z_src = None
                try:
                    z_tgt = int(descriptor.shape[zi]) if 0 <= zi < len(descriptor.shape) else int(descriptor.shape[0])
                except Exception:
                    z_tgt = None
                if (z_src is not None and z_tgt is not None and z_src > 0 and z_tgt > 0):
                    cur_z = None
                    # Prefer explicit step_hint (e.g., from a dims intent) over stale cached z_index
                    if step_hint is not None and len(step_hint) > zi:
                        cur_z = int(step_hint[zi])
                    elif self._z_index is not None:
                        cur_z = int(self._z_index)
                    else:
                        cur_z = 0
                    if z_src <= 1:
                        new_z = 0
                    else:
                        new_z = int(round(float(cur_z) * float(max(0, z_tgt - 1)) / float(max(1, z_src - 1))))
                    new_z = max(0, min(int(new_z), int(z_tgt - 1)))
                    # (diagnostic logging removed)
                    sh = list(step_hint) if step_hint is not None else [0] * len(descriptor.shape)
                    if len(sh) < len(descriptor.shape):
                        sh.extend([0] * (len(descriptor.shape) - len(sh)))
                    sh[zi] = int(new_z)
                    step_hint = tuple(int(x) for x in sh)
        except Exception:
            logger.debug("proportional Z remap failed", exc_info=True)

        with self._state_lock:
            step = source.set_current_level(level, step=step_hint)

        self._active_ms_level = level
        # Keep viewer slider aligned with the applied level's clamped step
        # IMPORTANT: set dims.range first to avoid clamping against previous level
        if self._viewer is not None:
            try:
                self._set_dims_range_for_level(source, level)
                self._viewer.dims.current_step = tuple(int(x) for x in step)
                self._last_step = tuple(int(x) for x in step)
            except Exception:
                logger.debug("apply_level: syncing viewer dims failed", exc_info=True)
        self._zarr_level = descriptor.path or None
        self._zarr_shape = descriptor.shape
        self._zarr_axes = ''.join(axes)
        self._zarr_dtype = str(source.dtype)

        if axes_lower and 'z' in axes_lower and len(step) > axes_lower.index('z'):
            self._z_index = int(step[axes_lower.index('z')])
        elif step:
            self._z_index = int(step[0])
        else:
            self._z_index = None

        # (diagnostic logging removed)

        if self._viewer is not None:
            # Ensure range is applied before setting step to avoid transient clamp
            self._set_dims_range_for_level(source, level)
            self._viewer.dims.current_step = tuple(int(x) for x in step)

        # Ensure contrast cache/limits are updated regardless of napari layer presence
        contrast = source.ensure_contrast(level=level)
        self._zarr_clim = contrast

        overs = None
        try:
            overs = self._oversampling_for_level(source, level)
        except Exception:
            logger.debug("oversampling compute failed", exc_info=True)

        layer = self._napari_layer
        update_layer = layer is not None

        if self.use_volume:
            volume = self._get_level_volume(source, level)
            d, h, w = int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])
            self._data_wh = (w, h)
            self._data_d = d
            if update_layer:
                layer.data = volume
                layer.contrast_limits = [float(contrast[0]), float(contrast[1])]
            self._log_layer_assignment('volume', level, None, (d, h, w), contrast)
        else:
            z_idx = self._z_index or 0
            # Apply per-level scale to the layer BEFORE computing ROI/slab so
            # world->index mapping is consistent across level changes.
            try:
                sy, sx = self._plane_scale_for_level(source, level)
                if update_layer:
                    layer.scale = (sy, sx)
            except Exception:
                logger.debug("apply_level: setting 2D layer scale pre-slab failed", exc_info=True)
                sy = sx = 1.0
            slab = self._load_slice(
                source,
                level,
                z_idx,
            )
            # Place the ROI slab at its world offset so the view doesn't nudge
            if update_layer:
                try:
                    roi_for_layer = None
                    try:
                        if getattr(self, '_last_roi', None) is not None and int(self._last_roi[0]) == int(level):
                            roi_for_layer = self._last_roi[1]
                    except Exception:
                        roi_for_layer = None
                    if roi_for_layer is not None:
                        yoff = float(roi_for_layer.y_start) * float(max(1e-12, sy))
                        xoff = float(roi_for_layer.x_start) * float(max(1e-12, sx))
                        layer.translate = (yoff, xoff)
                    else:
                        layer.translate = (0.0, 0.0)
                except Exception:
                    logger.debug("apply_level: setting 2D layer translate failed", exc_info=True)
            # (slab statistics logging removed)
            if update_layer:
                layer.data = slab
                # Optionally keep existing contrast limits stable across pans/level switches
                try:
                    if not getattr(self, '_sticky_contrast', True):
                        smin = float(np.nanmin(slab)) if hasattr(np, 'nanmin') else float(np.min(slab))
                        smax = float(np.nanmax(slab)) if hasattr(np, 'nanmax') else float(np.max(slab))
                        if not math.isfinite(smin) or not math.isfinite(smax) or smax <= smin:
                            layer.contrast_limits = [float(contrast[0]), float(contrast[1])]
                        else:
                            if 0.0 <= smin <= 1.0 and 0.0 <= smax <= 1.1:
                                layer.contrast_limits = [0.0, 1.0]
                            else:
                                layer.contrast_limits = [smin, smax]
                except Exception:
                    # If sticky, leave as-is; else fallback to provided contrast
                    if not getattr(self, '_sticky_contrast', True):
                        layer.contrast_limits = [float(contrast[0]), float(contrast[1])]
            h, w = self._plane_wh_for_level(source, level)
            self._data_wh = (w, h)
            try:
                if not getattr(self, '_preserve_view_on_switch', True):
                    if self.view is not None and hasattr(self.view, 'camera'):
                        world_w = float(w) * float(max(1e-12, sx))
                        world_h = float(h) * float(max(1e-12, sy))
                        self.view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
            except Exception:
                logger.debug("apply_level: camera set_range failed", exc_info=True)
            self._log_layer_assignment('slice', level, self._z_index, (h, w), contrast)

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
            self._maybe_select_level()
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
            self._enc_input_fmt = encoder.input_format

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
        self._apply_pending_state()
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
        self._scene_state_machine.queue_scene_state(normalized)

    def process_camera_commands(self, commands: Sequence[CameraCommand]) -> None:
        if not commands:
            return
        # Interaction observed (kept for future metrics)
        self._user_interaction_seen = True
        logger.debug("worker processing %d camera command(s)", len(commands))
        cam = getattr(self.view, 'camera', None)
        if cam is None:
            # Still capture zoom ratio so policy can react once camera is available
            for cmd in commands:
                if cmd.kind == 'zoom' and cmd.factor and cmd.factor > 0.0:
                    self._scene_state_machine.record_zoom_intent(float(cmd.factor))
            return

        canvas_wh: tuple[int, int]
        if self.canvas is not None and hasattr(self.canvas, 'size'):
            canvas_wh = (int(self.canvas.size[0]), int(self.canvas.size[1]))
        else:
            canvas_wh = (self.width, self.height)

        self._last_interaction_ts = time.perf_counter()
        self._mark_render_tick_needed()

        touched = False
        policy_touch = False

        for cmd in commands:
            kind = cmd.kind
            if kind == 'zoom':
                factor = float(cmd.factor) if cmd.factor is not None else 0.0
                if factor <= 0.0:
                    continue
                # Gate policy switching by env; still apply camera zoom
                anchor = cmd.anchor_px or (canvas_wh[0] * 0.5, canvas_wh[1] * 0.5)
                if isinstance(cam, scene.cameras.TurntableCamera):
                    camops.apply_zoom_3d(cam, factor)
                else:
                    camops.apply_zoom_2d(cam, factor, (float(anchor[0]), float(anchor[1])), canvas_wh, self.view)
                touched = True
                policy_touch = True
                if self._debug_zoom_drift and logger.isEnabledFor(logging.INFO):
                    logger.info("command zoom factor=%.4f anchor=(%.1f,%.1f)", factor, float(anchor[0]), float(anchor[1]))
            elif kind == 'pan':
                dx = float(cmd.dx_px)
                dy = float(cmd.dy_px)
                if dx == 0.0 and dy == 0.0:
                    continue
                if isinstance(cam, scene.cameras.TurntableCamera):
                    camops.apply_pan_3d(cam, dx, dy, canvas_wh)
                else:
                    camops.apply_pan_2d(cam, dx, dy, canvas_wh, self.view)
                touched = True
                policy_touch = True
                if self._debug_pan and logger.isEnabledFor(logging.INFO):
                    logger.info("command pan dx=%.2f dy=%.2f", dx, dy)
            elif kind == 'orbit':
                daz = float(cmd.d_az_deg)
                delv = float(cmd.d_el_deg)
                if not isinstance(cam, scene.cameras.TurntableCamera):
                    continue
                camops.apply_orbit(cam, daz, delv)
                touched = True
                policy_touch = True
                if self._debug_orbit and logger.isEnabledFor(logging.INFO):
                    logger.info("command orbit daz=%.2f del=%.2f", daz, delv)
            elif kind == 'reset':
                self._apply_camera_reset(cam)
                touched = True
                if self._debug_reset and logger.isEnabledFor(logging.INFO):
                    logger.info("command reset view")

        if touched:
            # Schedule a render tick and a single post-render selection if camera actually changed
            self._mark_render_tick_needed()
            if policy_touch:
                self._eval_after_render = True
            # No additional camera clamping beyond core mapping

    def _apply_camera_reset(self, cam) -> None:
        if cam is None or not hasattr(cam, 'set_range'):
            return
        w, h = getattr(self, '_data_wh', (self.width, self.height))
        d = getattr(self, '_data_d', None)
        # Convert pixel dims to world extents using current level scale
        sx = sy = 1.0
        try:
            if self._scene_source is not None and not self.use_volume:
                sy, sx = self._plane_scale_for_level(self._scene_source, int(self._active_ms_level))
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

    def _apply_pending_state(self) -> None:
        updates: PendingSceneUpdates = self._scene_state_machine.drain_pending_updates()

        if updates.display_mode is not None:
            try:
                self._apply_ndisplay_switch(updates.display_mode)
            except Exception:
                logger.exception("ndisplay switch failed")

        if updates.multiscale is not None:
            lvl = int(updates.multiscale.level)
            pth = updates.multiscale.path
            try:
                self._apply_multiscale_switch(lvl, pth)
            except Exception:
                logger.exception("multiscale switch failed")

        state = updates.scene_state
        if state is None:
            return

        cam = getattr(self.view, 'camera', None)
        # Update 2D slice when in slice mode and a new index is provided
        if not bool(self.use_volume) and state.current_step is not None:
            if self._viewer is not None:
                try:
                    steps = tuple(int(x) for x in state.current_step)
                    # Remember last dims for preserving indices across switches
                    self._last_step = steps
                    # Compute intended z index based on axes
                    z_new = None
                    if self._scene_source is not None:
                        axes = self._scene_source.axes
                        if 'z' in axes:
                            zi = axes.index('z')
                            if zi < len(steps):
                                z_new = int(steps[zi])
                    # Apply dims to viewer
                    self._viewer.dims.current_step = steps
                    # If z changed, update source/viewer and slab immediately and mark a render
                    if z_new is not None and (self._z_index is None or int(z_new) != int(self._z_index)):
                        try:
                            source = self._ensure_scene_source()
                            # Apply the z change directly without re-running level selection
                            try:
                                axes = source.axes
                                zi = axes.index('z') if 'z' in axes else 0
                                # Build a step with the new z, preserving other indices when available
                                base = list(source.current_step or steps)
                                if len(base) < len(source.level_shape(self._active_ms_level)):
                                    base = base + [0] * (len(source.level_shape(self._active_ms_level)) - len(base))
                                base[zi] = int(z_new)
                                with self._state_lock:
                                    _ = source.set_current_level(self._active_ms_level, step=tuple(int(x) for x in base))
                            except Exception:
                                logger.debug("apply_state: set_current_level(z) failed; will proceed to load slab", exc_info=True)
                            # On Z change, keep ROI-based slab to preserve world alignment
                            slab = self._load_slice(source, self._active_ms_level, int(z_new))
                            layer = getattr(self, '_napari_layer', None)
                            if layer is not None:
                                layer.data = slab
                                # Ensure translate matches the ROI placement
                                try:
                                    sy, sx = self._plane_scale_for_level(source, int(self._active_ms_level))
                                except Exception:
                                    sy = sx = 1.0
                                try:
                                    roi_for_layer = None
                                    if getattr(self, '_last_roi', None) is not None and int(self._last_roi[0]) == int(self._active_ms_level):
                                        roi_for_layer = self._last_roi[1]
                                    if roi_for_layer is not None:
                                        yoff = float(roi_for_layer.y_start) * float(max(1e-12, sy))
                                        xoff = float(roi_for_layer.x_start) * float(max(1e-12, sx))
                                        layer.translate = (yoff, xoff)
                                    else:
                                        layer.translate = (0.0, 0.0)
                                except Exception:
                                    logger.debug("apply_state: setting layer.translate on z-change failed", exc_info=True)
                                try:
                                    layer.visible = True
                                    layer.opacity = 1.0
                                    layer.blending = 'opaque'
                                except Exception:
                                    logger.debug("apply_state: ensuring layer visibility failed", exc_info=True)
                                # Update contrast limits for the new Z slice unless sticky
                                try:
                                    if not getattr(self, '_sticky_contrast', True):
                                        smin = float(np.nanmin(slab)) if hasattr(np, 'nanmin') else float(np.min(slab))
                                        smax = float(np.nanmax(slab)) if hasattr(np, 'nanmax') else float(np.max(slab))
                                        if not math.isfinite(smin) or not math.isfinite(smax) or smax <= smin:
                                            layer.contrast_limits = [0.0, 1.0]
                                        else:
                                            if 0.0 <= smin <= 1.0 and 0.0 <= smax <= 1.1:
                                                layer.contrast_limits = [0.0, 1.0]
                                            else:
                                                layer.contrast_limits = [smin, smax]
                                except Exception:
                                    logger.debug("apply_state: contrast update failed", exc_info=True)
                            vis = getattr(self, '_visual', None)
                            if vis is not None and hasattr(vis, 'set_data'):
                                vis.set_data(slab)  # type: ignore[attr-defined]
                            h, w = int(slab.shape[0]), int(slab.shape[1])
                            cam = getattr(self.view, 'camera', None)
                            # Ensure camera sees the new slice only if not preserving view
                            if cam is not None and hasattr(cam, 'set_range') and not getattr(self, '_preserve_view_on_switch', True):
                                try:
                                    sy, sx = (1.0, 1.0)
                                    if self._scene_source is not None:
                                        sy, sx = self._plane_scale_for_level(self._scene_source, int(self._active_ms_level))
                                    world_w = max(1.0, float(w) * float(max(1e-12, sx)))
                                    world_h = max(1.0, float(h) * float(max(1e-12, sy)))
                                    cam.set_range(x=(0.0, world_w), y=(0.0, world_h))
                                except Exception:
                                    logger.debug("apply_state: camera set_range on z-change failed", exc_info=True)
                            self._data_wh = (w, h)
                            self._z_index = int(z_new)
                            # Do not force fullframe horizon on Z; keep ROI for stability
                            # Optionally force IDR on Z for immediate visibility on clients
                            if getattr(self, '_idr_on_z', False):
                                self._request_encoder_idr()
                            self._notify_scene_refresh()
                            self._mark_render_tick_needed()
                            if self._log_layer_debug:
                                logger.info("apply_state: z updated -> %d (level=%d)", int(self._z_index), int(self._active_ms_level))
                        except Exception:
                            logger.debug("apply_state: immediate slab update failed", exc_info=True)
                    else:
                        # Even if z didn’t change, ensure a render to reflect dims
                        self._mark_render_tick_needed()
                except Exception:
                    logger.debug("apply_state: viewer dims update failed", exc_info=True)
            elif self._scene_source is not None:
                axes = self._scene_source.axes
                if 'z' in axes and len(state.current_step) > axes.index('z'):
                    try:
                        z_idx = int(state.current_step[axes.index('z')])
                    except (TypeError, ValueError):
                        logger.debug("apply_state: invalid z index in current_step=%r", state.current_step)
                        z_idx = self._z_index or 0
                else:
                    z_idx = self._z_index or 0
                if self._z_index is None or int(z_idx) != int(self._z_index):
                    source = self._ensure_scene_source()
                    # Directly set source/viewer to requested z without level re-apply
                    try:
                        axes = source.axes
                        zi = axes.index('z') if 'z' in axes else 0
                        base = list(source.current_step or [0] * len(source.level_shape(self._active_ms_level)))
                        if len(base) < len(source.level_shape(self._active_ms_level)):
                            base = base + [0] * (len(source.level_shape(self._active_ms_level)) - len(base))
                        base[zi] = int(z_idx)
                        with self._state_lock:
                            _ = source.set_current_level(self._active_ms_level, step=tuple(int(x) for x in base))
                    except Exception:
                        logger.debug("apply_state: set_current_level(z) failed (no viewer)", exc_info=True)
                    slab = self._load_slice(source, self._active_ms_level, int(z_idx))
                    vis = getattr(self, '_visual', None)
                    if vis is not None and hasattr(vis, 'set_data'):
                        vis.set_data(slab)  # type: ignore[attr-defined]
                    h, w = int(slab.shape[0]), int(slab.shape[1])
                    if getattr(self, '_data_wh', None) != (w, h) and cam is not None and hasattr(cam, 'set_range'):
                        cam.set_range(x=(0, w), y=(0, h))
                    self._data_wh = (w, h)
                    self._z_index = int(z_idx)
                    self._notify_scene_refresh()

        # Volume render parameters in 3D mode
        if bool(self.use_volume):
            vis = getattr(self, '_visual', None)
            if vis is not None:
                m = getattr(state, 'volume_mode', None)
                if m and hasattr(vis, 'method'):
                    mm = str(m).lower()
                    if mm in ('mip', 'translucent', 'iso'):
                        vis.method = mm  # type: ignore[attr-defined]
                cm = getattr(state, 'volume_colormap', None)
                if cm and hasattr(vis, 'cmap'):
                    name = str(cm).lower()
                    if name == 'gray':
                        name = 'grays'
                    vis.cmap = name  # type: ignore[attr-defined]
                cl = getattr(state, 'volume_clim', None)
                if isinstance(cl, tuple) and len(cl) >= 2 and hasattr(vis, 'clim'):
                    lo = float(cl[0]); hi = float(cl[1])
                    if hi < lo:
                        lo, hi = hi, lo
                    vis.clim = (lo, hi)  # type: ignore[attr-defined]
                op = getattr(state, 'volume_opacity', None)
                if op is not None and hasattr(vis, 'opacity'):
                    vis.opacity = float(max(0.0, min(1.0, float(op))))  # type: ignore[attr-defined]
                ss = getattr(state, 'volume_sample_step', None)
                if ss is not None and hasattr(vis, 'relative_step_size'):
                    vis.relative_step_size = float(max(0.1, min(4.0, float(ss))))  # type: ignore[attr-defined]

        if cam is None:
            self._maybe_select_level()
            return

        # Legacy absolute camera fields
        if state.center is not None and hasattr(cam, 'center'):
            cam.center = state.center  # type: ignore[attr-defined]
        if state.zoom is not None and hasattr(cam, 'zoom'):
            cam.zoom = state.zoom  # type: ignore[attr-defined]
        if state.angles is not None and hasattr(cam, 'angles'):
            cam.angles = state.angles  # type: ignore[attr-defined]

        # Evaluate policy only if the absolute state actually changed
        if self._scene_state_machine.update_state_signature(state):
            self._maybe_select_level()

    # --- Public coalesced toggles ------------------------------------------------
    def request_ndisplay(self, ndisplay: int) -> None:
        """Queue a 2D/3D view switch to apply on the render thread."""
        self._scene_state_machine.queue_display_mode(3 if int(ndisplay) >= 3 else 2)

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
        render_ms = self._render_frame_and_maybe_eval()
        # Blit timing (GPU + CPU)
        t_b0 = time.perf_counter()
        blit_gpu_ns = self._capture_blit_gpu_ns()
        t_b1 = time.perf_counter()
        blit_cpu_ms = (t_b1 - t_b0) * 1000.0
        map_ms, copy_ms = self._map_and_copy_to_torch()
        dst, convert_ms = self._convert_for_encoder()
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
    def _render_frame_and_maybe_eval(self) -> float:
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
        self._apply_pending_state()
        # If in 2D slice mode, update the slab when panning moves the viewport
        # outside the last ROI or beyond the hysteresis threshold.
        try:
            if not self.use_volume and self._scene_source is not None and self._napari_layer is not None:
                source = self._scene_source
                level = int(self._active_ms_level)
                # Compute current render ROI (render-only path guarantees viewport containment)
                roi = self._viewport_roi_for_level(source, level)
                # Decide whether to fetch a new slab
                need_update = False
                last = getattr(self, '_last_roi', None)
                thr = int(getattr(self, '_roi_edge_threshold', 4))
                if last is None or int(last[0]) != level:
                    need_update = True
                else:
                    prev = last[1]
                    if (
                        abs(int(roi.y_start) - int(prev.y_start)) >= thr or
                        abs(int(roi.y_stop)  - int(prev.y_stop))  >= thr or
                        abs(int(roi.x_start) - int(prev.x_start)) >= thr or
                        abs(int(roi.x_stop)  - int(prev.x_stop))  >= thr
                    ):
                        need_update = True
                if need_update:
                    z_idx = int(self._z_index or 0)
                    slab = self._load_slice(source, level, z_idx)
                    # Place slab using current ROI
                    try:
                        sy, sx = self._plane_scale_for_level(source, level)
                    except Exception:
                        sy = sx = 1.0
                    try:
                        self._napari_layer.translate = (
                            float(roi.y_start) * float(max(1e-12, sy)),
                            float(roi.x_start) * float(max(1e-12, sx)),
                        )
                    except Exception:
                        logger.debug("render-frame: set translate failed", exc_info=True)
                    self._napari_layer.data = slab
                    self._last_roi = (level, roi)
        except Exception:
            logger.debug("render-frame: slab update on pan failed", exc_info=True)
        self.canvas.render()
        if self._eval_after_render:
            try:
                self._eval_after_render = False
                self._maybe_select_level()
            except Exception:
                logger.debug("post-render policy eval failed", exc_info=True)
        self._render_tick_required = False
        self._render_loop_started = True
        t_r1 = time.perf_counter()
        return (t_r1 - t_r0) * 1000.0

    def _map_and_copy_to_torch(self) -> tuple[float, float]:
        debug_cb = None
        if hasattr(self, "_debug") and self._debug.cfg.enabled and self._debug.cfg.frames_remaining > 0:
            def _cb(tex_id: int, w: int, h: int, frame: torch.Tensor) -> None:
                self._debug.dump_triplet(tex_id, w, h, frame)

            debug_cb = _cb
        return self._cuda.map_and_copy(debug_cb=debug_cb)

    def _convert_for_encoder(self):
        t_c0 = time.perf_counter()
        try:
            dump_raw = int(os.getenv('NAPARI_CUDA_DUMP_RAW', '0'))
        except Exception:
            dump_raw = 0
        frame = self._cuda.torch_frame
        if dump_raw > 0:
            try:
                self._debug.dump_cuda_rgba(frame, self.width, self.height, prefix="raw")
                os.environ['NAPARI_CUDA_DUMP_RAW'] = str(max(0, dump_raw - 1))
            except Exception as e:
                logger.debug("Pre-encode raw dump failed: %s", e)
        src = frame
        try:
            r = src[..., 0].float() / 255.0
            g = src[..., 1].float() / 255.0
            b = src[..., 2].float() / 255.0
            y_n = 0.2126 * r + 0.7152 * g + 0.0722 * b
            y = torch.clamp(16.0 + 219.0 * y_n, 0.0, 255.0)
            cb = torch.clamp(128.0 + 224.0 * (b - y_n) / 1.8556, 0.0, 255.0)
            cr = torch.clamp(128.0 + 224.0 * (r - y_n) / 1.5748, 0.0, 255.0)
            try:
                if not hasattr(self, '_logged_swizzle_stats'):
                    rm = float(r.mean().item()); gm = float(g.mean().item()); bm = float(b.mean().item())
                    ym = float(y.mean().item()); cbm = float(cb.mean().item()); crm = float(cr.mean().item())
                    logger.info("Pre-encode channel means: R=%.3f G=%.3f B=%.3f | Y=%.3f Cb=%.3f Cr=%.3f", rm, gm, bm, ym, cbm, crm)
                    try:
                        if (getattr(self, '_auto_reset_on_black', False) and not getattr(self, '_black_reset_done', False) and (rm + gm + bm) < 1e-4):
                            logger.info("Detected black initial frame; applying one-shot camera reset")
                            if hasattr(self, 'view') and hasattr(self.view, 'camera'):
                                self._apply_camera_reset(self.view.camera)
                            self._black_reset_done = True
                        if (rm + gm + bm) >= 1e-4:
                            self._orientation_ready = True
                    except Exception:
                        logger.debug("auto-reset-on-black failed", exc_info=True)
                    setattr(self, '_logged_swizzle_stats', True)
            except Exception as e:
                logger.debug("Swizzle stats log failed: %s", e)
            H, W = self.height, self.width
            if self._enc_input_fmt == 'YUV444':
                if not hasattr(self, '_logged_swizzle'):
                    logger.info("Pre-encode swizzle: RGBA -> YUV444 (BT709 LIMITED, planar)")
                    setattr(self, '_logged_swizzle', True)
                dst = torch.empty((H * 3, W), dtype=torch.uint8, device=src.device)
                dst[0:H, :] = y.to(torch.uint8)
                dst[H:2*H, :] = cb.to(torch.uint8)
                dst[2*H:3*H, :] = cr.to(torch.uint8)
            elif self._enc_input_fmt in ('ARGB', 'ABGR'):
                if not hasattr(self, '_logged_swizzle'):
                    logger.info("Pre-encode swizzle: RGBA -> %s (packed)", self._enc_input_fmt)
                    setattr(self, '_logged_swizzle', True)
                dst = src[..., [3, 0, 1, 2]].contiguous() if self._enc_input_fmt == 'ARGB' else src[..., [3, 2, 1, 0]].contiguous()
            else:
                if not hasattr(self, '_logged_swizzle'):
                    logger.info("Pre-encode swizzle: RGBA -> NV12 (BT709 LIMITED, 4:2:0 UV interleaved)")
                    setattr(self, '_logged_swizzle', True)
                cb2 = F.avg_pool2d(cb.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
                cr2 = F.avg_pool2d(cr.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
                H2, W2 = cb2.shape
                dst = torch.empty((H + H2, W), dtype=torch.uint8, device=src.device)
                dst[0:H, :] = y.to(torch.uint8)
                uv = dst[H:, :]
                uv[:, 0::2] = cb2.to(torch.uint8)
                uv[:, 1::2] = cr2.to(torch.uint8)
        except Exception as e:
            logger.exception("Pre-encode color conversion failed: %s", e)
            dst = src[..., [3, 0, 1, 2]]
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            logger.debug("torch.cuda.synchronize failed", exc_info=True)
        t_c1 = time.perf_counter()
        return dst, (t_c1 - t_c0) * 1000.0

    # ---- C6 selection (napari-anchored) -------------------------------------
    def _maybe_select_level(self) -> None:
        """Select and apply a multiscale level based on current view.

        Uses an oversampling estimate per level and a small stabilizer to avoid
        oscillation. Enforces budgets via existing helpers and preserves Z
        proportionally in apply.
        """
        if not self._zarr_path:
            return
        try:
            source = self._ensure_scene_source()
        except Exception:
            logger.debug("ensure_scene_source failed in selection", exc_info=True)
            return
        levels = list(range(len(source.level_descriptors)))
        if not levels:
            return
        # Build oversampling map: viewport-to-slice size ratio per level
        overs_map: Dict[int, float] = {}
        for lvl in levels:
            try:
                overs_map[int(lvl)] = float(self._oversampling_for_level(source, int(lvl)))
            except Exception:
                continue
        if not overs_map:
            return
        # Prefer napari's own selected level when available; fall back to heuristic
        desired = None
        napari_reason = 'heuristic'

        # Recent zoom command (explicit user intent) takes precedence
        zoom_ratio = None
        zoom_hint = self._scene_state_machine.consume_zoom_intent(max_age=0.5)
        if zoom_hint is not None:
            zoom_ratio = float(zoom_hint.ratio)
        if zoom_ratio is not None:
            eps = 1e-3
            if zoom_ratio < 1.0 - eps and current > 0:
                desired = current - 1
                napari_reason = 'zoom-in'
            elif zoom_ratio > 1.0 + eps and current < max_level:
                desired = current + 1
                napari_reason = 'zoom-out'
        current = int(self._active_ms_level)
        max_level = max(overs_map.keys())
        try:
            thr_in = float(os.getenv('NAPARI_CUDA_LEVEL_THRESHOLD_IN', '1.05') or '1.05')
        except Exception:
            thr_in = 1.05
        try:
            thr_out = float(os.getenv('NAPARI_CUDA_LEVEL_THRESHOLD_OUT', '1.35') or '1.35')
        except Exception:
            thr_out = 1.35
        fine_threshold = max(thr_in, 1.05)

        if desired is None:
            # Heuristic: step one level finer when its oversampling is within threshold,
            # otherwise consider stepping coarser if current slice is oversampled.
            desired = current
            if current > 0:
                finer = current - 1
                ratio = overs_map.get(finer)
                if ratio is not None and ratio <= fine_threshold:
                    desired = finer
            if desired == current and current < max_level:
                ratio_cur = overs_map.get(current)
                if ratio_cur is not None and ratio_cur > thr_out:
                    desired = current + 1
            napari_reason = 'heuristic'
        else:
            # Napari hint: only allow single-step moves to avoid leaps.
            desired = max(0, min(int(desired), max_level))
            if desired < current:
                desired = max(desired, current - 1)
            elif desired > current:
                desired = min(desired, current + 1)
        # Stabilize against previous level to avoid jitter
        try:
            hyst = float(os.getenv('NAPARI_CUDA_LEVEL_HYST', '0.0') or '0.0')
        except Exception:
            hyst = 0.0
        selected = stabilize_level(int(desired), int(current), hysteresis=hyst)
        # Direction-sensitive band to avoid alternating near the boundary
        sel = int(selected)
        action_reason = napari_reason or 'heuristic'
        if action_reason not in ('zoom-in', 'zoom-out'):
            if sel < current:
                # Zooming in: only allow stepping to a finer level if that level is clearly under target
                if overs_map.get(sel, float('inf')) > thr_in:
                    sel = current
            elif sel > current:
                # Zooming out: only allow stepping to a coarser level if current level is clearly over target
                if overs_map.get(current, 0.0) <= thr_out:
                    sel = current
        selected = sel
        reason = action_reason
        if selected < current:
            logger.info(
                "lod.zoom_in: current=%d -> selected=%d overs=%.3f reason=%s",
                current,
                selected,
                overs_map.get(selected, float('nan')),
                reason,
            )
        elif selected > current:
            logger.info(
                "lod.zoom_out: current=%d -> selected=%d overs=%.3f reason=%s",
                current,
                selected,
                overs_map.get(selected, float('nan')),
                reason,
            )
        # Lock override
        lock = getattr(self, '_lock_level', None)
        if lock is not None:
            selected = int(lock)
            reason = 'locked'
        # Apply if changed; budgets enforced inside _set_level_with_budget
        if int(selected) != int(current):
            # Enforce a cooldown between level switches to avoid oscillation
            now_ts = time.perf_counter()
            try:
                cool_ms = float(getattr(self, '_level_switch_cooldown_ms', 150.0))
            except Exception:
                cool_ms = 150.0
            if self._last_level_switch_ts > 0.0:
                elapsed_ms = (now_ts - float(self._last_level_switch_ts)) * 1000.0
                if elapsed_ms < float(cool_ms):
                    return
            prev = current
            try:
                self._set_level_with_budget(int(selected), reason=reason)
                self._last_level_switch_ts = now_ts
                logger.info(
                    "ms.switch: %d -> %d (reason=%s) overs=%s",
                    int(prev), int(self._active_ms_level), reason,
                    '{' + ', '.join(f"{k}:{overs_map[k]:.2f}" for k in sorted(overs_map)) + '}',
                )
            except Exception as e:
                logger.info("ms.switch: hold=%d (budget reject %s)", int(current), str(e))
        else:
            # No change; emit a compact decision trace to aid testing
            logger.debug(
                "lod.hold: level=%d desired=%d selected=%d overs=%s",
                int(current), int(desired), int(selected),
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
