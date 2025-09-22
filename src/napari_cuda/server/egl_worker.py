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
from typing import Any, Callable, Deque, Dict, Optional, Mapping, Sequence, Tuple
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
from napari_cuda.server.level_budget import apply_level_with_budget, LevelBudgetError
from napari_cuda.server.config import LevelPolicySettings, ServerCtx
from napari_cuda.server.rendering.egl_context import EglContext
from napari_cuda.server.rendering.encoder import Encoder
from napari_cuda.server.capture import CaptureFacade, capture_frame_for_encoder
from napari_cuda.server.roi_applier import SliceUpdatePlanner
from napari_cuda.server.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
    viewport_debug_snapshot,
    resolve_viewport_roi,
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
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.state_machine import (
    CameraCommand,
    PendingSceneUpdate,
    SceneStateQueue,
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
                 *,
                 ctx: ServerCtx) -> None:
        self.width = int(width)
        self.height = int(height)
        self.use_volume = bool(use_volume)
        self.fps = int(fps)
        self.volume_depth = int(volume_depth)
        self.volume_dtype = str(volume_dtype)
        self.volume_relative_step = volume_relative_step
        self._ctx: ServerCtx = ctx
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

        self._capture = CaptureFacade(width=self.width, height=self.height)

        self._encoder: Optional[Encoder] = None

        # Track current data extents in world units (W,H)
        self._data_wh: tuple[int, int] = (int(width), int(height))
        self._data_d: Optional[int] = None
        # Gate pixel streaming until orientation/camera are settled and a non-black frame is observed
        self._orientation_ready: bool = False

        # Encoder access synchronization (reset vs encode across threads)
        self._enc_lock = threading.Lock()

        # Atomic state application
        self._state_lock = threading.Lock()
        self._scene_state_queue = SceneStateQueue()
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
        policy_cfg = self._ctx.policy
        self._log_policy_eval = bool(policy_cfg.log_policy_eval)
        # Legacy zoom-policy toggles removed; selection uses viewport oversampling
        # Optional: lock multiscale level (int index) to keep client/server in sync
        try:
            _lock = os.getenv('NAPARI_CUDA_LOCK_LEVEL')
            self._lock_level: Optional[int] = int(_lock) if (_lock is not None and _lock != '') else None
        except Exception:
            self._lock_level = None
        self._level_threshold_in = float(policy_cfg.threshold_in)
        self._level_threshold_out = float(policy_cfg.threshold_out)
        self._level_hysteresis = float(policy_cfg.hysteresis)
        self._level_fine_threshold = float(policy_cfg.fine_threshold)
        # Preserve current camera view on level switches (avoid resetting zoom)
        self._preserve_view_on_switch = bool(policy_cfg.preserve_view_on_switch)
        # Keep contrast limits stable across pans/level switches unless disabled
        self._sticky_contrast = bool(policy_cfg.sticky_contrast)
        # Last known dims.current_step from intents; used to preserve Z across
        # multiscale level switches regardless of viewer timing.
        self._last_step: Optional[tuple[int, ...]] = None
        # Coalesce selection to run once after a render when camera changed
        self._level_policy_refresh_needed: bool = False
        # Policy init happens naturally; no deferral needed
        # One-shot safety: reset camera if initial frames are black
        try:
            self._auto_reset_on_black = env_bool('NAPARI_CUDA_AUTO_RESET_ON_BLACK', True)
        except Exception:
            self._auto_reset_on_black = True
        self._capture.configure_auto_reset(self._auto_reset_on_black)
        self._black_reset_done: bool = False
        # Min time between level switches to avoid shimmering during pan
        self._level_switch_cooldown_ms = float(policy_cfg.cooldown_ms)
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
        self._capture.set_debug(self._debug)
        # Log format/size once for clarity
        logger.info(
            "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s, zarr=%s",
            self.width, self.height, self._capture.enc_input_format, self.fps, self._animate,
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
        assert self.view is not None, "adapter must supply a VisPy view"
        cam = self.view.camera
        if cam is not None:
            self._apply_camera_reset(cam)

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

        def _budget_check(scene: ZarrSceneSource, level: int) -> None:
            try:
                if self.use_volume:
                    self._volume_budget_allows(scene, level)
                else:
                    self._slice_budget_allows(scene, level)
            except _LevelBudgetError as exc:
                raise LevelBudgetError(str(exc)) from exc

        def _apply(scene: ZarrSceneSource, level: int, prev_level: Optional[int]) -> None:
            cache = getattr(self, '_roi_cache', None)
            if isinstance(cache, dict):
                cache.pop(int(level), None)
            roi_log = getattr(self, '_roi_log_state', None)
            if isinstance(roi_log, dict):
                roi_log.pop(int(level), None)
            self._apply_level_internal(scene, level, prev_level=prev_level)

        def _on_switch(prev_level: int, applied: int, elapsed_ms: float) -> None:
            self._log_ms_switch(
                previous=prev_level,
                applied=applied,
                source=source,
                elapsed_ms=elapsed_ms,
                reason=reason,
            )
            self._mark_render_tick_needed()

        try:
            applied_level, downgraded = apply_level_with_budget(
                desired_level=desired_level,
                use_volume=self.use_volume,
                source=source,
                current_level=int(self._active_ms_level),
                log_layer_debug=self._log_layer_debug,
                budget_check=_budget_check,
                apply_level_cb=_apply,
                on_switch=_on_switch,
            )
        except LevelBudgetError as exc:
            raise _LevelBudgetError(str(exc)) from exc
        self._active_ms_level = int(applied_level)
        self._level_downgraded = bool(downgraded)

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
        self._scene_state_queue.queue_multiscale_level(int(level), str(path) if path else None)
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
            "data_depth": int(self._data_d) if self._data_d is not None else None,
        }
        view = self.view
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
        return viewport_debug_snapshot(
            view=self.view,
            canvas_size=(int(self.width), int(self.height)),
            data_wh=self._data_wh,
            data_depth=self._data_d,
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
        align_chunks = (not for_policy) and bool(getattr(self, "_roi_align_chunks", True))
        ensure_contains = (not for_policy) and bool(getattr(self, "_roi_ensure_contains_viewport", True))
        edge_threshold = int(getattr(self, "_roi_edge_threshold", 4))
        chunk_pad = int(getattr(self, "_roi_pad_chunks", 1))

        roi_log = getattr(self, "_roi_log_state", None)
        log_state = roi_log if isinstance(roi_log, dict) else None

        cam = view.camera
        if not isinstance(cam, scene.cameras.PanZoomCamera):
            ensured = self._ensure_panzoom_camera(reason="roi-request")
            cam = ensured or cam
        assert isinstance(cam, scene.cameras.PanZoomCamera), "PanZoomCamera required for ROI compute"

        return resolve_viewport_roi(
            view=view,
            canvas_size=(int(self.width), int(self.height)),
            source=source,
            level=int(level),
            align_chunks=align_chunks,
            chunk_pad=chunk_pad,
            ensure_contains_viewport=ensure_contains,
            edge_threshold=edge_threshold,
            for_policy=for_policy,
            cache=self._roi_cache,
            log_state=log_state,
            snapshot_cb=self._viewport_debug_snapshot,
            log_layer_debug=self._log_layer_debug,
            quiet=quiet,
            logger_ref=logger,
        )

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
            ctx = self._build_scene_state_context(self.view.camera if self.view is not None else None)
            data_wh, data_d = SceneStateApplier.apply_volume_layer(
                ctx,
                volume=volume,
                contrast=applied.contrast,
            )
            self._data_wh = data_wh
            self._data_d = data_d
            self._log_layer_assignment('volume', applied.level, None, (data_d, data_wh[1], data_wh[0]), applied.contrast)
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
            if self._last_roi is not None and int(self._last_roi[0]) == int(applied.level):
                roi_for_layer = self._last_roi[1]
            if layer is not None:
                view = self.view
                assert view is not None
                ctx = self._build_scene_state_context(view.camera)
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
            if not self._preserve_view_on_switch:
                view = self.view
                assert view is not None and view.camera is not None
                world_w = float(w) * float(max(1e-12, sx))
                world_h = float(h) * float(max(1e-12, sy))
                view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
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
        if self._visual is not None:
            self._visual.parent = None  # type: ignore[attr-defined]
        self._visual = None

    def _apply_ndisplay_switch(self, ndisplay: int) -> None:
        """Apply 2D/3D toggle on the render thread."""
        assert self.view is not None and self.view.scene is not None, "VisPy view must be initialized"
        # Adapter-only: do not rebuild visuals; just update viewer dims
        target = 3 if int(ndisplay) >= 3 else 2
        self.use_volume = bool(target == 3)
        if self._viewer is not None:
            self._viewer.dims.ndisplay = target
        logger.info("ndisplay switch: %s", "3D" if target == 3 else "2D")
        # Evaluate policy once after display mode changes
        self._evaluate_level_policy()

    def _init_capture(self) -> None:
        self._capture.ensure()

    def _init_cuda_interop(self) -> None:
        """Init CUDA-GL interop and allocate destination via one-time DLPack bridge."""
        self._capture.initialize_cuda_interop()

    def _init_encoder(self) -> None:
        with self._enc_lock:
            if self._encoder is None:
                self._encoder = Encoder(self.width, self.height, fps_hint=int(self.fps))
            encoder = self._encoder
            encoder.set_fps_hint(int(self.fps))
            encoder.setup(self._ctx)
            self._capture.set_enc_input_format(encoder.input_format)

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
        if azimuth_deg is not None:
            assert self.view is not None and self.view.camera is not None
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
            center=tuple(float(c) for c in state.center) if state.center is not None else None,
            zoom=float(state.zoom) if state.zoom is not None else None,
            angles=tuple(float(a) for a in state.angles) if state.angles is not None else None,
            current_step=tuple(int(s) for s in state.current_step) if state.current_step is not None else None,
            volume_mode=str(state.volume_mode) if state.volume_mode is not None else None,
            volume_colormap=str(state.volume_colormap) if state.volume_colormap is not None else None,
            volume_clim=tuple(float(v) for v in state.volume_clim) if state.volume_clim is not None else None,
            volume_opacity=float(state.volume_opacity) if state.volume_opacity is not None else None,
            volume_sample_step=float(state.volume_sample_step) if state.volume_sample_step is not None else None,
        )
        self._scene_state_queue.queue_scene_state(normalized)

    def process_camera_commands(self, commands: Sequence[CameraCommand]) -> None:
        if not commands:
            return
        # Interaction observed (kept for future metrics)
        self._user_interaction_seen = True
        logger.debug("worker processing %d camera command(s)", len(commands))
        view = self.view
        assert view is not None, "process_camera_commands requires an active VisPy view"
        cam = view.camera
        if cam is None:
            for cmd in commands:
                if cmd.kind == 'zoom' and cmd.factor is not None and cmd.factor > 0.0:
                    self._scene_state_queue.record_zoom_intent(float(cmd.factor))
            return

        canvas_wh: tuple[int, int]
        if self.canvas is not None:
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
            view=view,
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
                self._scene_state_queue.record_zoom_intent(zoom_ratio)
                self._last_zoom_hint_ts = now_ts

        self._last_interaction_ts = time.perf_counter()

        if outcome.camera_changed:
            self._mark_render_tick_needed()
        if outcome.policy_triggered:
            self._level_policy_refresh_needed = True

    def _apply_camera_reset(self, cam) -> None:
        assert cam is not None, "VisPy camera expected"
        assert hasattr(cam, "set_range"), "Camera missing set_range handler"
        w, h = self._data_wh
        d = self._data_d
        # Convert pixel dims to world extents using current level scale
        sx = sy = 1.0
        if self._scene_source is not None and not self.use_volume:
            sy, sx = plane_scale_for_level(self._scene_source, int(self._active_ms_level))
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
        updates: PendingSceneUpdate = self._scene_state_queue.drain_pending_updates()

        if updates.display_mode is not None:
            self._apply_ndisplay_switch(updates.display_mode)

        if updates.multiscale is not None:
            lvl = int(updates.multiscale.level)
            pth = updates.multiscale.path
            self._apply_multiscale_switch(lvl, pth)

        state = updates.scene_state
        if state is None:
            return

        view = self.view
        assert view is not None, "drain_scene_updates requires an active VisPy view"
        cam = view.camera

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
                mode=state.volume_mode,
                colormap=state.volume_colormap,
                clim=state.volume_clim,
                opacity=state.volume_opacity,
                sample_step=state.volume_sample_step,
            )

        if cam is None:
            self._evaluate_level_policy()
            return

        # Legacy absolute camera fields
        if state.center is not None:
            assert hasattr(cam, "center"), "Camera missing center property"
            cam.center = state.center  # type: ignore[attr-defined]
        if state.zoom is not None:
            assert hasattr(cam, "zoom"), "Camera missing zoom property"
            cam.zoom = state.zoom  # type: ignore[attr-defined]
        if state.angles is not None:
            assert hasattr(cam, "angles"), "Camera missing angles property"
            cam.angles = state.angles  # type: ignore[attr-defined]

        # Evaluate policy only if the absolute state actually changed
        if self._scene_state_queue.update_state_signature(state):
            self._evaluate_level_policy()

    # --- Public coalesced toggles ------------------------------------------------
    def request_ndisplay(self, ndisplay: int) -> None:
        """Queue a 2D/3D view switch to apply on the render thread."""
        self._scene_state_queue.queue_display_mode(3 if int(ndisplay) >= 3 else 2)

    def _capture_blit_gpu_ns(self) -> Optional[int]:
        return self._capture.capture_blit_gpu_ns()


    def capture_and_encode_packet(self) -> tuple[FrameTimings, Optional[bytes], int, int]:
        """Same as capture_and_encode, but also returns the packet and flags.

        Flags bit 0x01 indicates keyframe (IDR/CRA).
        """
        # Debug env is logged once at init by DebugDumper
        # Capture wall and monotonic timestamps at frame start
        wall_ts = time.time()
        t0 = time.perf_counter()
        render_ms = self.render_tick()
        debug_cb = None
        if self._debug.cfg.enabled and self._debug.cfg.frames_remaining > 0:
            def _cb(tex_id: int, w: int, h: int, frame) -> None:
                self._debug.dump_triplet(tex_id, w, h, frame)  # type: ignore[attr-defined]

            debug_cb = _cb
        reset_cb: Optional[Callable[[], None]] = None
        if self._auto_reset_on_black and self.view is not None and self.view.camera is not None:
            def _reset_camera() -> None:
                cam = self.view.camera
                assert cam is not None
                self._apply_camera_reset(cam)

            reset_cb = _reset_camera
        capture = capture_frame_for_encoder(
            self._capture,
            debug_cb=debug_cb,
            reset_camera=reset_cb,
        )
        dst = capture.frame
        blit_gpu_ns = capture.blit_gpu_ns
        blit_cpu_ms = capture.blit_cpu_ms
        map_ms = capture.map_ms
        copy_ms = capture.copy_ms
        convert_ms = capture.convert_ms
        self._orientation_ready = capture.orientation_ready
        self._black_reset_done = capture.black_reset_done
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
        if self._level_policy_refresh_needed:
            self._level_policy_refresh_needed = False
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
        zoom_hint = self._scene_state_queue.consume_zoom_intent(max_age=0.5)
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
            self._capture.cleanup()
        except Exception:
            logger.debug("Cleanup: capture cleanup failed", exc_info=True)

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
