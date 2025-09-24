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
from napari_cuda.server.camera_animator import animate_if_enabled
from . import policy as level_policy
import napari_cuda.server.lod as lod
from napari_cuda.server.lod import apply_level
from napari_cuda.server.level_budget import LevelBudgetError
from napari_cuda.server.config import LevelPolicySettings, ServerCtx
from napari_cuda.server.rendering.egl_context import EglContext
from napari_cuda.server.rendering.encoder import Encoder
from napari_cuda.server.capture import CaptureFacade, FrameTimings, encode_frame
from napari_cuda.server.render_loop import run_render_tick
from napari_cuda.server.roi_applier import (
    SliceUpdatePlanner,
    refresh_slice,
    refresh_slice_for_worker,
)
from napari_cuda.server.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
    viewport_debug_snapshot,
    resolve_worker_viewport_roi,
)
from napari_cuda.server.scene_state_applier import (
    SceneStateApplier,
    SceneStateApplyContext,
    SceneDrainResult,
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
from napari_cuda.server.policy_metrics import PolicyMetrics
from napari_cuda.server.level_logging import LayerAssignmentLogger, LevelSwitchLogger
from napari_cuda.server.level_runtime import (
    apply_worker_level,
    apply_worker_volume_level,
    apply_worker_slice_level,
    format_worker_level_roi,
)

logger = logging.getLogger(__name__)

# Back-compat for tests patching the selector directly
select_level = lod.select_level

## Camera ops now live in napari_cuda.server.camera_ops as free functions.

class _LevelBudgetError(RuntimeError):
    """Raised when a multiscale level exceeds memory/voxel budgets."""
    pass


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

        self._configure_animation(animate, animate_dps)
        self._init_render_components()
        self._init_scene_state(scene_refresh_cb)
        self._init_locks()
        self._configure_debug_flags()
        self._configure_policy(self._ctx.policy)
        self._configure_roi_settings()
        self._configure_auto_reset()
        self._configure_budget_limits()

        if policy_name:
            try:
                self.set_policy(policy_name)
            except Exception:
                logger.exception("policy init set failed; continuing with default")

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

    def _frame_volume_camera(self, w: float, h: float, d: float) -> None:
        """Choose stable initial center and distance for TurntableCamera.

        ``w``, ``h``, ``d`` are world-space extents (after scale). We center the
        camera on the volume midpoint and set distance so the full height fits.
        """
        cam = self.view.camera
        if not isinstance(cam, scene.cameras.TurntableCamera):
            return
        center = (float(w) * 0.5, float(h) * 0.5, float(d) * 0.5)
        cam.center = center  # type: ignore[attr-defined]
        fov_deg = float(getattr(cam, 'fov', 60.0) or 60.0)
        fov_rad = math.radians(max(1e-3, min(179.0, fov_deg)))
        dist = (0.5 * float(h)) / max(1e-6, math.tan(0.5 * fov_rad))
        cam.distance = float(dist * 1.1)  # type: ignore[attr-defined]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "frame_volume_camera extent=(%.3f, %.3f, %.3f) center=%s dist=%.3f",
                w,
                h,
                d,
                center,
                float(cam.distance),
            )

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

        def _apply(scene: ZarrSceneSource, level: int, prev_level: Optional[int]) -> lod.AppliedLevel:
            return apply_worker_level(self, scene, level, prev_level=prev_level)

        def _on_switch(prev_level: int, applied: int, elapsed_ms: float) -> None:
            roi_desc = format_worker_level_roi(self, source, applied)
            self._switch_logger.log(
                enabled=self._log_layer_debug,
                previous=prev_level,
                applied=applied,
                roi_desc=roi_desc,
                reason=reason,
                elapsed_ms=elapsed_ms,
            )
            self._mark_render_tick_needed()

        try:
            applied_snapshot, downgraded = lod.apply_level_with_context(
                desired_level=desired_level,
                use_volume=self.use_volume,
                source=source,
                current_level=int(self._active_ms_level),
                log_layer_debug=self._log_layer_debug,
                budget_check=_budget_check,
                apply_level_fn=_apply,
                on_switch=_on_switch,
                roi_cache=self._roi_cache,
                roi_log_state=self._roi_log_state,
            )
        except LevelBudgetError as exc:
            raise _LevelBudgetError(str(exc)) from exc
        self._active_ms_level = int(applied_snapshot.level)
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

    def _mark_render_tick_complete(self) -> None:
        self._render_tick_required = False

    def _mark_render_loop_started(self) -> None:
        self._render_loop_started = True

    def _evaluate_policy_if_needed(self) -> None:
        if not self._level_policy_refresh_needed or self.use_volume:
            return
        self._level_policy_refresh_needed = False
        self._evaluate_level_policy()

    # ---- Init helpers -------------------------------------------------------

    def _configure_animation(self, animate: bool, animate_dps: float) -> None:
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception as exc:
            logger.debug("Invalid animate_dps=%r; using default 30.0: %s", animate_dps, exc)
            self._animate_dps = 30.0
        self._anim_start = time.perf_counter()

    def _init_render_components(self) -> None:
        self.cuda_ctx: Optional[cuda.Context] = None
        self.canvas: Optional[scene.SceneCanvas] = None
        self.view = None
        self._visual = None
        self._egl = EglContext(self.width, self.height)
        self._capture = CaptureFacade(width=self.width, height=self.height)
        self._encoder: Optional[Encoder] = None

    def _init_scene_state(self, scene_refresh_cb: Optional[Callable[[], None]]) -> None:
        self._viewer: Optional[ViewerModel] = None
        self._napari_layer = None
        self._scene_source: Optional[ZarrSceneSource] = None
        self._active_ms_level = 0
        self._level_downgraded = False
        self._scene_refresh_cb = scene_refresh_cb
        self._zoom_accumulator = 0.0
        self._last_zoom_hint_ts = 0.0
        self._zoom_hint_hold_s = 0.35
        self._data_wh = (int(self.width), int(self.height))
        self._data_d = None
        self._volume_scale = (1.0, 1.0, 1.0)
        self._orientation_ready = False
        self._policy_metrics = PolicyMetrics()
        self._layer_logger = LayerAssignmentLogger(logger)
        self._switch_logger = LevelSwitchLogger(logger)

    def _init_locks(self) -> None:
        self._enc_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._scene_state_queue = SceneStateQueue()
        self._last_ensure_log: Optional[tuple[int, Optional[str]]] = None
        self._last_ensure_log_ts = 0.0
        self._ensure_log_interval_s = 1.0
        self._render_tick_required = False
        self._render_loop_started = False
        self._level_policy_refresh_needed = False

    def _configure_debug_flags(self) -> None:
        self._debug_zoom_drift = env_bool('NAPARI_CUDA_DEBUG_ZOOM_DRIFT', False)
        self._debug_pan = env_bool('NAPARI_CUDA_DEBUG_PAN', False)
        self._debug_reset = env_bool('NAPARI_CUDA_DEBUG_RESET', False)
        self._debug_orbit = env_bool('NAPARI_CUDA_DEBUG_ORBIT', False)
        try:
            self._orbit_el_min = float(os.getenv('NAPARI_CUDA_ORBIT_ELEV_MIN', '-85.0') or '-85.0')
        except Exception:
            self._orbit_el_min = -85.0
        try:
            self._orbit_el_max = float(os.getenv('NAPARI_CUDA_ORBIT_ELEV_MAX', '85.0') or '85.0')
        except Exception:
            self._orbit_el_max = 85.0

    def _configure_policy(self, policy_cfg: LevelPolicySettings) -> None:
        self._policy_func = level_policy.resolve_policy('oversampling')
        self._policy_name = 'oversampling'
        self._last_interaction_ts = time.perf_counter()
        self._policy_metrics.reset()
        self._log_layer_debug = env_bool('NAPARI_CUDA_LAYER_DEBUG', False)
        self._log_roi_anchor = env_bool('NAPARI_CUDA_LOG_ROI_ANCHOR', False)
        self._log_policy_eval = bool(policy_cfg.log_policy_eval)
        try:
            _lock = os.getenv('NAPARI_CUDA_LOCK_LEVEL')
            self._lock_level = int(_lock) if (_lock and _lock != '') else None
        except Exception:
            self._lock_level = None
        self._level_threshold_in = float(policy_cfg.threshold_in)
        self._level_threshold_out = float(policy_cfg.threshold_out)
        self._level_hysteresis = float(policy_cfg.hysteresis)
        self._level_fine_threshold = float(policy_cfg.fine_threshold)
        self._preserve_view_on_switch = bool(policy_cfg.preserve_view_on_switch)
        self._sticky_contrast = bool(policy_cfg.sticky_contrast)
        self._last_step: Optional[tuple[int, ...]] = None
        self._level_switch_cooldown_ms = float(policy_cfg.cooldown_ms)
        self._last_level_switch_ts = 0.0

    def _configure_roi_settings(self) -> None:
        self._roi_cache: Dict[int, tuple[Optional[tuple[float, ...]], SliceROI]] = {}
        self._roi_log_state: Dict[int, tuple[SliceROI, float]] = {}
        self._roi_log_interval = 0.25
        try:
            self._roi_edge_threshold = int(os.getenv('NAPARI_CUDA_ROI_EDGE_THRESHOLD', '4') or '4')
        except Exception:
            self._roi_edge_threshold = 4
        self._roi_align_chunks = env_bool('NAPARI_CUDA_ROI_ALIGN_CHUNKS', False)
        self._roi_ensure_contains_viewport = env_bool('NAPARI_CUDA_ROI_ENSURE_CONTAINS_VIEWPORT', True)
        self._roi_pad_chunks = 1
        self._idr_on_z = False
        self._last_roi: Optional[tuple[int, SliceROI]] = None

    def _configure_auto_reset(self) -> None:
        try:
            self._auto_reset_on_black = env_bool('NAPARI_CUDA_AUTO_RESET_ON_BLACK', True)
        except Exception:
            self._auto_reset_on_black = True
        self._capture.configure_auto_reset(self._auto_reset_on_black)
        self._black_reset_done = False

    def _configure_budget_limits(self) -> None:
        cfg = self._ctx.cfg
        self._slice_max_bytes = int(max(0, getattr(cfg, 'max_slice_bytes', 0)))
        self._volume_max_bytes = int(max(0, getattr(cfg, 'max_volume_bytes', 0)))
        self._volume_max_voxels = int(max(0, getattr(cfg, 'max_volume_voxels', 0)))

    # ---- Level helpers ------------------------------------------------------

    def _update_level_metadata(self, descriptor, applied) -> None:
        self._active_ms_level = applied.level
        self._last_step = applied.step
        self._z_index = applied.z_index
        self._zarr_level = descriptor.path or None
        self._zarr_shape = descriptor.shape
        self._zarr_axes = applied.axes
        self._zarr_dtype = applied.dtype
        self._zarr_clim = applied.contrast

    def set_policy(self, name: str) -> None:
        new_name = str(name or '').strip().lower()
        # Accept only simplified oversampling policy (and close synonyms)
        allowed = {'oversampling', 'thresholds', 'ratio'}
        if new_name not in allowed:
            raise ValueError(f"Unsupported policy: {new_name}")
        self._policy_name = new_name
        # Map all aliases to the same selector
        self._policy_func = level_policy.resolve_policy(new_name)
        self._policy_metrics.reset()
        # No history maintenance; snapshot excludes history
        if self._log_layer_debug:
            logger.info("policy set: name=%s", new_name)

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

    def _viewport_roi_for_level(self, source: ZarrSceneSource, level: int, *, quiet: bool = False, for_policy: bool = False) -> SliceROI:
        view = getattr(self, "view", None)
        align_chunks = (not for_policy) and bool(getattr(self, "_roi_align_chunks", True))
        ensure_contains = (not for_policy) and bool(getattr(self, "_roi_ensure_contains_viewport", True))
        edge_threshold = int(getattr(self, "_roi_edge_threshold", 4))
        chunk_pad = int(getattr(self, "_roi_pad_chunks", 1))

        roi_log = getattr(self, "_roi_log_state", None)
        log_state = roi_log if isinstance(roi_log, dict) else None

        def _snapshot() -> Dict[str, Any]:
            return viewport_debug_snapshot(
                view=self.view,
                canvas_size=(int(self.width), int(self.height)),
                data_wh=self._data_wh,
                data_depth=self._data_d,
            )

        reason = "policy-roi" if for_policy else "roi-request"

        return resolve_worker_viewport_roi(
            view=view,
            canvas_size=(int(self.width), int(self.height)),
            source=source,
            level=int(level),
            align_chunks=align_chunks,
            chunk_pad=chunk_pad,
            ensure_contains_viewport=ensure_contains,
            edge_threshold=edge_threshold,
            for_policy=for_policy,
            roi_cache=self._roi_cache,
            roi_log_state=log_state,
            snapshot_cb=_snapshot,
            log_layer_debug=self._log_layer_debug,
            quiet=quiet,
            data_wh=self._data_wh,
            reason=reason,
            logger_ref=logger,
        )

    def policy_metrics_snapshot(self) -> Dict[str, object]:
        return self._policy_metrics.snapshot(
            policy=self._policy_name,
            active_level=int(self._active_ms_level),
            downgraded=bool(self._level_downgraded),
        )

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

    def _apply_level(
        self,
        source: ZarrSceneSource,
        level: int,
        *,
        prev_level: Optional[int] = None,
    ) -> lod.AppliedLevel:
        return apply_worker_level(self, source, level, prev_level=prev_level)

    def _apply_volume_level(
        self,
        source: ZarrSceneSource,
        applied: lod.AppliedLevel,
    ) -> None:
        apply_worker_volume_level(self, source, applied)

    def _apply_slice_level(
        self,
        source: ZarrSceneSource,
        applied: lod.AppliedLevel,
    ) -> None:
        apply_worker_slice_level(self, source, applied)

    def _format_level_roi(self, source: ZarrSceneSource, level: int) -> str:
        return format_worker_level_roi(self, source, level)

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
        self._policy_metrics.record(
            policy=self._policy_name,
            intent_level=intent_level if intent_level is not None else ctx.intent_level,
            selected_level=selected_level if selected_level is not None else target_level,
            desired_level=target_level,
            applied_level=int(self._active_ms_level),
            reason=reason,
            idle_ms=idle_ms,
            oversampling=ctx.level_oversampling,
            downgraded=bool(self._level_downgraded),
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

    @staticmethod
    def _coarsest_level_index(source) -> Optional[int]:
        """Return the largest level index available on the source."""

        descriptors = getattr(source, "level_descriptors", None)
        if not descriptors:
            return None
        try:
            last = descriptors[-1]
        except Exception:
            return None
        try:
            return int(getattr(last, "index"))
        except Exception:
            return int(len(descriptors) - 1)

    def _configure_camera_for_mode(self) -> None:
        view = self.view
        if view is None:
            return
        try:
            if self.use_volume:
                if not isinstance(view.camera, scene.cameras.TurntableCamera):
                    view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                extent = self._volume_world_extents()
                if extent is None:
                    w_px, h_px = self._data_wh
                    d_px = self._data_d or 1
                    extent = (float(w_px), float(h_px), float(d_px))
                world_w, world_h, world_d = extent
                view.camera.set_range(
                    x=(0.0, max(1.0, world_w)),
                    y=(0.0, max(1.0, world_h)),
                    z=(0.0, max(1.0, world_d)),
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "configure_camera_for_mode extent=(%.3f, %.3f, %.3f)",
                        world_w,
                        world_h,
                        world_d,
                    )
                self._frame_volume_camera(world_w, world_h, world_d)
                logger.debug(
                    "configure_camera_for_mode: use_volume extent=(%.3f, %.3f, %.3f) camera_center=%s distance=%.3f",
                    world_w,
                    world_h,
                    world_d,
                    getattr(view.camera, 'center', None),
                    float(getattr(view.camera, 'distance', 0.0)),
                )
            else:
                if not isinstance(view.camera, scene.cameras.PanZoomCamera):
                    view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                cam = view.camera
                if cam is not None:
                    self._apply_camera_reset(cam)
        except Exception:
            logger.debug("configure_camera_for_mode failed", exc_info=True)

    def _volume_shape_for_view(self) -> Optional[tuple[int, int, int]]:
        if self._data_d is not None and self._data_wh is not None:
            w, h = self._data_wh
            return int(self._data_d), int(h), int(w)
        if self._zarr_shape is not None and len(self._zarr_shape) >= 3:
            z, y, x = self._zarr_shape[:3]
            return int(z), int(y), int(x)
        return None

    def _volume_world_extents(self) -> Optional[tuple[float, float, float]]:
        shape = self._volume_shape_for_view()
        if shape is None:
            return None
        z, y, x = shape
        try:
            sz, sy, sx = self._volume_scale
        except Exception:
            sz = sy = sx = 1.0
        width = float(x) * float(sx)
        height = float(y) * float(sy)
        depth = float(z) * float(sz)
        return (width, height, depth)

    def _reset_volume_step(self, source, level: int) -> None:
        descriptor = None
        try:
            descriptor = source.level_descriptors[int(level)]
        except Exception:
            return
        axes = [str(a).lower() for a in getattr(source, "axes", []) or []]
        z_axis = 0
        if "z" in axes:
            z_axis = axes.index("z")
        step = list(getattr(source, "current_step", ()) or [])
        shape = list(getattr(descriptor, "shape", ()) or [])
        dims = max(len(step), len(shape), 3)
        if len(step) < dims:
            step.extend([0] * (dims - len(step)))
        if len(shape) < dims:
            shape.extend([1] * (dims - len(shape)))
        if z_axis >= len(step):
            step.extend([0] * (z_axis - len(step) + 1))
        if z_axis >= len(shape):
            shape.extend([1] * (z_axis - len(shape) + 1))
        max_z = max(1, int(shape[z_axis]))
        new_z = 0
        step[z_axis] = new_z
        clamped_step = []
        for idx, val in enumerate(step):
            sh = max(1, int(shape[idx] if idx < len(shape) else 1))
            clamped_step.append(int(max(0, min(int(val), sh - 1))))
        try:
            with self._state_lock:
                source.set_current_level(int(level), step=tuple(clamped_step))
        except Exception:
            logger.debug("ndisplay switch: resetting volume step failed", exc_info=True)
        self._z_index = int(new_z)
        self._last_step = tuple(clamped_step)

    def _apply_ndisplay_switch(self, ndisplay: int) -> None:
        """Apply 2D/3D toggle on the render thread."""
        assert self.view is not None and self.view.scene is not None, "VisPy view must be initialized"
        target = 3 if int(ndisplay) >= 3 else 2
        previous_volume = bool(self.use_volume)
        self.use_volume = bool(target == 3)

        if self.use_volume:
            self._last_roi = None
            source = None
            try:
                source = self._ensure_scene_source()
            except Exception:
                logger.debug("ndisplay switch: ensure_scene_source failed", exc_info=True)
            if source is not None:
                level = self._coarsest_level_index(source)
                if level is not None:
                    try:
                        self._perform_level_switch(
                            target_level=int(level),
                            reason="ndisplay-3d",
                            intent_level=int(level),
                            selected_level=int(level),
                            source=source,
                        )
                        self._reset_volume_step(source, int(level))
                    except Exception:
                        logger.exception("ndisplay switch: failed to apply coarsest level")
            self._level_policy_refresh_needed = False
        else:
            if previous_volume:
                self._level_policy_refresh_needed = True
                self._mark_render_tick_needed()

        self._configure_camera_for_mode()

        if self._viewer is not None:
            try:
                viewer_dims = self._viewer.dims
                viewer_dims.ndisplay = target
                if target == 3:
                    layer = self._napari_layer
                    layer_ndim = int(getattr(layer, "ndim", 0)) if layer is not None else 0
                    data = getattr(layer, "data", None) if layer is not None else None
                    data_ndim = int(getattr(data, "ndim", 0)) if data is not None else 0
                    ndim = max(layer_ndim, data_ndim, int(getattr(viewer_dims, "ndim", 0)), 3)
                    viewer_dims.ndim = ndim
                    try:
                        displayed = tuple(range(max(0, ndim - 3), ndim))
                        viewer_dims.displayed = displayed  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("ndisplay switch: displayed update failed", exc_info=True)
                    steps = list(getattr(viewer_dims, "current_step", () ) or ())
                    if len(steps) < ndim:
                        steps.extend([0] * (ndim - len(steps)))
                    elif len(steps) > ndim:
                        steps = steps[:ndim]
                    if steps:
                        try:
                            steps[0] = int(self._z_index) if self._z_index is not None else int(steps[0])
                        except Exception:
                            logger.debug("ndisplay switch: z index step adjust failed", exc_info=True)
                    shape = self._volume_shape_for_view()
                    if shape is not None:
                        axes = tuple(getattr(viewer_dims, "axis_labels", []) or [])
                        for axis_idx in range(min(len(steps), len(shape))):
                            size = int(max(1, shape[axis_idx]))
                            try:
                                steps[axis_idx] = int(max(0, min(int(steps[axis_idx]), size - 1)))
                            except Exception:
                                logger.debug("ndisplay switch: step clamp failed axis=%d", axis_idx, exc_info=True)
                        # Clamp any displayed axes based on computed shape and axis labels
                        if axes and shape:
                            axis_map = {label.lower(): idx for idx, label in enumerate(axes)}
                            for label_key, dim_size in zip(("z", "y", "x"), shape):
                                idx = axis_map.get(label_key)
                                if idx is not None and idx < len(steps):
                                    steps[idx] = int(max(0, min(int(steps[idx]), int(dim_size) - 1)))
                    try:
                        viewer_dims.current_step = tuple(int(s) for s in steps)
                    except Exception:
                        logger.debug("ndisplay switch: current_step update failed", exc_info=True)
            except Exception:
                logger.debug("ndisplay switch: viewer dims update failed", exc_info=True)

        logger.info("ndisplay switch: %s", "3D" if target == 3 else "2D")
        if not self.use_volume:
            self._evaluate_level_policy()
            self._level_policy_refresh_needed = False
        self._mark_render_tick_needed()

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

        def _mark_render() -> None:
            self._mark_render_tick_needed()

        def _trigger_policy() -> None:
            self._level_policy_refresh_needed = True

        def _record_zoom_intent(ratio: float) -> None:
            self._scene_state_queue.record_zoom_intent(float(ratio))

        outcome: CameraCommandOutcome = apply_camera_commands(
            commands,
            camera=cam,
            view=view,
            canvas_size=canvas_wh,
            reset_camera=self._apply_camera_reset,
            debug_flags=debug_flags,
            mark_render_tick_needed=_mark_render,
            trigger_policy_refresh=_trigger_policy,
            record_zoom_intent=_record_zoom_intent,
            last_zoom_hint_ts=self._last_zoom_hint_ts,
            zoom_hint_hold_s=self._zoom_hint_hold_s,
        )

        if outcome.last_zoom_hint_ts is not None:
            self._last_zoom_hint_ts = float(outcome.last_zoom_hint_ts)
        if outcome.interaction_ts is not None:
            self._last_interaction_ts = float(outcome.interaction_ts)

    def _apply_camera_reset(self, cam) -> None:
        assert cam is not None, "VisPy camera expected"
        assert hasattr(cam, "set_range"), "Camera missing set_range handler"
        w, h = self._data_wh
        d = self._data_d
        # Convert pixel dims to world extents using current level scale
        if self.use_volume:
            extent = self._volume_world_extents()
            if extent is None:
                extent = (float(w), float(h), float(d or 1))
            world_w, world_h, world_d = extent
            cam.set_range(
                x=(0.0, max(1.0, world_w)),
                y=(0.0, max(1.0, world_h)),
                z=(0.0, max(1.0, world_d)),
            )
            self._frame_volume_camera(world_w, world_h, world_d)
        else:
            sx = sy = 1.0
            if self._scene_source is not None:
                sy, sx = plane_scale_for_level(self._scene_source, int(self._active_ms_level))
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
            volume_scale=getattr(self, '_volume_scale', None),
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

        ctx = self._build_scene_state_context(cam)
        drain_res: SceneDrainResult = SceneStateApplier.drain_updates(
            ctx,
            state=state,
            queue=self._scene_state_queue,
        )

        if drain_res.z_index is not None:
            self._z_index = int(drain_res.z_index)
        if drain_res.data_wh is not None:
            self._data_wh = (int(drain_res.data_wh[0]), int(drain_res.data_wh[1]))
        if drain_res.last_step is not None:
            self._last_step = tuple(int(x) for x in drain_res.last_step)

        if drain_res.policy_refresh_needed and not self.use_volume:
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
        reset_cb: Optional[Callable[[], None]] = None
        if self._auto_reset_on_black and self.view is not None and self.view.camera is not None:
            def _reset_camera() -> None:
                cam = self.view.camera
                assert cam is not None
                self._apply_camera_reset(cam)

            reset_cb = _reset_camera

        encoded = encode_frame(
            capture=self._capture,
            render_frame=self.render_tick,
            obtain_encoder=lambda: self._encoder,
            encoder_lock=self._enc_lock,
            debug_dumper=self._debug,
            reset_camera=reset_cb,
        )

        self._orientation_ready = encoded.orientation_ready
        self._black_reset_done = encoded.black_reset_done

        return encoded.timings, encoded.packet, encoded.flags, encoded.sequence

    # ---- C5 helpers (pure refactor; no behavior change) ---------------------
    def _refresh_slice_if_needed(self) -> None:
        if self.use_volume or self._scene_source is None or self._napari_layer is None:
            return

        source = self._scene_source

        def _apply_slice(slab: np.ndarray, roi_to_apply: SliceROI) -> None:
            cam = getattr(self.view, "camera", None)
            ctx = self._build_scene_state_context(cam)
            SceneStateApplier.apply_slice_to_layer(
                ctx,
                source=source,
                slab=slab,
                roi=roi_to_apply,
                update_contrast=False,
            )

        refreshed, new_last = refresh_slice_for_worker(
            source=source,
            level=int(self._active_ms_level),
            last_roi=self._last_roi,
            z_index=self._z_index,
            edge_threshold=int(self._roi_edge_threshold),
            viewport_roi_for_level=self._viewport_roi_for_level,
            load_slice=self._load_slice,
            apply_slice=_apply_slice,
        )
        if refreshed:
            self._last_roi = new_last

    def render_tick(self) -> float:
        canvas = self.canvas
        assert canvas is not None, "render_tick requires an initialized canvas"

        return run_render_tick(
            animate_camera=lambda: animate_if_enabled(
                enabled=bool(self._animate),
                view=getattr(self, "view", None),
                width=self.width,
                height=self.height,
                animate_dps=float(self._animate_dps),
                anim_start=float(self._anim_start),
            ),
            drain_scene_updates=self.drain_scene_updates,
            refresh_slice=self._refresh_slice_if_needed,
            render_canvas=canvas.render,
            evaluate_policy_if_needed=self._evaluate_policy_if_needed,
            mark_tick_complete=self._mark_render_tick_complete,
            mark_loop_started=self._mark_render_loop_started,
        )

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

        current = int(self._active_ms_level)
        zoom_hint = self._scene_state_queue.consume_zoom_intent(max_age=0.5)
        zoom_ratio = float(zoom_hint.ratio) if zoom_hint is not None else None

        config = lod.LevelPolicyConfig(
            threshold_in=float(self._level_threshold_in),
            threshold_out=float(self._level_threshold_out),
            fine_threshold=float(self._level_fine_threshold),
            hysteresis=float(self._level_hysteresis),
            cooldown_ms=float(self._level_switch_cooldown_ms),
        )

        try:
            outcome = lod.run_policy_switch(
                source=source,
                current_level=current,
                oversampling_for_level=self._oversampling_for_level,
                zoom_ratio=zoom_ratio,
                lock_level=self._lock_level,
                last_switch_ts=float(self._last_level_switch_ts),
                config=config,
                log_policy_eval=self._log_policy_eval,
                apply_level=lambda level, reason: self._set_level_with_budget(level, reason=reason),
                budget_error=_LevelBudgetError,
                select_level_fn=select_level,
                logger_ref=logger,
            )
        except Exception as exc:
            action = getattr(getattr(exc, "policy_decision", None), "action", "unknown")
            logger.exception("ms.switch: level apply failed (reason=%s)", action)
            raise

        if outcome is None:
            return

        prev = current
        self._last_level_switch_ts = float(outcome.timestamp)
        if outcome.decision.action in {"zoom-in", "zoom-out"}:
            self._last_zoom_hint_ts = float(outcome.timestamp)

        logger.info(
            "ms.switch: %d -> %d (reason=%s) overs=%s",
            int(prev),
            int(self._active_ms_level),
            outcome.decision.action,
            lod.format_oversampling(outcome.oversampling),
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
