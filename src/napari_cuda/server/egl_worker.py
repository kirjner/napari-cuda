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
import ctypes
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
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

from OpenGL import GL, EGL  # type: ignore
from vispy import scene  # type: ignore

import cupy as cp  # type: ignore
import pycuda.driver as cuda  # type: ignore
import pycuda.gl  # type: ignore
from pycuda.gl import RegisteredImage, graphics_map_flags  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import PyNvVideoCodec as pnvc  # type: ignore
# Bitstream packing happens in the server layer
from .patterns import make_rgba_image
from .debug_tools import DebugConfig, DebugDumper
from .hw_limits import get_hw_limits

logger = logging.getLogger(__name__)


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


@dataclass
class ServerSceneState:
    """Minimal scene state snapshot applied atomically per frame.

    Extended to support coalesced camera operations applied once per frame.
    """
    # Legacy absolute camera state (kept for compatibility)
    center: Optional[tuple[float, float, float]] = None
    zoom: Optional[float] = None
    angles: Optional[tuple[float, float, float]] = None
    # Dims/Z state
    current_step: Optional[tuple[int, ...]] = None
    # Coalesced camera ops (consumed once by the worker)
    zoom_factor: Optional[float] = None
    zoom_anchor_px: Optional[tuple[float, float]] = None
    pan_dx_px: float = 0.0
    pan_dy_px: float = 0.0
    reset_view: bool = False
    # One-shot volume render params (applied on render thread)
    volume_mode: Optional[str] = None
    volume_colormap: Optional[str] = None
    volume_clim: Optional[tuple[float, float]] = None
    volume_opacity: Optional[float] = None
    volume_sample_step: Optional[float] = None
    # One-shot orbit deltas (degrees) for TurntableCamera
    orbit_daz_deg: Optional[float] = None
    orbit_del_deg: Optional[float] = None


class EGLRendererWorker:
    """Headless VisPy renderer using EGL with CUDA interop and NVENC."""

    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False, fps: int = 60,
                 volume_depth: int = 64, volume_dtype: str = "float32", volume_relative_step: Optional[float] = None,
                 animate: bool = False, animate_dps: float = 30.0,
                 zarr_path: Optional[str] = None, zarr_level: Optional[str] = None,
                 zarr_axes: Optional[str] = None, zarr_z: Optional[int] = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.use_volume = bool(use_volume)
        self.fps = int(fps)
        self.volume_depth = int(volume_depth)
        self.volume_dtype = str(volume_dtype)
        self.volume_relative_step = volume_relative_step
        # Optional simple turntable animation
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception as e:
            logger.debug("Invalid animate_dps=%r; using default 30.0: %s", animate_dps, e)
            self._animate_dps = 30.0
        self._anim_start = time.perf_counter()

        self.egl_display = None
        self.egl_context = None
        self.egl_surface = None

        self.cuda_ctx: Optional[cuda.Context] = None

        self.canvas: Optional[scene.SceneCanvas] = None
        self.view = None
        self._visual = None

        self._texture: Optional[int] = None
        self._fbo: Optional[int] = None

        self._registered_tex: Optional[RegisteredImage] = None
        # Destination frame (Torch CUDA tensor) via one-time DLPack handoff
        self._torch_frame: Optional[torch.Tensor] = None
        self._dst_pitch_bytes: Optional[int] = None

        self._encoder = None

        # Track current data extents in world units (W,H)
        self._data_wh: tuple[int, int] = (int(width), int(height))

        # Timer query double-buffering for capture blit
        self._query_ids = None  # type: Optional[tuple]
        self._query_idx = 0
        self._query_started = False

        # Encoder access synchronization (reset vs encode across threads)
        self._enc_lock = threading.Lock()

        # Atomic state application
        self._state_lock = threading.Lock()
        self._pending_state: Optional[ServerSceneState] = None

        # Encoding / instrumentation state
        self._frame_index = 0
        self._force_next_idr = False
        try:
            self._log_keyframes = bool(int(os.getenv('NAPARI_CUDA_LOG_KEYFRAMES', '0') or '0'))
        except Exception:
            self._log_keyframes = False
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

        # Ensure partial initialization is cleaned up if any step fails
        try:
            # Zarr/NGFF dataset configuration (optional)
            self._zarr_path: Optional[str] = zarr_path or os.getenv('NAPARI_CUDA_ZARR_PATH') or None
            self._zarr_level: Optional[str] = zarr_level or os.getenv('NAPARI_CUDA_ZARR_LEVEL') or None
            self._zarr_axes: str = (zarr_axes or os.getenv('NAPARI_CUDA_ZARR_AXES') or 'zyx')
            try:
                _z = zarr_z if zarr_z is not None else int(os.getenv('NAPARI_CUDA_ZARR_Z', '-1'))
                self._zarr_init_z: Optional[int] = _z if _z >= 0 else None
            except Exception:
                self._zarr_init_z = None

            # Lazy dataset handles
            self._da_volume = None  # type: ignore[assignment]
            self._zarr_shape: Optional[tuple[int, ...]] = None
            self._zarr_dtype: Optional[str] = None
            self._z_index: Optional[int] = None

            self._init_egl()
            self._init_cuda()
            self._init_vispy_scene()
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
        try:
            logger.info(
                "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s, zarr=%s",
                self.width, self.height, getattr(self, '_enc_input_fmt', 'unknown'), self.fps, self._animate,
                bool(self._zarr_path),
            )
        except Exception as e:
            logger.debug("Init info log failed: %s", e)

    def _frame_volume_camera(self, w: int, h: int, d: int) -> None:
        """Choose stable initial center and distance for TurntableCamera.

        Center at the volume centroid and set distance so the full height fits
        in view (adds a small margin). This avoids dead pan response before the
        first zoom and prevents the initial zoom from overshooting.
        """
        try:
            from vispy.scene.cameras import TurntableCamera  # type: ignore
        except Exception:
            TurntableCamera = None  # type: ignore
        try:
            cam = self.view.camera
        except Exception:
            return
        if TurntableCamera is None or not isinstance(cam, TurntableCamera):
            return
        try:
            cam.center = (float(w) * 0.5, float(h) * 0.5, float(d) * 0.5)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("frame_volume: set center failed", exc_info=True)
        try:
            fov_deg = float(getattr(cam, 'fov', 60.0) or 60.0)
            fov_rad = math.radians(max(1e-3, min(179.0, fov_deg)))
            dist = (0.5 * float(h)) / max(1e-6, math.tan(0.5 * fov_rad))
            cam.distance = float(dist * 1.1)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("frame_volume: set distance failed", exc_info=True)

    def _init_egl(self) -> None:
        egl_display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if egl_display == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("Failed to get EGL display")
        major = EGL.EGLint()
        minor = EGL.EGLint()
        if not EGL.eglInitialize(egl_display, major, minor):
            raise RuntimeError("Failed to initialize EGL")
        config_attribs = [
            EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RED_SIZE, 8,
            EGL.EGL_GREEN_SIZE, 8,
            EGL.EGL_BLUE_SIZE, 8,
            EGL.EGL_ALPHA_SIZE, 8,
            EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
            EGL.EGL_NONE,
        ]
        config_attribs_p = (EGL.EGLint * len(config_attribs))(*config_attribs)
        egl_config = EGL.EGLConfig()
        num_configs = EGL.EGLint()
        if not EGL.eglChooseConfig(egl_display, config_attribs_p, ctypes.byref(egl_config), 1, ctypes.byref(num_configs)):
            raise RuntimeError("Failed to choose EGL config")
        if num_configs.value < 1:
            raise RuntimeError("No EGL configs matched requested attributes")
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        egl_context = EGL.eglCreateContext(egl_display, egl_config, EGL.EGL_NO_CONTEXT, None)
        if egl_context == EGL.EGL_NO_CONTEXT:
            raise RuntimeError("Failed to create EGL context")
        pbuffer_attribs = [
            EGL.EGL_WIDTH, self.width,
            EGL.EGL_HEIGHT, self.height,
            EGL.EGL_NONE,
        ]
        pbuffer_attribs_p = (EGL.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
        egl_surface = EGL.eglCreatePbufferSurface(egl_display, egl_config, pbuffer_attribs_p)
        if egl_surface == EGL.EGL_NO_SURFACE:
            raise RuntimeError("Failed to create EGL pbuffer surface")
        if not EGL.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context):
            raise RuntimeError("Failed to make EGL context current")
        self.egl_display = egl_display
        self.egl_context = egl_context
        self.egl_surface = egl_surface

    def _init_cuda(self) -> None:
        cuda.init()
        dev = cuda.Device(0)
        ctx = dev.retain_primary_context()
        ctx.push()
        self.cuda_ctx = ctx

    def _init_vispy_scene(self) -> None:
        canvas = scene.SceneCanvas(size=(self.width, self.height), bgcolor="black", show=False, app="egl")
        view = canvas.central_widget.add_view()
        # Ensure attributes are set before any first render
        self.canvas = canvas
        self.view = view

        # Determine scene content in priority order:
        # 1) OME-Zarr 3D volume (when --volume with --zarr is provided)
        # 2) OME-Zarr 2D slice (preferred fallback for real data MVP)
        # 3) Explicit synthetic 3D volume demo if requested
        # 4) Synthetic 2D image pattern (fallback)
        scene_src = "synthetic"
        scene_meta = ""

        if self._zarr_path and self.use_volume:
            try:
                # 3D volume from OME-Zarr (axes assumed zyx unless overridden)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                vol3d = self._load_zarr_volume3d()
                # Normalize to float32 0..1 already performed by loader
                visual = scene.visuals.Volume(vol3d, parent=view.scene, method="mip", cmap="grays")
                try:
                    if self.volume_relative_step is not None and hasattr(visual, "relative_step_size"):
                        visual.relative_step_size = float(self.volume_relative_step)
                    d, h, w = int(vol3d.shape[0]), int(vol3d.shape[1]), int(vol3d.shape[2])
                    self._data_wh = (w, h)
                    self._data_d = int(d)
                    # Ensure initial camera frames the volume so first render is visible
                    try:
                        view.camera.set_range(x=(0, w), y=(0, h), z=(0, d))
                    except Exception as e:
                        # Fallback to XY only if TurntableCamera lacks z-range
                        try:
                            view.camera.set_range(x=(0, w), y=(0, h))
                        except Exception:
                            logger.debug("zarr-volume: initial camera set_range(xy) failed", exc_info=True)
                    # Stabilize initial pan/zoom by framing center + distance
                    try:
                        self._frame_volume_camera(w, h, d)
                    except Exception:
                        logger.debug("zarr-volume: frame camera failed", exc_info=True)
                    scene_src = "zarr-volume"
                    scene_meta = f"level={self._zarr_level or 'auto'} shape={d}x{h}x{w}"
                except Exception as e:
                    logger.debug("zarr-volume metadata set failed", exc_info=True)
            except Exception as e:
                logger.exception("Failed to initialize OME-Zarr volume; falling back to 2D slice: %s", e)
                # Fall through to 2D slice fallback
                self.use_volume = False

        elif self._zarr_path and not self.use_volume:
            try:
                # 2D slice mode from OME-Zarr (axes assumed zyx unless overridden)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                arr2d = self._load_zarr_initial_slice()
                visual = scene.visuals.Image(arr2d, parent=view.scene, cmap='grays', clim=None, method='auto')
                # Camera range to data extents (note: array is (H, W))
                try:
                    h, w = int(arr2d.shape[0]), int(arr2d.shape[1])
                    self._data_wh = (w, h)
                    view.camera.set_range(x=(0, w), y=(0, h))
                    scene_src = "zarr"
                    scene_meta = f"level={self._zarr_level or 'auto'} z={self._z_index} shape={h}x{w}"
                except Exception as e:
                    logger.debug("set_range(zarr) failed: %s", e)
            except Exception as e:
                logger.exception("Failed to initialize OME-Zarr slice; falling back to synthetic image: %s", e)
                # Fallback to synthetic 2D image
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                image = make_rgba_image(self.width, self.height)
                visual = scene.visuals.Image(image, parent=view.scene)
                try:
                    self._data_wh = (self.width, self.height)
                    view.camera.set_range(x=(0, self.width), y=(0, self.height))
                    scene_src = "synthetic"
                    scene_meta = f"shape={self.height}x{self.width}"
                except Exception:
                    logger.debug("synthetic 2D: initial camera set_range failed", exc_info=True)
        elif self.use_volume:
            # 3D volume demo
            view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60, distance=500)
            if self.volume_dtype == "float16":
                dtype = np.float16
            elif self.volume_dtype == "uint8":
                dtype = np.uint8
            else:
                dtype = np.float32
            volume = np.random.rand(self.volume_depth, self.height, self.width).astype(dtype)
            visual = scene.visuals.Volume(volume, parent=view.scene, method="mip", cmap="viridis")
            try:
                if self.volume_relative_step is not None and hasattr(visual, "relative_step_size"):
                    visual.relative_step_size = float(self.volume_relative_step)
                # Frame the synthetic volume to avoid needing a manual reset
                try:
                    d = int(self.volume_depth)
                    h = int(self.height)
                    w = int(self.width)
                    self._data_wh = (w, h)
                    self._data_d = int(d)
                    view.camera.set_range(x=(0, w), y=(0, h), z=(0, d))
                    try:
                        self._frame_volume_camera(w, h, d)
                    except Exception:
                        logger.debug("synthetic volume: frame camera failed", exc_info=True)
                except Exception:
                    try:
                        view.camera.set_range(x=(0, self.width), y=(0, self.height))
                    except Exception:
                        logger.debug("synthetic volume: initial camera set_range fallback failed", exc_info=True)
                scene_src = "volume"
                scene_meta = f"depth={self.volume_depth} shape={self.height}x{self.width} dtype={self.volume_dtype}"
            except Exception as e:
                logger.debug("Set volume relative_step_size failed: %s", e)
        else:
            # 2D synthetic image fallback
            view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
            image = make_rgba_image(self.width, self.height)
            visual = scene.visuals.Image(image, parent=view.scene)
            try:
                self._data_wh = (self.width, self.height)
                view.camera.set_range(x=(0, self.width), y=(0, self.height))
                scene_src = "synthetic"
                scene_meta = f"shape={self.height}x{self.width}"
            except Exception as e:
                logger.debug("PanZoom set_range failed: %s", e)

        # Assign visual before first draw and log concise init info
        self._visual = visual
        try:
            logger.info("Scene init: source=%s %s", scene_src, scene_meta)
        except Exception:
            logger.debug("Scene init log failed", exc_info=True)
        try:
            logger.info("Camera class: %s", type(view.camera).__name__)
        except Exception:
            pass
        # First render after attributes are fully set
        canvas.render()

    def _infer_zarr_level(self, root: str) -> Optional[str]:
        """Infer a dataset level path from NGFF .zattrs if level not provided."""
        try:
            zattrs = Path(root) / '.zattrs'
            if not zattrs.exists():
                return None
            data = json.loads(zattrs.read_text())
            ms = data.get('multiscales') or []
            if not ms:
                return None
            datasets = ms[0].get('datasets') or []
            # Prefer a mid-level if available (pick the second if >=2)
            if not datasets:
                return None
            if self._zarr_level:
                return self._zarr_level
            if len(datasets) >= 2:
                return datasets[1].get('path')
            return datasets[0].get('path')
        except Exception as e:
            logger.debug("infer zarr level failed", exc_info=True)
            return None

    def _load_zarr_initial_slice(self):
        """Load initial 2D slice from a OME-Zarr (ZYX) dataset as float32 array.

        - Chooses dataset level from provided level or .zattrs multiscales.
        - Picks initial Z (provided or mid-slice).
        - Returns a float32 normalized array (0..1) suitable for vispy Image.
        """
        assert self._zarr_path is not None
        root = self._zarr_path
        level = self._zarr_level or self._infer_zarr_level(root) or ''
        store_path = Path(root) / level if level else Path(root)
        if da is None:
            raise RuntimeError("dask.array is required for OME-Zarr loading but is not available")
        darr = da.from_zarr(str(store_path))
        if darr.ndim != 3:
            raise RuntimeError(f"Expected 3D (zyx) dataset; got shape {darr.shape}")
        self._da_volume = darr
        self._zarr_shape = tuple(int(s) for s in darr.shape)  # (Z,Y,X)
        # Choose Z
        z_init = self._zarr_init_z
        if z_init is None:
            z_init = int(self._zarr_shape[0] // 2) if self._zarr_shape else 0
        z_init = max(0, min(z_init, int(darr.shape[0]) - 1))
        self._z_index = int(z_init)

        # Compute percentiles for contrast once (on a small sub-sample to keep fast)
        try:
            # Read a center crop or the chosen slice to estimate clims
            sample = darr[self._z_index, :, :].astype('float32').compute()
            p1, p99 = np.percentile(sample, [0.5, 99.5])
            self._zarr_clim = (float(p1), float(p99))
            # Avoid zero span
            if self._zarr_clim[1] <= self._zarr_clim[0]:
                self._zarr_clim = (float(sample.min()), float(sample.max()))
        except Exception as e:
            logger.debug("zarr percentile/clim compute failed", exc_info=True)
            self._zarr_clim = None

        return self._load_zarr_slice(self._z_index)

    def _load_zarr_volume3d(self):
        """Load entire OME-Zarr 3D volume (ZYX) as float32 0..1 array for Volume visual.

        Uses the same dataset selection logic as _load_zarr_initial_slice and
        normalizes with cached percentiles if available.
        """
        assert self._zarr_path is not None
        root = self._zarr_path
        level = self._zarr_level or self._infer_zarr_level(root) or ''
        store_path = Path(root) / level if level else Path(root)
        if da is None:
            raise RuntimeError("dask.array is required for OME-Zarr loading but is not available")
        darr = da.from_zarr(str(store_path))
        if darr.ndim != 3:
            raise RuntimeError(f"Expected 3D (zyx) dataset; got shape {darr.shape}")
        self._da_volume = darr
        self._zarr_shape = tuple(int(s) for s in darr.shape)  # (Z,Y,X)
        # Budget check before compute
        try:
            hz, hy, hx = (int(self._zarr_shape[0]), int(self._zarr_shape[1]), int(self._zarr_shape[2]))
            vox = hz * hy * hx
            limits = get_hw_limits()
            if vox > limits.volume_max_voxels:
                raise RuntimeError(
                    f"Volume too large for budget: {hz}x{hy}x{hx} vox={vox} > cap={limits.volume_max_voxels}"
                )
        except Exception as e:
            logger.exception("Zarr volume size exceeds budget or check failed")
            raise
        # Ensure we have a contrast estimate
        if not hasattr(self, '_zarr_clim') or self._zarr_clim is None:
            try:
                zc = int((self._zarr_shape[0] or 1) // 2)
                sample = darr[zc, :, :].astype('float32').compute()
                p1, p99 = np.percentile(sample, [0.5, 99.5])
                c0, c1 = float(p1), float(p99)
                if c1 <= c0:
                    c0, c1 = float(sample.min()), float(sample.max())
                self._zarr_clim = (c0, c1)
            except Exception as e:
                logger.debug("zarr volume clim compute failed", exc_info=True)
                self._zarr_clim = None
        # Load full volume to float32 and normalize to 0..1
        vol = darr.astype('float32').compute()
        c0, c1 = None, None
        try:
            if hasattr(self, '_zarr_clim') and self._zarr_clim is not None:
                c0, c1 = self._zarr_clim
        except Exception as e:
            logger.debug("zarr volume clim fetch failed", exc_info=True)
        if c0 is None or c1 is None or c1 <= c0:
            c0 = float(vol.min()); c1 = float(vol.max())
            if not _np.isfinite(c0) or not _np.isfinite(c1) or c1 <= c0:
                c0, c1 = 0.0, 1.0
        vol = (vol - c0) / max(1e-12, (c1 - c0))
        np.clip(vol, 0.0, 1.0, out=vol)
        return vol

    def _load_zarr_slice(self, z: int):
        """Load and normalize a single Z slice to float32 (0..1)."""
        assert self._da_volume is not None
        z = max(0, min(int(z), int(self._da_volume.shape[0]) - 1))
        slab = self._da_volume[z, :, :].astype('float32').compute()
        self._z_index = int(z)
        # Normalize using cached clim if available, else min/max
        c0, c1 = None, None
        try:
            if hasattr(self, '_zarr_clim') and self._zarr_clim is not None:
                c0, c1 = self._zarr_clim
        except Exception as e:
            logger.debug("zarr clim fetch failed", exc_info=True)
        if c0 is None or c1 is None or c1 <= c0:
            c0 = float(slab.min())
            c1 = float(slab.max())
            if not np.isfinite(c0) or not np.isfinite(c1) or c1 <= c0:
                c0, c1 = 0.0, 1.0
        # Scale to 0..1
        slab = (slab - c0) / max(1e-12, (c1 - c0))
        slab = np.clip(slab, 0.0, 1.0, out=slab)
        return slab

    def _ensure_capture_buffers(self) -> None:
        if self._texture is None:
            self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.width, self.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        if self._fbo is None:
            self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._texture, 0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Capture FBO incomplete: 0x{status:x}")

    def _init_capture(self) -> None:
        self._ensure_capture_buffers()

    def _init_cuda_interop(self) -> None:
        """Init CUDA-GL interop and allocate destination via one-time DLPack bridge."""
        pycuda.gl.init()
        self._registered_tex = RegisteredImage(int(self._texture), GL.GL_TEXTURE_2D, graphics_map_flags.READ_ONLY)
        # Allocate a CuPy device array once, convert to Torch via DLPack (consumes capsule), then drop CuPy
        dev_cp = cp.empty((self.height, self.width, 4), dtype=cp.uint8)
        # Avoid deprecated toDlpack(): pass the array object to Torch's DLPack importer
        if hasattr(torch, 'from_dlpack'):
            self._torch_frame = torch.from_dlpack(dev_cp)
        else:
            self._torch_frame = torch.utils.dlpack.from_dlpack(dev_cp)
        del dev_cp  # do not use the CuPy array after handoff
        # Force contiguous layout (tight packing) and compute pitch
        try:
            self._torch_frame = self._torch_frame.contiguous()
        except Exception as e:
            logger.debug("Making torch frame contiguous failed (continuing): %s", e)
        # Pre-allocate auxiliary buffers for conversions to avoid per-frame alloc jitter
        try:
            self._torch_frame_argb = torch.empty_like(self._torch_frame)
        except Exception:
            self._torch_frame_argb = None  # type: ignore[attr-defined]
        try:
            self._yuv444 = torch.empty((self.height * 3, self.width), dtype=torch.uint8, device=self._torch_frame.device)
        except Exception:
            self._yuv444 = None  # type: ignore[attr-defined]
        # Compute pitch (bytes per row) from Torch tensor stride
        self._dst_pitch_bytes = int(self._torch_frame.stride(0) * self._torch_frame.element_size())
        row_bytes = int(self.width * 4)
        # Optional override to force tight pitch (debug/testing)
        try:
            if int(os.getenv('NAPARI_CUDA_FORCE_TIGHT_PITCH', '0')):
                self._dst_pitch_bytes = row_bytes
        except Exception as e:
            logger.debug("Reading NAPARI_CUDA_FORCE_TIGHT_PITCH failed: %s", e)
        try:
            logger.info("CUDA dst pitch: %d bytes (expected %d)", self._dst_pitch_bytes, row_bytes)
        except Exception as e:
            logger.debug("Pitch info log failed: %s", e)
        if self._dst_pitch_bytes != row_bytes:
            try:
                logger.warning(
                    "Non-tight pitch: dst_pitch=%d, expected=%d (width*4). This may cause right-edge artifacts.",
                    self._dst_pitch_bytes, row_bytes,
                )
            except Exception as e:
                logger.debug("Pitch warn log failed: %s", e)
        # Warn on odd dimensions which can cause crop issues in H.264 4:2:0
        if (self.width % 2) or (self.height % 2):
            try:
                logger.warning("Odd dimensions %dx%d may reveal right/bottom padding without SPS cropping.", self.width, self.height)
            except Exception as e:
                logger.debug("Odd-dimension warn log failed: %s", e)

    def _init_encoder(self) -> None:
        # Encoder input format: allow YUV444 (default), NV12 (4:2:0), or ARGB/ABGR (packed RGB)
        self._enc_input_fmt = os.getenv('NAPARI_CUDA_ENCODER_INPUT_FMT', 'YUV444').upper()
        if self._enc_input_fmt not in {'YUV444', 'NV12', 'ARGB', 'ABGR'}:
            self._enc_input_fmt = 'YUV444'
        try:
            logger.info("Encoder input format: %s", self._enc_input_fmt)
        except Exception as e:
            logger.debug("Encoder fmt info log failed: %s", e)
        # Configure encoder for low-latency streaming; repeat SPS/PPS on keyframes
        # Optional bitrate control for low-jitter CBR; if unsupported, fallback path below will be used
        try:
            bitrate = int(os.getenv('NAPARI_CUDA_BITRATE', '10000000'))
        except Exception:
            bitrate = None  # type: ignore[assignment]
        # Low-jitter CBR and low-latency settings
        rc_mode = os.getenv('NAPARI_CUDA_RC', 'cbr').lower()
        try:
            lookahead = int(os.getenv('NAPARI_CUDA_LOOKAHEAD', '0'))
        except Exception:
            lookahead = 0
        try:
            aq = int(os.getenv('NAPARI_CUDA_AQ', '0'))
        except Exception:
            aq = 0
        try:
            temporalaq = int(os.getenv('NAPARI_CUDA_TEMPORALAQ', '0'))
        except Exception:
            temporalaq = 0
        # Non-ref P frames (low-delay) toggle
        try:
            nonrefp = int(os.getenv('NAPARI_CUDA_NONREFP', '0'))
        except Exception:
            nonrefp = 0
        # Preset-based path only (stable, low-variance)
        preset_env = os.getenv('NAPARI_CUDA_PRESET', '')
        # IDR period (frames) for preset path (and for logging fallback)
        try:
            idr_period = int(os.getenv('NAPARI_CUDA_IDR_PERIOD', '600') or '600')
        except Exception:
            idr_period = 600

        # Preset/tuning path: low-latency tuning, explicit preset, no B-frames, repeat SPS/PPS, fixed IDR period
        preset = preset_env.strip() if preset_env.strip() else 'P3'
        kwargs = {
            'codec': 'h264',
            'tuning_info': 'low_latency',
            'preset': preset,
            'bf': 0,
            'repeatspspps': 1,
            'idrperiod': int(idr_period),
            'rc': rc_mode,
        }
        if bitrate and bitrate > 0:
            kwargs['bitrate'] = int(bitrate)
            kwargs['maxbitrate'] = int(bitrate)
        # Also set explicit framerate to align RC pacing with server fps
        kwargs['frameRateNum'] = int(max(1, int(self.fps)))
        kwargs['frameRateDen'] = 1
        try:
            self._encoder = pnvc.CreateEncoder(width=self.width, height=self.height, fmt=self._enc_input_fmt, usecpuinputbuffer=False, **kwargs)
            self._idr_period_cfg = int(idr_period)
            try:
                if int(os.getenv('NAPARI_CUDA_LOG_ENCODER_SETTINGS', '1') or '1'):
                    self._log_encoder_settings('preset', kwargs)
            except Exception:
                logger.debug("Log encoder settings failed", exc_info=True)
            try:
                logger.info(
                    "NVENC encoder created (preset): %dx%d fmt=%s preset=%s tuning=low_latency bf=0 rc=%s idrperiod=%d repeatspspps=1",
                    self.width, self.height, self._enc_input_fmt, preset, rc_mode.upper(), int(idr_period)
                )
            except Exception:
                logger.debug("Encoder creation info log failed", exc_info=True)
            # Force IDR next frame
            self._force_next_idr = True
            return
        except Exception as e:
            logger.warning("Preset NVENC path failed (%s)", e, exc_info=True)

        # If preset path fails for any reason, fall back to minimal encoder without tuning
        enc = getattr(self, '_encoder', None)
        if enc is None:
            try:
                self._encoder = pnvc.CreateEncoder(width=self.width, height=self.height, fmt=self._enc_input_fmt, usecpuinputbuffer=False)
                logger.warning("NVENC encoder created without preset kwargs; cadence may vary")
            except Exception as e:
                logger.exception("NVENC fallback encoder creation failed: %s", e)

    def _log_encoder_settings(self, path: str, init_kwargs: dict) -> None:
        """Emit a single-line summary of encoder settings.

        Combines the initialization kwargs we passed to NVENC with any
        reconfigurable params reported by the encoder at runtime.

        path label used in logs (e.g., 'preset').
        """
        try:
            # Canonicalize known fields from init kwargs
            k = {**init_kwargs}
            canon: dict[str, object] = {}
            def take(src_key: str, dst_key: str | None = None):
                dst_key = dst_key or src_key
                if src_key in k and k[src_key] is not None:
                    canon[dst_key] = k[src_key]

            # Common
            take('codec', 'codec')
            # Preset keys
            take('preset', 'preset')
            take('tuning_info', 'tuning')
            take('bf', 'bf')
            take('repeatspspps', 'repeatSPSPPS')
            take('idrperiod', 'idrPeriod')
            take('rc', 'rcMode')
            take('bitrate', 'bitrate')
            take('maxbitrate', 'maxBitrate')
            # Modern keys
            take('frameIntervalP', 'frameIntervalP')
            take('repeatSPSPPS', 'repeatSPSPPS')
            take('gopLength', 'gopLength')
            take('idrPeriod', 'idrPeriod')
            take('rcMode', 'rcMode')
            take('enablePTD', 'enablePTD')
            take('enableLookahead', 'enableLookahead')
            take('enableAQ', 'enableAQ')
            take('enableTemporalAQ', 'enableTemporalAQ')
            take('enableNonRefP', 'enableNonRefP')
            take('frameRateNum', 'frameRateNum')
            take('frameRateDen', 'frameRateDen')
            take('enableIntraRefresh', 'enableIntraRefresh')
            take('maxNumRefFrames', 'maxNumRefFrames')
            take('vbvBufferSize', 'vbvBufferSize')
            take('vbvInitialDelay', 'vbvInitialDelay')

            # Merge in live reconfigure params if available
            live = {}
            try:
                if hasattr(self._encoder, 'GetEncodeReconfigureParams'):
                    params = self._encoder.GetEncodeReconfigureParams()
                    # Pull known fields if present
                    for attr in (
                        'rateControlMode', 'averageBitrate', 'maxBitRate',
                        'vbvBufferSize', 'vbvInitialDelay', 'frameRateNum', 'frameRateDen', 'multiPass',
                    ):
                        if hasattr(params, attr):
                            live[attr] = getattr(params, attr)
            except Exception:
                logger.debug("GetEncodeReconfigureParams failed", exc_info=True)

            # Compose a flat, readable line
            # Prefer explicit canon values; supplement with live where not present
            for src, dst in (
                ('rateControlMode', 'rcMode'),
                ('averageBitrate', 'bitrate'),
                ('maxBitRate', 'maxBitrate'),
                ('vbvBufferSize', 'vbvBufferSize'),
                ('vbvInitialDelay', 'vbvInitialDelay'),
                ('frameRateNum', 'frameRateNum'),
                ('frameRateDen', 'frameRateDen'),
            ):
                if dst not in canon and src in live:
                    canon[dst] = live[src]

            # Stable order for readability
            order = [
                'codec','preset','tuning','frameIntervalP','bf','maxNumRefFrames',
                'gopLength','idrPeriod','repeatSPSPPS',
                'rcMode','bitrate','maxBitrate','vbvBufferSize','vbvInitialDelay',
                'enablePTD','enableLookahead','enableAQ','enableTemporalAQ','enableNonRefP',
                'frameRateNum','frameRateDen','enableIntraRefresh'
            ]
            parts = []
            for key in order:
                if key in canon:
                    parts.append(f"{key}={canon[key]}")
            logger.info("Encoder settings (%s): %s", path, ", ".join(parts))
        except Exception:
            # Best-effort only
            pass

    def reset_encoder(self) -> None:
        """Tear down and recreate the encoder to force a new GOP/IDR.

        Guarded by a lock to avoid races with concurrent Encode() calls.
        """
        with self._enc_lock:
            try:
                if self._encoder is not None:
                    try:
                        self._encoder.EndEncode()
                    except Exception as e:
                        logger.debug("EndEncode failed during reset: %s", e)
            finally:
                self._encoder = None
            try:
                self._init_encoder()
                logger.debug("NVENC encoder reset; next frame should include IDR")
            except Exception as e:
                logger.exception("Failed to reset NVENC encoder: %s", e)

    def force_idr(self) -> None:
        """Best-effort request to force next frame as IDR via Reconfigure."""
        logger.debug("Requesting encoder force IDR")
        try:
            if hasattr(self._encoder, 'GetEncodeReconfigureParams') and hasattr(self._encoder, 'Reconfigure'):
                params = self._encoder.GetEncodeReconfigureParams()
                # Some bindings expose forceIDR; ignore if not present
                if hasattr(params, 'forceIDR'):
                    setattr(params, 'forceIDR', 1)
                self._encoder.Reconfigure(params)
        except Exception as e:
            logger.debug("force_idr not supported: %s", e)

    def render_frame(self, azimuth_deg: Optional[float] = None) -> None:
        if azimuth_deg is not None and hasattr(self.view, "camera"):
            self.view.camera.azimuth = float(azimuth_deg)
        self._apply_pending_state()
        self.canvas.render()

    def apply_state(self, state: ServerSceneState) -> None:
        """Queue a complete scene state snapshot for the next frame."""
        with self._state_lock:
            self._pending_state = ServerSceneState(
                center=tuple(state.center) if state.center is not None else None,
                zoom=float(state.zoom) if state.zoom is not None else None,
                angles=tuple(state.angles) if state.angles is not None else None,
                current_step=tuple(state.current_step) if state.current_step is not None else None,
                zoom_factor=float(state.zoom_factor) if getattr(state, 'zoom_factor', None) is not None else None,
                zoom_anchor_px=tuple(state.zoom_anchor_px) if getattr(state, 'zoom_anchor_px', None) is not None else None,
                pan_dx_px=float(getattr(state, 'pan_dx_px', 0.0) or 0.0),
                pan_dy_px=float(getattr(state, 'pan_dy_px', 0.0) or 0.0),
                reset_view=bool(getattr(state, 'reset_view', False)),
                # Include volume render params if present
                volume_mode=str(getattr(state, 'volume_mode', None)) if getattr(state, 'volume_mode', None) is not None else None,
                volume_colormap=str(getattr(state, 'volume_colormap', None)) if getattr(state, 'volume_colormap', None) is not None else None,
                volume_clim=tuple(getattr(state, 'volume_clim')) if getattr(state, 'volume_clim', None) is not None else None,
                volume_opacity=float(getattr(state, 'volume_opacity')) if getattr(state, 'volume_opacity', None) is not None else None,
                volume_sample_step=float(getattr(state, 'volume_sample_step')) if getattr(state, 'volume_sample_step', None) is not None else None,
                # Include orbit deltas (one-shot)
                orbit_daz_deg=float(getattr(state, 'orbit_daz_deg')) if getattr(state, 'orbit_daz_deg', None) is not None else None,
                orbit_del_deg=float(getattr(state, 'orbit_del_deg')) if getattr(state, 'orbit_del_deg', None) is not None else None,
            )

    def _apply_pending_state(self) -> None:
        state: Optional[ServerSceneState] = None
        with self._state_lock:
            if self._pending_state is not None:
                state = self._pending_state
                self._pending_state = None
        if state is None:
            return
        try:
            cam = self.view.camera
            # Handle dims (Z step) if OME-Zarr volume is active FIRST so camera can be authoritative after
            if state.current_step is not None and getattr(self, '_da_volume', None) is not None:
                try:
                    # Interpret current_step against axes (default 'zyx')
                    z_idx = None
                    axes = (self._zarr_axes or 'zyx').lower()
                    # Find index of 'z' in provided dims; assume dims tuple matches axes order
                    if 'z' in axes:
                        pos = axes.index('z')
                        if pos < len(state.current_step):
                            z_idx = int(state.current_step[pos])
                    # Fallback: take first element
                    if z_idx is None and len(state.current_step) > 0:
                        z_idx = int(state.current_step[0])
                    if z_idx is not None and (self._z_index is None or int(z_idx) != int(self._z_index)):
                        slab = self._load_zarr_slice(int(z_idx))
                        # Update visual on the render thread
                        try:
                            self._visual.set_data(slab)
                            # Only update camera extents if the slab dimensions changed; avoid recenter/zoom churn
                            try:
                                h, w = int(slab.shape[0]), int(slab.shape[1])
                                # Track last seen slab size
                                last_wh = getattr(self, '_last_slab_wh', None)
                                cur_wh = (w, h)
                                if last_wh != cur_wh and hasattr(self.view, 'camera'):
                                    self.view.camera.set_range(x=(0, w), y=(0, h))
                                self._last_slab_wh = cur_wh
                                self._data_wh = cur_wh
                            except Exception:
                                logger.debug("zarr slice: camera set_range update failed", exc_info=True)
                        except Exception as e:
                            logger.debug("Update z-slice failed: %s", e)
                except Exception as e:
                    logger.debug("Apply dims state failed: %s", e)

            # Apply volume render params (one-shot, coalesced on render thread)
            try:
                vis = getattr(self, '_visual', None)
                if vis is not None and hasattr(vis, 'set_data') and hasattr(vis, 'method'):
                    # Render mode
                    vmode = getattr(state, 'volume_mode', None)
                    if vmode:
                        try:
                            m = str(vmode).lower()
                            if m in ('mip', 'translucent', 'iso'):
                                vis.method = m  # type: ignore[attr-defined]
                        except Exception:
                            logger.debug("apply volume mode failed", exc_info=True)
                    # Colormap
                    vcm = getattr(state, 'volume_colormap', None)
                    if vcm:
                        try:
                            name = str(vcm).lower()
                            if name == 'gray':
                                name = 'grays'
                            vis.cmap = name  # type: ignore[attr-defined]
                        except Exception:
                            logger.debug("apply volume colormap failed", exc_info=True)
                    # CLim
                    vcl = getattr(state, 'volume_clim', None)
                    if isinstance(vcl, tuple) and len(vcl) >= 2:
                        try:
                            lo = float(vcl[0]); hi = float(vcl[1])
                            if hi < lo:
                                lo, hi = hi, lo
                            vis.clim = (lo, hi)  # type: ignore[attr-defined]
                        except Exception:
                            logger.debug("apply volume clim failed", exc_info=True)
                    # Opacity
                    vo = getattr(state, 'volume_opacity', None)
                    if vo is not None:
                        try:
                            a = max(0.0, min(1.0, float(vo)))
                            if hasattr(vis, 'opacity'):
                                vis.opacity = float(a)  # type: ignore[attr-defined]
                        except Exception:
                            logger.debug("apply volume opacity failed", exc_info=True)
                    # Sample step (relative)
                    vs = getattr(state, 'volume_sample_step', None)
                    if vs is not None and hasattr(vis, 'relative_step_size'):
                        try:
                            rr = max(0.1, min(4.0, float(vs)))
                            vis.relative_step_size = float(rr)  # type: ignore[attr-defined]
                        except Exception:
                            logger.debug("apply volume sample_step failed", exc_info=True)
            except Exception:
                logger.debug("Apply volume params failed", exc_info=True)

            # Apply camera ops LAST so they remain authoritative
            try:
                # Optional one-shot reset to data extents
                if bool(getattr(state, 'reset_view', False)) and hasattr(self.view, 'camera'):
                    try:
                        w, h = getattr(self, '_data_wh', (self.width, self.height))
                        # If we have depth and a Turntable camera, also reset z range and distance
                        try:
                            from vispy.scene.cameras import TurntableCamera  # type: ignore
                        except Exception:
                            TurntableCamera = None  # type: ignore
                        cam = self.view.camera
                        d = getattr(self, '_data_d', None)
                        if TurntableCamera is not None and isinstance(cam, TurntableCamera) and d is not None:
                            # Reset full 3D range and re-frame distance
                            self.view.camera.set_range(x=(0, int(w)), y=(0, int(h)), z=(0, int(d)))
                            try:
                                self._frame_volume_camera(int(w), int(h), int(d))
                            except Exception:
                                logger.debug("reset_view: frame camera failed", exc_info=True)
                            # Reset orbit to defaults
                            setattr(cam, 'azimuth', 30.0)
                            setattr(cam, 'elevation', 30.0)
                        else:
                            # 2D or unknown camera: reset XY range
                            self.view.camera.set_range(x=(0, int(w)), y=(0, int(h)))
                        if self._debug_reset:
                            logger.info("reset_view applied: wh=(%d,%d) d=%s", int(w), int(h), str(d))
                    except Exception as e:
                        logger.debug("camera.reset failed: %s", e)
                # Orbit deltas (degrees) for TurntableCamera
                try:
                    from vispy.scene.cameras import TurntableCamera  # type: ignore
                except Exception:
                    TurntableCamera = None  # type: ignore
                if TurntableCamera is not None and isinstance(cam, TurntableCamera):
                    daz = getattr(state, 'orbit_daz_deg', None)
                    delv = getattr(state, 'orbit_del_deg', None)
                    if (daz is not None and float(daz) != 0.0) or (delv is not None and float(delv) != 0.0):
                        try:
                            cur_az = float(getattr(cam, 'azimuth', 0.0) or 0.0)
                            cur_el = float(getattr(cam, 'elevation', 0.0) or 0.0)
                            new_az = cur_az + float(daz or 0.0)
                            # Wrap azimuth to [-180, 180]
                            if new_az > 180.0:
                                new_az = ((new_az + 180.0) % 360.0) - 180.0
                            if new_az < -180.0:
                                new_az = ((new_az - 180.0) % 360.0) + 180.0
                            new_el = cur_el + float(delv or 0.0)
                            # Clamp elevation
                            el_min = float(getattr(self, '_orbit_el_min', -85.0))
                            el_max = float(getattr(self, '_orbit_el_max', 85.0))
                            if el_min > el_max:
                                el_min, el_max = el_max, el_min
                            if new_el < el_min:
                                new_el = el_min
                            if new_el > el_max:
                                new_el = el_max
                            setattr(cam, 'azimuth', float(new_az))
                            setattr(cam, 'elevation', float(new_el))
                            if self._debug_orbit:
                                logger.info(
                                    "orbit: daz=%.2f del=%.2f -> az=%.2f el=%.2f (clamp=[%.1f,%.1f])",
                                    float(daz or 0.0), float(delv or 0.0), float(new_az), float(new_el), float(el_min), float(el_max)
                                )
                        except Exception:
                            logger.debug("camera.orbit apply failed", exc_info=True)

                # Anchored zoom (canvas/video pixel anchor)
                zf = getattr(state, 'zoom_factor', None)
                anc = getattr(state, 'zoom_anchor_px', None)
                if zf is not None and float(zf) > 0.0 and anc is not None and hasattr(self.view, 'camera'):
                    try:
                        cw, ch = (self.canvas.size if (self.canvas is not None and hasattr(self.canvas, 'size')) else (self.width, self.height))
                        # Special-case TurntableCamera: adjust distance for zoom
                        if TurntableCamera is not None and isinstance(cam, TurntableCamera):
                            # Map server zoom factor to distance scale directly:
                            # factor < 1 -> zoom IN (distance decreases), factor > 1 -> zoom OUT (distance increases)
                            try:
                                cur_d = float(getattr(cam, 'distance', 1.0) or 1.0)
                                zf_raw = float(zf)
                                new_d = max(1e-6, cur_d * zf_raw)
                                cam.distance = new_d  # type: ignore[attr-defined]
                                if self._debug_zoom_drift:
                                    try:
                                        logger.info("zoom_3d: f=%.4f dist_before=%.4f dist_after=%.4f", float(zf_raw), cur_d, new_d)
                                    except Exception:
                                        logger.debug("zoom_3d log failed", exc_info=True)
                            except Exception:
                                logger.debug("turntable zoom(distance) failed", exc_info=True)
                        else:
                            # 2D PanZoom path: map anchor to world and use camera.zoom()
                            # Map anchor from canvas pixels (top-left origin for transform) to world
                            center_world = None
                            try:
                                ax_px = float(anc[0])
                                ay_bl = float(anc[1])
                                ay_tl = float(ch) - ay_bl
                                tr = self.view.transform * self.view.scene.transform
                                mapped = tr.imap([ax_px, ay_tl, 0, 1])
                                wx = float(mapped[0]); wy = float(mapped[1])
                                center_world = (wx, wy)
                            except Exception:
                                try:
                                    rect = getattr(cam, 'rect', None)
                                    if rect is not None and cw and ch:
                                        ax_px = float(anc[0])
                                        ay_bl = float(anc[1])
                                        sx = ax_px / float(cw)
                                        sy_bl = ay_bl / float(ch)
                                        wx = float(rect.left) + sx * float(rect.width)
                                        wy = float(rect.bottom) + sy_bl * float(rect.height)
                                        center_world = (wx, wy)
                                except Exception:
                                    center_world = None
                            if center_world is None:
                                center_world = (float(anc[0]), float(anc[1]))
                            if self._debug_zoom_drift:
                                self._log_zoom_drift(float(zf), (float(anc[0]), float(anc[1])), (float(center_world[0]), float(center_world[1])), int(cw), int(ch))
                            else:
                                self.view.camera.zoom(float(zf), center=center_world)  # type: ignore[call-arg]
                    except Exception as e:
                        logger.debug("camera.zoom_at failed: %s", e)
                # Pixel-space pan: in 3D turn vertical drag into dolly (distance)
                # and horizontal into lateral world pan; 2D maps to world delta.
                dx = float(getattr(state, 'pan_dx_px', 0.0) or 0.0)
                dy = float(getattr(state, 'pan_dy_px', 0.0) or 0.0)
                if (dx != 0.0 or dy != 0.0) and hasattr(cam, 'center'):
                    try:
                        cw, ch = (self.canvas.size if (self.canvas is not None and hasattr(self.canvas, 'size')) else (self.width, self.height))
                        # Special-case TurntableCamera: derive per-pixel world scale from fov and distance
                        try:
                            from vispy.scene.cameras import TurntableCamera  # type: ignore
                        except Exception:
                            TurntableCamera = None  # type: ignore
                        if TurntableCamera is not None and isinstance(cam, TurntableCamera):
                            try:
                                fov_deg = float(getattr(cam, 'fov', 60.0) or 60.0)
                                dist = float(getattr(cam, 'distance', 1.0) or 1.0)
                                fov_rad = math.radians(max(1e-3, min(179.0, fov_deg)))
                                scale_y = 2.0 * dist * math.tan(0.5 * fov_rad) / max(1e-6, float(ch))
                                scale_x = scale_y * (float(cw) / max(1e-6, float(ch)))
                                # Lateral pan from horizontal drag
                                dwx = dx * scale_x
                                # Dolly from vertical drag: pan down (dy>0) -> closer (reduce distance)
                                if dy != 0.0:
                                    try:
                                        new_d = max(1e-6, dist - (dy * scale_y))
                                        cam.distance = float(new_d)  # type: ignore[attr-defined]
                                    except Exception:
                                        logger.debug("turntable dolly(distance) failed", exc_info=True)
                                # No vertical world pan when in 3D dolly mode
                                dwy = 0.0
                                if self._debug_pan:
                                    try:
                                        c0 = getattr(cam, 'center', None)
                                        logger.info("pan_3d: dpx=(%.2f,%.2f) scale=(%.6f,%.6f) world_pan=(%.6f,%.6f) dolly %.4f->%.4f center_before=%s fov=%.2f canvas=%dx%d",
                                                    float(dx), float(dy), float(scale_x), float(scale_y), float(dwx), float(dwy), float(dist), float(getattr(cam, 'distance', dist)), str(c0), float(fov_deg), int(cw), int(ch))
                                    except Exception:
                                        logger.debug("pan_3d log(pre) failed", exc_info=True)
                            except Exception:
                                # Fallback to small proportional shift if unavailable
                                dwx = dx * 0.01
                                dwy = 0.0
                                logger.debug("turntable pan scale compute failed; using tiny fallback world delta", exc_info=True)
                        else:
                            # 2D mapping using inverse transform at canvas center
                            if self._debug_pan:
                                # Use detailed 2D instrumented path which applies the pan
                                self._log_pan_mapping(dx, dy, int(cw), int(ch))
                                # Already applied; skip duplicate center update
                                dwx = dwy = 0.0
                            else:
                                try:
                                    cx_px = float(cw) * 0.5
                                    cy_px = float(ch) * 0.5
                                    tr = self.view.transform * self.view.scene.transform
                                    p0 = tr.imap((cx_px, cy_px))
                                    p1 = tr.imap((cx_px + dx, cy_px + dy))
                                    dwx = float(p1[0] - p0[0])
                                    dwy = float(p1[1] - p0[1])
                                except Exception:
                                    # Fallback: approximate with zoom scale if available
                                    z = float(getattr(cam, 'zoom', 1.0) or 1.0)
                                    inv = 1.0 / max(1e-6, z)
                                    dwx = dx * inv
                                    dwy = dy * inv
                        c = getattr(cam, 'center', None)
                        if isinstance(c, (tuple, list)) and len(c) >= 2:
                            cam.center = (float(c[0]) - dwx, float(c[1]) - dwy)  # type: ignore[attr-defined]
                        else:
                            cam.center = (-dwx, -dwy)  # type: ignore[attr-defined]
                        if TurntableCamera is not None and isinstance(cam, TurntableCamera) and self._debug_pan:
                            try:
                                c1 = getattr(cam, 'center', None)
                                logger.info(
                                    "pan_3d: applied center_after=%s (added world delta=(%.6f,%.6f))",
                                    str(c1), float(dwx), float(dwy)
                                )
                            except Exception:
                                logger.debug("pan_3d log(post) failed", exc_info=True)
                    except Exception as e:
                        logger.debug("camera.pan_px failed: %s", e)
                # Legacy absolute camera fields (kept for back-compat)
                if state.center is not None and hasattr(cam, 'center'):
                    try:
                        cam.center = state.center  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.debug("Apply center failed: %s", e)
                if state.zoom is not None and hasattr(cam, 'zoom'):
                    try:
                        cam.zoom = state.zoom  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.debug("Apply zoom failed: %s", e)
                if state.angles is not None and hasattr(cam, 'angles'):
                    try:
                        cam.angles = state.angles  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.debug("Apply angles failed: %s", e)
            except Exception as e:
                logger.debug("Apply camera ops failed: %s", e)
        except Exception as e:
            logger.debug("Apply state failed: %s", e)

    def _capture_blit_gpu_ns(self) -> Optional[int]:
        try:
            # Ensure we have a pair of timer queries
            if self._query_ids is None:
                ids = GL.glGenQueries(2)
                # PyOpenGL may return a list/tuple or single int for count=2
                if isinstance(ids, (list, tuple)) and len(ids) == 2:
                    self._query_ids = (int(ids[0]), int(ids[1]))
                else:
                    # Fallback: generate individually
                    q0 = int(GL.glGenQueries(1))
                    q1 = int(GL.glGenQueries(1))
                    self._query_ids = (q0, q1)

            qids = self._query_ids
            cur = self._query_idx
            prev = 1 - cur

            # Begin query for this frame
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, qids[cur])

            # Perform the blit
            bound_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
            read_fbo = int(bound_fbo)
            draw_fbo = int(self._fbo)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
            GL.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)

            # End this frame's query
            GL.glEndQuery(GL.GL_TIME_ELAPSED)

            # Read previous frame's result (blocking only if needed, typically ready)
            gpu_ns = None
            if self._query_started:
                result = GL.GLuint64(0)
                GL.glGetQueryObjectui64v(qids[prev], GL.GL_QUERY_RESULT, result)
                gpu_ns = int(result.value)
            else:
                # First frame: nothing to read yet
                self._query_started = True

            # Restore default binds
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)

            # Swap query index for next frame
            self._query_idx = prev
            return gpu_ns
        except Exception as e:
            logger.debug("Blit with timer query failed; falling back without timing: %s", e)
            try:
                bound_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
                read_fbo = int(bound_fbo)
                draw_fbo = int(self._fbo)
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
                GL.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, read_fbo)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)
            except Exception as e2:
                logger.debug("Blit fallback failed: %s", e2)
            return None


    def capture_and_encode_packet(self) -> tuple[FrameTimings, Optional[bytes], int, int]:
        """Same as capture_and_encode, but also returns the packet and flags.

        Flags bit 0x01 indicates keyframe (IDR/CRA).
        """
        # Debug env is logged once at init by DebugDumper
        # Capture wall and monotonic timestamps at frame start
        wall_ts = time.time()
        t0 = time.perf_counter()
        t_r0 = time.perf_counter()
        # Optional animation
        if self._animate and hasattr(self.view, "camera"):
            t = time.perf_counter() - self._anim_start
            cam = self.view.camera
            # 3D: simple turntable
            try:
                from vispy.scene.cameras import TurntableCamera  # type: ignore
                if isinstance(cam, TurntableCamera):
                    try:
                        cam.azimuth = (self._animate_dps * t) % 360.0
                    except Exception as e:
                        logger.debug("Animate(3D) failed: %s", e)
                else:
                    # 2D PanZoom: gentle pan + zoom pulse around center
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
            except Exception as e:
                logger.debug("Animate(camera) failed to classify: %s", e)
        self._apply_pending_state()
        self.canvas.render()
        t_r1 = time.perf_counter()
        render_ms = (t_r1 - t_r0) * 1000.0
        # Blit timing (GPU + CPU)
        t_b0 = time.perf_counter()
        blit_gpu_ns = self._capture_blit_gpu_ns()
        t_b1 = time.perf_counter()
        blit_cpu_ms = (t_b1 - t_b0) * 1000.0
        t_m0 = time.perf_counter()
        mapping = self._registered_tex.map()
        cuda_array = mapping.array(0, 0)
        t_m1 = time.perf_counter()
        map_ms = (t_m1 - t_m0) * 1000.0
        start_evt = cuda.Event()
        end_evt = cuda.Event()
        start_evt.record()
        m = cuda.Memcpy2D()
        m.set_src_array(cuda_array)
        m.set_dst_device(int(self._torch_frame.data_ptr()))
        m.width_in_bytes = self.width * 4
        m.height = self.height
        m.dst_pitch = self._dst_pitch_bytes if self._dst_pitch_bytes is not None else (self.width * 4)
        m(aligned=True)
        end_evt.record()
        end_evt.synchronize()
        copy_ms = start_evt.time_till(end_evt)
        try:
            # Optional deep debug: unified triplet dump
            if hasattr(self, "_debug") and self._debug.cfg.enabled and self._debug.cfg.frames_remaining > 0:
                self._debug.dump_triplet(int(self._texture), self.width, self.height, self._torch_frame)
        except Exception as e:
            logger.warning("Debug triplet failed: %s", e, exc_info=True)
        finally:
            mapping.unmap()
        # Measure color conversion (swizzle) and NVENC separately
        t_c0 = time.perf_counter()
        # Optional pre-encode raw dump (RGBA) for isolation
        try:
            dump_raw = int(os.getenv('NAPARI_CUDA_DUMP_RAW', '0'))
        except Exception:
            dump_raw = 0
        if dump_raw > 0:
            try:
                self._debug.dump_cuda_rgba(self._torch_frame, self.width, self.height, prefix="raw")
                os.environ['NAPARI_CUDA_DUMP_RAW'] = str(max(0, dump_raw - 1))
            except Exception as e:
                logger.debug("Pre-encode raw dump failed: %s", e)
        # Convert RGBA (GL) -> encoder input format (BT.709 limited) on GPU
        src = self._torch_frame
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
                    rm = float(r.mean().item())
                    gm = float(g.mean().item())
                    bm = float(b.mean().item())
                    ym = float(y.mean().item())
                    cbm = float(cb.mean().item())
                    crm = float(cr.mean().item())
                    logger.info(
                        "Pre-encode channel means: R=%.3f G=%.3f B=%.3f | Y=%.3f Cb=%.3f Cr=%.3f",
                        rm, gm, bm, ym, cbm, crm,
                    )
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
                # Reorder channels only; keep interleaved 4-channel layout
                if self._enc_input_fmt == 'ARGB':
                    dst = src[..., [3, 0, 1, 2]].contiguous()
                else:
                    # ABGR
                    dst = src[..., [3, 2, 1, 0]].contiguous()
            else:
                # NV12 (4:2:0): Y plane HxW + interleaved UV plane (H/2 x W)
                if not hasattr(self, '_logged_swizzle'):
                    logger.info("Pre-encode swizzle: RGBA -> NV12 (BT709 LIMITED, 4:2:0 UV interleaved)")
                    setattr(self, '_logged_swizzle', True)
                # Average pool Cb/Cr over 2x2 blocks
                cb2 = F.avg_pool2d(cb.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
                cr2 = F.avg_pool2d(cr.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
                H2, W2 = cb2.shape
                dst = torch.empty((H + H2, W), dtype=torch.uint8, device=src.device)
                dst[0:H, :] = y.to(torch.uint8)
                uv = dst[H:, :]
                # Interleave subsampled chroma: U at even columns, V at odd columns
                uv[:, 0::2] = cb2.to(torch.uint8)
                uv[:, 1::2] = cr2.to(torch.uint8)
        except Exception as e:
            logger.exception("Pre-encode color conversion failed: %s", e)
            dst = src[..., [3, 0, 1, 2]]
        # Ensure GPU kernels complete before timing conversion cost
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            logger.debug("torch.cuda.synchronize failed", exc_info=True)
        t_c1 = time.perf_counter()
        convert_ms = (t_c1 - t_c0) * 1000.0
        with self._enc_lock:
            enc = self._encoder
        if enc is None:
            pkt_obj = None
        else:
            t_e0 = time.perf_counter()
            # Force IDR on first frame after init/reset for decoder sync
            pic_flags = 0
            try:
                if getattr(self, '_force_next_idr', False):
                    pic_flags |= int(pnvc.NV_ENC_PIC_FLAGS.FORCEIDR)
                    pic_flags |= int(pnvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
                    self._force_next_idr = False
            except Exception:
                logger.debug("force_next_idr flag check failed", exc_info=True)
            if pic_flags:
                pkt_obj = enc.Encode(dst, pic_flags)
            else:
                pkt_obj = enc.Encode(dst)
            t_e1 = time.perf_counter()
            t_p0 = t_e1
            t_p1 = t_e1
            # Increment index and optional keyframe logging
            try:
                self._frame_index += 1
                if self._log_keyframes:
                    def _packet_is_keyframe(obj) -> bool:
                        try:
                            data = bytes(obj)
                        except Exception:
                            return False
                        # Try Annex B (start codes)
                        i = 0
                        n = len(data)
                        seen = False
                        while i + 3 < n:
                            if data[i] == 0 and data[i+1] == 0 and data[i+2] == 1:
                                if i + 3 < n:
                                    nal_type = data[i+3] & 0x1F
                                    if nal_type == 5:
                                        seen = True
                                i += 3
                            elif i + 4 < n and data[i] == 0 and data[i+1] == 0 and data[i+2] == 0 and data[i+3] == 1:
                                if i + 4 < n:
                                    nal_type = data[i+4] & 0x1F
                                    if nal_type == 5:
                                        seen = True
                                i += 4
                            else:
                                i += 1
                        if seen:
                            return True
                        # Try AVCC (length-prefixed)
                        i = 0
                        while i + 4 <= n:
                            ln = int.from_bytes(data[i:i+4], 'big')
                            i += 4
                            if ln <= 0 or i + ln > n:
                                break
                            nal_type = data[i] & 0x1F if ln >= 1 else 0
                            if nal_type == 5:
                                return True
                            i += ln
                        return False

                    keyframe = False
                    if pkt_obj is not None:
                        if isinstance(pkt_obj, (list, tuple)):
                            for part in pkt_obj:
                                if _packet_is_keyframe(part):
                                    keyframe = True
                                    break
                        else:
                            keyframe = _packet_is_keyframe(pkt_obj)
                    logger.debug("Encode frame %d: keyframe=%s", self._frame_index, bool(keyframe))
            except Exception as e:
                logger.debug("Keyframe instrumentation skipped: %s", e)
        # If encoder wasn't available, set NVENC time to 0
        encode_ms = ((t_e1 - t_e0) * 1000.0) if enc is not None else 0.0
        pack_ms = ((t_p1 - t_p0) * 1000.0) if enc is not None else 0.0
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
        seq = int(self._frame_index)
        return timings, pkt_obj, flags, seq

    # (packer is now provided by bitstream.py)

    @staticmethod
    def _torch_from_cupy(arr: cp.ndarray):
        # Legacy helper not used; kept for compatibility if needed elsewhere
        dl = arr.toDlpack()
        return torch.utils.dlpack.from_dlpack(dl)

    def cleanup(self) -> None:
        try:
            if self._registered_tex is not None:
                self._registered_tex.unregister()
        except Exception as e:
            logger.debug("Cleanup: unregister RegisteredImage failed: %s", e)
        self._registered_tex = None
        try:
            if self._fbo is not None:
                GL.glDeleteFramebuffers(int(self._fbo))
        except Exception as e:
            logger.debug("Cleanup: delete FBO failed: %s", e)
        self._fbo = None
        try:
            if self._texture is not None:
                GL.glDeleteTextures(int(self._texture))
        except Exception as e:
            logger.debug("Cleanup: delete texture failed: %s", e)
        self._texture = None
        try:
            with self._enc_lock:
                if self._encoder is not None:
                    self._encoder.EndEncode()
                self._encoder = None
        except Exception as e:
            logger.debug("Cleanup: encoder EndEncode failed: %s", e)
        try:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
        except Exception as e:
            logger.debug("Cleanup: CUDA context pop/detach failed: %s", e)
        self.cuda_ctx = None
        try:
            if self.canvas is not None:
                self.canvas.close()
        except Exception as e:
            logger.debug("Cleanup: canvas close failed: %s", e)
        self.canvas = None
        try:
            if self.egl_display is not None:
                EGL.eglTerminate(self.egl_display)
        except Exception as e:
            logger.debug("Cleanup: eglTerminate failed: %s", e)
        self.egl_display = None

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
