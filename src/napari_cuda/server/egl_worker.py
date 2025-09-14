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
import threading
import math

import numpy as np

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
    """Minimal scene state snapshot applied atomically per frame."""
    center: Optional[tuple[float, float, float]] = None
    zoom: Optional[float] = None
    angles: Optional[tuple[float, float, float]] = None
    current_step: Optional[tuple[int, ...]] = None


class EGLRendererWorker:
    """Headless VisPy renderer using EGL with CUDA interop and NVENC."""

    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False, fps: int = 60,
                 volume_depth: int = 64, volume_dtype: str = "float32", volume_relative_step: Optional[float] = None,
                 animate: bool = False, animate_dps: float = 30.0) -> None:
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

        # Ensure partial initialization is cleaned up if any step fails
        try:
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
                "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s",
                self.width, self.height, getattr(self, '_enc_input_fmt', 'unknown'), self.fps, self._animate,
            )
        except Exception as e:
            logger.debug("Init info log failed: %s", e)

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
        # Use a 3D camera for volumes, but a 2D PanZoom camera for images to avoid perspective skew
        if self.use_volume:
            view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60, distance=500)
            # Full XY resolution for apples-to-apples comparison with Image
            if self.volume_dtype == "float16":
                dtype = np.float16
            elif self.volume_dtype == "uint8":
                dtype = np.uint8
            else:
                dtype = np.float32
            volume = np.random.rand(self.volume_depth, self.height, self.width).astype(dtype)
            visual = scene.visuals.Volume(volume, parent=view.scene, method="mip", cmap="viridis")
            # Optional sampling control
            try:
                if self.volume_relative_step is not None and hasattr(visual, "relative_step_size"):
                    visual.relative_step_size = float(self.volume_relative_step)
            except Exception as e:
                logger.debug("Set volume relative_step_size failed: %s", e)
        else:
            # 2D image: orthographic view that fills the frame (no trapezoid)
            view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
            image = make_rgba_image(self.width, self.height)
            visual = scene.visuals.Image(image, parent=view.scene)
            # Ensure the full image extents are visible without perspective distortion
            try:
                view.camera.set_range(x=(0, self.width), y=(0, self.height))
            except Exception as e:
                logger.debug("PanZoom set_range failed: %s", e)
        canvas.render()
        self.canvas = canvas
        self.view = view
        self._visual = visual

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
                pass
            try:
                logger.info(
                    "NVENC encoder created (preset): %dx%d fmt=%s preset=%s tuning=low_latency bf=0 rc=%s idrperiod=%d repeatspspps=1",
                    self.width, self.height, self._enc_input_fmt, preset, rc_mode.upper(), int(idr_period)
                )
            except Exception:
                pass
            # Force IDR next frame
            self._force_next_idr = True
            return
        except Exception as e:
            logger.warning("Preset NVENC path failed (%s)", e, exc_info=True)

        # If preset path fails for any reason, fall back to minimal encoder without tuning
        try:
            self._encoder  # type: ignore[attr-defined]
        except Exception:
            pass
        if self._encoder is None:
            try:
                self._encoder = pnvc.CreateEncoder(width=self.width, height=self.height, fmt=self._enc_input_fmt, usecpuinputbuffer=False)
                logger.warning("NVENC encoder created without preset kwargs; cadence may vary")
            except Exception:
                pass

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
                pass

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
        try:
            logger.debug("Requesting encoder force IDR")
        except Exception:
            pass
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
            pass
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
                pass
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
