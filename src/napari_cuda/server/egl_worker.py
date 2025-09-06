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
from typing import Optional, List, Tuple
import threading

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
import PyNvVideoCodec as pnvc  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class FrameTimings:
    render_ms: float
    blit_gpu_ns: Optional[int]
    map_ms: float
    copy_ms: float
    encode_ms: float
    total_ms: float
    packet_bytes: Optional[int]


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
                 volume_depth: int = 64, volume_dtype: str = "float32", volume_relative_step: Optional[float] = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.use_volume = bool(use_volume)
        self.fps = int(fps)
        self.volume_depth = int(volume_depth)
        self.volume_dtype = str(volume_dtype)
        self.volume_relative_step = volume_relative_step

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

        # Atomic state application
        self._state_lock = threading.Lock()
        self._pending_state: Optional[ServerSceneState] = None

        self._init_egl()
        self._init_cuda()
        self._init_vispy_scene()
        self._init_capture()
        self._init_cuda_interop()
        self._init_encoder()
        # Cached parameter sets for robust streaming
        self._cached_sps: Optional[bytes] = None
        self._cached_pps: Optional[bytes] = None
        self._cached_vps: Optional[bytes] = None

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
        view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60, distance=500)
        if self.use_volume:
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
            except Exception:
                pass
        else:
            image = np.random.randint(0, 255, (self.height, self.width, 4), dtype=np.uint8)
            visual = scene.visuals.Image(image, parent=view.scene)
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
        # Compute pitch (bytes per row) from Torch tensor stride
        self._dst_pitch_bytes = int(self._torch_frame.stride(0) * self._torch_frame.element_size())

    def _init_encoder(self) -> None:
        # Configure encoder for low-latency streaming; repeat SPS/PPS on keyframes
        kwargs = {
            'codec': 'h264',
            'tuning_info': 'low_latency',
            'preset': 'P3',
            'bf': 0,
            'repeatspspps': 1,
        }
        try:
            self._encoder = pnvc.CreateEncoder(width=self.width, height=self.height, fmt="ABGR", usecpuinputbuffer=False, **kwargs)
        except Exception:
            # Fallback if kwargs unsupported by this binding version
            self._encoder = pnvc.CreateEncoder(width=self.width, height=self.height, fmt="ABGR", usecpuinputbuffer=False)

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
                except Exception:
                    pass
            if state.zoom is not None and hasattr(cam, 'zoom'):
                try:
                    cam.zoom = state.zoom  # type: ignore[attr-defined]
                except Exception:
                    pass
            if state.angles is not None and hasattr(cam, 'angles'):
                try:
                    cam.angles = state.angles  # type: ignore[attr-defined]
                except Exception:
                    pass
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

    def capture_and_encode(self) -> FrameTimings:
        t0 = time.perf_counter()
        t_r0 = time.perf_counter()
        self._apply_pending_state()
        self.canvas.render()
        t_r1 = time.perf_counter()
        render_ms = (t_r1 - t_r0) * 1000.0
        blit_gpu_ns = self._capture_blit_gpu_ns()
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
        mapping.unmap()
        t_e0 = time.perf_counter()
        pkt_obj = self._encoder.Encode(self._torch_frame)
        # PyNvVideoCodec may return a list of packets (e.g., SPS/PPS + frame)
        if isinstance(pkt_obj, (list, tuple)):
            try:
                pkt = b"".join(bytes(p) for p in pkt_obj)
            except Exception:
                pkt = None
        else:
            pkt = pkt_obj
        t_e1 = time.perf_counter()
        encode_ms = (t_e1 - t_e0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        pkt_bytes = len(pkt) if pkt else None
        return FrameTimings(render_ms, blit_gpu_ns, map_ms, copy_ms, encode_ms, total_ms, pkt_bytes)

    def capture_and_encode_packet(self) -> tuple[FrameTimings, Optional[bytes], int]:
        """Same as capture_and_encode, but also returns the packet and flags.

        Flags bit 0x01 indicates keyframe (IDR/CRA).
        """
        t0 = time.perf_counter()
        t_r0 = time.perf_counter()
        self._apply_pending_state()
        self.canvas.render()
        t_r1 = time.perf_counter()
        render_ms = (t_r1 - t_r0) * 1000.0
        blit_gpu_ns = self._capture_blit_gpu_ns()
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
        mapping.unmap()
        t_e0 = time.perf_counter()
        pkt_obj = self._encoder.Encode(self._torch_frame)
        pkt, is_key = self._pack_to_annexb(pkt_obj)
        t_e1 = time.perf_counter()
        encode_ms = (t_e1 - t_e0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        pkt_bytes = len(pkt) if pkt else None
        timings = FrameTimings(render_ms, blit_gpu_ns, map_ms, copy_ms, encode_ms, total_ms, pkt_bytes)
        flags = 0x01 if is_key else 0
        return timings, (pkt if pkt else None), flags

    def _pack_to_annexb(self, packets) -> Tuple[Optional[bytes], bool]:
        """Normalize encoder output to Annex B and detect keyframe.

        Accepts bytes or a list/tuple of bytes. Handles AVCC length-prefixed NALs.
        Returns (payload bytes or None, is_keyframe).
        """
        if packets is None:
            return None, False
        chunks: List[bytes] = []
        if isinstance(packets, (bytes, bytearray, memoryview)):
            chunks = [bytes(packets)]
        elif isinstance(packets, (list, tuple)):
            for p in packets:
                if p is None:
                    continue
                chunks.append(bytes(p))
        else:
            try:
                chunks = [bytes(packets)]
            except Exception:
                return None, False

        def is_annexb(buf: bytes) -> bool:
            return buf.startswith(b"\x00\x00\x01") or buf.startswith(b"\x00\x00\x00\x01")

        def parse_nals(buf: bytes) -> List[bytes]:
            nals: List[bytes] = []
            if not buf:
                return nals
            if is_annexb(buf):
                i = 0
                l = len(buf)
                idx: List[int] = []
                while i < l - 3:
                    if buf[i:i+3] == b"\x00\x00\x01":
                        idx.append(i); i += 3
                    elif i < l - 4 and buf[i:i+4] == b"\x00\x00\x00\x01":
                        idx.append(i); i += 4
                    else:
                        i += 1
                if not idx:
                    return [buf]
                idx.append(l)
                for a, b in zip(idx, idx[1:]):
                    j = a
                    while j < b and buf[j] == 0:
                        j += 1
                    if j < b and buf[j] == 1:
                        j += 1
                    nal = buf[j:b]
                    if nal:
                        nals.append(nal)
                return nals
            else:
                # AVCC: 4-byte big-endian length prefixes
                i = 0
                l = len(buf)
                while i + 4 <= l:
                    ln = int.from_bytes(buf[i:i+4], 'big'); i += 4
                    if ln <= 0 or i + ln > l:
                        return [buf]
                    nals.append(buf[i:i+ln]); i += ln
                return nals if nals else [buf]

        nal_units: List[bytes] = []
        for c in chunks:
            nal_units.extend(parse_nals(c))

        def t264(b0: int) -> int: return b0 & 0x1F
        def t265(b0: int) -> int: return (b0 >> 1) & 0x3F

        is_key = False
        saw_sps = saw_pps = saw_vps = False
        for nal in nal_units:
            if not nal:
                continue
            b0 = nal[0]
            n264 = t264(b0)
            n265 = t265(b0)
            if n264 == 7:  # SPS
                saw_sps = True; self._cached_sps = nal
            elif n264 == 8:  # PPS
                saw_pps = True; self._cached_pps = nal
            elif n264 == 5:  # IDR
                is_key = True
            if n265 == 32:  # VPS
                saw_vps = True; self._cached_vps = nal
            elif n265 == 33:  # SPS
                saw_sps = True; self._cached_sps = nal
            elif n265 == 34:  # PPS
                saw_pps = True; self._cached_pps = nal
            elif n265 in (19, 20, 21):  # IDR/CRA
                is_key = True

        out: List[bytes] = []
        if is_key:
            if self._cached_vps and not saw_vps:
                out.append(self._cached_vps)
            if self._cached_sps and not saw_sps:
                out.append(self._cached_sps)
            if self._cached_pps and not saw_pps:
                out.append(self._cached_pps)
        out.extend(nal_units)
        if not out:
            return None, False
        ba = bytearray()
        first = True
        for nal in out:
            ba.extend(b"\x00\x00\x00\x01" if first else b"\x00\x00\x01"); first = False
            ba.extend(nal)
        return bytes(ba), is_key

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
            if self._encoder is not None:
                self._encoder.EndEncode()
        except Exception as e:
            logger.debug("Cleanup: encoder EndEncode failed: %s", e)
        self._encoder = None
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
