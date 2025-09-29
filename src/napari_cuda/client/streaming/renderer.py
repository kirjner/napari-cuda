from __future__ import annotations

import importlib
import logging
import os
from collections import deque
from contextlib import contextmanager
from typing import Optional, Tuple, Any, Callable

import numpy as np
from vispy.gloo import Texture2D, Program

from OpenGL import GL as GL  # type: ignore

from napari_cuda.client.streaming.vt_frame import FrameLease

logger = logging.getLogger(__name__)

_MISSING = object()


def _safe_call(
    func: Callable[..., Any],
    *args: Any,
    log_message: Optional[str] = None,
    default: Any = _MISSING,
    **kwargs: Any,
) -> Any:
    """Invoke ``func`` and log boundary failures when a fallback is acceptable."""

    try:
        return func(*args, **kwargs)
    except Exception:  # pragma: no cover - subsystem boundary guard
        if log_message:
            logger.exception(log_message)
        else:
            logger.exception("boundary call failed: %r", func)
        if default is _MISSING:
            raise
        return default
VERTEX_SHADER = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

FRAGMENT_SHADER = """
uniform sampler2D texture;
varying vec2 v_texcoord;

void main() {
    gl_FragColor = texture2D(texture, v_texcoord);
}
"""


class VTReleaseQueue:
    """Track VT texture releases until the GPU signals completion."""

    def __init__(self) -> None:
        # Each entry stores (GLsync handle, tex_cap)
        self._entries: deque[Tuple[object, object]] = deque()

    def enqueue(self, tex_cap: object) -> bool:
        sync = _safe_call(
            GL.glFenceSync,
            GL.GL_SYNC_GPU_COMMANDS_COMPLETE,
            0,
            log_message="VT release queue: glFenceSync failed",
            default=None,
        )
        if sync is None:
            return False
        self._entries.append((sync, tex_cap))
        return True

    def drain(self, vt_module) -> None:
        if not self._entries:
            return
        # Drain head entries whose fences have signaled
        for _ in range(len(self._entries)):
            sync, tex_cap = self._entries[0]
            finished = False
            if sync is None:
                finished = True
            else:
                res = _safe_call(
                    GL.glClientWaitSync,
                    sync,
                    0,
                    0,
                    log_message="VT release queue: glClientWaitSync failed",
                    default=GL.GL_WAIT_FAILED,
                )
                if res in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED, GL.GL_WAIT_FAILED):
                    finished = True
            if not finished:
                break
            if sync is not None:
                _safe_call(
                    GL.glDeleteSync,
                    sync,
                    log_message="VT release queue: glDeleteSync failed",
                    default=None,
                )
            self._entries.popleft()
            _safe_call(
                vt_module.gl_release_tex,
                tex_cap,
                log_message="VT release queue: gl_release_tex failed",
                default=None,
            )

    def reset(self, vt_module) -> None:
        while self._entries:
            sync, tex_cap = self._entries.popleft()
            if sync is not None:
                _safe_call(
                    GL.glDeleteSync,
                    sync,
                    log_message="VT release queue reset: glDeleteSync failed",
                    default=None,
                )
            _safe_call(
                vt_module.gl_release_tex,
                tex_cap,
                log_message="VT release queue reset: gl_release_tex failed",
                default=None,
            )

    def __len__(self) -> int:
        return len(self._entries)


class GLRenderer:
    """Minimal GL renderer for streaming frames.

    Owns a texture and shader program and draws an RGB frame.
    """

    def __init__(self, scene_canvas) -> None:
        self._scene_canvas = scene_canvas
        self._video_texture: Optional[Texture2D] = None
        self._video_program: Optional[Program] = None
        # VT zero-copy state (initialized lazily inside draw when needed)
        self._vt = None  # type: ignore
        self._vt_cache = None  # GLCache capsule
        self._gl_frame_counter: int = 0
        # Debug safety: force CPU mapping instead of zero-copy
        self._vt_force_cpu = (os.getenv('NAPARI_CUDA_VT_FORCE_CPU', '0') or '0').lower() in (
            '1',
            'true',
            'yes',
            'on',
        )
        self._vt_gl_safe = (os.getenv('NAPARI_CUDA_VT_GL_SAFE', '0') or '0').lower() in (
            '1',
            'true',
            'yes',
            'on',
        )
        flush_env = (os.getenv('NAPARI_CUDA_VT_GL_FLUSH_EVERY', '').strip() or '30')
        if flush_env.lstrip('-').isdigit():
            self._vt_gl_flush_every = int(flush_env)
        else:
            if flush_env and flush_env != '30':
                logger.debug(
                    "Invalid NAPARI_CUDA_VT_GL_FLUSH_EVERY=%s; defaulting to 30", flush_env
                )
            self._vt_gl_flush_every = 30
        # Raw-GL programs/VBO for rectangle/2D textures
        self._gl_prog_rect: Optional[int] = None
        self._gl_prog_2d: Optional[int] = None
        self._gl_pos_loc_rect: Optional[int] = None
        self._gl_tex_loc_rect: Optional[int] = None
        self._gl_u_tex_size_loc: Optional[int] = None
        self._gl_pos_loc_2d: Optional[int] = None
        self._gl_tex_loc_2d: Optional[int] = None
        self._gl_vbo: Optional[int] = None
        self._vt_first_draw_logged: bool = False
        self._has_drawn: bool = False
        self._vt_release_queue = VTReleaseQueue()
        self._gl_context_id: Optional[int] = None
        # Track current video texture shape for resize detection
        self._vid_shape: Optional[tuple[int, int]] = None
        # Set swap/pacing preferences once to reduce compositor jitter
        qt_gui = _safe_call(
            importlib.import_module,
            'qtpy.QtGui',
            log_message="QSurfaceFormat module import failed",
            default=None,
        )
        if qt_gui is not None:
            QSurfaceFormat = getattr(qt_gui, 'QSurfaceFormat', None)
            if QSurfaceFormat is None:
                logger.debug("QSurfaceFormat symbol missing in qtpy.QtGui")
            else:
                fmt = QSurfaceFormat()
                _safe_call(
                    fmt.setSwapBehavior,
                    QSurfaceFormat.DoubleBuffer,
                    log_message="QSurfaceFormat.setSwapBehavior failed",
                    default=None,
                )
                _safe_call(
                    fmt.setSwapInterval,
                    1,
                    log_message="QSurfaceFormat.setSwapInterval failed",
                    default=None,
                )
                _safe_call(
                    QSurfaceFormat.setDefaultFormat,
                    fmt,
                    log_message="QSurfaceFormat.setDefaultFormat failed",
                    default=None,
                )
        self._init_resources()

    def _init_resources(self) -> None:
        # Ensure unpack alignment for tight RGB rows
        _safe_call(
            GL.glPixelStorei,
            GL.GL_UNPACK_ALIGNMENT,
            1,
            log_message="glPixelStorei UNPACK_ALIGNMENT failed",
            default=None,
        )
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        texture = _safe_call(
            Texture2D,
            dummy_frame,
            internalformat='rgb8',
            log_message="Texture2D rgb8 initialization failed",
            default=None,
        )
        if texture is None:
            texture = Texture2D(dummy_frame)
        self._video_texture = texture
        # Minimize filtering artifacts and edge sampling
        assert self._video_texture is not None
        self._video_texture.interpolation = 'nearest'
        self._video_texture.wrapping = 'clamp_to_edge'
        self._vid_shape = (dummy_frame.shape[0], dummy_frame.shape[1])
        self._video_program = Program(VERTEX_SHADER, FRAGMENT_SHADER)
        vertices = np.array([
            [-1, -1, 0, 1],
            [ 1, -1, 1, 1],
            [-1,  1, 0, 0],
            [ 1,  1, 1, 0],
        ], dtype=np.float32)
        self._video_program['position'] = np.ascontiguousarray(vertices[:, :2])
        self._video_program['texcoord'] = np.ascontiguousarray(vertices[:, 2:])
        self._video_program['texture'] = self._video_texture
        logger.debug("GLRenderer resources initialized")

    def _current_context_id(self) -> Optional[int]:
        canvas_ctx = getattr(self._scene_canvas, 'context', None)
        if canvas_ctx is None:
            return None
        ctx = getattr(canvas_ctx, 'context', None)
        return None if ctx is None else id(ctx)

    def _drain_vt_release_queue(self) -> None:
        if not self._vt_gl_safe or self._vt is None:
            return
        self._vt_release_queue.drain(self._vt)

    def _destroy_raw_gl_resources(self) -> None:
        if self._gl_prog_rect is not None:
            _safe_call(
                GL.glDeleteProgram,
                int(self._gl_prog_rect),
                log_message="delete rect program failed",
                default=None,
            )
        if self._gl_prog_2d is not None:
            _safe_call(
                GL.glDeleteProgram,
                int(self._gl_prog_2d),
                log_message="delete 2d program failed",
                default=None,
            )
        if self._gl_vbo is not None:
            _safe_call(
                GL.glDeleteBuffers,
                1,
                [int(self._gl_vbo)],
                log_message="delete VBO failed",
                default=None,
            )
        self._gl_prog_rect = None
        self._gl_prog_2d = None
        self._gl_pos_loc_rect = None
        self._gl_tex_loc_rect = None
        self._gl_u_tex_size_loc = None
        self._gl_pos_loc_2d = None
        self._gl_tex_loc_2d = None
        self._gl_vbo = None

    def _on_context_changed(self) -> None:
        if self._vt is not None and len(self._vt_release_queue):
            self._vt_release_queue.reset(self._vt)
        self._destroy_raw_gl_resources()
        self._vt_cache = None
        self._vt_first_draw_logged = False
        self._gl_frame_counter = 0
        logger.debug("GLRenderer: GL context change detected; resources rebuilt")

    def _maybe_handle_context_change(self) -> None:
        ctx_id = self._current_context_id()
        if ctx_id is None:
            return
        if self._gl_context_id is None:
            self._gl_context_id = ctx_id
            return
        if ctx_id != self._gl_context_id:
            self._on_context_changed()
            self._gl_context_id = ctx_id

    @contextmanager
    def _vt_draw_state(self, pos_loc: Optional[int], tex_loc: Optional[int], target: int):
        prev_program = _safe_call(
            GL.glGetIntegerv,
            GL.GL_CURRENT_PROGRAM,
            default=None,
            log_message="VT draw: query current program failed",
        )
        prev_buffer = _safe_call(
            GL.glGetIntegerv,
            GL.GL_ARRAY_BUFFER_BINDING,
            default=None,
            log_message="VT draw: query array buffer binding failed",
        )
        prev_active_tex = _safe_call(
            GL.glGetIntegerv,
            GL.GL_ACTIVE_TEXTURE,
            default=None,
            log_message="VT draw: query active texture failed",
        )
        try:
            yield
        finally:
            if prev_active_tex is not None:
                _safe_call(
                    GL.glActiveTexture,
                    int(prev_active_tex),
                    log_message="VT draw: restore active texture failed",
                    default=None,
                )
            _safe_call(
                GL.glBindTexture,
                int(target),
                0,
                log_message="VT draw: unbind texture failed",
                default=None,
            )
            if tex_loc is not None and int(tex_loc) != -1:
                _safe_call(
                    GL.glDisableVertexAttribArray,
                    int(tex_loc),
                    log_message="VT draw: disable tex attrib failed",
                    default=None,
                )
            if pos_loc is not None and int(pos_loc) != -1:
                _safe_call(
                    GL.glDisableVertexAttribArray,
                    int(pos_loc),
                    log_message="VT draw: disable pos attrib failed",
                    default=None,
                )
            target_buffer = int(prev_buffer) if prev_buffer is not None else 0
            _safe_call(
                GL.glBindBuffer,
                GL.GL_ARRAY_BUFFER,
                target_buffer,
                log_message="VT draw: restore array buffer failed",
                default=None,
            )
            program_target = int(prev_program) if prev_program is not None else 0
            _safe_call(
                GL.glUseProgram,
                program_target,
                log_message="VT draw: restore program failed",
                default=None,
            )

    def draw(self, frame: Optional[Any]) -> None:
        ctx = self._scene_canvas.context
        self._maybe_handle_context_change()
        self._drain_vt_release_queue()
        # Accept either numpy RGB frames or a (CVPixelBufferRef, release_cb) tuple
        payload = frame
        release_cb = None
        release_after_draw_cb: Optional[Callable[[object], None]] = None
        release_after_draw_payload: Optional[object] = None
        if isinstance(frame, tuple) and len(frame) == 2:
            payload, release_cb = frame  # type: ignore[assignment]

        lease = payload if isinstance(payload, FrameLease) else None
        vt_capsule = lease.capsule if lease is not None else None

        # Try VT zero-copy path if payload looks like a CVPixelBuffer and OpenGL is available
        drew_vt = False
        vt_attempted = False
        if vt_capsule is not None:
            if self._vt is None:
                vt_module = _safe_call(
                    importlib.import_module,
                    'napari_cuda._vt',
                    log_message="VT zero-copy draw: import failed",
                    default=None,
                )
                if vt_module is not None:
                    self._vt = vt_module
            if self._vt is not None:
                pf = _safe_call(
                    self._vt.pixel_format,  # type: ignore[misc]
                    vt_capsule,
                    default=None,
                    log_message="VT zero-copy draw: pixel_format failed",
                )
                if pf is not None and not self._vt_force_cpu:
                    if self._vt_cache is None:
                        self._vt_cache = _safe_call(
                            self._vt.gl_cache_init_for_current_context,  # type: ignore[attr-defined]
                            log_message="VT zero-copy draw: cache init failed",
                            default=None,
                        )
                    if self._vt_cache:
                        vt_attempted = True
                        drew_vt = bool(
                            _safe_call(
                                self._draw_vt_texture,
                                self._vt_cache,
                                vt_capsule,
                                default=False,
                                log_message="VT zero-copy draw: _draw_vt_texture failed",
                            )
                        )
                        if drew_vt and release_cb is not None:
                            release_after_draw_cb = release_cb
                            release_after_draw_payload = payload
                            release_cb = None
        elif payload is None:
            logger.debug("draw: payload=None vt_attempted=%s", vt_attempted)

        did_draw = False
        if not drew_vt:
            # If VT was attempted but failed, preserve last frame to avoid flicker
            if vt_attempted and self._has_drawn:
                if release_cb is not None and payload is not None:
                    _safe_call(
                        release_cb,
                        payload,
                        log_message="VT drop-frame release failed",
                        default=None,
                    )
                    release_cb = None
                return
            # If no new frame, keep last drawn content to avoid flicker
            if payload is None and self._has_drawn:
                return
            # Fallback: upload numpy RGB
            if self._video_program is None or self._video_texture is None:
                self._init_resources()
            arr: Optional[np.ndarray] = None
            if lease is not None and vt_capsule is not None:
                if self._vt is not None:
                    pf = _safe_call(
                        self._vt.pixel_format,  # type: ignore[misc]
                        vt_capsule,
                        default=None,
                        log_message="Frame normalization: pixel_format failed",
                    )
                    if pf is not None and self._vt_force_cpu:
                        data = _safe_call(
                            self._vt.map_to_rgb,  # type: ignore[attr-defined]
                            vt_capsule,
                            default=None,
                            log_message="Frame normalization: map_to_rgb failed",
                        )
                        if data is None:
                            arr = None
                        else:
                            arr = np.frombuffer(data[0], dtype=np.uint8).reshape(
                                (int(data[2]), int(data[1]), 3)
                            )
            elif payload is not None:
                try:
                    arr = np.asarray(payload)
                except Exception:
                    arr = None
                if arr is not None and arr.ndim == 3:
                    arr = np.flipud(arr)
            if arr is not None and (arr.dtype != np.uint8 or not arr.flags.c_contiguous):
                arr = np.ascontiguousarray(arr, dtype=np.uint8)
            if arr is None:
                logger.debug("Frame normalization failed; skipping draw")
            if arr is not None:
                # Recreate texture only if size changed to avoid realloc churn
                h, w = int(arr.shape[0]), int(arr.shape[1])
                if self._vid_shape != (h, w):
                    texture = _safe_call(
                        Texture2D,
                        arr,
                        internalformat='rgb8',
                        log_message="Texture resize rgb8 init failed",
                        default=None,
                    )
                    if texture is None:
                        texture = Texture2D(arr)
                    self._video_texture = texture
                    assert self._video_texture is not None
                    self._video_texture.interpolation = 'nearest'
                    self._video_texture.wrapping = 'clamp_to_edge'
                    self._video_program['texture'] = self._video_texture
                    self._vid_shape = (h, w)
                else:
                        assert self._video_texture is not None
                        _safe_call(
                            self._video_texture.set_data,
                            arr,
                            log_message="Texture2D set_data failed",
                            default=None,
                        )
                if release_cb is not None and payload is not None:
                    _safe_call(
                        release_cb,
                        payload,
                        log_message="VT fallback release failed",
                        default=None,
                    )
                    release_cb = None
                    payload = None
            # Avoid clearing every frame to prevent visible flashes on some drivers
            if not self._has_drawn:
                ctx.clear('black')
            self._video_program.draw('triangle_strip')
            self._has_drawn = True
            did_draw = True

        # Invoke release callback (if provided) after submission
        if release_after_draw_cb is not None and release_after_draw_payload is not None:
            _safe_call(
                release_after_draw_cb,
                release_after_draw_payload,
                log_message="release_cb failed after VT draw",
                default=None,
            )

    # --- VT helpers ---
    def _compile_glsl(self, vert_src: str, frag_src: str) -> Optional[int]:
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_src)
        GL.glCompileShader(vs)
        ok = GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS)
        if ok != GL.GL_TRUE:
            log = GL.glGetShaderInfoLog(vs)
            GL.glDeleteShader(vs)
            logger.debug("Vertex shader compile failed: %s", log)
            return None
        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, frag_src)
        GL.glCompileShader(fs)
        ok = GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS)
        if ok != GL.GL_TRUE:
            log = GL.glGetShaderInfoLog(fs)
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            logger.debug("Fragment shader compile failed: %s", log)
            return None
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glBindAttribLocation(prog, 0, 'a_position')
        GL.glBindAttribLocation(prog, 1, 'a_tex')
        GL.glLinkProgram(prog)
        link_ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        if link_ok != GL.GL_TRUE:
            log = GL.glGetProgramInfoLog(prog)
            GL.glDeleteProgram(prog)
            logger.debug("Program link failed: %s", log)
            return None
        return int(prog)

    def _ensure_raw_gl_resources(self) -> None:
        # Create VBO if needed
        if self._gl_vbo is None:
            import numpy as _np
            quad = _np.array([
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ], dtype=_np.float32)
            vbo = GL.glGenBuffers(1)
            self._gl_vbo = int(vbo)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._gl_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        # Compile programs if needed (try 3.2 core shaders first, then 1.20)
        if self._gl_prog_rect is None:
            V150 = """
            #version 150
            in vec2 a_position; in vec2 a_tex; uniform vec2 u_tex_size; out vec2 v_tex_px;
            void main(){ gl_Position=vec4(a_position,0.0,1.0); v_tex_px=a_tex*u_tex_size; }
            """
            F150 = """
            #version 150
            uniform sampler2DRect u_texRect; in vec2 v_tex_px; out vec4 fragColor;
            void main(){ fragColor = texture(u_texRect, v_tex_px); }
            """
            V120 = """
            #version 120
            attribute vec2 a_position; attribute vec2 a_tex; uniform vec2 u_tex_size; varying vec2 v_tex_px;
            void main(){ gl_Position=vec4(a_position,0.0,1.0); v_tex_px=a_tex*u_tex_size; }
            """
            F120 = """
            #version 120
            #extension GL_ARB_texture_rectangle : enable
            uniform sampler2DRect u_texRect; varying vec2 v_tex_px;
            void main(){ gl_FragColor = texture2DRect(u_texRect, v_tex_px); }
            """
            prog = self._compile_glsl(V150, F150)
            if prog is None:
                prog = self._compile_glsl(V120, F120)
            if prog is not None:
                self._gl_prog_rect = prog
                self._gl_pos_loc_rect = int(GL.glGetAttribLocation(prog, 'a_position'))
                self._gl_tex_loc_rect = int(GL.glGetAttribLocation(prog, 'a_tex'))
                self._gl_u_tex_size_loc = int(GL.glGetUniformLocation(prog, 'u_tex_size'))
        if self._gl_prog_2d is None:
            V150_2D = """
            #version 150
            in vec2 a_position; in vec2 a_tex; out vec2 v_tex;
            void main(){ gl_Position=vec4(a_position,0.0,1.0); v_tex=a_tex; }
            """
            F150_2D = """
            #version 150
            uniform sampler2D u_tex2d; in vec2 v_tex; out vec4 fragColor;
            void main(){ fragColor = texture(u_tex2d, v_tex); }
            """
            V120_2D = """
            #version 120
            attribute vec2 a_position; attribute vec2 a_tex; varying vec2 v_tex;
            void main(){ gl_Position=vec4(a_position,0.0,1.0); v_tex=a_tex; }
            """
            F120_2D = """
            #version 120
            uniform sampler2D u_tex2d; varying vec2 v_tex;
            void main(){ gl_FragColor = texture2D(u_tex2d, v_tex); }
            """
            prog2 = self._compile_glsl(V150_2D, F150_2D)
            if prog2 is None:
                prog2 = self._compile_glsl(V120_2D, F120_2D)
            if prog2 is not None:
                self._gl_prog_2d = prog2
                self._gl_pos_loc_2d = int(GL.glGetAttribLocation(prog2, 'a_position'))
                self._gl_tex_loc_2d = int(GL.glGetAttribLocation(prog2, 'a_tex'))

    def _draw_vt_texture(self, cache_cap, cvpixelbuffer_cap) -> bool:
        if self._vt is None:
            return False
        # Ensure sane GL state; avoid clearing so previous image persists on errors
        _safe_call(
            GL.glDisable,
            GL.GL_DEPTH_TEST,
            log_message="GL state setup (DEPTH_TEST) failed",
            default=None,
        )
        _safe_call(
            GL.glDisable,
            GL.GL_BLEND,
            log_message="GL state setup (BLEND) failed",
            default=None,
        )
        res = self._vt.gl_tex_from_cvpixelbuffer(cache_cap, cvpixelbuffer_cap)
        if not res:
            return False
        tex_cap, name, target, tw, th = res
        enqueued = False
        try:
            target_i = int(target)
            name_i = int(name)
            w_i = int(tw); h_i = int(th)
            self._ensure_raw_gl_resources()
            if target_i == 0x84F5:  # GL_TEXTURE_RECTANGLE
                prog = self._gl_prog_rect
                if prog is None or self._gl_vbo is None:
                    return False
                if not self._vt_first_draw_logged:
                    logger.info("GLRenderer: VT zero-copy draw engaged (RECT target)")
                    self._vt_first_draw_logged = True
                with self._vt_draw_state(self._gl_pos_loc_rect, self._gl_tex_loc_rect, target_i):
                    GL.glUseProgram(prog)
                    if self._gl_u_tex_size_loc is not None and int(self._gl_u_tex_size_loc) != -1:
                        GL.glUniform2f(int(self._gl_u_tex_size_loc), float(w_i), float(h_i))
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(target_i, name_i)
                    loc_sampler = GL.glGetUniformLocation(prog, 'u_texRect')
                    if int(loc_sampler) != -1:
                        GL.glUniform1i(loc_sampler, 0)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._gl_vbo))
                    stride = 4 * 4
                    import ctypes as _ctypes
                    if self._gl_pos_loc_rect is not None and int(self._gl_pos_loc_rect) != -1:
                        GL.glEnableVertexAttribArray(int(self._gl_pos_loc_rect))
                        GL.glVertexAttribPointer(int(self._gl_pos_loc_rect), 2, GL.GL_FLOAT, GL.GL_FALSE, stride, _ctypes.c_void_p(0))
                    if self._gl_tex_loc_rect is not None and int(self._gl_tex_loc_rect) != -1:
                        GL.glEnableVertexAttribArray(int(self._gl_tex_loc_rect))
                        GL.glVertexAttribPointer(int(self._gl_tex_loc_rect), 2, GL.GL_FLOAT, GL.GL_FALSE, stride, _ctypes.c_void_p(8))
                    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            else:
                # GL_TEXTURE_2D path
                prog = self._gl_prog_2d
                if prog is None or self._gl_vbo is None:
                    return False
                if not self._vt_first_draw_logged:
                    logger.info("GLRenderer: VT zero-copy draw engaged (2D target)")
                    self._vt_first_draw_logged = True
                with self._vt_draw_state(self._gl_pos_loc_2d, self._gl_tex_loc_2d, target_i):
                    GL.glUseProgram(prog)
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(target_i, name_i)
                    loc_sampler = GL.glGetUniformLocation(prog, 'u_tex2d')
                    if int(loc_sampler) != -1:
                        GL.glUniform1i(loc_sampler, 0)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._gl_vbo))
                    stride = 4 * 4
                    import ctypes as _ctypes
                    if self._gl_pos_loc_2d is not None and int(self._gl_pos_loc_2d) != -1:
                        GL.glEnableVertexAttribArray(int(self._gl_pos_loc_2d))
                        GL.glVertexAttribPointer(int(self._gl_pos_loc_2d), 2, GL.GL_FLOAT, GL.GL_FALSE, stride, _ctypes.c_void_p(0))
                    if self._gl_tex_loc_2d is not None and int(self._gl_tex_loc_2d) != -1:
                        GL.glEnableVertexAttribArray(int(self._gl_tex_loc_2d))
                        GL.glVertexAttribPointer(int(self._gl_tex_loc_2d), 2, GL.GL_FLOAT, GL.GL_FALSE, stride, _ctypes.c_void_p(8))
                    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            _safe_call(GL.glFlush, log_message="VT draw: glFlush failed", default=None)
            # Release the GL texture object created by VT (immediately or deferred)
            logger.debug("VT release tex=%s", hex(name_i))
            if self._vt_gl_safe:
                enqueued = self._vt_release_queue.enqueue(tex_cap)
                if not enqueued:
                    logger.debug("VT release queue: enqueue failed; releasing immediately")
                    _safe_call(
                        self._vt.gl_release_tex,
                        tex_cap,
                        log_message="VT draw: immediate gl_release_tex failed",
                        default=None,
                    )
            else:
                _safe_call(
                    self._vt.gl_release_tex,
                    tex_cap,
                    log_message="VT draw: gl_release_tex failed",
                    default=None,
                )
            # Optionally flush the texture cache periodically to release internal resources
            self._gl_frame_counter += 1
            if self._vt_gl_flush_every > 0 and (
                self._gl_frame_counter % int(max(1, self._vt_gl_flush_every))
            ) == 0:
                _safe_call(
                    self._vt.gl_cache_flush,
                    cache_cap,
                    log_message="gl_cache_flush failed",
                    default=None,
                )
            return True
        except Exception:
            logger.debug("VT draw failed", exc_info=True)
            if not enqueued:
                _safe_call(
                    self._vt.gl_release_tex,
                    tex_cap,
                    log_message="gl_release_tex (cleanup) failed",
                    default=None,
                )
            return False
