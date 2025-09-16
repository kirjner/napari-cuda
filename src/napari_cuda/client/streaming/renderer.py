from __future__ import annotations

import logging
from typing import Optional, Tuple, Any

import numpy as np
from vispy.gloo import Texture2D, Program

from OpenGL import GL as GL  # type: ignore

logger = logging.getLogger(__name__)


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
        self._gl_dbg_last_log: float = 0.0
        self._gl_frame_counter: int = 0
        # Debug safety: force CPU mapping instead of zero-copy
        try:
            import os as _os
            self._vt_force_cpu = (_os.getenv('NAPARI_CUDA_VT_FORCE_CPU', '0') or '0') in ('1','true','yes','on')
            self._vt_gl_safe = (_os.getenv('NAPARI_CUDA_VT_GL_SAFE', '0') or '0') in ('1','true','yes','on')
            self._vt_gl_flush_every = int(_os.getenv('NAPARI_CUDA_VT_GL_FLUSH_EVERY', '30') or '30')
        except Exception:
            self._vt_force_cpu = False
            self._vt_gl_safe = False
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
        # Track current video texture shape for resize detection
        self._vid_shape: Optional[tuple[int, int]] = None
        # Set swap/pacing preferences once to reduce compositor jitter
        try:
            from qtpy.QtGui import QSurfaceFormat
            fmt = QSurfaceFormat()
            fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
            fmt.setSwapInterval(1)
            QSurfaceFormat.setDefaultFormat(fmt)
        except Exception:
            logger.debug("QSurfaceFormat default config failed", exc_info=True)
        self._init_resources()

    def _init_resources(self) -> None:
        # Ensure unpack alignment for tight RGB rows
        try:
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        except Exception:
            logger.debug("glPixelStorei UNPACK_ALIGNMENT failed", exc_info=True)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        try:
            self._video_texture = Texture2D(dummy_frame, internalformat='rgb8')
            # Minimize filtering artifacts and edge sampling
            self._video_texture.interpolation = 'nearest'  # type: ignore[attr-defined]
            self._video_texture.wrapping = 'clamp_to_edge'  # type: ignore[attr-defined]
        except Exception:
            # Fallback without internalformat if not supported
            self._video_texture = Texture2D(dummy_frame)
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

    def draw(self, frame: Optional[Any]) -> None:
        ctx = self._scene_canvas.context
        # Accept either numpy RGB frames or a (CVPixelBufferRef, release_cb) tuple
        payload = frame
        release_cb = None
        release_after_draw_cb: Optional[Callable[[object], None]] = None
        release_after_draw_payload: Optional[object] = None
        if isinstance(frame, tuple) and len(frame) == 2:
            payload, release_cb = frame  # type: ignore[assignment]

        # Try VT zero-copy path if payload looks like a CVPixelBuffer and OpenGL is available
        drew_vt = False
        vt_attempted = False
        if payload is not None:
            try:
                # Lazy import to detect VT capsule
                if self._vt is None:
                    from napari_cuda import _vt as vt  # type: ignore
                    self._vt = vt
                pf = self._vt.pixel_format(payload)  # type: ignore[misc]
                if pf is not None and not self._vt_force_cpu:
                    if self._vt_cache is None:
                        self._vt_cache = self._vt.gl_cache_init_for_current_context()
                    if self._vt_cache:
                        vt_attempted = True
                        drew_vt = self._draw_vt_texture(self._vt_cache, payload)
                        if drew_vt and release_cb is not None:
                            release_after_draw_cb = release_cb
                            release_after_draw_payload = payload
                            release_cb = None
            except Exception:
                # Not a CVPixelBuffer or VT path failed; fall back below
                logger.debug("VT zero-copy draw attempt failed; falling back", exc_info=True)
                drew_vt = False
        else:
            logger.debug("draw: payload=%s vt_attempted=%s", type(payload).__name__, vt_attempted)

        did_draw = False
        if not drew_vt:
            # If VT was attempted but failed, preserve last frame to avoid flicker
            if vt_attempted and self._has_drawn:
                if release_cb is not None and payload is not None:
                    try:
                        release_cb(payload)  # type: ignore[misc]
                    except Exception:
                        logger.debug("VT drop-frame release failed", exc_info=True)
                    release_cb = None
                return
            # If no new frame, keep last drawn content to avoid flicker
            if payload is None and self._has_drawn:
                return
            # Fallback: upload numpy RGB
            if self._video_program is None or self._video_texture is None:
                self._init_resources()
            if payload is not None:
                try:
                    # If payload is a CVPixelBuffer and force_cpu is on, map to RGB bytes
                    if self._vt is not None:
                        try:
                            pf = self._vt.pixel_format(payload)  # type: ignore[misc]
                        except Exception:
                            pf = None
                        if pf is not None and self._vt_force_cpu:
                            data = self._vt.map_to_rgb(payload)
                            arr = np.frombuffer(data[0], dtype=np.uint8).reshape((int(data[2]), int(data[1]), 3))
                        else:
                            arr = np.asarray(payload)
                    else:
                        arr = np.asarray(payload)
                    if arr.dtype != np.uint8 or not arr.flags.c_contiguous:
                        arr = np.ascontiguousarray(arr, dtype=np.uint8)
                except Exception:
                    logger.debug("Frame normalization failed", exc_info=True)
                    arr = None
                if arr is not None:
                    # Recreate texture only if size changed to avoid realloc churn
                    h, w = int(arr.shape[0]), int(arr.shape[1])
                    if self._vid_shape != (h, w):
                        try:
                            self._video_texture = Texture2D(arr, internalformat='rgb8')
                            self._video_texture.interpolation = 'nearest'  # type: ignore[attr-defined]
                            self._video_texture.wrapping = 'clamp_to_edge'  # type: ignore[attr-defined]
                            self._video_program['texture'] = self._video_texture
                        except Exception:
                            self._video_texture = Texture2D(arr)
                            self._video_program['texture'] = self._video_texture
                        self._vid_shape = (h, w)
                    else:
                        self._video_texture.set_data(arr)
                if release_cb is not None and payload is not None:
                    try:
                        release_cb(payload)  # type: ignore[misc]
                    except Exception:
                        logger.debug("VT fallback release failed", exc_info=True)
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
            try:
                release_after_draw_cb(release_after_draw_payload)  # type: ignore[misc]
            except Exception:
                logger.debug("release_cb failed after VT draw", exc_info=True)

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
        try:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
        except Exception:
            logger.debug("GL state setup failed", exc_info=True)
        res = self._vt.gl_tex_from_cvpixelbuffer(cache_cap, cvpixelbuffer_cap)
        if not res:
            return False
        tex_cap, name, target, tw, th = res
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
                GL.glUseProgram(prog)
                # u_tex_size
                if self._gl_u_tex_size_loc is not None and int(self._gl_u_tex_size_loc) != -1:
                    GL.glUniform2f(int(self._gl_u_tex_size_loc), float(w_i), float(h_i))
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(target_i, name_i)
                loc_sampler = GL.glGetUniformLocation(prog, 'u_texRect')
                if int(loc_sampler) != -1:
                    GL.glUniform1i(loc_sampler, 0)
                # VBO attributes
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._gl_vbo))
                stride = 4 * 4
                import ctypes as _ctypes
                if self._gl_pos_loc_rect is not None and int(self._gl_pos_loc_rect) != -1:
                    GL.glEnableVertexAttribArray(int(self._gl_pos_loc_rect))
                    GL.glVertexAttribPointer(int(self._gl_pos_loc_rect), 2, GL.GL_FLOAT, GL.GL_FALSE, stride, _ctypes.c_void_p(0))
                if self._gl_tex_loc_rect is not None and int(self._gl_tex_loc_rect) != -1:
                    GL.glEnableVertexAttribArray(int(self._gl_tex_loc_rect))
                    GL.glVertexAttribPointer(int(self._gl_tex_loc_rect), 2, GL.GL_FLOAT, GL.GL_FALSE, stride, _ctypes.c_void_p(8))
                # Draw
                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
                # Cleanup binds
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glUseProgram(0)
            else:
                # GL_TEXTURE_2D path
                prog = self._gl_prog_2d
                if prog is None or self._gl_vbo is None:
                    return False
                if not self._vt_first_draw_logged:
                    logger.info("GLRenderer: VT zero-copy draw engaged (2D target)")
                    self._vt_first_draw_logged = True
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
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glUseProgram(0)
            # Ensure GPU finished using the texture if debug safety is enabled
            try:
                if self._vt_gl_safe:
                    GL.glFinish()
                else:
                    GL.glFlush()
            except Exception:
                pass
            # Release the GL texture object created by VT
            try:
                logger.debug("VT release tex=%s", hex(name_i))
                self._vt.gl_release_tex(tex_cap)
            except Exception:
                logger.debug("gl_release_tex failed", exc_info=True)
            # Optionally flush the texture cache periodically to release internal resources
            try:
                self._gl_frame_counter += 1
                if self._vt_gl_flush_every > 0 and (self._gl_frame_counter % int(max(1, self._vt_gl_flush_every))) == 0:
                    self._vt.gl_cache_flush(cache_cap)
            except Exception:
                logger.debug("gl_cache_flush failed", exc_info=True)
            # Optional GL cache debug
            try:
                import os as _os, time as _time
                if (_os.getenv('NAPARI_CUDA_VT_GL_DEBUG', '0') or '0') in ('1','true','yes','on'):
                    now = _time.time()
                    if now - float(self._gl_dbg_last_log or 0.0) >= 1.0:
                        try:
                            creates, releases = self._vt.gl_cache_counts(cache_cap)  # type: ignore[misc]
                            logger.info("VT GL dbg: cache creates=%d releases=%d", int(creates), int(releases))
                        except Exception:
                            pass
                        self._gl_dbg_last_log = now
            except Exception:
                pass
            return True
        except Exception:
            logger.debug("VT draw failed", exc_info=True)
            try:
                self._vt.gl_release_tex(tex_cap)
            except Exception:
                logger.debug("gl_release_tex (cleanup) failed", exc_info=True)
            return False
