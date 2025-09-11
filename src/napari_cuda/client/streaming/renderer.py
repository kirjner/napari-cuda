from __future__ import annotations

import logging
from typing import Optional, Tuple, Any

import numpy as np
from vispy.gloo import Texture2D, Program

try:
    from OpenGL import GL as _GL  # type: ignore
except Exception:  # pragma: no cover
    _GL = None  # type: ignore

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
        self._init_resources()

    def _init_resources(self) -> None:
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._video_texture = Texture2D(dummy_frame)
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
        if isinstance(frame, tuple) and len(frame) == 2:
            payload, release_cb = frame  # type: ignore[assignment]

        # Try VT zero-copy path if payload looks like a CVPixelBuffer and OpenGL is available
        drew_vt = False
        vt_attempted = False
        if payload is not None and _GL is not None:
            try:
                # Lazy import to detect VT capsule
                if self._vt is None:
                    from napari_cuda import _vt as vt  # type: ignore
                    self._vt = vt
                pf = self._vt.pixel_format(payload)  # type: ignore[misc]
                if pf is not None:
                    if self._vt_cache is None:
                        self._vt_cache = self._vt.gl_cache_init_for_current_context()
                    if self._vt_cache:
                        vt_attempted = True
                        drew_vt = self._draw_vt_texture(self._vt_cache, payload)
            except Exception:
                # Not a CVPixelBuffer or VT path failed; fall back below
                logger.debug("VT zero-copy draw attempt failed; falling back", exc_info=True)
                drew_vt = False

        if not drew_vt:
            # If VT was attempted but failed, preserve last frame to avoid flicker
            if vt_attempted and self._has_drawn:
                return
            # If no new frame, keep last drawn content to avoid flicker
            if payload is None and self._has_drawn:
                return
            # Fallback: upload numpy RGB
            if self._video_program is None or self._video_texture is None:
                self._init_resources()
            if payload is not None:
                try:
                    arr = np.asarray(payload)
                    if arr.dtype != np.uint8 or not arr.flags.c_contiguous:
                        arr = np.ascontiguousarray(arr, dtype=np.uint8)
                except Exception:
                    logger.debug("Frame normalization failed", exc_info=True)
                    arr = None
                if arr is not None:
                    self._video_texture.set_data(arr)
            # Avoid clearing every frame to prevent visible flashes on some drivers
            if not self._has_drawn:
                ctx.clear('black')
            self._video_program.draw('triangle_strip')
            self._has_drawn = True

        # Invoke release callback (if provided) after submission
        if release_cb is not None and payload is not None:
            try:
                release_cb(payload)  # type: ignore[misc]
            except Exception:
                logger.debug("release_cb failed after draw", exc_info=True)

    # --- VT helpers ---
    def _compile_glsl(self, vert_src: str, frag_src: str) -> Optional[int]:
        if _GL is None:
            return None
        vs = _GL.glCreateShader(_GL.GL_VERTEX_SHADER)
        _GL.glShaderSource(vs, vert_src)
        _GL.glCompileShader(vs)
        ok = _GL.glGetShaderiv(vs, _GL.GL_COMPILE_STATUS)
        if ok != _GL.GL_TRUE:
            log = _GL.glGetShaderInfoLog(vs)
            _GL.glDeleteShader(vs)
            logger.debug("Vertex shader compile failed: %s", log)
            return None
        fs = _GL.glCreateShader(_GL.GL_FRAGMENT_SHADER)
        _GL.glShaderSource(fs, frag_src)
        _GL.glCompileShader(fs)
        ok = _GL.glGetShaderiv(fs, _GL.GL_COMPILE_STATUS)
        if ok != _GL.GL_TRUE:
            log = _GL.glGetShaderInfoLog(fs)
            _GL.glDeleteShader(vs)
            _GL.glDeleteShader(fs)
            logger.debug("Fragment shader compile failed: %s", log)
            return None
        prog = _GL.glCreateProgram()
        _GL.glAttachShader(prog, vs)
        _GL.glAttachShader(prog, fs)
        _GL.glBindAttribLocation(prog, 0, 'a_position')
        _GL.glBindAttribLocation(prog, 1, 'a_tex')
        _GL.glLinkProgram(prog)
        link_ok = _GL.glGetProgramiv(prog, _GL.GL_LINK_STATUS)
        _GL.glDeleteShader(vs)
        _GL.glDeleteShader(fs)
        if link_ok != _GL.GL_TRUE:
            log = _GL.glGetProgramInfoLog(prog)
            _GL.glDeleteProgram(prog)
            logger.debug("Program link failed: %s", log)
            return None
        return int(prog)

    def _ensure_raw_gl_resources(self) -> None:
        if _GL is None:
            return
        # Create VBO if needed
        if self._gl_vbo is None:
            import numpy as _np
            quad = _np.array([
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ], dtype=_np.float32)
            vbo = _GL.glGenBuffers(1)
            self._gl_vbo = int(vbo)
            _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, self._gl_vbo)
            _GL.glBufferData(_GL.GL_ARRAY_BUFFER, quad.nbytes, quad, _GL.GL_STATIC_DRAW)
            _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, 0)
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
                self._gl_pos_loc_rect = int(_GL.glGetAttribLocation(prog, 'a_position'))
                self._gl_tex_loc_rect = int(_GL.glGetAttribLocation(prog, 'a_tex'))
                self._gl_u_tex_size_loc = int(_GL.glGetUniformLocation(prog, 'u_tex_size'))
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
                self._gl_pos_loc_2d = int(_GL.glGetAttribLocation(prog2, 'a_position'))
                self._gl_tex_loc_2d = int(_GL.glGetAttribLocation(prog2, 'a_tex'))

    def _draw_vt_texture(self, cache_cap, cvpixelbuffer_cap) -> bool:
        if _GL is None or self._vt is None:
            return False
        # Ensure sane GL state; avoid clearing so previous image persists on errors
        try:
            if _GL is not None:
                _GL.glDisable(_GL.GL_DEPTH_TEST)
                _GL.glDisable(_GL.GL_BLEND)
        except Exception:
            pass
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
                    try:
                        logger.info("GLRenderer: VT zero-copy draw engaged (RECT target)")
                    except Exception:
                        pass
                    self._vt_first_draw_logged = True
                _GL.glUseProgram(prog)
                # u_tex_size
                if self._gl_u_tex_size_loc is not None and int(self._gl_u_tex_size_loc) != -1:
                    _GL.glUniform2f(int(self._gl_u_tex_size_loc), float(w_i), float(h_i))
                _GL.glActiveTexture(_GL.GL_TEXTURE0)
                _GL.glBindTexture(target_i, name_i)
                loc_sampler = _GL.glGetUniformLocation(prog, 'u_texRect')
                if int(loc_sampler) != -1:
                    _GL.glUniform1i(loc_sampler, 0)
                # VBO attributes
                _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, int(self._gl_vbo))
                stride = 4 * 4
                import ctypes as _ctypes
                if self._gl_pos_loc_rect is not None and int(self._gl_pos_loc_rect) != -1:
                    _GL.glEnableVertexAttribArray(int(self._gl_pos_loc_rect))
                    _GL.glVertexAttribPointer(int(self._gl_pos_loc_rect), 2, _GL.GL_FLOAT, _GL.GL_FALSE, stride, _ctypes.c_void_p(0))
                if self._gl_tex_loc_rect is not None and int(self._gl_tex_loc_rect) != -1:
                    _GL.glEnableVertexAttribArray(int(self._gl_tex_loc_rect))
                    _GL.glVertexAttribPointer(int(self._gl_tex_loc_rect), 2, _GL.GL_FLOAT, _GL.GL_FALSE, stride, _ctypes.c_void_p(8))
                # Draw
                _GL.glDrawArrays(_GL.GL_TRIANGLE_STRIP, 0, 4)
                # Cleanup binds
                _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, 0)
                _GL.glUseProgram(0)
            else:
                # GL_TEXTURE_2D path
                prog = self._gl_prog_2d
                if prog is None or self._gl_vbo is None:
                    return False
                if not self._vt_first_draw_logged:
                    try:
                        logger.info("GLRenderer: VT zero-copy draw engaged (2D target)")
                    except Exception:
                        pass
                    self._vt_first_draw_logged = True
                _GL.glUseProgram(prog)
                _GL.glActiveTexture(_GL.GL_TEXTURE0)
                _GL.glBindTexture(target_i, name_i)
                loc_sampler = _GL.glGetUniformLocation(prog, 'u_tex2d')
                if int(loc_sampler) != -1:
                    _GL.glUniform1i(loc_sampler, 0)
                _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, int(self._gl_vbo))
                stride = 4 * 4
                import ctypes as _ctypes
                if self._gl_pos_loc_2d is not None and int(self._gl_pos_loc_2d) != -1:
                    _GL.glEnableVertexAttribArray(int(self._gl_pos_loc_2d))
                    _GL.glVertexAttribPointer(int(self._gl_pos_loc_2d), 2, _GL.GL_FLOAT, _GL.GL_FALSE, stride, _ctypes.c_void_p(0))
                if self._gl_tex_loc_2d is not None and int(self._gl_tex_loc_2d) != -1:
                    _GL.glEnableVertexAttribArray(int(self._gl_tex_loc_2d))
                    _GL.glVertexAttribPointer(int(self._gl_tex_loc_2d), 2, _GL.GL_FLOAT, _GL.GL_FALSE, stride, _ctypes.c_void_p(8))
                _GL.glDrawArrays(_GL.GL_TRIANGLE_STRIP, 0, 4)
                _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, 0)
                _GL.glUseProgram(0)
            # Release the GL texture object created by VT
            try:
                self._vt.gl_release_tex(tex_cap)
            except Exception:
                logger.debug("gl_release_tex failed", exc_info=True)
            return True
        except Exception:
            logger.debug("VT draw failed", exc_info=True)
            try:
                self._vt.gl_release_tex(tex_cap)
            except Exception:
                pass
            return False
