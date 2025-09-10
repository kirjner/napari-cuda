#!/usr/bin/env python3
"""
VT + CoreVideo/OpenGL zero-copy sanity check (macOS only).

Two parts:
1) Offscreen GL context smoke: initialize GL cache, allocate BGRA CVPixelBuffer,
   create GL texture via shim, and print name/target/size.
2) VisPy draw check: create a VisPy Canvas and render the texture once using a
   fragment shader compatible with either GL_TEXTURE_2D or GL_TEXTURE_RECTANGLE.

Usage:
  uv run python scripts/vt_gl_sanity.py [width height]
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    # Try PyQt6 first, then fall back to PyQt5
    try:
        from PyQt6 import QtGui  # type: ignore
    except Exception:
        try:
            from PyQt5 import QtGui  # type: ignore
        except Exception as e:
            print(f"Qt import failed (need PyQt6 or PyQt5): {e}")
            return 2
    try:
        from napari_cuda import _vt as vt  # type: ignore
    except Exception as e:
        print(f"napari_cuda._vt import failed: {e}")
        return 2

    # Dimensions (default 64x64)
    try:
        w = int(sys.argv[1]) if len(sys.argv) > 1 else 64
        h = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        if w <= 0 or h <= 0:
            raise ValueError
    except Exception:
        print("Invalid dimensions; expected positive integers")
        return 2

    try:
        app = QtGui.QGuiApplication(sys.argv)
        # Part 1: Offscreen smoke
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 2)
        # Handle enum scoping differences between Qt5 and Qt6
        try:
            profile_enum = QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile  # Qt6
        except AttributeError:
            profile_enum = QtGui.QSurfaceFormat.CoreProfile  # Qt5
        fmt.setProfile(profile_enum)
        surf = QtGui.QOffscreenSurface()
        surf.setFormat(fmt)
        surf.create()
        ctx = QtGui.QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            print("QOpenGLContext.create() failed")
            return 1
        if not ctx.makeCurrent(surf):
            print("QOpenGLContext.makeCurrent() failed")
            return 1

        cache = vt.gl_cache_init_for_current_context()
        print("GL cache init:", bool(cache))
        if not cache:
            return 1

        # Create BGRA CVPixelBuffer compatible with OpenGL
        pb = vt.alloc_pixelbuffer_bgra(w, h, True)
        if not pb:
            print("alloc_pixelbuffer_bgra failed")
            return 1

        pf = vt.pixel_format(pb)
        print("Pixel format:", hex(int(pf)) if pf else pf)

        res = vt.gl_tex_from_cvpixelbuffer(cache, pb)
        if not res:
            print("gl_tex_from_cvpixelbuffer returned None")
            return 1
        tex_cap, name, target, tw, th = res
        print(f"GL texture: name={name} target=0x{int(target):x} size={tw}x{th}")
        if int(name) == 0 or int(tw) != w or int(th) != h:
            print("Unexpected GL texture result")
            return 1
        vt.gl_release_tex(tex_cap)
        print("Zero-copy BGRA path: OK")

        # Part 2: VisPy draw check with target-aware shader
        from vispy import app, gloo  # type: ignore

        VERT = """
        attribute vec2 a_position;
        attribute vec2 a_tex;
        uniform vec2 u_tex_size;  // in pixels
        varying vec2 v_tex;       // normalized
        varying vec2 v_tex_px;    // pixels
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_tex = a_tex;
            v_tex_px = a_tex * u_tex_size;
        }
        """

        FRAG_2D = """
        uniform sampler2D u_tex2d;
        varying vec2 v_tex;
        void main() {
            gl_FragColor = texture2D(u_tex2d, v_tex);
        }
        """

        FRAG_RECT = """
        #extension GL_ARB_texture_rectangle : enable
        uniform sampler2DRect u_texRect;
        varying vec2 v_tex_px;
        void main() {
            gl_FragColor = texture2DRect(u_texRect, v_tex_px);
        }
        """

        class _Canvas(app.Canvas):
            def __init__(self, vt_mod, pixelbuffer, size):
                app.Canvas.__init__(self, size=(size[0], size[1]), show=False)
                self._vt = vt_mod
                self._pb = pixelbuffer
                self._tsize = size
                self._target = None
                self._name = None
                self._tex_cap = None
                self.program = None
                self._gl_prog = None
                self._gl_pos_loc = None
                self._gl_tex_loc = None
                self._gl_utex_loc = None
                self._gl_vbo = None

            def _init_gl(self):
                import numpy as _np
                if int(self._target) == 0x84F5:
                    # Rectangle textures: use raw OpenGL program (gloo lacks sampler2DRect support)
                    from OpenGL import GL as _GL  # type: ignore
                    # Try GL 3.2 core profile first, then fall back to GLSL 1.20
                    VERT_150 = """
                    #version 150
                    in vec2 a_position;
                    in vec2 a_tex;
                    uniform vec2 u_tex_size;
                    out vec2 v_tex_px;
                    void main() {
                        gl_Position = vec4(a_position, 0.0, 1.0);
                        v_tex_px = a_tex * u_tex_size;
                    }
                    """
                    FRAG_150 = """
                    #version 150
                    uniform sampler2DRect u_texRect;
                    in vec2 v_tex_px;
                    out vec4 fragColor;
                    void main() {
                        fragColor = texture(u_texRect, v_tex_px);
                    }
                    """
                    VERT_120 = """
                    #version 120
                    attribute vec2 a_position;
                    attribute vec2 a_tex;
                    uniform vec2 u_tex_size;
                    varying vec2 v_tex_px;
                    void main() {
                        gl_Position = vec4(a_position, 0.0, 1.0);
                        v_tex_px = a_tex * u_tex_size;
                    }
                    """
                    FRAG_120 = """
                    #version 120
                    #extension GL_ARB_texture_rectangle : enable
                    uniform sampler2DRect u_texRect;
                    varying vec2 v_tex_px;
                    void main() {
                        gl_FragColor = texture2DRect(u_texRect, v_tex_px);
                    }
                    """
                    def _compile_link(vsrc, fsrc):
                        vs = _GL.glCreateShader(_GL.GL_VERTEX_SHADER)
                        _GL.glShaderSource(vs, vsrc)
                        _GL.glCompileShader(vs)
                        status = _GL.glGetShaderiv(vs, _GL.GL_COMPILE_STATUS)
                        if status != _GL.GL_TRUE:
                            log = _GL.glGetShaderInfoLog(vs)
                            _GL.glDeleteShader(vs)
                            raise RuntimeError(f"Vertex shader compile failed: {log}")
                        fs = _GL.glCreateShader(_GL.GL_FRAGMENT_SHADER)
                        _GL.glShaderSource(fs, fsrc)
                        _GL.glCompileShader(fs)
                        status = _GL.glGetShaderiv(fs, _GL.GL_COMPILE_STATUS)
                        if status != _GL.GL_TRUE:
                            log = _GL.glGetShaderInfoLog(fs)
                            _GL.glDeleteShader(vs)
                            _GL.glDeleteShader(fs)
                            raise RuntimeError(f"Fragment shader compile failed: {log}")
                        prog_ = _GL.glCreateProgram()
                        # Bind attribute locations before linking to avoid -1
                        _GL.glAttachShader(prog_, vs)
                        _GL.glAttachShader(prog_, fs)
                        _GL.glBindAttribLocation(prog_, 0, 'a_position')
                        _GL.glBindAttribLocation(prog_, 1, 'a_tex')
                        _GL.glLinkProgram(prog_)
                        link_ok = _GL.glGetProgramiv(prog_, _GL.GL_LINK_STATUS)
                        # Shaders can be deleted after linking
                        _GL.glDeleteShader(vs)
                        _GL.glDeleteShader(fs)
                        if link_ok != _GL.GL_TRUE:
                            log = _GL.glGetProgramInfoLog(prog_)
                            _GL.glDeleteProgram(prog_)
                            raise RuntimeError(f"Program link failed: {log}")
                        return prog_
                    prog = None
                    try:
                        prog = _compile_link(VERT_150, FRAG_150)
                    except Exception:
                        prog = _compile_link(VERT_120, FRAG_120)
                    # Get attribute/uniform locations
                    self._gl_prog = prog
                    self._gl_pos_loc = _GL.glGetAttribLocation(prog, 'a_position')
                    self._gl_tex_loc = _GL.glGetAttribLocation(prog, 'a_tex')
                    self._gl_utex_loc = _GL.glGetUniformLocation(prog, 'u_tex_size')
                    # Create interleaved VBO: [pos.x, pos.y, tex.u, tex.v] * 4
                    quad = _np.array([
                        -1.0, -1.0, 0.0, 0.0,
                         1.0, -1.0, 1.0, 0.0,
                        -1.0,  1.0, 0.0, 1.0,
                         1.0,  1.0, 1.0, 1.0,
                    ], dtype=_np.float32)
                    vbo = _GL.glGenBuffers(1)
                    self._gl_vbo = vbo
                    _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, vbo)
                    _GL.glBufferData(_GL.GL_ARRAY_BUFFER, quad.nbytes, quad, _GL.GL_STATIC_DRAW)
                    _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, 0)
                else:
                    # Regular 2D texture path via gloo
                    prog = gloo.Program(VERT, FRAG_2D)
                    positions = _np.array([
                        [-1.0, -1.0],
                        [ 1.0, -1.0],
                        [-1.0,  1.0],
                        [ 1.0,  1.0],
                    ], dtype=_np.float32)
                    texcoords = _np.array([
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ], dtype=_np.float32)
                    prog['a_position'] = _np.ascontiguousarray(positions)
                    prog['a_tex'] = _np.ascontiguousarray(texcoords)
                    self.program = prog

            def on_draw(self, event):
                gloo.set_clear_color('black')
                gloo.clear()
                from OpenGL import GL as _GL  # type: ignore
                # Lazily create GL texture in VisPy's context
                if self._tex_cap is None:
                    cache = self._vt.gl_cache_init_for_current_context()
                    if not cache:
                        print("GL cache init (vispy) failed")
                        return
                    res = self._vt.gl_tex_from_cvpixelbuffer(cache, self._pb)
                    if not res:
                        print("vispy: gl_tex_from_cvpixelbuffer failed")
                        return
                    self._tex_cap, self._name, self._target, tw, th = res
                    # Initialize program now that target is known
                    self._init_gl()
                    print("VisPy target:", hex(int(self._target)))

                _GL.glActiveTexture(_GL.GL_TEXTURE0)
                _GL.glBindTexture(int(self._target), int(self._name))
                if int(self._target) == 0x84F5:
                    # Raw GL draw path
                    import ctypes as _ctypes
                    _GL.glUseProgram(self._gl_prog)
                    # u_tex_size (may be -1 for GLSL that doesn't use it)
                    if self._gl_utex_loc is not None and int(self._gl_utex_loc) != -1:
                        _GL.glUniform2f(self._gl_utex_loc, float(self._tsize[0]), float(self._tsize[1]))
                    # Vertex attribs from interleaved VBO
                    _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, self._gl_vbo)
                    stride = 4 * 4  # 4 floats per vertex * 4 bytes
                    if int(self._gl_pos_loc) != -1:
                        _GL.glEnableVertexAttribArray(self._gl_pos_loc)
                        _GL.glVertexAttribPointer(self._gl_pos_loc, 2, _GL.GL_FLOAT, _GL.GL_FALSE, stride, _ctypes.c_void_p(0))
                    if int(self._gl_tex_loc) != -1:
                        _GL.glEnableVertexAttribArray(self._gl_tex_loc)
                        _GL.glVertexAttribPointer(self._gl_tex_loc, 2, _GL.GL_FLOAT, _GL.GL_FALSE, stride, _ctypes.c_void_p(8))
                    # Sampler uniform to texture unit 0
                    loc_sampler = _GL.glGetUniformLocation(self._gl_prog, 'u_texRect')
                    _GL.glUniform1i(loc_sampler, 0)
                    _GL.glDrawArrays(_GL.GL_TRIANGLE_STRIP, 0, 4)
                    _GL.glBindBuffer(_GL.GL_ARRAY_BUFFER, 0)
                    _GL.glUseProgram(0)
                else:
                    # gloo draw path
                    # Sampler uniform to texture unit 0
                    self.program['u_tex2d'] = 0
                    self.program.draw('triangle_strip')
                # Release once after drawing
                if self._tex_cap is not None:
                    self._vt.gl_release_tex(self._tex_cap)
                    self._tex_cap = None
                    print("VisPy draw: OK")
                    self.close()

        # Draw once inside VisPy's context
        canv = _Canvas(vt, pb, (w, h))
        canv.show()
        canv.update()
        app.process_events()
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
