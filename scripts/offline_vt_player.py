#!/usr/bin/env python3
"""
Offline zero-copy VT→OpenGL player (macOS).

Renders a synthetic animated pattern by writing into a CVPixelBuffer (BGRA),
creating a zero-copy GL texture via the VT shim, and drawing with the client
GLRenderer. No server, receiver, or decoder required.

Usage:
  uv run python scripts/offline_vt_player.py [width height fps seconds]

Env:
  QT_API=pyqt6 VISPY_USE_APP=pyqt6  # recommended for PyQt6
"""

from __future__ import annotations

import math
import os
import sys
import time


def _parse_args():
    try:
        w = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
        h = int(sys.argv[2]) if len(sys.argv) > 2 else 720
        fps = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
        secs = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
        mode = (sys.argv[5] if len(sys.argv) > 5 else os.getenv('OFFLINE_MODE', 'video')).lower()
        w = max(8, w); h = max(8, h); fps = max(1.0, fps); secs = max(0.5, secs)
        return w, h, fps, secs, mode
    except Exception:
        print("Invalid args; expected [width height fps seconds [video|turntable|cube]]")
        sys.exit(2)

def _maybe_hint_qt_backend():
    # Do not import Qt here; let VisPy pick based on env (VISPY_USE_APP/QT_API)
    # This avoids conflicting imports when both PyQt5 and PyQt6 are installed.
    want = (os.getenv('VISPY_USE_APP') or os.getenv('QT_API') or '').lower()
    if want not in ('pyqt5', 'pyqt6'):
        return


def main() -> int:
    w, h, fps, secs, mode = _parse_args()
    _maybe_hint_qt_backend()
    try:
        from vispy import app
    except Exception as e:
        print(f"VisPy not available: {e}")
        return 2
    try:
        from napari_cuda.client.rendering.renderer import GLRenderer
    except Exception as e:
        print(f"GLRenderer import failed: {e}")
        return 2
    try:
        from napari_cuda import _vt as vt
    except Exception as e:
        print(f"VT shim not available: {e}")
        return 2

    # Optional: use PyObjC Quartz to write into CVPixelBuffer base address
    # Not using PyObjC; use vt.pb_lock_base / pb_unlock_base for direct access

    class Player(app.Canvas):
        def __init__(self):
            app.Canvas.__init__(self, size=(w, h), show=True)
            self._renderer = GLRenderer(self)
            self._t0 = time.time()
            self._next = self._t0
            self._interval = 1.0 / fps
            self._end = self._t0 + secs
            self._frame_n = 0
            # Preallocate a reusable BGRA CVPixelBuffer to avoid per-frame alloc
            self._pb = vt.alloc_pixelbuffer_bgra(w, h, True)
            if self._pb is None:
                raise RuntimeError("alloc_pixelbuffer_bgra failed")
            # Precompute gradient bases
            import numpy as _np
            self._x = _np.linspace(0, 255, w, dtype=_np.uint8)[None, :]
            self._y = _np.linspace(0, 255, h, dtype=_np.uint8)[:, None]
            # Drive redraws via timer on GUI thread
            self._timer = app.Timer(interval=self._interval, connect=self._tick, start=True)

        def on_draw(self, event):
            # Draw latest frame if any
            payload = getattr(self, '_cur', None)
            if payload is not None:
                self._renderer.draw(payload)
                self._cur = None

        def _tick(self, ev):
            if time.time() >= self._end:
                try:
                    # Release our persistent PB
                    vt.pb_unlock_base(self._pb)
                except Exception:
                    pass
                try:
                    vt.release_frame(self._pb)
                except Exception:
                    pass
                self.close()
                return
            pb = self._pb
            # Fill BGRA via vt.pb_lock_base and numpy (vectorized)
            try:
                addr, bpr, width, height = vt.pb_lock_base(pb)
            except Exception:
                addr = None
            if addr:
                import ctypes
                import numpy as _np
                width = int(width); height = int(height); bpr = int(bpr)
                size = bpr * height
                raw = (ctypes.c_ubyte * size).from_address(int(addr))
                arr = _np.ctypeslib.as_array(raw)
                img = arr.reshape((height, bpr // 4, 4))
                # Animated offsets
                t = time.time() - self._t0
                ox = int((math.sin(t * 2.0) * 0.5 + 0.5) * 64)
                oy = int((math.cos(t * 1.7) * 0.5 + 0.5) * 64)
                # Build checker mask vectorized
                xx = (_np.arange(width, dtype=_np.int32)[None, :] + ox) // 32
                yy = (_np.arange(height, dtype=_np.int32)[:, None] + oy) // 32
                mask = ((xx ^ yy) & 1).astype(_np.uint8)
                r = mask * 255
                # Gradients from precomputed bases
                g = self._x.copy()
                g = _np.broadcast_to(g, (height, width))
                b = self._y.copy()
                b = _np.broadcast_to(b, (height, width))
                # Write BGRA into the IOSurface-backed buffer (first width columns)
                img[:height, :width, 0] = b
                img[:height, :width, 1] = g
                img[:height, :width, 2] = r
                img[:height, :width, 3] = 255
                try:
                    vt.pb_unlock_base(pb)
                except Exception:
                    pass
            # Hand off to renderer; we keep PB alive and reuse it
            self._cur = pb
            self.update()

    # 3D turntable (video path): vectorized Lambert shading of a rotating sphere into CVPixelBuffer
    import numpy as _np
    class TurntableVideo(app.Canvas):
        def __init__(self):
            app.Canvas.__init__(self, size=(w, h), show=True)
            self._renderer = GLRenderer(self)
            self._t0 = time.time()
            self._interval = 1.0 / fps
            self._end = self._t0 + secs
            # Persistent PB + precomputed XY grid
            self._pb = vt.alloc_pixelbuffer_bgra(w, h, True)
            if self._pb is None:
                raise RuntimeError("alloc_pixelbuffer_bgra failed")
            # Build normalized pixel grid, circle mask, and base Z
            aspect = w / float(h)
            xs = _np.linspace(-1.0, 1.0, w, dtype=_np.float32) * aspect
            ys = _np.linspace(-1.0, 1.0, h, dtype=_np.float32)
            self._X = _np.broadcast_to(xs[None, :], (h, w))
            self._Y = _np.broadcast_to(ys[:, None], (h, w))
            self._mask = (self._X**2 + self._Y**2) <= 1.0
            self._L = _np.array([0.35, 0.6, 1.0], dtype=_np.float32)
            self._L = self._L / _np.linalg.norm(self._L)
            self._timer = app.Timer(interval=self._interval, connect=self._tick, start=True)

        def on_draw(self, event):
            payload = getattr(self, '_cur', None)
            if payload is not None:
                self._renderer.draw(payload)
                self._cur = None

        def _tick(self, ev):
            if time.time() >= self._end:
                try:
                    vt.pb_unlock_base(self._pb)
                except Exception:
                    pass
                try:
                    vt.release_frame(self._pb)
                except Exception:
                    pass
                self.close(); return
            pb = self._pb
            # Lock and view BGRA image
            try:
                addr, bpr, width, height = vt.pb_lock_base(pb)
            except Exception:
                addr = None
            if addr:
                import ctypes
                width = int(width); height = int(height); bpr = int(bpr)
                size = bpr * height
                raw = (ctypes.c_ubyte * size).from_address(int(addr))
                img = _np.ctypeslib.as_array(raw).reshape((height, bpr // 4, 4))
                # Sphere Z from X,Y
                X = self._X; Y = self._Y; mask = self._mask
                Z = _np.zeros_like(X)
                Z[mask] = _np.sqrt(_np.maximum(0.0, 1.0 - (X[mask]**2 + Y[mask]**2)))
                # Rotate normals around Y axis by theta
                t = time.time() - self._t0
                c = math.cos(t * 0.8); s = math.sin(t * 0.8)
                Nx = X * c + Z * s
                Ny = Y
                Nz = -X * s + Z * c
                # Checker pattern on sphere using lon/lat
                lon = _np.arctan2(Nz, Nx)  # [-pi, pi]
                lat = _np.arcsin(_np.clip(Ny, -1.0, 1.0))  # [-pi/2, pi/2]
                u = (lon / (2.0 * _np.pi) + 0.5)
                v = (lat / _np.pi + 0.5)
                check = (((u * 8).astype(_np.int32) + (v * 8).astype(_np.int32)) & 1).astype(_np.float32)
                base_r = check * 1.0 + (1.0 - check) * 0.2
                base_g = 0.3 + 0.7 * (1.0 - check)
                base_b = 0.2 + 0.8 * check
                # Lambert shading
                L = self._L
                lam = _np.maximum(0.0, Nx * L[0] + Ny * L[1] + Nz * L[2])
                col_r = ((0.2 + 0.8 * lam) * base_r * 255.0).astype(_np.uint8)
                col_g = ((0.2 + 0.8 * lam) * base_g * 255.0).astype(_np.uint8)
                col_b = ((0.2 + 0.8 * lam) * base_b * 255.0).astype(_np.uint8)
                # Write BGRA in masked area; rest black
                img[:height, :width, 0] = 0
                img[:height, :width, 1] = 0
                img[:height, :width, 2] = 0
                img[:height, :width, 3] = 255
                img[mask, 0] = col_b[mask]
                img[mask, 1] = col_g[mask]
                img[mask, 2] = col_r[mask]
                try:
                    vt.pb_unlock_base(pb)
                except Exception:
                    pass
            self._cur = pb
            self.update()

    # 3D cube using VisPy gloo (separate path; useful to stress GL)
    from vispy import gloo

    VERT_3D = """
    uniform mat4 u_mvp;
    attribute vec3 a_position;
    attribute vec3 a_color;
    varying vec3 v_color;
    void main() {
        v_color = a_color;
        gl_Position = u_mvp * vec4(a_position, 1.0);
    }
    """

    FRAG_3D = """
    varying vec3 v_color;
    void main() {
        gl_FragColor = vec4(v_color, 1.0);
    }
    """

    def _perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> _np.ndarray:
        f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
        m = _np.zeros((4, 4), dtype=_np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (zfar + znear) / (znear - zfar)
        m[2, 3] = (2 * zfar * znear) / (znear - zfar)
        m[3, 2] = -1.0
        return m

    def _look_at(eye, target, up) -> _np.ndarray:
        eye = _np.array(eye, dtype=_np.float32)
        target = _np.array(target, dtype=_np.float32)
        up = _np.array(up, dtype=_np.float32)
        f = target - eye; f = f / _np.linalg.norm(f)
        s = _np.cross(f, up); s = s / _np.linalg.norm(s)
        u = _np.cross(s, f)
        m = _np.eye(4, dtype=_np.float32)
        m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
        t = _np.eye(4, dtype=_np.float32)
        t[:3, 3] = -eye
        return m @ t

    def _rotation_y(theta: float) -> _np.ndarray:
        c = math.cos(theta); s = math.sin(theta)
        m = _np.eye(4, dtype=_np.float32)
        m[0, 0] = c; m[0, 2] = s
        m[2, 0] = -s; m[2, 2] = c
        return m

    class Turntable(app.Canvas):
        def __init__(self):
            app.Canvas.__init__(self, size=(w, h), show=True)
            # Cube geometry (positions and colors)
            pos = _np.array([
                # Front
                [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
                # Back
                [-1, -1, -1], [-1,  1, -1], [ 1,  1, -1], [ 1, -1, -1],
            ], dtype=_np.float32) * 0.5
            col = _np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.2, 0.2, 0.2],
            ], dtype=_np.float32)
            idx = _np.array([
                # Front
                0, 1, 2, 0, 2, 3,
                # Top
                3, 2, 6, 3, 6, 5,
                # Back
                5, 6, 7, 5, 7, 4,
                # Bottom
                4, 7, 1, 4, 1, 0,
                # Left
                4, 0, 3, 4, 3, 5,
                # Right
                1, 7, 6, 1, 6, 2,
            ], dtype=_np.uint32)
            self._prog = gloo.Program(VERT_3D, FRAG_3D)
            self._prog['a_position'] = pos
            self._prog['a_color'] = col
            self._idx = gloo.IndexBuffer(idx)
            self._start = time.time()
            self._proj = _perspective(45.0, w / float(h), 0.1, 10.0)
            self._view = _look_at([1.8, 1.5, 2.2], [0, 0, 0], [0, 1, 0])
            self._timer = app.Timer(interval=1.0 / fps, connect=self.update, start=True)

        def on_draw(self, ev):
            gloo.set_state(clear_color='black', depth_test=True)
            gloo.clear(color=True, depth=True)
            t = time.time() - self._start
            model = _rotation_y(t * 0.8)
            mvp = self._proj @ self._view @ model
            self._prog['u_mvp'] = mvp.astype(_np.float32)
            self._prog.draw('triangles', self._idx)

        def on_resize(self, ev):
            sz = ev.size
            gloo.set_viewport(0, 0, *sz)
            aspect = sz[0] / float(max(1, sz[1]))
            self._proj = _perspective(45.0, aspect, 0.1, 10.0)

    if mode == 'turntable':
        TurntableVideo()
    elif mode == 'cube':
        Turntable()
    elif mode == 'volume':
        # Mirror server's --volume + --animate behavior using a standalone SceneCanvas
        from vispy import scene  # type: ignore
        canvas = scene.SceneCanvas(size=(w, h), bgcolor='black', show=True)
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60, distance=500)
        vol = (_np.random.rand(64, h, w).astype(_np.float32))
        _ = scene.visuals.Volume(vol, parent=view.scene, method='mip', cmap='viridis')
        start_t = time.time()
        def _tick(ev):
            t = time.time() - start_t
            try:
                view.camera.azimuth = (30.0 + t * 30.0) % 360.0
            except Exception:
                pass
            canvas.update()
        tmr = app.Timer(interval=1.0/fps, connect=_tick, start=True)
        # Stop after secs
        def _stop(ev):
            try:
                tmr.stop()
            except Exception:
                pass
            try:
                canvas.close()
            except Exception:
                pass
        app.Timer(interval=secs, iterations=1, connect=_stop, start=True)
    else:
        Player()
    app.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
