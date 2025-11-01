#!/usr/bin/env python3
"""
Minimal CUDA-OpenGL interop validation for HPC.

Creates an offscreen OpenGL context, allocates a texture, registers it with
CUDA via pycuda.gl, maps/unmaps it, and reports success/failure.

Run:
  uv run python scripts/test_cuda_gl.py
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger('test_cuda_gl')


def main() -> int:
    # Headless-friendly defaults
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

    try:
        import OpenGL.GL as GL
        import pycuda.driver as cuda
        import pycuda.gl
        from pycuda.gl import RegisteredImage, graphics_map_flags
        from qtpy.QtGui import (
            QOffscreenSurface,
            QOpenGLContext,
            QSurfaceFormat,
        )
        from qtpy.QtWidgets import QApplication
    except Exception as e:
        log.error(f"Import failure (Qt/OpenGL/CUDA): {e}")
        return 1

    app = QApplication.instance() or QApplication([])

    # Configure an OpenGL context
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)

    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    if not ctx.create():
        log.error('Failed to create QOpenGLContext')
        return 1

    surf = QOffscreenSurface()
    surf.setFormat(fmt)
    surf.create()
    if not surf.isValid():
        log.error('Failed to create QOffscreenSurface')
        return 1

    if not ctx.makeCurrent(surf):
        log.error('Failed to make context current')
        return 1

    # Create a simple RGBA8 texture
    width, height = 256, 256
    tex_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    import numpy as np
    data = (np.random.rand(height, width, 4) * 255).astype(np.uint8)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data
    )

    # Initialize CUDA and interop
    cuda.init()
    if cuda.Device.count() == 0:
        log.error('No CUDA devices found')
        return 1
    dev = cuda.Device(0)
    ctx_cuda = dev.make_context()
    pycuda.gl.init()

    try:
        reg = RegisteredImage(int(tex_id), GL.GL_TEXTURE_2D, graphics_map_flags.READ_ONLY)
        reg.map()
        arr = reg.get_mapped_array(0, 0)
        # Basic attribute access to ensure mapping worked
        _ = (GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_WIDTH),
             GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_HEIGHT))
        reg.unmap()
        log.info('CUDA-OpenGL interop: SUCCESS (map/unmap)')
        return 0
    except Exception as e:
        log.error(f'CUDA-OpenGL interop: FAILED ({e})')
        return 2
    finally:
        try:
            ctx_cuda.pop()
            ctx_cuda.detach()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())

