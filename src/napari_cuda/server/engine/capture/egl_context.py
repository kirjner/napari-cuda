from __future__ import annotations

import ctypes
import logging
from typing import Optional

from OpenGL import EGL, GL  # type: ignore

logger = logging.getLogger(__name__)


class EglContext:
    """Manage adoption or creation of an EGL context for headless rendering."""

    def __init__(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._display: Optional[int] = None
        self._context: Optional[int] = None
        self._surface: Optional[int] = None
        self._owns: bool = False

    @property
    def display(self) -> Optional[int]:
        return self._display

    @property
    def context(self) -> Optional[int]:
        return self._context

    @property
    def surface(self) -> Optional[int]:
        return self._surface

    @property
    def owns_context(self) -> bool:
        return self._owns

    def ensure(self) -> None:
        """Adopt the current EGL context or create a new pbuffer context."""
        if self._context is not None:
            return

        cur_ctx = None
        try:
            cur_ctx = EGL.eglGetCurrentContext()
        except Exception:
            cur_ctx = None

        if cur_ctx and cur_ctx != EGL.EGL_NO_CONTEXT:
            try:
                self._display = EGL.eglGetCurrentDisplay()
                self._surface = EGL.eglGetCurrentSurface(EGL.EGL_DRAW)
                self._context = cur_ctx
                self._owns = False
                self._configure_read_buffer()
                return
            except Exception:
                logger.debug("Adopting current EGL context failed; falling back to new context", exc_info=True)
                self._display = None
                self._surface = None
                self._context = None

        self._create_pbuffer()
        self._configure_read_buffer()

    def cleanup(self) -> None:
        """Terminate the EGL display if this manager created it."""
        if self._owns and self._display is not None:
            try:
                EGL.eglTerminate(self._display)
            except Exception:
                logger.debug("cleanup: eglTerminate failed", exc_info=True)
        self._display = None
        self._context = None
        self._surface = None
        self._owns = False

    # ------------------------------------------------------------------
    # Internal helpers

    def _configure_read_buffer(self) -> None:
        try:
            dbl = bool(GL.glGetBooleanv(GL.GL_DOUBLEBUFFER))
        except Exception:
            dbl = False
        try:
            GL.glReadBuffer(GL.GL_BACK if dbl else GL.GL_FRONT)
            logger.debug(
                "EGL context ready: GL_DOUBLEBUFFER=%s -> GL_READ_BUFFER=%s",
                dbl,
                'BACK' if dbl else 'FRONT',
            )
        except Exception:
            logger.debug("Setting GL_READ_BUFFER failed", exc_info=True)

    def _create_pbuffer(self) -> None:
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
            EGL.EGL_WIDTH, self._width,
            EGL.EGL_HEIGHT, self._height,
            EGL.EGL_NONE,
        ]
        pbuffer_attribs_p = (EGL.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
        egl_surface = EGL.eglCreatePbufferSurface(egl_display, egl_config, pbuffer_attribs_p)
        if egl_surface == EGL.EGL_NO_SURFACE:
            raise RuntimeError("Failed to create EGL pbuffer surface")

        if not EGL.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context):
            raise RuntimeError("Failed to make EGL context current")

        self._display = egl_display
        self._context = egl_context
        self._surface = egl_surface
        self._owns = True


__all__ = ["EglContext"]
