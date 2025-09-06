"""
OpenGL framebuffer interceptor for napari/VisPy.

Captures the final rendered frame by blitting the currently bound
framebuffer into an owned texture-attached FBO (GPU-only copy).

Exposes the texture id for CUDA interop, and optional GPU timing via
GL timer queries. Designed to be attached to napari's VispyCanvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from OpenGL import GL


@dataclass
class CaptureStats:
    width: int
    height: int
    gpu_time_ns: Optional[int]


class GLFrameInterceptor:
    """
    Intercepts VisPy canvas draws and captures the composited framebuffer
    into a texture for downstream GPU processing.

    Usage
    -----
    interceptor = GLFrameInterceptor(qt_viewer.canvas)
    interceptor.attach()
    # ... after each draw, call interceptor.texture_id
    # to retrieve the latest captured texture.
    """

    def __init__(self, canvas) -> None:  # canvas: napari._vispy.canvas.VispyCanvas
        self._canvas = canvas
        self._scene_canvas = canvas._scene_canvas
        self._texture: Optional[int] = None
        self._fbo: Optional[int] = None
        self._size: Tuple[int, int] = (0, 0)
        self._last_stats: Optional[CaptureStats] = None
        self._attached = False

    # Public API
    @property
    def texture_id(self) -> Optional[int]:
        return self._texture

    @property
    def last_stats(self) -> Optional[CaptureStats]:
        return self._last_stats

    def attach(self) -> None:
        """Attach to canvas draw + resize events."""
        if self._attached:
            return
        self._scene_canvas.events.draw.connect(self._on_draw, position='last')
        self._scene_canvas.events.resize.connect(self._on_resize)
        # Don't create buffers yet - wait for first draw when GL context exists
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        try:
            self._scene_canvas.events.draw.disconnect(self._on_draw)
        except Exception:
            pass
        try:
            self._scene_canvas.events.resize.disconnect(self._on_resize)
        except Exception:
            pass
        self._delete_buffers()
        self._attached = False

    # Internal
    def _on_resize(self, event) -> None:
        new_size = tuple(event.size)
        if new_size != self._size:
            # Try to resize buffers, but don't fail if context not ready
            self._ensure_buffers(new_size)

    def _ensure_buffers(self, size: Tuple[int, int]) -> bool:
        """
        Create or resize GL buffers. Returns True on success, False on failure.
        Failures are expected before GL context is ready.
        """
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            return False  # Invalid size
        
        self._size = (width, height)
        
        try:
            # Try to make context current if possible
            if hasattr(self._scene_canvas, 'context') and self._scene_canvas.context:
                try:
                    self._scene_canvas.context.make_current()
                except Exception:
                    pass  # Context may not be ready
            
            # Create or resize texture and FBO
            if self._texture is None:
                self._texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA,
                width,
                height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                None,
            )
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            if self._fbo is None:
                self._fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D,
                self._texture,
                0,
            )
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                # FBO creation failed - clean up and retry later
                raise RuntimeError(f"Capture FBO incomplete: 0x{status:x}")
            
            return True  # Success
            
        except (GL.error.GLError, RuntimeError) as e:
            # GL context not ready or FBO creation failed
            # Clean up and retry on next draw
            if self._fbo is not None:
                try:
                    GL.glDeleteFramebuffers(int(self._fbo))
                except Exception:
                    pass
                self._fbo = None
            if self._texture is not None:
                try:
                    GL.glDeleteTextures(int(self._texture))
                except Exception:
                    pass
                self._texture = None
            return False  # Will retry on next draw

    def _delete_buffers(self) -> None:
        try:
            if self._fbo is not None:
                GL.glDeleteFramebuffers(int(self._fbo))
        finally:
            self._fbo = None
        try:
            if self._texture is not None:
                GL.glDeleteTextures(int(self._texture))
        finally:
            self._texture = None

    def _on_draw(self, event) -> None:
        """After the scene has drawn, capture into our texture.

        We capture the currently bound framebuffer by blitting to our
        own FBO. This keeps the copy on-GPU.
        """
        width, height = self._canvas.size
        if (width, height) != self._size:
            # Size changed, need to recreate buffers
            if not self._ensure_buffers((width, height)):
                return  # Context not ready, try next frame
        
        # If we don't have buffers yet, try to create them
        if self._fbo is None or self._texture is None:
            if not self._ensure_buffers((width, height)):
                return  # Context not ready, try next frame
        
        # Now we should have valid buffers
        if self._fbo is None or self._texture is None:
            return  # Still no buffers somehow
        
        # Query currently bound framebuffer (Qt/VisPy FBO)
        try:
            bound_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        except GL.error.GLError:
            return  # GL context not ready
        
        read_fbo = int(bound_fbo)
        draw_fbo = int(self._fbo)
        
        gpu_time_ns: Optional[int] = None
        query_id = None
        try:
            # GPU timer query (optional)
            query_id = GL.glGenQueries(1)
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, query_id)

            # Blit color buffer into our texture-attached FBO
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
            GL.glBlitFramebuffer(
                0,
                0,
                width,
                height,
                0,
                0,
                width,
                height,
                GL.GL_COLOR_BUFFER_BIT,
                GL.GL_NEAREST,
            )

            GL.glEndQuery(GL.GL_TIME_ELAPSED)

            # Fetch timer result (blocking here is acceptable post-draw)
            available = GL.GLint(0)
            max_wait = 100
            while not available.value and max_wait > 0:
                GL.glGetQueryObjectiv(query_id, GL.GL_QUERY_RESULT_AVAILABLE, available)
                max_wait -= 1
            
            if available.value:
                result = GL.GLuint64(0)
                GL.glGetQueryObjectui64v(query_id, GL.GL_QUERY_RESULT, result)
                gpu_time_ns = int(result.value)
        except Exception:
            # Fallback without timing
            try:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
                GL.glBlitFramebuffer(
                    0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
                )
            except Exception:
                pass
        finally:
            # Restore default bind
            try:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, read_fbo)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)
            except Exception:
                pass
            if query_id is not None:
                try:
                    GL.glDeleteQueries(1, [query_id])
                except Exception:
                    pass

        self._last_stats = CaptureStats(width=width, height=height, gpu_time_ns=gpu_time_ns)