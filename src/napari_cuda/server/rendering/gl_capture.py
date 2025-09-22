from __future__ import annotations

import logging
from typing import Optional, Tuple

from OpenGL import GL  # type: ignore


logger = logging.getLogger(__name__)


class GLCapture:
    """Manage the FBO/texture used for GPU blits plus timer queries."""

    def __init__(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._texture: Optional[int] = None
        self._fbo: Optional[int] = None
        self._query_ids: Optional[Tuple[int, int]] = None
        self._query_idx = 0
        self._query_started = False

    @property
    def texture_id(self) -> Optional[int]:
        return self._texture

    @property
    def framebuffer_id(self) -> Optional[int]:
        return self._fbo

    def ensure(self) -> None:
        if self._texture is None:
            self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            self._width,
            self._height,
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
            raise RuntimeError(f"Capture FBO incomplete: 0x{status:x}")

    def _query_pair(self) -> Tuple[int, int]:
        if self._query_ids is None:
            ids = GL.glGenQueries(2)
            if isinstance(ids, (list, tuple)) and len(ids) == 2:
                self._query_ids = (int(ids[0]), int(ids[1]))
            else:
                q0 = int(GL.glGenQueries(1))
                q1 = int(GL.glGenQueries(1))
                self._query_ids = (q0, q1)
        return self._query_ids

    def blit_with_timing(self) -> Optional[int]:
        if self._fbo is None:
            raise RuntimeError("Capture framebuffer not initialized")
        qids = self._query_pair()
        cur = self._query_idx
        prev = 1 - cur
        bound_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        read_fbo = int(bound_fbo)
        draw_fbo = int(self._fbo)

        try:
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, qids[cur])
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
            GL.glBlitFramebuffer(
                0,
                0,
                self._width,
                self._height,
                0,
                0,
                self._width,
                self._height,
                GL.GL_COLOR_BUFFER_BIT,
                GL.GL_NEAREST,
            )
            GL.glEndQuery(GL.GL_TIME_ELAPSED)
        except Exception:
            logger.debug("Blit with timer query failed; falling back to untimed blit", exc_info=True)
            self._blit_without_timing()
            return None
        finally:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)

        gpu_ns = None
        if self._query_started:
            try:
                result = GL.GLuint64(0)
                GL.glGetQueryObjectui64v(qids[prev], GL.GL_QUERY_RESULT, result)
                gpu_ns = int(result.value)
            except Exception:
                logger.debug("Fetching timer query result failed", exc_info=True)
                gpu_ns = None
        else:
            self._query_started = True

        self._query_idx = prev
        return gpu_ns

    def _blit_without_timing(self) -> None:
        if self._fbo is None:
            raise RuntimeError("Capture framebuffer not initialized")
        bound_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        read_fbo = int(bound_fbo)
        draw_fbo = int(self._fbo)
        try:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, draw_fbo)
            GL.glBlitFramebuffer(
                0,
                0,
                self._width,
                self._height,
                0,
                0,
                self._width,
                self._height,
                GL.GL_COLOR_BUFFER_BIT,
                GL.GL_NEAREST,
            )
        finally:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, read_fbo)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)

    def cleanup(self) -> None:
        try:
            if self._fbo is not None:
                GL.glDeleteFramebuffers(int(self._fbo))
        except Exception:
            logger.debug("Cleanup: delete FBO failed", exc_info=True)
        try:
            if self._texture is not None:
                GL.glDeleteTextures(int(self._texture))
        except Exception:
            logger.debug("Cleanup: delete texture failed", exc_info=True)
        if self._query_ids is not None:
            for q in self._query_ids:
                try:
                    GL.glDeleteQueries(1, [int(q)])
                except Exception:
                    logger.debug("Cleanup: delete query %s failed", q, exc_info=True)
        self._texture = None
        self._fbo = None
        self._query_ids = None
        self._query_idx = 0
        self._query_started = False


__all__ = ["GLCapture"]
