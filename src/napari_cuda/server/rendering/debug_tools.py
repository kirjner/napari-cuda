"""
Debug tools for dumping intermediate GPU frames to disk for diagnosis.

Provides a small configuration object and a dumper class that can save:
- Default framebuffer via glReadPixels (flipped upright for viewing)
- FBO texture via glGetTexImage (flipped upright)
- CUDA-copied RGBA tensor (optionally flipped for viewing)

This module centralizes debug dumping and logging so the render loop stays lean.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
import ctypes
from typing import Optional

import numpy as np

from napari_cuda.server.data.logging_policy import DumpControls

try:
    from OpenGL import GL  # type: ignore
except Exception as e:  # pragma: no cover - only used in GPU runtime
    GL = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class DebugConfig:
    enabled: bool = False
    frames_remaining: int = 0
    out_dir: str = "benchmarks/frames"
    flip_cuda_for_view: bool = False
    have_imageio: bool = False

    @classmethod
    def from_policy(cls, policy: DumpControls) -> "DebugConfig":
        have_iio = False
        try:
            import imageio.v3 as _iio  # type: ignore
            have_iio = True
        except Exception:
            have_iio = False
        frames_budget = max(0, int(policy.frames_budget))
        return cls(
            enabled=bool(policy.enabled and frames_budget > 0),
            frames_remaining=frames_budget,
            out_dir=str(policy.output_dir),
            flip_cuda_for_view=bool(policy.flip_cuda_for_view),
            have_imageio=have_iio,
        )


class DebugDumper:
    def __init__(self, cfg: DebugConfig) -> None:
        self.cfg = cfg
        self._env_logged = False

    def log_env_once(self) -> None:
        if self._env_logged:
            return
        logger.info(
            "Debug: frames=%d, out_dir=%s, flip_cuda=%s, have_imageio=%s",
            self.cfg.frames_remaining,
            self.cfg.out_dir,
            self.cfg.flip_cuda_for_view,
            self.cfg.have_imageio,
        )
        self._env_logged = True

    def ensure_out_dir(self) -> None:
        try:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create debug out_dir %s: %s", self.cfg.out_dir, e)

    def _save_array(self, arr: np.ndarray, prefix: str, width: int, height: int) -> Optional[str]:
        ts = int(time.time() * 1000)
        try:
            if self.cfg.have_imageio:
                import imageio.v3 as iio  # type: ignore
                path = os.path.join(self.cfg.out_dir, f"{prefix}_{width}x{height}_{ts}.png")
                iio.imwrite(path, arr)
            else:
                path = os.path.join(self.cfg.out_dir, f"{prefix}_{width}x{height}_{ts}.npy")
                np.save(path, arr)
            logger.info("Debug dump: %s", path)
            return path
        except Exception as e:
            logger.warning("Failed to save %s dump: %s", prefix, e, exc_info=True)
            return None

    def dump_framebuffer(self, width: int, height: int) -> Optional[str]:  # GPU-only path
        if GL is None:
            logger.warning("GL not available; cannot dump framebuffer")
            return None
        try:
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            raw = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            fb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
            fb = np.flipud(fb)
            return self._save_array(fb, "fb", width, height)
        except Exception as e:
            logger.warning("Framebuffer dump failed: %s", e, exc_info=True)
            return None

    def dump_texture(self, texture_id: int, width: int, height: int) -> Optional[str]:  # GPU-only path
        if GL is None:
            logger.warning("GL not available; cannot dump texture")
            return None
        try:
            with _BindTexture2D(texture_id):
                buf = (ctypes.c_ubyte * (width * height * 4))()
                GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, buf)
                tex = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
                tex = np.flipud(tex)
                return self._save_array(tex, "tex", width, height)
        except Exception as e:
            logger.warning("Texture dump failed: %s", e, exc_info=True)
            return None

    def dump_cuda_rgba(self, tensor, width: int, height: int, prefix: str = "cuda") -> Optional[str]:
        try:
            host = tensor.detach().contiguous().to("cpu")
            arr = host.numpy()
            if self.cfg.flip_cuda_for_view:
                arr = np.flipud(arr)
            return self._save_array(arr, prefix, width, height)
        except Exception as e:
            logger.warning("CUDA dump failed: %s", e, exc_info=True)
            return None

    def dump_triplet(self, texture_id: int, width: int, height: int, tensor) -> None:
        if not self.cfg.enabled or self.cfg.frames_remaining <= 0:
            return
        self.log_env_once()
        self.ensure_out_dir()
        try:
            self.dump_framebuffer(width, height)
            self.dump_texture(texture_id, width, height)
            self.dump_cuda_rgba(tensor, width, height, prefix="cuda")
        finally:
            # Ensure we always decrement to avoid spamming if errors persist
            self.cfg.frames_remaining = max(0, self.cfg.frames_remaining - 1)


class _BindTexture2D:
    def __init__(self, tex_id: int) -> None:
        self.tex_id = tex_id
        self.prev: Optional[int] = None

    def __enter__(self):  # pragma: no cover - GL runtime
        if GL is None:
            return
        try:
            self.prev = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        except Exception:
            self.prev = None
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - GL runtime
        if GL is None:
            return False
        try:
            if self.prev is not None:
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.prev)
            else:
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        except Exception as e:
            logger.debug("Restore GL texture bind failed: %s", e)
        return False
