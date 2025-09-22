from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

import cupy as cp  # type: ignore
import pycuda.driver as cuda  # type: ignore
import pycuda.gl  # type: ignore
from pycuda.gl import RegisteredImage, graphics_map_flags  # type: ignore
import torch  # type: ignore
from OpenGL import GL  # type: ignore


logger = logging.getLogger(__name__)


DebugCallback = Callable[[int, int, int, torch.Tensor], None]


class CudaInterop:
    """Manage GL texture registration and copy into a Torch tensor."""

    def __init__(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._registered: Optional[RegisteredImage] = None
        self._torch_frame: Optional[torch.Tensor] = None
        self._torch_frame_argb: Optional[torch.Tensor] = None
        self._yuv444: Optional[torch.Tensor] = None
        self._dst_pitch_bytes: Optional[int] = None
        self._texture_id: Optional[int] = None

    @property
    def torch_frame(self) -> torch.Tensor:
        if self._torch_frame is None:
            raise RuntimeError("CUDA interop not initialized")
        return self._torch_frame

    @property
    def torch_frame_argb(self) -> Optional[torch.Tensor]:
        return self._torch_frame_argb

    @property
    def yuv444_buffer(self) -> Optional[torch.Tensor]:
        return self._yuv444

    @property
    def dst_pitch_bytes(self) -> Optional[int]:
        return self._dst_pitch_bytes

    @property
    def texture_id(self) -> Optional[int]:
        return self._texture_id

    def initialize(self, texture_id: int) -> None:
        pycuda.gl.init()
        self._texture_id = int(texture_id)
        self._registered = RegisteredImage(int(texture_id), GL.GL_TEXTURE_2D, graphics_map_flags.READ_ONLY)  # type: ignore[attr-defined]

        dev_cp = cp.empty((self._height, self._width, 4), dtype=cp.uint8)
        if hasattr(torch, "from_dlpack"):
            self._torch_frame = torch.from_dlpack(dev_cp)
        else:
            self._torch_frame = torch.utils.dlpack.from_dlpack(dev_cp)
        del dev_cp
        self._torch_frame = self._torch_frame.contiguous()
        try:
            self._torch_frame_argb = torch.empty_like(self._torch_frame)
        except Exception:
            logger.debug("Allocating ARGB working buffer failed", exc_info=True)
            self._torch_frame_argb = None
        try:
            self._yuv444 = torch.empty((self._height * 3, self._width), dtype=torch.uint8, device=self._torch_frame.device)
        except Exception:
            logger.debug("Allocating YUV buffer failed", exc_info=True)
            self._yuv444 = None

        self._dst_pitch_bytes = int(self._torch_frame.stride(0) * self._torch_frame.element_size())
        row_bytes = int(self._width * 4)
        if int(os.getenv("NAPARI_CUDA_FORCE_TIGHT_PITCH", "0") or "0"):
            self._dst_pitch_bytes = row_bytes
        logger.info("CUDA dst pitch: %d bytes (expected %d)", self._dst_pitch_bytes, row_bytes)
        if self._dst_pitch_bytes != row_bytes:
            logger.warning(
                "Non-tight pitch: dst_pitch=%d expected=%d (width*4). Right-edge artifacts possible.",
                self._dst_pitch_bytes,
                row_bytes,
            )
        if (self._width % 2) or (self._height % 2):
            logger.warning(
                "Odd dimensions %dx%d may need SPS cropping to avoid padding artifacts.",
                self._width,
                self._height,
            )

    def map_and_copy(self, debug_cb: Optional[DebugCallback] = None) -> tuple[float, float]:
        if self._registered is None or self._torch_frame is None:
            raise RuntimeError("CUDA interop not initialized")
        mapping = self._registered.map()
        try:
            t_m0 = time.perf_counter()
            cuda_array = mapping.array(0, 0)
            t_m1 = time.perf_counter()
            map_ms = (t_m1 - t_m0) * 1000.0

            start_evt = cuda.Event()
            end_evt = cuda.Event()
            start_evt.record()
            copy = cuda.Memcpy2D()
            copy.set_src_array(cuda_array)
            copy.set_dst_device(int(self._torch_frame.data_ptr()))
            copy.width_in_bytes = self._width * 4
            copy.height = self._height
            copy.dst_pitch = self._dst_pitch_bytes if self._dst_pitch_bytes is not None else (self._width * 4)
            copy(aligned=True)
            end_evt.record()
            end_evt.synchronize()
            copy_ms = start_evt.time_till(end_evt)

            if debug_cb is not None:
                try:
                    debug_cb(
                        int(self._texture_id) if self._texture_id is not None else 0,
                        self._width,
                        self._height,
                        self._torch_frame,
                    )
                except Exception:
                    logger.warning("Debug triplet callback failed", exc_info=True)

            return map_ms, copy_ms
        finally:
            try:
                mapping.unmap()
            except Exception:
                logger.exception("CUDA mapping unmap failed")
                raise

    def cleanup(self) -> None:
        try:
            if self._registered is not None:
                self._registered.unregister()
        except Exception:
            logger.debug("Cleanup: unregister RegisteredImage failed", exc_info=True)
        self._registered = None
        self._torch_frame = None
        self._torch_frame_argb = None
        self._yuv444 = None
        self._dst_pitch_bytes = None
        self._texture_id = None


__all__ = ["CudaInterop"]
