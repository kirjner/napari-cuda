"""Capture façade that composes GL capture, CUDA interop, and frame pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any
import logging
import time

from napari_cuda.server.rendering.gl_capture import GLCapture
from napari_cuda.server.rendering.cuda_interop import CudaInterop
from napari_cuda.server.rendering.frame_pipeline import FramePipeline


logger = logging.getLogger(__name__)


class CaptureFacade:
    """Own the capture stack (GL capture, CUDA interop, frame pipeline)."""

    def __init__(self, *, width: int, height: int) -> None:
        self._gl_capture = GLCapture(width, height)
        self._cuda = CudaInterop(width, height)
        self._pipeline = FramePipeline(
            gl_capture=self._gl_capture,
            cuda=self._cuda,
            width=width,
            height=height,
            debug=None,
        )

    # --- Configuration --------------------------------------------------

    def configure_auto_reset(self, enabled: bool) -> None:
        self._pipeline.configure_auto_reset(enabled)

    def set_debug(self, debug: object | None) -> None:
        self._pipeline.set_debug(debug)

    @property
    def enc_input_format(self) -> str:
        return self._pipeline.enc_input_format

    def set_enc_input_format(self, fmt: str) -> None:
        self._pipeline.set_enc_input_format(fmt)

    # --- Lifecycle ------------------------------------------------------

    def ensure(self) -> None:
        self._gl_capture.ensure()

    @property
    def texture_id(self) -> Optional[int]:
        return self._gl_capture.texture_id

    def initialize_cuda_interop(self) -> None:
        tex = self._gl_capture.texture_id
        if tex is None:
            raise RuntimeError("Capture texture is not available for CUDA interop")
        self._cuda.initialize(tex)

    def cleanup(self) -> None:
        try:
            self._cuda.cleanup()
        except Exception:  # pragma: no cover - defensive cleanup path
            logger.debug("CaptureFacade CUDA cleanup failed", exc_info=True)
        try:
            self._gl_capture.cleanup()
        except Exception:  # pragma: no cover - defensive cleanup path
            logger.debug("CaptureFacade GL cleanup failed", exc_info=True)

    # --- Capture --------------------------------------------------------

    def capture_blit_gpu_ns(self) -> Optional[int]:
        return self._pipeline.capture_blit_gpu_ns()

    def map_and_copy_to_torch(self, debug_cb: Optional[Callable[[int, int, int, object], None]]) -> tuple[float, float]:
        return self._pipeline.map_and_copy_to_torch(debug_cb)

    def convert_for_encoder(
        self,
        *,
        reset_camera: Optional[Callable[[], None]] = None,
    ) -> tuple[object, float]:
        return self._pipeline.convert_for_encoder(reset_camera=reset_camera)

    @property
    def orientation_ready(self) -> bool:
        return self._pipeline.orientation_ready

    @property
    def black_reset_done(self) -> bool:
        return self._pipeline.black_reset_done


@dataclass
class FrameCapture:
    frame: Any
    blit_gpu_ns: Optional[int]
    blit_cpu_ms: float
    map_ms: float
    copy_ms: float
    convert_ms: float
    orientation_ready: bool
    black_reset_done: bool


def capture_frame_for_encoder(
    facade: CaptureFacade,
    *,
    debug_cb: Optional[Callable[[int, int, int, object], None]] = None,
    reset_camera: Optional[Callable[[], None]] = None,
) -> FrameCapture:
    t_b0 = time.perf_counter()
    blit_gpu_ns = facade.capture_blit_gpu_ns()
    blit_cpu_ms = (time.perf_counter() - t_b0) * 1000.0
    map_ms, copy_ms = facade.map_and_copy_to_torch(debug_cb)
    frame, convert_ms = facade.convert_for_encoder(reset_camera=reset_camera)
    return FrameCapture(
        frame=frame,
        blit_gpu_ns=blit_gpu_ns,
        blit_cpu_ms=blit_cpu_ms,
        map_ms=map_ms,
        copy_ms=copy_ms,
        convert_ms=convert_ms,
        orientation_ready=facade.orientation_ready,
        black_reset_done=facade.black_reset_done,
    )


__all__ = ["CaptureFacade", "FrameCapture", "capture_frame_for_encoder"]
