"""Frame capture and pre-encode helpers for the EGL worker.

Encapsulates the GPU blit, CUDA map/copy, and RGBA→encoder conversions so the
worker can delegate heavy lifting and shed defensive state.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

from .gl_capture import GLCapture
from .cuda_interop import CudaInterop

logger = logging.getLogger(__name__)


class FramePipeline:
    """Owns capture + CUDA interop and pre-encode colour conversion."""

    def __init__(
        self,
        *,
        gl_capture: GLCapture,
        cuda: CudaInterop,
        width: int,
        height: int,
        debug: Optional[object] = None,
    ) -> None:
        self._gl_capture = gl_capture
        self._cuda = cuda
        self._width = int(width)
        self._height = int(height)
        self._debug = debug
        self._auto_reset_on_black: bool = True
        self._black_reset_done: bool = False
        self._orientation_ready: bool = False
        self._enc_input_format: str = "YUV444"
        self._logged_swizzle = False
        self._logged_swizzle_stats = False
        self._raw_dump_budget: int = 0

    # --- Configuration ------------------------------------------------------

    def set_debug(self, debug: Optional[object]) -> None:
        self._debug = debug

    def configure_auto_reset(self, enabled: bool) -> None:
        self._auto_reset_on_black = bool(enabled)

    def set_raw_dump_budget(self, budget: int) -> None:
        self._raw_dump_budget = max(0, int(budget))

    def set_dimensions(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)

    @property
    def enc_input_format(self) -> str:
        return self._enc_input_format

    def set_enc_input_format(self, fmt: str) -> None:
        fmt_str = str(fmt)
        if fmt_str != self._enc_input_format:
            self._logged_swizzle = False
        self._enc_input_format = fmt_str

    @property
    def orientation_ready(self) -> bool:
        return self._orientation_ready

    @property
    def black_reset_done(self) -> bool:
        return self._black_reset_done

    # --- Capture helpers ----------------------------------------------------

    def capture_blit_gpu_ns(self) -> Optional[int]:
        return self._gl_capture.blit_with_timing()

    def map_and_copy_to_torch(self, debug_cb: Optional[Callable[[int, int, int, torch.Tensor], None]]) -> tuple[float, float]:
        return self._cuda.map_and_copy(debug_cb=debug_cb)

    # --- Conversion ---------------------------------------------------------

    def convert_for_encoder(self, *, reset_camera: Optional[Callable[[], None]] = None) -> tuple[torch.Tensor, float]:
        """Convert the CUDA frame to the encoder's expected input format."""
        t_c0 = time.perf_counter()
        frame = self._cuda.torch_frame
        if self._raw_dump_budget > 0 and self._debug is not None:
            try:
                self._debug.dump_cuda_rgba(frame, self._width, self._height, prefix="raw")  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug("Pre-encode raw dump failed: %s", exc)
            finally:
                self._raw_dump_budget = max(0, self._raw_dump_budget - 1)
        src = frame
        try:
            r = src[..., 0].float() / 255.0
            g = src[..., 1].float() / 255.0
            b = src[..., 2].float() / 255.0
            y_n = 0.2126 * r + 0.7152 * g + 0.0722 * b
            y = torch.clamp(16.0 + 219.0 * y_n, 0.0, 255.0)
            cb = torch.clamp(128.0 + 224.0 * (b - y_n) / 1.8556, 0.0, 255.0)
            cr = torch.clamp(128.0 + 224.0 * (r - y_n) / 1.5748, 0.0, 255.0)
            if not self._logged_swizzle_stats:
                try:
                    rm = float(r.mean().item())
                    gm = float(g.mean().item())
                    bm = float(b.mean().item())
                    ym = float(y.mean().item())
                    cbm = float(cb.mean().item())
                    crm = float(cr.mean().item())
                    logger.info(
                        "Pre-encode channel means: R=%.3f G=%.3f B=%.3f | Y=%.3f Cb=%.3f Cr=%.3f",
                        rm,
                        gm,
                        bm,
                        ym,
                        cbm,
                        crm,
                    )
                    if self._auto_reset_on_black and not self._black_reset_done and (rm + gm + bm) < 1e-4:
                        logger.info("Detected black initial frame; applying one-shot camera reset")
                        if reset_camera is not None:
                            try:
                                reset_camera()
                            except Exception:  # pragma: no cover - debug path
                                logger.debug("auto-reset-on-black failed", exc_info=True)
                        self._black_reset_done = True
                    if (rm + gm + bm) >= 1e-4:
                        self._orientation_ready = True
                except Exception as exc:  # pragma: no cover - statistics only
                    logger.debug("Swizzle stats log failed: %s", exc)
                self._logged_swizzle_stats = True
            H, W = self._height, self._width
            if self._enc_input_format == 'YUV444':
                if not self._logged_swizzle:
                    logger.info("Pre-encode swizzle: RGBA -> YUV444 (BT709 LIMITED, planar)")
                    self._logged_swizzle = True
                dst = torch.empty((H * 3, W), dtype=torch.uint8, device=src.device)
                dst[0:H, :] = y.to(torch.uint8)
                dst[H:2 * H, :] = cb.to(torch.uint8)
                dst[2 * H:3 * H, :] = cr.to(torch.uint8)
            elif self._enc_input_format in ('ARGB', 'ABGR'):
                if not self._logged_swizzle:
                    logger.info("Pre-encode swizzle: RGBA -> %s (packed)", self._enc_input_format)
                    self._logged_swizzle = True
                indices = [3, 0, 1, 2] if self._enc_input_format == 'ARGB' else [3, 2, 1, 0]
                dst = src[..., indices].contiguous()
            else:
                if not self._logged_swizzle:
                    logger.info("Pre-encode swizzle: RGBA -> NV12 (BT709 LIMITED, 4:2:0 UV interleaved)")
                    self._logged_swizzle = True
                cb2 = F.avg_pool2d(cb.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
                cr2 = F.avg_pool2d(cr.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
                H2, W2 = cb2.shape
                dst = torch.empty((H + H2, W), dtype=torch.uint8, device=src.device)
                dst[0:H, :] = y.to(torch.uint8)
                uv = dst[H:, :]
                uv[:, 0::2] = cb2.to(torch.uint8)
                uv[:, 1::2] = cr2.to(torch.uint8)
        except Exception as exc:  # pragma: no cover - conversion failure path
            logger.exception("Pre-encode color conversion failed: %s", exc)
            dst = src[..., [3, 0, 1, 2]]
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover - debug path
            logger.debug("torch.cuda.synchronize failed", exc_info=True)
        t_c1 = time.perf_counter()
        return dst, (t_c1 - t_c0) * 1000.0
