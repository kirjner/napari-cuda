"""Render resource helpers for the EGL worker."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import pycuda.driver as cuda  # type: ignore

from napari_cuda.server.app.config import ServerCtx
from napari_cuda.server.engine import CaptureFacade, EglContext, Encoder


@dataclass
class WorkerResources:
    """Own the long-lived GL, CUDA, capture, and encoder handles."""

    width: int
    height: int
    egl: EglContext = field(init=False)
    capture: CaptureFacade = field(init=False)
    enc_lock: threading.Lock = field(init=False)
    encoder: Optional[Encoder] = None
    cuda_ctx: Optional[cuda.Context] = None
    enc_input_fmt: Optional[str] = None

    def __post_init__(self) -> None:
        self.egl = EglContext(self.width, self.height)
        self.capture = CaptureFacade(width=self.width, height=self.height)
        self.enc_lock = threading.Lock()

    def init_cuda(self) -> cuda.Context:
        """Retain and push the primary CUDA context for device 0."""

        cuda.init()
        device = cuda.Device(0)
        context = device.retain_primary_context()
        context.push()
        self.cuda_ctx = context
        return context

    def bootstrap(self, *, server_ctx: ServerCtx, fps_hint: int) -> Optional[cuda.Context]:
        """Initialize EGL, CUDA, capture interop, and encoder in sequence."""

        self.egl.ensure()
        context = self.init_cuda()
        self.capture.ensure()
        self.capture.initialize_cuda_interop()
        self.ensure_encoder(fps_hint=int(fps_hint), server_ctx=server_ctx)
        return context

    def ensure_encoder(self, *, fps_hint: int, server_ctx: ServerCtx) -> Encoder:
        """Create or refresh the NVENC encoder and sync capture format."""

        with self.enc_lock:
            if self.encoder is None:
                self.encoder = Encoder(self.width, self.height, fps_hint=int(fps_hint))
            encoder = self.encoder
            encoder.set_fps_hint(int(fps_hint))
            encoder.setup(server_ctx)
            self.capture.pipeline.set_enc_input_format(encoder.input_format)
            self.enc_input_fmt = encoder.input_format
            return encoder

    def reset_encoder(self, *, fps_hint: int, server_ctx: ServerCtx) -> None:
        """Reset the encoder after resolution or policy changes."""

        with self.enc_lock:
            encoder = self.encoder
            if encoder is None:
                return
            encoder.set_fps_hint(int(fps_hint))
            encoder.reset(server_ctx)
            self.enc_input_fmt = encoder.input_format

    def request_idr(self) -> None:
        """Ask the encoder to emit an IDR frame."""

        with self.enc_lock:
            encoder = self.encoder
            if encoder is None:
                return
            encoder.request_idr()

    def force_idr(self) -> None:
        """Force the encoder to output an IDR immediately."""

        with self.enc_lock:
            encoder = self.encoder
            if encoder is None:
                return
            encoder.force_idr()

    def cleanup(self) -> None:
        """Tear down capture, encoder, EGL, and CUDA resources."""

        self.capture.cleanup()

        with self.enc_lock:
            encoder = self.encoder
            if encoder is not None:
                encoder.shutdown()
            self.encoder = None

        if self.cuda_ctx is not None:
            self.cuda_ctx.pop()
            self.cuda_ctx.detach()
            self.cuda_ctx = None

        self.egl.cleanup()


__all__ = ["WorkerResources"]
