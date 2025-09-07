"""napari-cuda server components for GPU-accelerated streaming.

Avoid importing heavy modules at package import time; import these explicitly:
- napari_cuda.server.headless_server: legacy Qt-based server
- napari_cuda.server.egl_headless_server: asyncio EGL server (no Qt)
"""

__all__ = ["CudaStreamingLayer", "HeadlessServer"]
