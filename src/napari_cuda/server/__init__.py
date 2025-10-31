"""napari-cuda server components for GPU-accelerated streaming.

Active server entry points live in :mod:`napari_cuda.server.app`. The layout
separates application bootstrap (`server/app`), authoritative state logic
(`server/control`), shared viewer state (`server/viewstate`), GPU/capture
pipelines (`server/engine`), and transport control helpers (`server/control`).
"""

__all__ = []
