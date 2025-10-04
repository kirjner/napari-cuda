"""napari-cuda server components for GPU-accelerated streaming.

Active server entry points live in :mod:`napari_cuda.server.app`. The new
layout separates application bootstrap (`server/app`), authoritative state
logic (`server/state`), data sources and policies (`server/data`), rendering
pipelines (`server/rendering`), and transport control (`server/control`).
"""

__all__ = []
