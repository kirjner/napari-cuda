"""napari-cuda client components for receiving GPU-accelerated streams."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ProxyViewer", "StreamingCanvas", "launch_streaming_client"]


def _lazy_attr(name: str) -> Any:
    module_map = {
        "ProxyViewer": ("napari_cuda.client.proxy_viewer", "ProxyViewer"),
        "StreamingCanvas": ("napari_cuda.client.streaming_canvas", "StreamingCanvas"),
        "launch_streaming_client": ("napari_cuda.client.launcher", "launch_streaming_client"),
    }
    if name not in module_map:
        raise AttributeError(name)
    module_path, attr = module_map[name]
    module = import_module(module_path)
    return getattr(module, attr)


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    return _lazy_attr(name)
