"""Application-facing entry points for the napari-cuda client."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ProxyViewer", "StreamingCanvas", "launch_streaming_client"]

_MODULE_MAP = {
    "ProxyViewer": ("napari_cuda.client.app.proxy_viewer", "ProxyViewer"),
    "StreamingCanvas": ("napari_cuda.client.app.streaming_canvas", "StreamingCanvas"),
    "launch_streaming_client": ("napari_cuda.client.app.launcher", "launch_streaming_client"),
}


def _lazy_attr(name: str) -> Any:
    module_path, attr = _MODULE_MAP[name]
    module = import_module(module_path)
    return getattr(module, attr)


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    if name not in _MODULE_MAP:
        raise AttributeError(name)
    return _lazy_attr(name)
