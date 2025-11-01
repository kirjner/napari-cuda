"""Client-side remote layer helpers with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LayerRecord",
    "RegistrySnapshot",
    "RemoteArray",
    "RemoteImageLayer",
    "RemoteLayerRegistry",
    "RemoteMultiscale",
    "RemotePreview",
    "build_remote_data",
]


def __getattr__(name: str) -> Any:
    if name in {"RemoteArray", "RemoteMultiscale", "RemotePreview", "build_remote_data"}:
        module = import_module(".remote_data", __name__)
        return getattr(module, name)
    if name in {"RemoteLayerRegistry", "RegistrySnapshot", "LayerRecord"}:
        module = import_module(".registry", __name__)
        return getattr(module, name)
    if name == "RemoteImageLayer":
        module = import_module(".remote_image_layer", __name__)
        return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
