"""Helper utilities for the client streaming loop."""

from importlib import import_module
from types import ModuleType

__all__ = ["control"]


def __getattr__(name: str) -> ModuleType:
    if name == "control":
        module = import_module(f"{__name__}.control")
        globals()[name] = module
        return module
    raise AttributeError(name)
