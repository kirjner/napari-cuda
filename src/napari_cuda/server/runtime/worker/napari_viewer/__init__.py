"""Compatibility shim for relocated viewer bootstrap helpers."""

from __future__ import annotations

from napari_cuda.server.runtime.bootstrap.interface import ViewerBootstrapInterface
from napari_cuda.server.runtime.bootstrap.setup_viewer import (
    CanonicalAxes,
    ViewerBuilder,
    apply_canonical_axes,
    canonical_axes_from_source,
)

__all__ = [
    "CanonicalAxes",
    "ViewerBuilder",
    "ViewerBootstrapInterface",
    "apply_canonical_axes",
    "canonical_axes_from_source",
]
