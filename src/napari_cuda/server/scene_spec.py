"""Scene and dims composition stubs.

This module provides function signatures and docstrings for building scene
metadata and serialization payloads. In PR1 we will have the class-based
ViewerSceneManager delegate to these functions. For now, these stubs are
no-ops to avoid behavior changes until wiring is approved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Minimal protocol-like stubs to keep signatures readable without importing
# heavy modules at this stage. These can be replaced with actual protocol types
# when wiring (or we can keep dicts to reduce coupling).

@dataclass
class LayerRenderHints:
    rendering: Optional[str] = None
    colormap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    opacity: Optional[float] = None
    sample_step: Optional[float] = None


def build_layer_spec(
    *,
    shape: Sequence[int],
    is_volume: bool,
    zarr_path: Optional[str],
    multiscale_levels: Optional[List[Dict[str, Any]]],
    multiscale_current_level: Optional[int],
    render_hints: Optional[LayerRenderHints] = None,
    extras: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a layer spec dict. No-op placeholder for now.

    Intended to replace `ViewerSceneManager._build_layer_spec` with a pure
    function returning a serializable dict.
    """
    return {}


def build_dims_spec(
    *,
    layer_spec: Dict[str, Any],
    current_step: Optional[Iterable[int]],
    ndisplay: Optional[int],
    axis_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a dims spec dict. No-op placeholder for now."""
    return {}


def build_camera_spec(
    *,
    center: Optional[Tuple[float, float, float]],
    zoom: Optional[float],
    angles: Optional[Tuple[float, float, float]],
    ndisplay: Optional[int],
) -> Dict[str, Any]:
    """Build a camera spec dict. No-op placeholder for now."""
    return {}


def scene_message(
    *,
    layers: List[Dict[str, Any]],
    dims: Dict[str, Any],
    camera: Dict[str, Any],
    capabilities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a scene message dict for the state channel. Placeholder."""
    return {
        "type": "scene.update",
        "timestamp": timestamp,
        "scene": {
            "layers": layers,
            "dims": dims,
            "camera": camera,
            "capabilities": capabilities or [],
            "metadata": metadata or {},
        },
    }


def dims_metadata(scene: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract dims metadata for HUD and downstream intents. Placeholder."""
    return {}


__all__ = [
    "LayerRenderHints",
    "build_layer_spec",
    "build_dims_spec",
    "build_camera_spec",
    "scene_message",
    "dims_metadata",
]

