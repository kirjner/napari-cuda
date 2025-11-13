"""Temporary adapters between LayerBlock payloads and legacy visuals."""

from __future__ import annotations

from typing import Any

from napari_cuda.server.scene.blocks import (
    LayerBlock,
    LayerControlsBlock,
    layer_thumbnail_to_payload,
)
from napari_cuda.server.scene.models import LayerVisualState


def layer_block_controls_as_kwargs(controls: LayerControlsBlock) -> dict[str, Any]:
    """Return a mapping compatible with LayerVisualState kwargs."""

    kwargs: dict[str, Any] = {
        "visible": controls.visible,
        "opacity": controls.opacity,
        "blending": controls.blending,
        "interpolation": controls.interpolation,
        "colormap": controls.colormap,
        "gamma": controls.gamma,
    }
    if controls.contrast_limits is not None:
        kwargs["contrast_limits"] = controls.contrast_limits
    if controls.depiction is not None:
        kwargs["depiction"] = controls.depiction
    if controls.rendering is not None:
        kwargs["rendering"] = controls.rendering
    if controls.attenuation is not None:
        kwargs["attenuation"] = controls.attenuation
    if controls.iso_threshold is not None:
        kwargs["iso_threshold"] = controls.iso_threshold
    if controls.projection_mode is not None:
        kwargs["projection_mode"] = controls.projection_mode
    if controls.plane_thickness is not None:
        kwargs["plane_thickness"] = controls.plane_thickness
    return kwargs


def layer_block_to_visual_state(block: LayerBlock) -> LayerVisualState:
    """Convert a LayerBlock into the legacy LayerVisualState shim."""

    controls = layer_block_controls_as_kwargs(block.controls)
    metadata = dict(block.metadata) if block.metadata else {}
    thumbnail = (
        layer_thumbnail_to_payload(block.thumbnail)
        if block.thumbnail is not None
        else None
    )
    extras = dict(block.extras)
    versions = {str(key): int(value) for key, value in (block.versions or {}).items()}
    return LayerVisualState(
        layer_id=str(block.layer_id),
        metadata=metadata,
        thumbnail=thumbnail,
        extra=extras,
        versions=versions,
        **controls,
    )


__all__ = [
    "layer_block_controls_as_kwargs",
    "layer_block_to_visual_state",
]
