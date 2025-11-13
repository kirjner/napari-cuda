"""Shared helpers for diffing LayerBlock payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from napari_cuda.server.scene.blocks import LayerBlock

VersionKey = tuple[str, str, str]
AppliedVersions = MutableMapping[VersionKey, int]

_MANDATORY_LAYER_CONTROLS = (
    "visible",
    "opacity",
    "blending",
    "interpolation",
    "colormap",
    "gamma",
)

_OPTIONAL_LAYER_CONTROLS = {
    "contrast_limits",
    "depiction",
    "rendering",
    "attenuation",
    "iso_threshold",
    "projection_mode",
    "plane_thickness",
}


@dataclass(frozen=True)
class LayerBlockDelta:
    """Describes which sections in a LayerBlock mutated."""

    block: LayerBlock
    controls: tuple[str, ...]
    metadata_changed: bool = False
    thumbnail_changed: bool = False


def index_layer_blocks(blocks: Sequence[LayerBlock]) -> dict[str, LayerBlock]:
    """Build a mapping keyed by layer_id for quick lookup."""

    return {str(block.layer_id): block for block in blocks}


def compute_layer_block_deltas(
    applied_versions: AppliedVersions | None,
    layer_blocks: Sequence[LayerBlock],
) -> dict[str, LayerBlockDelta]:
    """Return only the mutated portions of each LayerBlock."""

    deltas: dict[str, LayerBlockDelta] = {}
    for block in layer_blocks:
        layer_id = str(block.layer_id)
        versions = block.versions or {}
        controls_changed: list[str] = []
        metadata_changed = False
        thumbnail_changed = False

        if versions:
            for prop, version in versions.items():
                version_value = int(version)
                if applied_versions is not None:
                    key = ("layer", layer_id, str(prop))
                    previous = applied_versions.get(key)
                    if previous is not None and previous == version_value:
                        continue
                    applied_versions[key] = version_value
                prop_key = str(prop)
                if prop_key == "metadata":
                    metadata_changed = True
                    continue
                if prop_key == "thumbnail":
                    thumbnail_changed = True
                    continue
                if (
                    prop_key in _OPTIONAL_LAYER_CONTROLS
                    and getattr(block.controls, prop_key) is None
                ):
                    continue
                controls_changed.append(prop_key)
        else:
            base_keys: list[str] = list(_MANDATORY_LAYER_CONTROLS)
            for key in _OPTIONAL_LAYER_CONTROLS:
                if getattr(block.controls, key) is None:
                    continue
                base_keys.append(key)
            controls_changed = base_keys
            metadata_changed = bool(block.metadata)
            thumbnail_changed = block.thumbnail is not None

        if not controls_changed and not metadata_changed and not thumbnail_changed:
            continue

        deltas[layer_id] = LayerBlockDelta(
            block=block,
            controls=tuple(sorted(controls_changed)),
            metadata_changed=metadata_changed,
            thumbnail_changed=thumbnail_changed,
        )

    return deltas


__all__ = [
    "AppliedVersions",
    "LayerBlockDelta",
    "compute_layer_block_deltas",
    "index_layer_blocks",
]
