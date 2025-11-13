"""Shared helpers for diffing LayerBlock payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from napari_cuda.server.scene.blocks import (
    LayerBlock,
    layer_thumbnail_to_payload,
)

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
    extras_changed: bool = False


def index_layer_blocks(blocks: Sequence[LayerBlock]) -> dict[str, LayerBlock]:
    """Build a mapping keyed by layer_id for quick lookup."""

    return {str(block.layer_id): block for block in blocks}


def compute_layer_block_deltas(
    applied_versions: AppliedVersions | None,
    layer_blocks: Sequence[LayerBlock],
    *,
    previous_blocks: Mapping[str, LayerBlock] | None = None,
) -> dict[str, LayerBlockDelta]:
    """Return only the mutated portions of each LayerBlock."""

    deltas: dict[str, LayerBlockDelta] = {}
    for block in layer_blocks:
        layer_id = str(block.layer_id)
        versions = block.versions or {}
        controls_changed: list[str] = []
        metadata_changed = False
        thumbnail_changed = False
        extras_changed = False
        previous_block = previous_blocks.get(layer_id) if previous_blocks else None  # type: ignore[arg-type]

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
                if not hasattr(block.controls, prop_key):
                    extras_changed = True
                    continue
                if (
                    prop_key in _OPTIONAL_LAYER_CONTROLS
                    and getattr(block.controls, prop_key) is None
                ):
                    continue
                controls_changed.append(prop_key)
                extras_changed = extras_changed or prop_key.startswith("volume.")
        else:
            base_keys: list[str] = list(_MANDATORY_LAYER_CONTROLS)
            for key in _OPTIONAL_LAYER_CONTROLS:
                if getattr(block.controls, key) is None:
                    continue
                base_keys.append(key)
            controls_changed = base_keys
            metadata_changed = bool(block.metadata)
            thumbnail_changed = block.thumbnail is not None
        if previous_block is None and block.extras:
            extras_changed = True
        elif previous_block is not None and block.extras != previous_block.extras:
            extras_changed = True

        if not controls_changed and not metadata_changed and not thumbnail_changed and not extras_changed:
            continue

        deltas[layer_id] = LayerBlockDelta(
            block=block,
            controls=tuple(sorted(controls_changed)),
            metadata_changed=metadata_changed,
            thumbnail_changed=thumbnail_changed,
            extras_changed=extras_changed,
        )

    return deltas


def layer_block_delta_updates(
    delta: LayerBlockDelta,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Return dicts suitable for legacy LayerVisualState shims."""

    block = delta.block
    updates: dict[str, Any] = {}
    version_updates: dict[str, int] = {}
    versions = block.versions or {}

    for key in delta.controls:
        value = getattr(block.controls, key)
        updates[str(key)] = value
        version = versions.get(str(key))
        if version is not None:
            version_updates[str(key)] = int(version)

    if delta.metadata_changed:
        updates["metadata"] = dict(block.metadata)
        version = versions.get("metadata")
        if version is not None:
            version_updates["metadata"] = int(version)

    if delta.thumbnail_changed:
        updates["thumbnail"] = (
            layer_thumbnail_to_payload(block.thumbnail) if block.thumbnail is not None else None
        )
        version = versions.get("thumbnail")
        if version is not None:
            version_updates["thumbnail"] = int(version)

    if block.extras and delta.extras_changed:
        for key, value in block.extras.items():
            updates[str(key)] = value

    return updates, version_updates or None


def layer_block_delta_sections(
    delta: LayerBlockDelta,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Return notify.layers payload sections for a LayerBlock delta."""

    block = delta.block
    controls: dict[str, Any] = {}
    for key in delta.controls:
        controls[str(key)] = getattr(block.controls, key)

    metadata = dict(block.metadata) if delta.metadata_changed and block.metadata else None

    thumbnail = (
        layer_thumbnail_to_payload(block.thumbnail) if delta.thumbnail_changed and block.thumbnail is not None else None
    )

    data_payload: dict[str, Any] | None = None
    removed = False
    if delta.extras_changed and block.extras:
        data_payload = {}
        for key, value in block.extras.items():
            skey = str(key)
            if skey == "removed":
                if value:
                    removed = True
                continue
            data_payload[skey] = value
        if not data_payload:
            data_payload = None

    return (
        controls or None,
        metadata,
        data_payload,
        thumbnail,
        removed,
    )


__all__ = [
    "AppliedVersions",
    "LayerBlockDelta",
    "compute_layer_block_deltas",
    "index_layer_blocks",
    "layer_block_delta_sections",
    "layer_block_delta_updates",
]
