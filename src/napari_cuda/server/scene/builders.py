"""Server-side helpers to build scene snapshots from the control ledger."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import (
    Any,
    Mapping,
    Optional,
)

import numpy as np

from napari_cuda.protocol.axis_labels import normalize_axis_labels
from napari_cuda.protocol.snapshots import (
    LayerSnapshot,
    SceneSnapshot,
    scene_snapshot,
    viewer_snapshot_from_blocks,
)
from napari_cuda.server.scene.defaults import default_volume_state
from napari_cuda.server.scene.models import (
    CameraDeltaCommand,
    LayerVisualState,
    RenderLedgerSnapshot,
)
from napari_cuda.server.state_ledger import LedgerEntry, ServerStateLedger
from napari_cuda.shared.dims_spec import (
    build_dims_spec_from_ledger,
    validate_ledger_against_dims_spec,
)

logger = logging.getLogger(__name__)


CONTROL_KEYS: tuple[str, ...] = (
    "visible",
    "opacity",
    "blending",
    "interpolation",
    "gamma",
    "colormap",
    "contrast_limits",
    "depiction",
    "rendering",
    "attenuation",
    "iso_threshold",
)

_DEFAULT_LAYER_CONTROLS: dict[str, Any] = {
    "visible": True,
    "opacity": 1.0,
    "blending": "opaque",
    "interpolation": "linear",
    "gamma": 1.0,
    "colormap": "gray",
}


__all__ = [
    "CONTROL_KEYS",
    "CameraDeltaCommand",
    "build_ledger_snapshot",
    "default_volume_state",
    "pull_render_snapshot",
    "snapshot_dims_metadata",
    "snapshot_layer_controls",
    "snapshot_multiscale_state",
    "snapshot_render_state",
    "snapshot_scene",
    "snapshot_viewport_state",
    "snapshot_volume_state",
]


# ---------------------------------------------------------------------------
# Ledger adapters


def snapshot_multiscale_state(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> dict[str, Any]:
    """Derive multiscale policy metadata from a ledger snapshot."""

    def _normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _normalize(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_normalize(v) for v in value]
        return value

    state: dict[str, Any] = {}

    policy_entry = snapshot.get(("multiscale", "main", "policy"))
    if policy_entry is not None and policy_entry.value is not None:
        state["policy"] = str(policy_entry.value)

    index_space_entry = snapshot.get(("multiscale", "main", "index_space"))
    if index_space_entry is not None and index_space_entry.value is not None:
        state["index_space"] = str(index_space_entry.value)

    level_entry = snapshot.get(("multiscale", "main", "level"))
    if level_entry is not None and level_entry.value is not None:
        level_value = level_entry.value
        if isinstance(level_value, (int, float)):
            state["current_level"] = int(level_value)
        else:
            state["current_level"] = level_value

    levels_entry = snapshot.get(("multiscale", "main", "levels"))
    if levels_entry is not None and isinstance(levels_entry.value, Sequence):
        levels_payload = []
        for entry in levels_entry.value:
            if isinstance(entry, Mapping):
                normalized = _normalize(entry)
                if "downsample" not in normalized:
                    shape_value = normalized.get("shape")
                    if isinstance(shape_value, Sequence) and shape_value:
                        normalized["downsample"] = [1.0 for _ in shape_value]
                levels_payload.append(normalized)
        if levels_payload:
            state["levels"] = levels_payload

    level_shapes_entry = snapshot.get(("multiscale", "main", "level_shapes"))
    if level_shapes_entry is not None and isinstance(level_shapes_entry.value, Sequence):
        shapes_payload = []
        for shape in level_shapes_entry.value:
            if isinstance(shape, Sequence):
                shapes_payload.append([int(dim) for dim in shape])
        if shapes_payload:
            state["level_shapes"] = shapes_payload

    downgraded_entry = snapshot.get(("multiscale", "main", "downgraded"))
    if downgraded_entry is not None and downgraded_entry.value is not None:
        state["downgraded"] = bool(downgraded_entry.value)

    return state


def snapshot_render_state(
    ledger: ServerStateLedger,
    *,
    plane_center: Any = None,
    plane_zoom: Any = None,
    plane_rect: Any = None,
    volume_center: Any = None,
    volume_angles: Any = None,
    volume_distance: Any = None,
    volume_fov: Any = None,
    current_step: Any = None,
    volume_mode: Any = None,
    volume_colormap: Any = None,
    volume_clim: Any = None,
    volume_opacity: Any = None,
    volume_sample_step: Any = None,
) -> RenderLedgerSnapshot:
    """Build a render-scene snapshot from the ledger with optional overrides."""

    snapshot_map = ledger.snapshot()
    base = build_ledger_snapshot(ledger, snapshot_map)

    volume_defaults = snapshot_volume_state(snapshot_map)
    if not volume_defaults:
        volume_defaults = default_volume_state()
    default_mode = volume_defaults.get("mode")
    default_colormap = volume_defaults.get("colormap")
    default_clim_value = volume_defaults.get("clim")
    default_clim = None
    if isinstance(default_clim_value, Sequence):
        default_clim = tuple(float(v) for v in default_clim_value)
    default_opacity = volume_defaults.get("opacity")
    default_sample_step = volume_defaults.get("sample_step")

    overrides = {
        "plane_center": _snapshot_coalesce_tuple(plane_center, base.plane_center, float),
        "plane_zoom": _snapshot_coalesce_float(plane_zoom, base.plane_zoom),
        "plane_rect": _snapshot_coalesce_tuple(plane_rect, base.plane_rect, float),
        "volume_center": _snapshot_coalesce_tuple(volume_center, base.volume_center, float),
        "volume_angles": _snapshot_coalesce_tuple(volume_angles, base.volume_angles, float),
        "volume_distance": _snapshot_coalesce_float(volume_distance, base.volume_distance),
        "volume_fov": _snapshot_coalesce_float(volume_fov, base.volume_fov),
        "current_step": _snapshot_coalesce_tuple(current_step, base.current_step, int),
        "volume_mode": _snapshot_coalesce_string(volume_mode, base.volume_mode, fallback=default_mode),
        "volume_colormap": _snapshot_coalesce_string(
            volume_colormap,
            base.volume_colormap,
            fallback=default_colormap,
        ),
        "volume_clim": _snapshot_coalesce_tuple(
            volume_clim,
            base.volume_clim,
            float,
            fallback=default_clim,
        ),
        "volume_opacity": _snapshot_coalesce_float(
            volume_opacity,
            base.volume_opacity,
            fallback=default_opacity,
        ),
        "volume_sample_step": _snapshot_coalesce_float(
            volume_sample_step,
            base.volume_sample_step,
            fallback=default_sample_step,
        ),
    }

    return replace(base, **overrides)


def snapshot_layer_controls(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> dict[str, dict[str, Any]]:
    """Collect per-layer control values from a ledger snapshot."""

    controls: dict[str, dict[str, Any]] = defaultdict(dict)
    for (scope, target, key), entry in snapshot.items():
        if scope != "layer":
            continue
        controls[str(target)][str(key)] = entry.value
    return {layer_id: props for layer_id, props in controls.items() if props}


def snapshot_volume_state(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> dict[str, Any]:
    """Extract volume render hints from the ledger snapshot."""

    mapping: dict[str, Any] = {}
    for ledger_key, field in (
        (("volume", "main", "rendering"), "rendering"),
        (("volume", "main", "colormap"), "colormap"),
        (("volume", "main", "contrast_limits"), "clim"),
        (("volume", "main", "opacity"), "opacity"),
        (("volume", "main", "sample_step"), "sample_step"),
    ):
        entry = snapshot.get(ledger_key)
        if entry is not None and entry.value is not None:
            mapping[field] = entry.value
    return mapping


def snapshot_viewport_state(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> dict[str, Any]:
    """Extract viewport mode and plane/volume payloads from the ledger snapshot."""

    def _clone(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _clone(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_clone(v) for v in value]
        return value

    payload: dict[str, Any] = {}

    mode_entry = snapshot.get(("viewport", "state", "mode"))
    if mode_entry is not None and mode_entry.value is not None:
        payload["mode"] = str(mode_entry.value)

    plane_entry = snapshot.get(("viewport", "plane", "state"))
    if plane_entry is not None and plane_entry.value is not None:
        payload["plane"] = _clone(plane_entry.value)

    volume_entry = snapshot.get(("viewport", "volume", "state"))
    if volume_entry is not None and volume_entry.value is not None:
        payload["volume"] = _clone(volume_entry.value)

    return payload


# ---------------------------------------------------------------------------
# Scene snapshot construction


def snapshot_scene(
    *,
    render_state: RenderLedgerSnapshot,
    ledger_snapshot: Mapping[tuple[str, str, str], LedgerEntry],
    canvas_size: tuple[int, int],
    fps_target: float,
    default_layer_id: str = "layer-0",
    default_layer_name: str = "napari-cuda",
    ndisplay: Optional[int] = None,
    zarr_path: Optional[str] = None,
    scene_source: Optional[object] = None,
    layer_controls: Optional[Mapping[str, Mapping[str, Any]]] = None,
    multiscale_state: Optional[dict[str, Any]] = None,
    volume_state: Optional[dict[str, Any]] = None,
    thumbnail_provider: Optional[Callable[[str], Optional[Any]]] = None,
) -> SceneSnapshot:
    """Build a ``SceneSnapshot`` suitable for ``notify.scene`` baselines."""

    ndisplay_value = int(ndisplay if ndisplay is not None else render_state.ndisplay or 2)
    ndisplay_value = max(1, ndisplay_value)

    ledger_multiscale = snapshot_multiscale_state(ledger_snapshot)
    if multiscale_state is None or (not multiscale_state and ledger_multiscale):
        multiscale_state = ledger_multiscale

    if not multiscale_state and scene_source is not None:
        multiscale_state = _fallback_multiscale_state(scene_source)

    ledger_controls = snapshot_layer_controls(ledger_snapshot)
    if layer_controls is None or (not layer_controls and ledger_controls):
        layer_controls = ledger_controls

    if volume_state is None or not volume_state:
        volume_state = _resolve_volume_state(render_state, ledger_snapshot)

    geometry = _resolve_layer_geometry(render_state, multiscale_state, ndisplay_value)

    dims_block = _build_dims_block(geometry)
    # Attach optional dims margins if present in ledger
    ml_entry = ledger_snapshot.get(("dims", "main", "margin_left"))
    mr_entry = ledger_snapshot.get(("dims", "main", "margin_right"))
    if ml_entry is not None and ml_entry.value is not None:
        vals = ml_entry.value
        if isinstance(vals, (list, tuple)):
            dims_block["margin_left"] = [float(v) for v in vals]
    if mr_entry is not None and mr_entry.value is not None:
        vals = mr_entry.value
        if isinstance(vals, (list, tuple)):
            dims_block["margin_right"] = [float(v) for v in vals]

    camera_block = _build_camera_block(render_state, geometry, ndisplay_value)

    viewer_settings = {
        "fps_target": float(fps_target),
        "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
        "volume_enabled": bool(geometry.is_volume),
    }

    viewer = viewer_snapshot_from_blocks(
        settings=viewer_settings,
        dims=dims_block,
        camera=camera_block,
    )

    layer_snapshots: list[LayerSnapshot] = []
    if default_layer_id:
        layer_id_str = str(default_layer_id)

        layer_overrides: LayerVisualState | Mapping[str, Any] | None = None
        if layer_controls:
            layer_overrides = layer_controls.get(layer_id_str)
        if layer_overrides is None and render_state.layer_values:
            layer_state = render_state.layer_values.get(layer_id_str)
            if isinstance(layer_state, LayerVisualState):
                layer_overrides = layer_state

        controls_block = _resolve_layer_controls(layer_overrides)

        layer_block: dict[str, Any] = {
            "layer_id": layer_id_str,
            "layer_type": "image",
            "name": default_layer_name,
            "ndim": len(geometry.shape),
            "shape": geometry.shape,
            "axis_labels": geometry.axis_labels,
            "volume": geometry.is_volume,
            "controls": controls_block,
            "level_shapes": geometry.level_shapes,
        }

        layer_metadata: dict[str, Any] = {}
        if thumbnail_provider is not None:
            raw_thumbnail = thumbnail_provider(layer_id_str)
            normalized_thumb = _normalize_thumbnail_array(raw_thumbnail)
            if normalized_thumb is not None:
                layer_metadata["thumbnail"] = normalized_thumb.tolist()

        layer_block["metadata"] = layer_metadata

        multiscale_block = _build_multiscale_block(multiscale_state, base_shape=geometry.shape)
        if multiscale_block:
            layer_block["multiscale"] = multiscale_block

        render_hints = _build_render_hints(volume_state)
        if render_hints:
            layer_block["render"] = render_hints

        contrast_limits = _resolve_contrast_limits(volume_state)
        if contrast_limits is not None:
            layer_block["contrast_limits"] = contrast_limits

        scale = _resolve_scale(scene_source)
        if scale is not None:
            layer_block["scale"] = scale

        translate = _resolve_translate(scene_source)
        if translate is not None:
            layer_block["translate"] = translate

        source_block = _resolve_source_block(scene_source)
        if source_block is not None:
            layer_block["source"] = source_block

        layer_snapshots.append(LayerSnapshot(layer_id=layer_id_str, block=layer_block))

    metadata = _scene_metadata(zarr_path=zarr_path, scene_source=scene_source)

    return scene_snapshot(
        viewer=viewer,
        layers=layer_snapshots,
        policies={},
        metadata=metadata,
    )


def snapshot_dims_metadata(scene_snapshot: SceneSnapshot) -> dict[str, Any]:
    """Extract dims metadata used for HUD payloads."""

    dims_block = dict(scene_snapshot.viewer.dims)
    meta: dict[str, Any] = dict(dims_block)

    if not scene_snapshot.layers:
        return meta

    layer_block = scene_snapshot.layers[0].block
    meta["volume"] = bool(layer_block.get("volume"))

    controls = layer_block.get("controls")
    if isinstance(controls, Mapping):
        meta["controls"] = {str(key): value for key, value in controls.items()}

    multiscale_block = layer_block.get("multiscale")
    if isinstance(multiscale_block, Mapping):
        ms_dict = dict(multiscale_block)
        metadata_block = ms_dict.pop("metadata", None)
        if isinstance(metadata_block, Mapping):
            for key, value in metadata_block.items():
                ms_dict[str(key)] = _normalize_scalar(value)
        meta["multiscale"] = ms_dict
        levels = ms_dict.get("levels")
        if isinstance(levels, Sequence) and levels:
            index = int(ms_dict.get("current_level", 0))
            if 0 <= index < len(levels):
                level_shape = levels[index].get("shape")
                if isinstance(level_shape, Sequence):
                    meta["sizes"] = [int(x) for x in level_shape]

    return meta


# ---------------------------------------------------------------------------
# Internal helpers


def _snapshot_coalesce_tuple(
    value: Any,
    base: Optional[tuple],
    mapper,
    *,
    fallback: Any = None,
) -> Optional[tuple]:
    source = value if value is not None else (base if base is not None else fallback)
    if source is None:
        return None
    if isinstance(source, (str, bytes, bytearray)):
        return base
    if not isinstance(source, Iterable):
        return base
    return tuple(mapper(v) for v in source)


def _snapshot_coalesce_float(value: Any, base: Optional[float], *, fallback: Any = None) -> Optional[float]:
    source = value if value is not None else (base if base is not None else fallback)
    if source is None:
        return None
    if isinstance(source, (int, float)):
        return float(source)
    return base


def _snapshot_coalesce_string(value: Any, base: Optional[str], *, fallback: Any = None) -> Optional[str]:
    source = value if value is not None else (base if base is not None else fallback)
    if source is None:
        return None
    text = str(source).strip()
    return text if text else base


def _resolve_layer_controls(overrides: LayerVisualState | Mapping[str, Any] | None) -> dict[str, Any]:
    controls = dict(_DEFAULT_LAYER_CONTROLS)
    if not overrides:
        return controls

    if isinstance(overrides, LayerVisualState):
        for key in CONTROL_KEYS:
            value = getattr(overrides, key)
            if value is not None:
                controls[key] = value
        return controls

    for key, value in overrides.items():
        key_str = str(key)
        if key_str not in CONTROL_KEYS:
            continue
        controls[key_str] = value
    return controls


def _resolve_volume_state(
    render_state: RenderLedgerSnapshot,
    ledger_snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> dict[str, Any]:
    defaults = default_volume_state()
    ledger_state = snapshot_volume_state(ledger_snapshot)

    mode = render_state.volume_mode or ledger_state.get("mode") or defaults["mode"]
    colormap = render_state.volume_colormap or ledger_state.get("colormap") or defaults["colormap"]
    clim = render_state.volume_clim or ledger_state.get("clim") or defaults["clim"]
    opacity = (
        render_state.volume_opacity
        if render_state.volume_opacity is not None
        else ledger_state.get("opacity", defaults["opacity"])
    )
    sample_step = (
        render_state.volume_sample_step
        if render_state.volume_sample_step is not None
        else ledger_state.get("sample_step", defaults["sample_step"])
    )

    clim_list: list[float]
    if isinstance(clim, Sequence):
        clim_list = [float(v) for v in clim[:2]]
        if len(clim_list) < 2:
            clim_list.extend([float(defaults["clim"][0]), float(defaults["clim"][1])])
    else:
        clim_list = [float(defaults["clim"][0]), float(defaults["clim"][1])]

    return {
        "mode": str(mode),
        "colormap": str(colormap),
        "clim": clim_list[:2],
        "opacity": float(opacity),
        "sample_step": float(sample_step),
    }


@dataclass
class _LayerGeometry:
    shape: list[int]
    axis_labels: list[str]
    order: list[int]
    current_step: list[int]
    displayed: list[int]
    level_shapes: list[list[int]]
    current_level: int
    ndisplay: int
    is_volume: bool


def _resolve_layer_geometry(
    render_state: RenderLedgerSnapshot,
    multiscale_state: Optional[Mapping[str, Any]],
    ndisplay_value: int,
) -> _LayerGeometry:
    level_shapes: list[list[int]] = []

    if render_state.level_shapes:
        level_shapes = [list(map(int, level)) for level in render_state.level_shapes]
    elif multiscale_state:
        levels = multiscale_state.get("levels")
        if isinstance(levels, Sequence):
            for entry in levels:
                if isinstance(entry, Mapping):
                    shape = entry.get("shape")
                    if isinstance(shape, Sequence):
                        level_shapes.append([int(x) for x in shape])

    current_level = int(render_state.current_level or 0)
    if multiscale_state and "current_level" in multiscale_state:
        level_value = multiscale_state["current_level"]
        if isinstance(level_value, (int, float)):
            current_level = int(level_value)

    if level_shapes:
        current_level = max(0, min(current_level, len(level_shapes) - 1))
        shape = level_shapes[current_level]
    else:
        axis_count = len(render_state.axis_labels or ())
        shape = [0 for _ in range(axis_count or 2)]
        level_shapes = [shape]
        current_level = 0

    axis_labels = list(render_state.axis_labels or ())
    axis_labels = normalize_axis_labels(axis_labels, len(shape))

    order = list(render_state.order or range(len(shape)))
    order = [idx for idx in order if 0 <= idx < len(shape)]
    seen = set(order)
    for idx in range(len(shape)):
        if idx not in seen:
            order.append(idx)

    current_step_src = list(render_state.current_step or [])
    if len(current_step_src) < len(shape):
        current_step_src.extend([0] * (len(shape) - len(current_step_src)))
    current_step = [int(value) for value in current_step_src[: len(shape)]]

    displayed_src = list(render_state.displayed or [])
    if not displayed_src:
        displayed_src = order[-ndisplay_value:]
    displayed = [idx for idx in displayed_src if 0 <= idx < len(shape)]
    if len(displayed) < ndisplay_value:
        extras = [idx for idx in order if idx not in displayed]
        displayed.extend(extras[: ndisplay_value - len(displayed)])
    displayed = displayed[: max(1, ndisplay_value)]

    ndisplay_clamped = max(1, min(ndisplay_value, len(shape))) if shape else ndisplay_value

    is_volume = bool(render_state.dims_mode == "volume" or ndisplay_value >= 3)

    return _LayerGeometry(
        shape=list(shape),
        axis_labels=list(axis_labels),
        order=order,
        current_step=current_step,
        displayed=displayed,
        level_shapes=[list(entry) for entry in level_shapes],
        current_level=current_level,
        ndisplay=ndisplay_clamped,
        is_volume=is_volume,
    )


def _build_multiscale_block(
    multiscale_state: Optional[Mapping[str, Any]],
    *,
    base_shape: Sequence[int],
) -> Optional[dict[str, Any]]:
    if not multiscale_state:
        return None

    levels_data = multiscale_state.get("levels")
    if not isinstance(levels_data, Sequence) or not levels_data:
        return None

    levels: list[dict[str, Any]] = []
    for entry in levels_data:
        if not isinstance(entry, Mapping):
            continue
        shape_value = entry.get("shape")
        downsample_value = entry.get("downsample")
        if not isinstance(shape_value, Sequence) or not isinstance(downsample_value, Sequence):
            continue
        levels.append(
            {
                "shape": [int(x) for x in shape_value],
                "downsample": [float(x) for x in downsample_value],
                "path": entry.get("path"),
            }
        )

    if not levels:
        return None

    payload: dict[str, Any] = {
        "levels": levels,
        "current_level": int(multiscale_state.get("current_level", 0)),
    }

    metadata: dict[str, Any] = {}
    if "policy" in multiscale_state:
        metadata["policy"] = multiscale_state["policy"]
    if "index_space" in multiscale_state:
        metadata["index_space"] = multiscale_state["index_space"]
    if metadata:
        payload["metadata"] = metadata

    return payload


def _build_render_hints(volume_state: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    shading = volume_state.get("shade") if volume_state else None
    if shading is None:
        return None
    return {"shading": str(shading)}


def _resolve_contrast_limits(volume_state: Mapping[str, Any]) -> Optional[list[float]]:
    if not volume_state:
        return None
    clim = volume_state.get("clim")
    if isinstance(clim, Sequence) and len(clim) >= 2:
        return [float(clim[0]), float(clim[1])]
    return None


def _build_dims_block(geometry: _LayerGeometry) -> dict[str, Any]:
    return {
        "ndim": len(geometry.shape),
        "axis_labels": list(geometry.axis_labels),
        "order": list(geometry.order),
        "level_shapes": [list(shape) for shape in geometry.level_shapes],
        "current_level": int(geometry.current_level),
        "current_step": list(geometry.current_step),
        "displayed": list(geometry.displayed),
        "ndisplay": int(geometry.ndisplay),
    }


def _build_camera_block(
    render_state: RenderLedgerSnapshot,
    geometry: _LayerGeometry,
    ndisplay_value: int,
) -> dict[str, Any]:
    block: dict[str, Any] = {"ndisplay": int(ndisplay_value)}

    if geometry.is_volume and render_state.volume_center is not None:
        block["center"] = [float(v) for v in render_state.volume_center]
        if render_state.volume_angles is not None:
            block["angles"] = [float(v) for v in render_state.volume_angles]
        if render_state.volume_distance is not None:
            block["distance"] = float(render_state.volume_distance)
        if render_state.volume_fov is not None:
            block["fov"] = float(render_state.volume_fov)
        return block

    if render_state.plane_center is not None:
        block["center"] = [float(v) for v in render_state.plane_center]
    if render_state.plane_zoom is not None:
        block["zoom"] = float(render_state.plane_zoom)
    if render_state.plane_rect is not None:
        block["rect"] = [float(v) for v in render_state.plane_rect]

    return block


def _scene_metadata(
    *,
    zarr_path: Optional[str],
    scene_source: Optional[object],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if zarr_path:
        metadata["status"] = "ready"
    else:
        metadata["status"] = "idle"
    if zarr_path:
        metadata["source_path"] = str(zarr_path)
    if scene_source is not None:
        metadata.setdefault("source_kind", "ome-zarr")
    return metadata


def _fallback_multiscale_state(scene_source: object) -> dict[str, Any]:
    descriptors = getattr(scene_source, "level_descriptors", [])
    if not descriptors:
        return {}
    return {
        "policy": "auto",
        "index_space": "base",
        "current_level": int(getattr(scene_source, "current_level", 0)),
        "levels": [
            {
                "path": getattr(desc, "path", ""),
                "shape": [int(x) for x in getattr(desc, "shape", ())],
                "downsample": list(getattr(desc, "downsample", ())),
            }
            for desc in descriptors
        ],
    }


def _resolve_scale(scene_source: Optional[object]) -> Optional[list[float]]:
    if scene_source is None:
        return None
    if not hasattr(scene_source, "current_level") or not hasattr(scene_source, "level_scale"):
        return None
    level_index = int(scene_source.current_level)
    scale_fn = scene_source.level_scale
    scale_tuple = scale_fn(level_index)
    if not isinstance(scale_tuple, Sequence):
        return None
    return [float(value) for value in scale_tuple]


def _resolve_translate(scene_source: Optional[object]) -> Optional[list[float]]:
    if scene_source is None or not hasattr(scene_source, "translate"):
        return None
    translate = scene_source.translate
    if not isinstance(translate, Sequence):
        return None
    return [float(value) for value in translate]


def _resolve_source_block(scene_source: Optional[object]) -> Optional[dict[str, Any]]:
    if scene_source is None:
        return None
    block: dict[str, Any] = {"kind": "ome-zarr"}
    if hasattr(scene_source, "data_id"):
        data_id = scene_source.data_id
        if data_id is not None:
            block["data_id"] = str(data_id)
    if hasattr(scene_source, "cache_version"):
        cache_version = scene_source.cache_version
        if cache_version is not None:
            block["cache_version"] = int(cache_version)
    return block or None


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    return value


def _normalize_thumbnail_array(data: Any) -> Optional[np.ndarray]:
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (list, tuple)):
        arr = np.asarray(data)
    else:
        return None

    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        return None
    return arr


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_contrast_limits(value: Any) -> Optional[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lo_val = _coerce_float(value[0])
        hi_val = _coerce_float(value[1])
        if lo_val is None or hi_val is None:
            return None
        lo = float(lo_val)
        hi = float(hi_val)
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)
    return None


# ---------------------------------------------------------------------------
# Render ledger snapshots


def build_ledger_snapshot(
    ledger: ServerStateLedger,
    snapshot: dict[tuple[str, str, str], LedgerEntry] | None = None,
) -> RenderLedgerSnapshot:
    """Construct a render snapshot from confirmed ledger state."""

    snapshot = snapshot if snapshot is not None else ledger.snapshot()
    dims_spec = build_dims_spec_from_ledger(snapshot)
    validate_ledger_against_dims_spec(dims_spec, snapshot)

    plane_center_tuple = _tuple_or_none(
        _ledger_value(snapshot, "camera_plane", "main", "center"),
        float,
    )
    plane_zoom_float = _float_or_none(_ledger_value(snapshot, "camera_plane", "main", "zoom"))
    plane_rect_tuple = _tuple_or_none(
        _ledger_value(snapshot, "camera_plane", "main", "rect"),
        float,
    )

    volume_center_tuple = _tuple_or_none(
        _ledger_value(snapshot, "camera_volume", "main", "center"),
        float,
    )
    volume_angles_tuple = _tuple_or_none(
        _ledger_value(snapshot, "camera_volume", "main", "angles"),
        float,
    )
    volume_distance_float = _float_or_none(_ledger_value(snapshot, "camera_volume", "main", "distance"))
    volume_fov_float = _float_or_none(_ledger_value(snapshot, "camera_volume", "main", "fov"))

    dims_entry = snapshot.get(("dims", "main", "current_step"))
    current_step = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "current_step"), int)
    dims_version = None if dims_entry is None else _version_or_none(dims_entry.version)

    view_entry_key = ("view", "main", "ndisplay")
    view_entry = snapshot.get(view_entry_key)
    ndisplay_val = _int_or_none(_ledger_value(snapshot, "view", "main", "ndisplay"))
    dims_mode = None
    if ndisplay_val is not None:
        dims_mode = "volume" if int(ndisplay_val) >= 3 else "plane"
    displayed_axes = _tuple_or_none(_ledger_value(snapshot, "view", "main", "displayed"), int)
    order_axes = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "order"), int)
    axis_labels = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "axis_labels"), str)
    dims_labels = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "labels"), str)
    level_shapes = _shape_sequence(_ledger_value(snapshot, "multiscale", "main", "level_shapes"))
    multiscale_level_entry_key = ("multiscale", "main", "level")
    multiscale_level_entry = snapshot.get(multiscale_level_entry_key)
    current_level = _int_or_none(_ledger_value(snapshot, "multiscale", "main", "level"))
    multiscale_level_version = (
        None if multiscale_level_entry is None else _version_or_none(multiscale_level_entry.version)
    )

    defaults = default_volume_state()
    volume_mode = _string_or_none(_ledger_value(snapshot, "volume", "main", "rendering"), fallback=defaults.get("mode"))
    volume_colormap = _string_or_none(
        _ledger_value(snapshot, "volume", "main", "colormap"),
        fallback=defaults.get("colormap"),
    )
    volume_clim = _tuple_or_none(
        _ledger_value(snapshot, "volume", "main", "contrast_limits"),
        float,
        fallback=defaults.get("clim"),
    )
    volume_opacity = _float_or_none(
        _ledger_value(snapshot, "volume", "main", "opacity"),
        fallback=defaults.get("opacity"),
    )
    volume_sample_step = _float_or_none(
        _ledger_value(snapshot, "volume", "main", "sample_step"),
        fallback=defaults.get("sample_step"),
    )

    layer_raw_values: dict[str, dict[str, Any]] = {}
    layer_versions_raw: dict[str, dict[str, int]] = {}
    for (scope, target, key), entry in snapshot.items():
        if scope != "layer":
            continue
        layer_id = str(target)
        prop = str(key)
        values = layer_raw_values.setdefault(layer_id, {})
        values[prop] = entry.value
        version_value = _version_or_none(entry.version)
        if version_value is not None:
            versions = layer_versions_raw.setdefault(layer_id, {})
            versions[prop] = version_value

    layer_states: dict[str, LayerVisualState] = {}
    for layer_id, props in layer_raw_values.items():
        raw_props = {str(k): v for k, v in props.items()}
        versions_map_raw = layer_versions_raw.get(layer_id, {})
        versions_map = {str(k): int(v) for k, v in versions_map_raw.items()}

        visible = _coerce_bool(raw_props.pop("visible", None))
        opacity = _coerce_float(raw_props.pop("opacity", None))
        blending = raw_props.pop("blending", None)
        blending = str(blending) if blending is not None else None
        interpolation = raw_props.pop("interpolation", None)
        interpolation = str(interpolation) if interpolation is not None else None
        colormap_value = raw_props.pop("colormap", None)
        colormap = str(colormap_value) if colormap_value is not None else None
        gamma = _coerce_float(raw_props.pop("gamma", None))
        contrast_limits = _coerce_contrast_limits(raw_props.pop("contrast_limits", None))
        depiction = raw_props.pop("depiction", None)
        depiction = str(depiction) if depiction is not None else None
        rendering = raw_props.pop("rendering", None)
        rendering = str(rendering) if rendering is not None else None
        attenuation = _coerce_float(raw_props.pop("attenuation", None))
        iso_threshold = _coerce_float(raw_props.pop("iso_threshold", None))

        metadata_value = raw_props.pop("metadata", None)
        metadata: dict[str, Any] = {}
        if isinstance(metadata_value, Mapping):
            metadata = {str(k): v for k, v in metadata_value.items()}

        thumbnail_value = raw_props.pop("thumbnail", None)
        thumbnail_payload: Optional[dict[str, Any]] = None
        if isinstance(thumbnail_value, Mapping):
            thumbnail_payload = {str(k): v for k, v in thumbnail_value.items()}

        extra = {str(k): v for k, v in raw_props.items()}

        layer_states[layer_id] = LayerVisualState(
            layer_id=str(layer_id),
            visible=visible,
            opacity=opacity,
            blending=blending,
            interpolation=interpolation,
            colormap=colormap,
            gamma=gamma,
            contrast_limits=contrast_limits,
            depiction=depiction,
            rendering=rendering,
            attenuation=attenuation,
            iso_threshold=iso_threshold,
            metadata=metadata,
            thumbnail=thumbnail_payload,
            extra=extra,
            versions=versions_map,
        )

    camera_versions: dict[str, int] = {}
    for scope_name, prefix in (("camera_plane", "plane"), ("camera_volume", "volume")):
        for camera_key in ("center", "zoom", "angles", "distance", "fov", "rect"):
            entry = snapshot.get((scope_name, "main", camera_key))
            if entry is None:
                continue
            version_value = _version_or_none(entry.version)
            if version_value is not None:
                camera_versions[f"{prefix}.{camera_key}"] = int(version_value)
    camera_versions_payload = camera_versions or None

    op_seq_entry = snapshot.get(("scene", "main", "op_seq"))
    op_seq_value = int(op_seq_entry.value) if op_seq_entry is not None and op_seq_entry.value is not None else 0

    return RenderLedgerSnapshot(
        plane_center=plane_center_tuple if plane_center_tuple is not None else None,
        plane_zoom=plane_zoom_float,
        plane_rect=plane_rect_tuple if plane_rect_tuple is not None else None,
        volume_center=volume_center_tuple if volume_center_tuple is not None else None,
        volume_angles=volume_angles_tuple if volume_angles_tuple is not None else None,
        volume_distance=volume_distance_float,
        volume_fov=volume_fov_float,
        current_step=current_step,
        ndisplay=ndisplay_val,
        view_version=_version_or_none(view_entry.version) if view_entry is not None else None,
        displayed=displayed_axes,
        order=order_axes,
        axis_labels=axis_labels,
        dims_labels=dims_labels,
        level_shapes=level_shapes,
        current_level=current_level,
        dims_mode=dims_mode,
        dims_version=dims_version,
        multiscale_level_version=multiscale_level_version,
        volume_mode=volume_mode,
        volume_colormap=volume_colormap,
        volume_clim=volume_clim,
        volume_opacity=volume_opacity,
        volume_sample_step=volume_sample_step,
        layer_values=layer_states or None,
        camera_versions=camera_versions_payload,
        op_seq=op_seq_value,
        axes_spec=axes_spec,
    )


def _ledger_value(
    snapshot: dict[tuple[str, str, str], LedgerEntry],
    scope: str,
    target: str,
    key: str,
) -> Any:
    entry = snapshot.get((scope, target, key))
    return None if entry is None else entry.value


def _tuple_or_none(value: Any, mapper, *, fallback: Any = None) -> Optional[tuple]:
    source = value if value is not None else fallback
    if source is None:
        return None
    assert isinstance(source, Iterable), "expected iterable for tuple conversion"
    assert not isinstance(source, (str, bytes)), "string-like values not supported for tuple conversion"
    return tuple(mapper(v) for v in source)


def _float_or_none(value: Any, *, fallback: Any = None) -> Optional[float]:
    source = value if value is not None else fallback
    if source is None:
        return None
    return float(source)


def _string_or_none(value: Any, *, fallback: Any = None) -> Optional[str]:
    source = value if value is not None else fallback
    if source is None:
        return None
    text = str(source).strip()
    return text if text else None


def _int_or_none(value: Any, *, fallback: Any = None) -> Optional[int]:
    source = value if value is not None else fallback
    if source is None:
        return None
    return int(source)


def _shape_sequence(value: Any) -> Optional[tuple[tuple[int, ...], ...]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raise TypeError("shape sequence must be iterable, not mapping")
    assert isinstance(value, Iterable), "shape sequence must be iterable"
    result: list[tuple[int, ...]] = []
    for entry in value:
        shape_tuple = _tuple_or_none(entry, int)
        assert shape_tuple is not None, "shape entries must be iterable"
        result.append(shape_tuple)
    return tuple(result)


def _version_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def pull_render_snapshot(server: Any) -> RenderLedgerSnapshot:
    """Return the current render snapshot."""

    with server._state_lock:
        snapshot = build_ledger_snapshot(server._state_ledger)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "render snapshot pulled step=%s ndisplay=%s",
            snapshot.current_step,
            snapshot.ndisplay,
        )
    return snapshot
