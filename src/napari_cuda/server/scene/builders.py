"""Server-side helpers to build scene snapshots from the control ledger."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import (
    Any,
    Mapping,
    Optional,
)

import numpy as np

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
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.shared.dims_spec import (
    dims_block_from_spec,
    dims_spec_from_payload,
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

    spec_entry = snapshot.get(("dims", "main", "dims_spec"))
    assert spec_entry is not None and spec_entry.value is not None, "dims_spec missing from snapshot"
    spec = dims_spec_from_payload(spec_entry.value)
    assert spec is not None, "dims_spec payload missing"
    state["current_level"] = int(spec.current_level)
    state["levels"] = [dict(level) for level in spec.levels]
    state["level_shapes"] = [
        [int(dim) for dim in shape] for shape in spec.level_shapes
    ]
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
        default_clim = tuple(default_clim_value)
    default_opacity = volume_defaults.get("opacity")
    default_sample_step = volume_defaults.get("sample_step")

    resolved_plane_center = plane_center if plane_center is not None else base.plane_center
    resolved_plane_zoom = plane_zoom if plane_zoom is not None else base.plane_zoom
    resolved_plane_rect = plane_rect if plane_rect is not None else base.plane_rect
    resolved_volume_center = volume_center if volume_center is not None else base.volume_center
    resolved_volume_angles = volume_angles if volume_angles is not None else base.volume_angles
    resolved_volume_distance = volume_distance if volume_distance is not None else base.volume_distance
    resolved_volume_fov = volume_fov if volume_fov is not None else base.volume_fov
    resolved_step = current_step if current_step is not None else base.current_step

    resolved_volume_mode = volume_mode
    if resolved_volume_mode is None:
        resolved_volume_mode = base.volume_mode if base.volume_mode is not None else default_mode

    resolved_volume_colormap = volume_colormap
    if resolved_volume_colormap is None:
        if base.volume_colormap is not None:
            resolved_volume_colormap = base.volume_colormap
        else:
            resolved_volume_colormap = default_colormap

    resolved_volume_clim = volume_clim
    if resolved_volume_clim is None:
        if base.volume_clim is not None:
            resolved_volume_clim = base.volume_clim
        else:
            resolved_volume_clim = default_clim

    resolved_volume_opacity = volume_opacity
    if resolved_volume_opacity is None:
        resolved_volume_opacity = base.volume_opacity if base.volume_opacity is not None else default_opacity

    resolved_volume_sample_step = volume_sample_step
    if resolved_volume_sample_step is None:
        if base.volume_sample_step is not None:
            resolved_volume_sample_step = base.volume_sample_step
        else:
            resolved_volume_sample_step = default_sample_step

    return replace(
        base,
        plane_center=resolved_plane_center,
        plane_zoom=resolved_plane_zoom,
        plane_rect=resolved_plane_rect,
        volume_center=resolved_volume_center,
        volume_angles=resolved_volume_angles,
        volume_distance=resolved_volume_distance,
        volume_fov=resolved_volume_fov,
        current_step=resolved_step,
        volume_mode=resolved_volume_mode,
        volume_colormap=resolved_volume_colormap,
        volume_clim=resolved_volume_clim,
        volume_opacity=resolved_volume_opacity,
        volume_sample_step=resolved_volume_sample_step,
    )


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

    payload: dict[str, Any] = {}

    spec_entry = snapshot.get(("dims", "main", "dims_spec"))
    if spec_entry is not None and spec_entry.value is not None:
        spec = dims_spec_from_payload(spec_entry.value)
        if spec is not None:
            payload["mode"] = RenderMode.VOLUME.name if int(spec.ndisplay) >= 3 else RenderMode.PLANE.name

    plane_entry = snapshot.get(("viewport", "plane", "state"))
    if plane_entry is not None and plane_entry.value is not None:
        payload["plane"] = plane_entry.value

    volume_entry = snapshot.get(("viewport", "volume", "state"))
    if volume_entry is not None and volume_entry.value is not None:
        payload["volume"] = volume_entry.value

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

    requested_ndisplay = ndisplay if ndisplay is not None else render_state.ndisplay

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

    spec = render_state.dims_spec
    if spec is None:
        spec_entry = ledger_snapshot.get(("dims", "main", "dims_spec"))
        assert spec_entry is not None, "ledger missing dims_spec entry"
        spec = dims_spec_from_payload(spec_entry.value)
        assert spec is not None, "dims spec payload missing"

    ndisplay_value = int(requested_ndisplay if requested_ndisplay is not None else spec.ndisplay or 2)
    ndisplay_value = max(1, ndisplay_value)

    dims_block = dims_block_from_spec(spec)

    camera_block = _build_camera_block(render_state, is_volume=ndisplay_value >= 3, ndisplay_value=ndisplay_value)

    viewer_settings = {
        "fps_target": float(fps_target),
        "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
        "volume_enabled": bool(ndisplay_value >= 3),
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

        current_level = int(spec.current_level)
        level_shapes = list(spec.level_shapes)
        if 0 <= current_level < len(level_shapes):
            layer_shape = [int(dim) for dim in level_shapes[current_level]]
        else:
            ndim = int(spec.ndim or 0) or 1
            layer_shape = [0 for _ in range(ndim)]

        axis_labels = [axis.label for axis in spec.axes]
        is_volume = bool(ndisplay_value >= 3)

        layer_block: dict[str, Any] = {
            "layer_id": layer_id_str,
            "layer_type": "image",
            "name": default_layer_name,
            "ndim": len(layer_shape),
            "shape": layer_shape,
            "axis_labels": axis_labels,
            "volume": is_volume,
            "controls": controls_block,
            "level_shapes": [list(shape) for shape in level_shapes],
        }

        layer_metadata: dict[str, Any] = {}
        if thumbnail_provider is not None:
            raw_thumbnail = thumbnail_provider(layer_id_str)
            normalized_thumb = _normalize_thumbnail_array(raw_thumbnail)
            if normalized_thumb is not None:
                layer_metadata["thumbnail"] = normalized_thumb.tolist()

        layer_block["metadata"] = layer_metadata

        multiscale_block = _build_multiscale_block(multiscale_state, base_shape=layer_shape)
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

    return {
        "mode": mode,
        "colormap": colormap,
        "clim": clim,
        "opacity": opacity,
        "sample_step": sample_step,
    }


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
        return list(clim[:2])
    return None


def _build_camera_block(
    render_state: RenderLedgerSnapshot,
    *,
    is_volume: bool,
    ndisplay_value: int,
) -> dict[str, Any]:
    block: dict[str, Any] = {"ndisplay": int(ndisplay_value)}

    if is_volume and render_state.volume_center is not None:
        block["center"] = list(render_state.volume_center)
        if render_state.volume_angles is not None:
            block["angles"] = list(render_state.volume_angles)
        if render_state.volume_distance is not None:
            block["distance"] = render_state.volume_distance
        if render_state.volume_fov is not None:
            block["fov"] = render_state.volume_fov
        return block

    if render_state.plane_center is not None:
        block["center"] = list(render_state.plane_center)
    if render_state.plane_zoom is not None:
        block["zoom"] = render_state.plane_zoom
    if render_state.plane_rect is not None:
        block["rect"] = list(render_state.plane_rect)

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


# ---------------------------------------------------------------------------
# Render ledger snapshots


def build_ledger_snapshot(
    ledger: ServerStateLedger,
    snapshot: dict[tuple[str, str, str], LedgerEntry] | None = None,
) -> RenderLedgerSnapshot:
    """Construct a render snapshot from confirmed ledger state."""

    snapshot = snapshot if snapshot is not None else ledger.snapshot()
    spec_entry = snapshot.get(("dims", "main", "dims_spec"))
    if spec_entry is None:
        raise AssertionError("ledger missing dims_spec entry for render snapshot")
    assert isinstance(spec_entry, LedgerEntry), "ledger dims_spec entry malformed"
    dims_spec = dims_spec_from_payload(spec_entry.value)
    assert dims_spec is not None, "dims spec ledger entry missing payload"
    validate_ledger_against_dims_spec(dims_spec, snapshot)

    plane_center_tuple = _ledger_value(snapshot, "camera_plane", "main", "center")
    plane_zoom_float = _ledger_value(snapshot, "camera_plane", "main", "zoom")
    plane_rect_tuple = _ledger_value(snapshot, "camera_plane", "main", "rect")

    volume_center_tuple = _ledger_value(snapshot, "camera_volume", "main", "center")
    volume_angles_tuple = _ledger_value(snapshot, "camera_volume", "main", "angles")
    volume_distance_float = _ledger_value(snapshot, "camera_volume", "main", "distance")
    volume_fov_float = _ledger_value(snapshot, "camera_volume", "main", "fov")

    current_step = tuple(int(v) for v in dims_spec.current_step)
    dims_version = _version_or_none(spec_entry.version)

    ndisplay_val = int(dims_spec.ndisplay)
    dims_mode = "volume" if ndisplay_val >= 3 else "plane"
    displayed_axes = tuple(int(idx) for idx in dims_spec.displayed)
    order_axes = tuple(int(idx) for idx in dims_spec.order)
    axis_labels = tuple(axis.label for axis in dims_spec.axes)
    dims_labels = tuple(str(lbl) for lbl in dims_spec.labels) if dims_spec.labels is not None else None
    level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in dims_spec.level_shapes)
    current_level = int(dims_spec.current_level)
    multiscale_level_version = dims_version

    volume_mode = _ledger_value(snapshot, "volume", "main", "rendering")
    volume_colormap = _ledger_value(snapshot, "volume", "main", "colormap")
    volume_clim = _ledger_value(snapshot, "volume", "main", "contrast_limits")
    volume_opacity = _ledger_value(snapshot, "volume", "main", "opacity")
    volume_sample_step = _ledger_value(snapshot, "volume", "main", "sample_step")

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

        visible = raw_props.pop("visible", None)
        opacity = raw_props.pop("opacity", None)
        blending = raw_props.pop("blending", None)
        interpolation = raw_props.pop("interpolation", None)
        colormap = raw_props.pop("colormap", None)
        gamma = raw_props.pop("gamma", None)
        contrast_limits = raw_props.pop("contrast_limits", None)
        depiction = raw_props.pop("depiction", None)
        rendering = raw_props.pop("rendering", None)
        attenuation = raw_props.pop("attenuation", None)
        iso_threshold = raw_props.pop("iso_threshold", None)

        metadata_value = raw_props.pop("metadata", None)
        metadata: Mapping[str, Any] = metadata_value if isinstance(metadata_value, Mapping) else {}

        thumbnail_value = raw_props.pop("thumbnail", None)
        thumbnail_payload: Mapping[str, Any] | None = thumbnail_value if isinstance(thumbnail_value, Mapping) else None

        extra = {str(k): v for k, v in raw_props.items()}

        layer_states[layer_id] = LayerVisualState(
            layer_id=str(layer_id),
            visible=visible,
            opacity=opacity,
            blending=blending,
            interpolation=interpolation,
            colormap=colormap,
            gamma=gamma,
            contrast_limits=tuple(contrast_limits) if isinstance(contrast_limits, Sequence) else contrast_limits,
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
        view_version=dims_version,
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
        dims_spec=dims_spec,
    )


def _ledger_value(
    snapshot: dict[tuple[str, str, str], LedgerEntry],
    scope: str,
    target: str,
    key: str,
) -> Any:
    entry = snapshot.get((scope, target, key))
    return None if entry is None else entry.value


def _shape_sequence(value: Any) -> Optional[tuple[tuple[int, ...], ...]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raise TypeError("shape sequence must be iterable, not mapping")
    assert isinstance(value, Iterable), "shape sequence must be iterable"
    result: list[tuple[int, ...]] = []
    for entry in value:
        assert isinstance(entry, Iterable), "shape entries must be iterable"
        assert not isinstance(entry, (str, bytes)), "shape entries may not be strings"
        result.append(tuple(int(dim) for dim in entry))
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
