"""Scene and dims composition helpers (protocol dataclasses).

Pure functions for building scene metadata using the protocol dataclasses in
``napari_cuda.protocol.messages``. Callers can delegate to these helpers to
avoid duplicating serialization logic.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from napari_cuda.protocol.messages import (
    LayerSpec,
    DimsSpec,
    CameraSpec,
    SceneSpec,
    SceneSpecMessage,
    MultiscaleSpec,
    MultiscaleLevelSpec,
    LayerRenderHints,
)


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
) -> LayerSpec:
    """Build a LayerSpec dataclass from simple inputs.

    Notes
    - Uses a single default layer id/name/type to match current server model.
    - If ``multiscale_levels`` provided, constructs a MultiscaleSpec.
    - ``extras`` is merged with ``{"is_volume": is_volume, "zarr_path": zarr_path}``.
    """
    ndim = int(len(tuple(shape)))
    base_extras: Dict[str, Any] = {"is_volume": bool(is_volume)}
    if zarr_path:
        base_extras["zarr_path"] = zarr_path
    if extras:
        base_extras.update({k: v for k, v in extras.items() if v is not None})

    ms: Optional[MultiscaleSpec] = None
    if multiscale_levels:
        levels: List[MultiscaleLevelSpec] = []
        for entry in multiscale_levels:
            if not isinstance(entry, dict):
                continue
            levels.append(
                MultiscaleLevelSpec(
                    shape=[int(x) for x in (entry.get("shape") or list(shape))],
                    downsample=[float(x) for x in (entry.get("downsample") or [1.0] * ndim)],
                    path=entry.get("path"),
                )
            )
        if levels:
            ms = MultiscaleSpec(levels=levels, current_level=int(multiscale_current_level or 0))

    # Map optional render hints directly; allow both dataclass and dict
    rh: Optional[LayerRenderHints]
    if isinstance(render_hints, dict):
        rh = LayerRenderHints.from_dict(render_hints)
    else:
        rh = render_hints

    return LayerSpec(
        layer_id="layer-0",
        layer_type="image",
        name="napari-cuda",
        ndim=ndim,
        shape=[int(x) for x in shape],
        dtype=None,
        axis_labels=None,
        render=rh,
        multiscale=ms,
        extras=base_extras or None,
        metadata=metadata or None,
    )


def build_dims_spec(
    *,
    layer_spec: Dict[str, Any],
    current_step: Optional[Iterable[int]],
    ndisplay: Optional[int],
    axis_labels: Optional[Sequence[str]] = None,
) -> DimsSpec:
    """Build a DimsSpec from a layer spec and provided dims state."""
    # Accept dict or dataclass
    if isinstance(layer_spec, dict):
        try:
            layer = LayerSpec.from_dict(layer_spec)  # type: ignore[attr-defined]
        except Exception:
            # Fallback minimal coercion
            shape = [int(x) for x in (layer_spec.get("shape") or [])]
            layer = LayerSpec(
                layer_id=str(layer_spec.get("layer_id", "layer-0")),
                layer_type=str(layer_spec.get("layer_type", "image")),
                name=str(layer_spec.get("name", "napari-cuda")),
                ndim=int(layer_spec.get("ndim", len(shape) or 2)),
                shape=shape or [0, 0],
            )
    else:
        layer = layer_spec

    ndim = int(layer.ndim or len(layer.shape))
    labels = list(axis_labels) if axis_labels is not None else (
        ["z", "y", "x"] if ndim == 3 else ["y", "x"] if ndim == 2 else [f"d{i}" for i in range(ndim)]
    )
    sizes = [int(x) for x in (layer.shape or [0] * ndim)]
    ranges = [[0, max(0, s - 1)] for s in sizes]
    try:
        current = list(current_step) if current_step is not None else None
    except Exception:
        current = None
    try:
        ndisp = int(ndisplay) if ndisplay is not None else min(2, ndim)
    except Exception:
        ndisp = min(2, ndim)

    # Pick displayed indices: last ndisp dims
    displayed = list(range(max(0, ndim - ndisp), ndim))

    return DimsSpec(
        ndim=ndim,
        axis_labels=labels,
        order=labels,
        sizes=sizes,
        range=ranges,
        current_step=current,
        displayed=displayed,
        ndisplay=ndisp,
    )


def build_camera_spec(
    *,
    center: Optional[Tuple[float, float, float]],
    zoom: Optional[float],
    angles: Optional[Tuple[float, float, float]],
    ndisplay: Optional[int],
) -> Optional[CameraSpec]:
    """Build a CameraSpec from provided fields; returns None if all are None."""
    if center is None and zoom is None and angles is None and ndisplay is None:
        return None
    center_list = list(center) if center is not None else None
    angles_list = list(angles) if angles is not None else None
    return CameraSpec(
        center=center_list,
        zoom=float(zoom) if zoom is not None else None,
        angles=angles_list,
        ndisplay=int(ndisplay) if ndisplay is not None else None,
    )


def scene_message(
    *,
    layers: List[Dict[str, Any]],
    dims: Dict[str, Any],
    camera: Dict[str, Any],
    capabilities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> SceneSpecMessage:
    """Build a SceneSpecMessage suitable for the state channel."""
    # Accept dicts or dataclasses
    built_layers: List[LayerSpec] = []
    for entry in layers:
        if isinstance(entry, LayerSpec):
            built_layers.append(entry)
        elif isinstance(entry, dict):
            built_layers.append(LayerSpec.from_dict(entry))  # type: ignore[attr-defined]
    dims_spec = dims if isinstance(dims, DimsSpec) else DimsSpec.from_dict(dims)  # type: ignore[attr-defined]
    cam_spec = camera if (camera is None or isinstance(camera, CameraSpec)) else CameraSpec.from_dict(camera)  # type: ignore[attr-defined]
    scene = SceneSpec(layers=built_layers, dims=dims_spec, camera=cam_spec, capabilities=list(capabilities or []), metadata=metadata)
    return SceneSpecMessage(timestamp=timestamp, scene=scene)


def dims_metadata(scene: Optional[Union[SceneSpec, Dict[str, Any]]]) -> Dict[str, Any]:
    """Extract dims metadata for HUD and downstream intents.

    Mirrors ViewerSceneManager.dims_metadata: returns a dict containing
    ndim, order, sizes, range, axis_labels, and derived layer extras including
    volume flag, render hints, and multiscale info (with policy/index_space).
    """
    if scene is None:
        return {}

    # Normalize to dict view
    if isinstance(scene, SceneSpec):
        scene_dict = scene.to_dict()
    else:
        scene_dict = scene

    try:
        dims = dict(scene_dict.get("dims") or {})
    except Exception:
        dims = {}
    meta: Dict[str, Any] = {
        "ndim": dims.get("ndim"),
        "order": dims.get("order"),
        "sizes": dims.get("sizes"),
        "range": dims.get("range"),
        "axis_labels": dims.get("axis_labels"),
    }

    layers = scene_dict.get("layers") or []
    layer0 = layers[0] if layers else None
    if isinstance(layer0, dict):
        extras = dict(layer0.get("extras") or {})
        if "is_volume" in extras:
            meta["volume"] = bool(extras.get("is_volume"))
        render = layer0.get("render")
        if render is not None:
            meta["render"] = render if isinstance(render, dict) else asdict(render)
        ms = layer0.get("multiscale")
        if isinstance(ms, dict):
            ms_dict = dict(ms)
        elif ms is not None:
            # Dataclass case
            try:
                ms_dict = ms.to_dict()  # type: ignore[attr-defined]
            except Exception:
                ms_dict = {}
        else:
            ms_dict = {}
        if ms_dict:
            md = ms_dict.pop("metadata", None)
            if isinstance(md, dict):
                policy = md.get("policy")
                index_space = md.get("index_space")
                if policy is not None:
                    ms_dict["policy"] = policy
                if index_space is not None:
                    ms_dict["index_space"] = index_space
            meta["multiscale"] = ms_dict
        # Derive sizes/range from active level when available
        try:
            eff_shape = None
            if ms_dict and ms_dict.get("levels"):
                cur = int(ms_dict.get("current_level", 0))
                levels = ms_dict.get("levels", [])
                if 0 <= cur < len(levels):
                    shape = (levels[cur] or {}).get("shape")
                    if shape:
                        eff_shape = [int(x) for x in shape]
            if eff_shape is None:
                eff_shape = [int(x) for x in (layer0.get("shape") or [])]
            if eff_shape:
                meta["sizes"] = list(eff_shape)
                meta["range"] = [[0, max(0, int(s) - 1)] for s in eff_shape]
        except Exception:
            # Keep dims-derived sizes/range
            pass

    # Strip None
    return {k: v for k, v in meta.items() if v is not None}


__all__ = [
    "LayerRenderHints",
    "build_layer_spec",
    "build_dims_spec",
    "build_camera_spec",
    "scene_message",
    "dims_metadata",
]
