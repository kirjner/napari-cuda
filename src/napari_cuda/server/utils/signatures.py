"""Canonical signature helpers for server dedupe flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Tuple

from napari_cuda.server.scene import LayerVisualState, RenderLedgerSnapshot

SignatureTuple = Tuple[Any, ...]
VersionKey = Tuple[str, str, str]


@dataclass(frozen=True)
class SignatureToken:
    """Immutable wrapper around signature tuples with change detection."""

    value: SignatureTuple

    def changed(self, previous: Optional["SignatureToken"]) -> bool:
        return previous is None or previous.value != self.value


@dataclass(frozen=True)
class VersionGate:
    """Represents the render-snapshot version counters we track."""

    entries: Tuple[Tuple[VersionKey, int], ...]

    def apply(self, mapping: MutableMapping[VersionKey, int]) -> None:
        for key, version in self.entries:
            mapping[key] = version


# ---------------------------------------------------------------------------
# Scene signatures


def scene_content_signature(
    snapshot: RenderLedgerSnapshot,
    *,
    dataset_id: Optional[str] = None,
) -> SignatureToken:
    """Signature describing the full scene content visible to the user."""

    token: SignatureTuple = (
        ("dataset", None if dataset_id is None else str(dataset_id)),
        ("dims", _dims_tuple_from_snapshot(snapshot)),
        ("view", _view_tuple_from_snapshot(snapshot)),
        ("layers", _layers_visuals_tuple(snapshot)),
        ("volume", _volume_settings_tuple(snapshot)),
    )
    return SignatureToken(token)


def layer_inputs_signature(
    snapshot: RenderLedgerSnapshot,
    layer_id: str,
    *,
    dataset_id: Optional[str] = None,
) -> SignatureToken:
    """Inputs-only signature for layer-specific worker operations (thumbnails)."""

    layer_values = snapshot.layer_values or {}
    state = layer_values.get(str(layer_id))
    visuals = _layer_visual_items(state, ndisplay=snapshot.ndisplay) if isinstance(state, LayerVisualState) else ()
    token: SignatureTuple = (
        ("layer", str(layer_id)),
        ("dataset", None if dataset_id is None else str(dataset_id)),
        ("dims", _dims_tuple_from_snapshot(snapshot)),
        ("view", _view_tuple_from_snapshot(snapshot)),
        ("visuals", visuals),
    )
    return SignatureToken(token)


# ---------------------------------------------------------------------------
# Layer signatures


def layer_content_signature(state: LayerVisualState) -> SignatureToken:
    """Signature of outbound layer payload values (notify.layers)."""

    keys = tuple(sorted(state.keys()))
    items = tuple((str(key), _canon(state.get(key))) for key in keys)
    return SignatureToken(items)


# ---------------------------------------------------------------------------
# Dims signatures


def dims_content_signature(
    *,
    current_step: Tuple[int, ...],
    current_level: int,
    ndisplay: int,
    mode: str,
    displayed: Optional[Tuple[int, ...]],
    axis_labels: Optional[Tuple[str, ...]],
    order: Optional[Tuple[int, ...]],
    labels: Optional[Tuple[str, ...]],
    levels: Tuple[Mapping[str, Any], ...],
    level_shapes: Tuple[Tuple[int, ...], ...],
    downgraded: Optional[bool],
) -> SignatureToken:
    """Signature describing the dims payload delivered to clients."""

    levels_sig = tuple(tuple(sorted((str(k), _canon(v)) for k, v in level.items())) for level in levels)
    token: SignatureTuple = (
        tuple(int(v) for v in current_step),
        int(current_level),
        int(ndisplay),
        str(mode),
        None if displayed is None else tuple(int(v) for v in displayed),
        None if axis_labels is None else tuple(str(v) for v in axis_labels),
        None if order is None else tuple(int(v) for v in order),
        None if labels is None else tuple(str(v) for v in labels),
        levels_sig,
        tuple(tuple(int(dim) for dim in shape) for shape in level_shapes),
        None if downgraded is None else bool(downgraded),
    )
    return SignatureToken(token)


def dims_content_signature_from_payload(payload: Any) -> SignatureToken:
    """Convenience wrapper that derives the signature from a NotifyDims payload."""

    spec = getattr(payload, "dims_spec", None)
    assert spec is not None, "notify.dims missing dims spec"

    return dims_content_signature(
        current_step=tuple(int(v) for v in spec.current_step),
        current_level=int(spec.current_level),
        ndisplay=int(spec.ndisplay),
        mode="plane" if spec.plane_mode else "volume",
        displayed=tuple(int(v) for v in spec.displayed),
        axis_labels=tuple(axis.label for axis in spec.axes),
        order=tuple(int(v) for v in spec.order),
        labels=None if payload.labels is None else tuple(str(v) for v in payload.labels),
        levels=tuple(dict(level) for level in payload.levels),
        level_shapes=tuple(tuple(int(dim) for dim in shape) for shape in spec.level_shapes),
        downgraded=payload.downgraded,
    )


# ---------------------------------------------------------------------------
# Version gates


def snapshot_versions(snapshot: RenderLedgerSnapshot) -> VersionGate:
    """Collect the ledger versions embedded in the render snapshot."""

    entries: list[Tuple[VersionKey, int]] = []
    if snapshot.dims_version is not None:
        entries.append((("dims", "main", "current_step"), int(snapshot.dims_version)))
    if snapshot.view_version is not None:
        entries.append((("view", "main", "ndisplay"), int(snapshot.view_version)))
    if snapshot.multiscale_level_version is not None:
        entries.append((("multiscale", "main", "level"), int(snapshot.multiscale_level_version)))
    if snapshot.camera_versions:
        for attr, version in snapshot.camera_versions.items():
            scope, key = _split_camera_attr(str(attr))
            entries.append(((scope, "main", key), int(version)))
    return VersionGate(tuple(entries))


# ---------------------------------------------------------------------------
# Internal helpers


def _split_camera_attr(attr: str) -> Tuple[str, str]:
    if "." not in attr:
        return "camera", attr
    prefix, remainder = attr.split(".", 1)
    if prefix == "plane":
        return "camera_plane", remainder
    if prefix == "volume":
        return "camera_volume", remainder
    if prefix == "legacy":
        return "camera", remainder
    return "camera", attr


def _dims_tuple_from_snapshot(snapshot: RenderLedgerSnapshot) -> SignatureTuple:
    spec = snapshot.dims_spec
    assert spec is not None, "render snapshot missing dims spec"

    return (
        int(spec.ndisplay),
        tuple(int(v) for v in spec.order),
        tuple(int(v) for v in spec.displayed),
        tuple(int(v) for v in spec.current_step),
        int(spec.current_level),
        tuple(axis.label for axis in spec.axes),
        "plane" if spec.plane_mode else "volume",
    )


def _view_tuple_from_snapshot(snapshot: RenderLedgerSnapshot) -> SignatureTuple:
    return (
        _plane_pose_tuple(snapshot),
        _volume_pose_tuple(snapshot),
    )


def _layers_visuals_tuple(snapshot: RenderLedgerSnapshot) -> SignatureTuple:
    result: list[Tuple[str, SignatureTuple]] = []
    layer_values = snapshot.layer_values or {}
    for layer_id, state in sorted(layer_values.items()):
        if isinstance(state, LayerVisualState):
            items = _layer_visual_items(state, ndisplay=snapshot.ndisplay)
            result.append((str(layer_id), items))
    return tuple(result)


def _volume_settings_tuple(snapshot: RenderLedgerSnapshot) -> SignatureTuple:
    return (
        _canon(snapshot.volume_mode),
        _canon(snapshot.volume_colormap),
        _canon(snapshot.volume_clim),
        _canon(snapshot.volume_opacity),
        _canon(snapshot.volume_sample_step),
    )


def _plane_pose_tuple(snapshot: RenderLedgerSnapshot) -> SignatureTuple:
    center = None
    if snapshot.plane_center is not None and len(snapshot.plane_center) >= 2:
        center = (_round_float(snapshot.plane_center[0]), _round_float(snapshot.plane_center[1]))
    zoom = _round_float(snapshot.plane_zoom)
    rect = None
    if snapshot.plane_rect is not None and len(snapshot.plane_rect) >= 4:
        rect = tuple(_round_float(component) for component in snapshot.plane_rect[:4])
    return (
        center,
        zoom,
        rect,
    )


def _volume_pose_tuple(snapshot: RenderLedgerSnapshot) -> SignatureTuple:
    center = None
    if snapshot.volume_center is not None and len(snapshot.volume_center) >= 3:
        center = tuple(_round_float(component) for component in snapshot.volume_center[:3])
    angles = None
    if snapshot.volume_angles is not None and len(snapshot.volume_angles) >= 2:
        az = _round_float(snapshot.volume_angles[0])
        el = _round_float(snapshot.volume_angles[1])
        roll = _round_float(snapshot.volume_angles[2]) if len(snapshot.volume_angles) >= 3 else _round_float(0.0)
        angles = (az, el, roll)
    distance = _round_float(snapshot.volume_distance)
    fov = _round_float(snapshot.volume_fov)
    return (
        center,
        angles,
        distance,
        fov,
    )


def _layer_visual_items(
    state: Optional[LayerVisualState],
    *,
    ndisplay: Optional[int],
) -> SignatureTuple:
    if state is None:
        return ()
    items: list[Tuple[str, Any]] = []
    for key in sorted(state.keys()):
        if key == "thumbnail":
            continue
        if key == "metadata":
            continue
        if ndisplay is not None and int(ndisplay) < 3 and key.startswith("volume."):
            continue
        items.append((str(key), _normalize_visual_value(state.get(key))))
    return tuple(items)


def _round_float(value: Optional[float], places: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), places)


def _canon(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _canon(v)) for k, v in value.items()))
    return value


def _normalize_visual_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_visual_value(v) for v in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _normalize_visual_value(v)) for k, v in value.items()))
    return value


__all__ = [
    "SignatureToken",
    "VersionGate",
    "scene_content_signature",
    "layer_inputs_signature",
    "layer_content_signature",
    "dims_content_signature",
    "dims_content_signature_from_payload",
    "snapshot_versions",
]
