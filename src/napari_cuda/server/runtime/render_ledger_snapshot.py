"""Build immutable render snapshots from the server ledger."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from napari_cuda.server.control.state_ledger import ServerStateLedger, LedgerEntry

if TYPE_CHECKING:
    from napari_cuda.server.scene import ServerSceneData


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderLedgerSnapshot:
    """Authoritative scene state consumed by the render thread."""

    center: Optional[tuple[float, float, float]] = None
    zoom: Optional[float] = None
    angles: Optional[tuple[float, float, float]] = None
    distance: Optional[float] = None
    fov: Optional[float] = None
    rect: Optional[tuple[float, float, float, float]] = None
    current_step: Optional[tuple[int, ...]] = None
    dims_version: Optional[int] = None
    ndisplay: Optional[int] = None
    view_version: Optional[int] = None
    displayed: Optional[tuple[int, ...]] = None
    order: Optional[tuple[int, ...]] = None
    axis_labels: Optional[tuple[str, ...]] = None
    dims_labels: Optional[tuple[str, ...]] = None
    level_shapes: Optional[tuple[tuple[int, ...], ...]] = None
    current_level: Optional[int] = None
    multiscale_level_version: Optional[int] = None
    dims_mode: Optional[str] = None
    volume_mode: Optional[str] = None
    volume_colormap: Optional[str] = None
    volume_clim: Optional[tuple[float, float]] = None
    volume_opacity: Optional[float] = None
    volume_sample_step: Optional[float] = None
    layer_updates: Optional[Dict[str, Dict[str, Any]]] = None
    camera_versions: Optional[Dict[str, int]] = None


def build_ledger_snapshot(
    ledger: ServerStateLedger,
    scene: "ServerSceneData",
    *,
    layer_updates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    drain_pending_layers: bool = True,
) -> RenderLedgerSnapshot:
    """Construct the render snapshot by stitching together ledger + local state.

    Parameters
    ----------
    ledger
        ServerStateLedger containing confirmed property values.
    scene
        Mutable scene data bag holding local fallbacks (volume hints, pending layer deltas).
    layer_updates
        Optional explicit layer deltas to include instead of draining the pending store.
    drain_pending_layers
        When ``True`` (default) the helper will consume ``scene.pending_layer_updates``.
    """

    snapshot = ledger.snapshot()

    center_tuple = _tuple_or_none(_ledger_value(snapshot, "camera", "main", "center"), float)
    zoom_float = _float_or_none(_ledger_value(snapshot, "camera", "main", "zoom"))
    angles_tuple = _tuple_or_none(_ledger_value(snapshot, "camera", "main", "angles"), float)
    distance_float = _float_or_none(_ledger_value(snapshot, "camera", "main", "distance"))
    fov_float = _float_or_none(_ledger_value(snapshot, "camera", "main", "fov"))
    rect_tuple = _tuple_or_none(_ledger_value(snapshot, "camera", "main", "rect"), float)

    dims_entry = snapshot.get(("dims", "main", "current_step"))
    current_step = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "current_step"), int)
    dims_version = None if dims_entry is None else _version_or_none(dims_entry.version)
    pending_dims = getattr(scene, "pending_dims_step", None)
    if pending_dims is not None:
        current_step = tuple(int(v) for v in pending_dims)
        scene.pending_dims_step = None

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

    volume_mode = _string_or_none(
        _ledger_value(snapshot, "volume", "main", "render_mode"),
        fallback=scene.volume_state.get("mode"),
    )
    volume_colormap = _string_or_none(
        _ledger_value(snapshot, "volume", "main", "colormap"),
        fallback=scene.volume_state.get("colormap"),
    )
    volume_clim = _tuple_or_none(
        _ledger_value(snapshot, "volume", "main", "contrast_limits"),
        float,
        fallback=scene.volume_state.get("clim"),
    )
    volume_opacity = _float_or_none(
        _ledger_value(snapshot, "volume", "main", "opacity"),
        fallback=scene.volume_state.get("opacity"),
    )
    volume_sample_step = _float_or_none(
        _ledger_value(snapshot, "volume", "main", "sample_step"),
        fallback=scene.volume_state.get("sample_step"),
    )

    resolved_layers: Dict[str, Dict[str, Any]] = {}
    if layer_updates is not None:
        resolved_layers.update(_normalise_layer_updates(layer_updates))
    elif scene.pending_layer_updates:
        resolved_layers.update(_normalise_layer_updates(scene.pending_layer_updates))
        if drain_pending_layers:
            scene.pending_layer_updates.clear()

    camera_versions: Dict[str, int] = {}
    for camera_key in ("center", "zoom", "angles", "distance", "fov", "rect"):
        entry = snapshot.get(("camera", "main", camera_key))
        if entry is None:
            continue
        version_value = _version_or_none(entry.version)
        if version_value is not None:
            camera_versions[str(camera_key)] = int(version_value)
    camera_versions_payload = camera_versions or None

    return RenderLedgerSnapshot(
        center=center_tuple,
        zoom=zoom_float,
        angles=angles_tuple,
        current_step=current_step,
        distance=distance_float,
        fov=fov_float,
        rect=rect_tuple,
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
        layer_updates=resolved_layers or None,
        camera_versions=camera_versions_payload,
    )


def _ledger_value(
    snapshot: Dict[tuple[str, str, str], LedgerEntry],
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


def _normalise_layer_updates(updates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    normalised: Dict[str, Dict[str, Any]] = {}
    for layer_id, props in updates.items():
        entries: Dict[str, Any] = {}
        for key, value in props.items():
            entries[str(key)] = value
        if entries:
            normalised[str(layer_id)] = entries
    return normalised


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


__all__ = ["RenderLedgerSnapshot", "build_ledger_snapshot"]


def pull_render_snapshot(server: Any) -> RenderLedgerSnapshot:
    """Return the current render snapshot."""

    with server._state_lock:
        snapshot = build_ledger_snapshot(server._state_ledger, server._scene)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "render snapshot pulled step=%s ndisplay=%s",
            snapshot.current_step,
            snapshot.ndisplay,
        )
    return snapshot


__all__.append("pull_render_snapshot")
