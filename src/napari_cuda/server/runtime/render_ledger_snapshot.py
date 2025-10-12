"""Build immutable render snapshots from the server ledger."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from napari_cuda.server.control.state_ledger import ServerStateLedger, LedgerEntry
from napari_cuda.server.control.latest_intent import get_all as latest_get_all

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
    ndisplay: Optional[int] = None
    displayed: Optional[tuple[int, ...]] = None
    order: Optional[tuple[int, ...]] = None
    axis_labels: Optional[tuple[str, ...]] = None
    dims_labels: Optional[tuple[str, ...]] = None
    level_shapes: Optional[tuple[tuple[int, ...], ...]] = None
    current_level: Optional[int] = None
    dims_mode: Optional[str] = None
    volume_mode: Optional[str] = None
    volume_colormap: Optional[str] = None
    volume_clim: Optional[tuple[float, float]] = None
    volume_opacity: Optional[float] = None
    volume_sample_step: Optional[float] = None
    layer_updates: Optional[Dict[str, Dict[str, Any]]] = None


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

    current_step = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "current_step"), int)
    pending_dims = getattr(scene, "pending_dims_step", None)
    if pending_dims is not None:
        current_step = tuple(int(v) for v in pending_dims)
        scene.pending_dims_step = None

    dims_mode = _string_or_none(_ledger_value(snapshot, "dims", "main", "mode"))
    ndisplay_val = _int_or_none(_ledger_value(snapshot, "view", "main", "ndisplay"))
    displayed_axes = _tuple_or_none(_ledger_value(snapshot, "view", "main", "displayed"), int)
    order_axes = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "order"), int)
    axis_labels = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "axis_labels"), str)
    dims_labels = _tuple_or_none(_ledger_value(snapshot, "dims", "main", "labels"), str)
    level_shapes = _shape_sequence(_ledger_value(snapshot, "multiscale", "main", "level_shapes"))
    current_level = _int_or_none(_ledger_value(snapshot, "multiscale", "main", "level"))

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

    return RenderLedgerSnapshot(
        center=center_tuple,
        zoom=zoom_float,
        angles=angles_tuple,
        current_step=current_step,
        distance=distance_float,
        fov=fov_float,
        rect=rect_tuple,
        ndisplay=ndisplay_val,
        displayed=displayed_axes,
        order=order_axes,
        axis_labels=axis_labels,
        dims_labels=dims_labels,
        level_shapes=level_shapes,
        current_level=current_level,
        dims_mode=dims_mode,
        volume_mode=volume_mode,
        volume_colormap=volume_colormap,
        volume_clim=volume_clim,
        volume_opacity=volume_opacity,
        volume_sample_step=volume_sample_step,
        layer_updates=resolved_layers or None,
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


__all__ = ["RenderLedgerSnapshot", "build_ledger_snapshot"]


def pull_render_snapshot(
    server: Any,
) -> tuple[RenderLedgerSnapshot, Dict[str, int]]:
    """Return the current render snapshot plus latest intent metadata."""

    with server._state_lock:
        snapshot = build_ledger_snapshot(server._state_ledger, server._scene)

        desired_dims_seq = -1
        latest_dims = latest_get_all("dims")
        for _key, (seq, value) in latest_dims.items():
            seq_int = int(seq)
            if seq_int >= desired_dims_seq and isinstance(value, (list, tuple)):
                desired_dims_seq = seq_int

        desired_view_seq = -1
        latest_view = latest_get_all("view")
        for _key, (seq, value) in latest_view.items():
            seq_int = int(seq)
            # Retain latest sequence for applied comparison even if worker no longer consumes desired ndisplay.
            if seq_int >= desired_view_seq:
                desired_view_seq = seq_int

    desired_seqs = {"dims": int(desired_dims_seq), "view": int(desired_view_seq)}
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "render snapshot pulled step=%s ndisplay=%s desired_seqs=%s",
            snapshot.current_step,
            snapshot.ndisplay,
            desired_seqs,
        )
    return snapshot, desired_seqs


__all__.append("pull_render_snapshot")
