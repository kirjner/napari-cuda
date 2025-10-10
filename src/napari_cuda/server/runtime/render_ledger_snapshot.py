"""Build immutable render snapshots from the server ledger."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


def _clamp_step(
    step: Sequence[int],
    level_shapes: Sequence[Sequence[int]] | None,
    current_level: Optional[int],
) -> Tuple[int, ...]:
    values = [int(v) for v in step]
    if level_shapes is None or current_level is None:
        return tuple(values)
    li = int(current_level)
    if li < 0 or li >= len(level_shapes):
        return tuple(values)
    shape = level_shapes[li]
    for i in range(min(len(values), len(shape))):
        size = int(shape[i])
        if size <= 0:
            continue
        if values[i] < 0:
            values[i] = 0
        elif values[i] >= size:
            values[i] = size - 1
    return tuple(values)


def pull_render_snapshot(
    server: Any,
) -> tuple[RenderLedgerSnapshot, Dict[str, int], Optional[int]]:
    """Return the current render snapshot plus latest intent metadata."""

    with server._state_lock:
        snapshot = build_ledger_snapshot(server._state_ledger, server._scene)
        level_shapes = snapshot.level_shapes
        current_level = snapshot.current_level

        desired_dims_seq = -1
        desired_step: Optional[Tuple[int, ...]] = None
        latest_dims = latest_get_all("dims")
        for _key, (seq, value) in latest_dims.items():
            seq_int = int(seq)
            if seq_int >= desired_dims_seq and isinstance(value, (list, tuple)):
                desired_step = tuple(int(v) for v in value)
                desired_dims_seq = seq_int
        if desired_step is not None:
            clamped = _clamp_step(desired_step, level_shapes, current_level)
            if snapshot.current_step != clamped:
                snapshot = replace(snapshot, current_step=clamped)

        desired_view_seq = -1
        desired_ndisplay: Optional[int] = None
        latest_view = latest_get_all("view")
        for _key, (seq, value) in latest_view.items():
            seq_int = int(seq)
            if seq_int >= desired_view_seq and isinstance(value, (int, float)):
                desired_view_seq = seq_int
                desired_ndisplay = 3 if int(value) >= 3 else 2

    desired_seqs = {"dims": int(desired_dims_seq), "view": int(desired_view_seq)}
    logger.info(
        "render snapshot pulled step=%s desired_seqs=%s desired_ndisplay=%s",
        snapshot.current_step,
        desired_seqs,
        desired_ndisplay,
    )
    return snapshot, desired_seqs, desired_ndisplay


__all__.extend(["pull_render_snapshot", "_clamp_step"])
