"""Build immutable render snapshots from the server ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from napari_cuda.server.control.state_ledger import ServerStateLedger, LedgerEntry

if TYPE_CHECKING:
    from napari_cuda.server.scene import ServerSceneData


@dataclass(frozen=True)
class RenderSceneSnapshot:
    """Authoritative scene state consumed by the render thread."""

    center: Optional[tuple[float, float, float]] = None
    zoom: Optional[float] = None
    angles: Optional[tuple[float, float, float]] = None
    current_step: Optional[tuple[int, ...]] = None
    volume_mode: Optional[str] = None
    volume_colormap: Optional[str] = None
    volume_clim: Optional[tuple[float, float]] = None
    volume_opacity: Optional[float] = None
    volume_sample_step: Optional[float] = None
    layer_updates: Optional[Dict[str, Dict[str, Any]]] = None


def build_render_snapshot(
    ledger: ServerStateLedger,
    scene: "ServerSceneData",
    *,
    layer_updates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    drain_pending_layers: bool = True,
) -> RenderSceneSnapshot:
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

    return RenderSceneSnapshot(
        center=center_tuple,
        zoom=zoom_float,
        angles=angles_tuple,
        current_step=current_step,
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
    try:
        return tuple(mapper(v) for v in source)
    except Exception:
        return None


def _float_or_none(value: Any, *, fallback: Any = None) -> Optional[float]:
    source = value if value is not None else fallback
    if source is None:
        return None
    try:
        return float(source)
    except Exception:
        return None


def _string_or_none(value: Any, *, fallback: Any = None) -> Optional[str]:
    source = value if value is not None else fallback
    if source is None:
        return None
    try:
        text = str(source).strip()
        return text if text else None
    except Exception:
        return None


def _normalise_layer_updates(updates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    normalised: Dict[str, Dict[str, Any]] = {}
    for layer_id, props in updates.items():
        entries: Dict[str, Any] = {}
        for key, value in props.items():
            entries[str(key)] = value
        if entries:
            normalised[str(layer_id)] = entries
    return normalised


__all__ = ["RenderSceneSnapshot", "build_render_snapshot"]
