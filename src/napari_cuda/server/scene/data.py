"""Server control-channel data bag and helpers.

This module centralises the mutable scene metadata tracked by the
headless server so state-channel handlers can operate on a single bag of
data and emit immutable snapshots to the worker.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot, build_ledger_snapshot
from napari_cuda.server.control.state_ledger import LedgerEntry, ServerStateLedger


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


__all__ = [
    "CONTROL_KEYS",
    "CameraDeltaCommand",
    "ServerSceneData",
    "create_server_scene_data",
    "default_multiscale_state",
    "default_volume_state",
    "build_render_scene_state",
    "layer_controls_from_ledger",
    "volume_state_from_ledger",
    "ControlMeta",
    "get_control_meta",
    "increment_server_sequence",
]


def default_volume_state() -> Dict[str, Any]:
    """Return the canonical defaults for volume render hints."""

    return {
        "mode": "mip",
        "colormap": "gray",
        "clim": [0.0, 1.0],
        "opacity": 1.0,
        "sample_step": 1.0,
    }


def default_multiscale_state() -> Dict[str, Any]:
    """Return the canonical defaults for multiscale metadata."""

    return {
        "levels": [],
        "current_level": 0,
        "policy": "oversampling",
        "index_space": "base",
    }


@dataclass(frozen=True)
class CameraDeltaCommand:
    """Queued camera delta consumed by the render thread."""

    kind: Literal["zoom", "pan", "orbit", "reset"]
    target: str = "main"
    command_seq: int = 0
    factor: Optional[float] = None
    anchor_px: Optional[tuple[float, float]] = None
    dx_px: float = 0.0
    dy_px: float = 0.0
    d_az_deg: float = 0.0
    d_el_deg: float = 0.0


@dataclass
class ServerSceneData:
    """Mutable scene metadata owned by the headless server."""

    latest_state: RenderLedgerSnapshot = field(default_factory=RenderLedgerSnapshot)
    use_volume: bool = False
    volume_state: Dict[str, Any] = field(default_factory=default_volume_state)
    multiscale_state: Dict[str, Any] = field(default_factory=default_multiscale_state)
    last_scene_snapshot: Optional[Dict[str, Any]] = None
    state_ledger: Optional[ServerStateLedger] = None
    last_scene_seq: int = 0
    control_meta: Dict[str, Dict[str, Dict[str, "ControlMeta"]]] = field(default_factory=dict)
    # Removed optimistic dims caches; staged transactions drive desired state


def create_server_scene_data(
    *,
    state_ledger: Optional[ServerStateLedger] = None,
) -> ServerSceneData:
    """Instantiate :class:`ServerSceneData` with an optional policy log path."""

    return ServerSceneData(state_ledger=state_ledger)


@dataclass
class ControlMeta:
    """Bookkeeping for legacy reducer compatibility."""

    last_server_seq: int = 0
    last_timestamp: float = 0.0


def get_control_meta(store: ServerSceneData, scope: str, target: str, key: str) -> ControlMeta:
    """Return (and create if needed) control metadata for a property."""

    scope_map = store.control_meta.setdefault(scope, {})
    target_map = scope_map.setdefault(target, {})
    meta = target_map.get(key)
    if meta is None:
        meta = ControlMeta()
        target_map[key] = meta
    return meta


def increment_server_sequence(store: ServerSceneData) -> int:
    """Increment and return the scene-level sequence counter."""

    store.last_scene_seq += 1
    return store.last_scene_seq


def build_render_scene_state(
    ledger: ServerStateLedger,
    scene: ServerSceneData,
    *,
    center: Any = None,
    zoom: Any = None,
    angles: Any = None,
    distance: Any = None,
    fov: Any = None,
    rect: Any = None,
    current_step: Any = None,
    volume_mode: Any = None,
    volume_colormap: Any = None,
    volume_clim: Any = None,
    volume_opacity: Any = None,
    volume_sample_step: Any = None,
) -> RenderLedgerSnapshot:
    """Build a render-scene snapshot from the ledger with optional overrides."""

    base = build_ledger_snapshot(
        ledger,
        scene,
    )

    return RenderLedgerSnapshot(
        center=_coalesce_tuple(center, base.center, float),
        zoom=_coalesce_float(zoom, base.zoom),
        angles=_coalesce_tuple(angles, base.angles, float),
        distance=_coalesce_float(distance, base.distance),
        fov=_coalesce_float(fov, base.fov),
        rect=_coalesce_tuple(rect, base.rect, float),
        current_step=_coalesce_tuple(current_step, base.current_step, int),
        volume_mode=_coalesce_string(volume_mode, base.volume_mode, fallback=scene.volume_state.get("mode")),
        volume_colormap=_coalesce_string(
            volume_colormap,
            base.volume_colormap,
            fallback=scene.volume_state.get("colormap"),
        ),
        volume_clim=_coalesce_tuple(
            volume_clim,
            base.volume_clim,
            float,
            fallback=scene.volume_state.get("clim"),
        ),
        volume_opacity=_coalesce_float(volume_opacity, base.volume_opacity, fallback=scene.volume_state.get("opacity")),
        volume_sample_step=_coalesce_float(
            volume_sample_step,
            base.volume_sample_step,
            fallback=scene.volume_state.get("sample_step"),
        ),
        layer_values=base.layer_values,
        layer_versions=base.layer_versions,
    )


def _coalesce_tuple(
    value: Any,
    base: Optional[tuple],
    mapper,
    *,
    fallback: Any = None,
) -> Optional[tuple]:
    source = value if value is not None else (base if base is not None else fallback)
    if source is None:
        return None
    try:
        return tuple(mapper(v) for v in source)
    except Exception:
        return base


def _coalesce_float(value: Any, base: Optional[float], *, fallback: Any = None) -> Optional[float]:
    source = value if value is not None else (base if base is not None else fallback)
    if source is None:
        return None
    try:
        return float(source)
    except Exception:
        return base


def _coalesce_string(value: Any, base: Optional[str], *, fallback: Any = None) -> Optional[str]:
    source = value if value is not None else (base if base is not None else fallback)
    if source is None:
        return None
    try:
        text = str(source).strip()
        return text if text else base
    except Exception:
        return base


def layer_controls_from_ledger(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> Dict[str, Dict[str, Any]]:
    """Collect per-layer control values from a ledger snapshot."""

    controls: dict[str, dict[str, Any]] = defaultdict(dict)
    for (scope, target, key), entry in snapshot.items():
        if scope != "layer":
            continue
        controls[str(target)][str(key)] = entry.value
    return {layer_id: props for layer_id, props in controls.items() if props}


def volume_state_from_ledger(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> Dict[str, Any]:
    """Extract volume render hints from the ledger snapshot."""

    mapping: Dict[str, Any] = {}
    for ledger_key, field in (
        (("volume", "main", "render_mode"), "mode"),
        (("volume", "main", "colormap"), "colormap"),
        (("volume", "main", "contrast_limits"), "clim"),
        (("volume", "main", "opacity"), "opacity"),
        (("volume", "main", "sample_step"), "sample_step"),
    ):
        entry = snapshot.get(ledger_key)
        if entry is not None and entry.value is not None:
            mapping[field] = entry.value
    return mapping
