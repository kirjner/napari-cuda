"""Server-side helpers derived from the control ledger."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot, build_ledger_snapshot
from napari_cuda.server.control.state_ledger import LedgerEntry, ServerStateLedger
from napari_cuda.server.scene_defaults import default_volume_state


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
    "default_volume_state",
    "build_render_scene_state",
    "layer_controls_from_ledger",
    "volume_state_from_ledger",
    "multiscale_state_from_snapshot",
    "viewport_state_from_ledger",
]


def multiscale_state_from_snapshot(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> Dict[str, Any]:
    """Derive multiscale policy metadata from a ledger snapshot."""

    def _normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _normalize(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_normalize(v) for v in value]
        return value

    state: Dict[str, Any] = {}

    policy_entry = snapshot.get(("multiscale", "main", "policy"))
    if policy_entry is not None and policy_entry.value is not None:
        state["policy"] = str(policy_entry.value)

    index_space_entry = snapshot.get(("multiscale", "main", "index_space"))
    if index_space_entry is not None and index_space_entry.value is not None:
        state["index_space"] = str(index_space_entry.value)

    level_entry = snapshot.get(("multiscale", "main", "level"))
    if level_entry is not None and level_entry.value is not None:
        try:
            state["current_level"] = int(level_entry.value)
        except Exception:
            state["current_level"] = level_entry.value

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


def build_render_scene_state(
    ledger: ServerStateLedger,
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

    snapshot_map = ledger.snapshot()
    base = build_ledger_snapshot(ledger, snapshot_map)

    volume_defaults = volume_state_from_ledger(snapshot_map)
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

    return RenderLedgerSnapshot(
        center=_coalesce_tuple(center, base.center, float),
        zoom=_coalesce_float(zoom, base.zoom),
        angles=_coalesce_tuple(angles, base.angles, float),
        distance=_coalesce_float(distance, base.distance),
        fov=_coalesce_float(fov, base.fov),
        rect=_coalesce_tuple(rect, base.rect, float),
        current_step=_coalesce_tuple(current_step, base.current_step, int),
        volume_mode=_coalesce_string(volume_mode, base.volume_mode, fallback=default_mode),
        volume_colormap=_coalesce_string(
            volume_colormap,
            base.volume_colormap,
            fallback=default_colormap,
        ),
        volume_clim=_coalesce_tuple(
            volume_clim,
            base.volume_clim,
            float,
            fallback=default_clim,
        ),
        volume_opacity=_coalesce_float(volume_opacity, base.volume_opacity, fallback=default_opacity),
        volume_sample_step=_coalesce_float(
            volume_sample_step,
            base.volume_sample_step,
            fallback=default_sample_step,
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


def viewport_state_from_ledger(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> Dict[str, Any]:
    """Extract viewport mode and plane/volume payloads from the ledger snapshot."""

    def _clone(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _clone(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_clone(v) for v in value]
        return value

    payload: Dict[str, Any] = {}

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
