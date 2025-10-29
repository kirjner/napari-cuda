"""Control-side helpers for probing initial scene metadata."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

from napari_cuda.server.control.state_models import BootstrapSceneMetadata
from napari_cuda.server.data.zarr_source import ZarrSceneSource, LevelDescriptor
from napari_cuda.server.data.roi import plane_wh_for_level
from napari_cuda.server.data import lod


def _resolve_level_shapes(descriptors: Sequence[LevelDescriptor]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(size) for size in descriptor.shape) for descriptor in descriptors)


def _resolve_levels(descriptors: Sequence[LevelDescriptor]) -> Tuple[dict[str, object], ...]:
    payload: list[dict[str, object]] = []
    for descriptor in descriptors:
        entry: dict[str, object] = {
            "index": int(descriptor.index),
            "shape": [int(size) for size in descriptor.shape],
            "downsample": [float(value) for value in descriptor.downsample],
        }
        if descriptor.path:
            entry["path"] = str(descriptor.path)
        payload.append(entry)
    return tuple(payload)


def _normalize_step(
    *,
    initial: Tuple[int, ...],
    axes: Sequence[str],
    use_volume: bool,
) -> Tuple[int, ...]:
    values = list(int(value) for value in initial)
    if use_volume and axes:
        axes_lower = [axis.lower() for axis in axes]
        if "z" in axes_lower:
            z_index = axes_lower.index("z")
            while len(values) <= z_index:
                values.append(0)
            values[z_index] = 0
    return tuple(values)


def probe_scene_bootstrap(
    *,
    path: str,
    use_volume: bool,
    preferred_level: Optional[str] = None,
    axes_override: Optional[Sequence[str]] = None,
    z_override: Optional[int] = None,
    canvas_size: Tuple[int, int] = (1, 1),
    oversampling_thresholds: Optional[Mapping[int, float]] = None,
    oversampling_hysteresis: float = 0.1,
    threshold_in: float = 1.05,
    threshold_out: float = 1.35,
    fine_threshold: float = 1.05,
    policy_hysteresis: float = 0.0,
    cooldown_ms: float = 0.0,
) -> BootstrapSceneMetadata:
    """Inspect the scene source and return bootstrap metadata for the ledger."""

    source = ZarrSceneSource(
        path,
        preferred_level=preferred_level,
        axis_override=tuple(axes_override) if axes_override is not None else None,
    )

    current_level = source.current_level
    descriptors = source.level_descriptors
    level_shapes = _resolve_level_shapes(descriptors)
    levels = _resolve_levels(descriptors)

    axes = tuple(source.axes)
    viewport_w = max(1, int(canvas_size[0]))
    viewport_h = max(1, int(canvas_size[1]))
    overs_map: dict[int, float] = {}
    for descriptor in descriptors:
        level_index = int(descriptor.index)
        plane_h, plane_w = plane_wh_for_level(source, level_index)
        overs_map[level_index] = max(plane_h / viewport_h, plane_w / viewport_w)

    policy_config = lod.LevelPolicyConfig(
        threshold_in=float(threshold_in),
        threshold_out=float(threshold_out),
        fine_threshold=float(fine_threshold),
        hysteresis=float(policy_hysteresis),
        cooldown_ms=float(cooldown_ms),
    )

    oversampling_map = {int(k): float(v) for k, v in overs_map.items()}
    current_idx = int(current_level)
    remaining = max(1, len(descriptors) + 2)
    while remaining > 0:
        level_inputs = lod.LevelPolicyInputs(
            current_level=current_idx,
            oversampling=oversampling_map,
            zoom_ratio=None,
            lock_level=None,
            last_switch_ts=0.0,
            now_ts=0.0,
        )
        decision = lod.select_level(policy_config, level_inputs)
        selected_idx = int(decision.selected_level)
        if not decision.should_switch or selected_idx == current_idx:
            current_idx = selected_idx
            break
        current_idx = selected_idx
        remaining -= 1

    selected_level = current_idx
    level_count = len(descriptors)
    if level_count > 0:
        selected_level = max(0, min(int(selected_level), level_count - 1))
    else:
        selected_level = 0

    initial_step = source.initial_step(step_or_z=z_override, level=int(selected_level))
    applied_step = source.set_current_slice(initial_step, int(selected_level))
    resolved_step = _normalize_step(initial=applied_step, axes=axes, use_volume=use_volume)

    ndim = len(axes) if axes else len(resolved_step)
    if ndim <= 0:
        ndim = 1
    order = tuple(range(ndim))
    ndisplay = 3 if use_volume and ndim >= 3 else min(2, ndim)

    plane_h, plane_w = plane_wh_for_level(source, int(selected_level))
    rect = (0.0, 0.0, float(plane_w), float(plane_h))
    center = (float(plane_w) * 0.5, float(plane_h) * 0.5)
    zoom = 1.0

    return BootstrapSceneMetadata(
        step=resolved_step,
        axis_labels=axes if axes else tuple(f"axis-{idx}" for idx in range(ndim)),
        order=order,
        level_shapes=level_shapes,
        levels=levels,
        current_level=int(selected_level),
        ndisplay=int(ndisplay),
        plane_rect=rect,
        plane_center=center,
        plane_zoom=zoom,
    )


__all__ = ["probe_scene_bootstrap"]
