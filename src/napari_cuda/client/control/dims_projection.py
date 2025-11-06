"""Shared helpers for deriving viewer-facing dims state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.shared.dims_spec import DimsSpec


@dataclass(frozen=True)
class DimsProjection:
    step: tuple[int, ...]
    ndisplay: int
    primary_axis: int


def project_dims(spec: DimsSpec, ledger: ClientStateLedger | None = None) -> DimsProjection:
    if ledger is None:
        step = tuple(int(axis.current_step) for axis in spec.axes)
        ndisplay = int(spec.ndisplay)
    else:
        step = []
        for axis in spec.axes:
            pending = ledger.latest_pending_value('dims', axis.label, 'index')
            if pending is not None:
                step.append(int(pending))
                continue
            confirmed = ledger.confirmed_value('dims', axis.label, 'index')
            if confirmed is not None:
                step.append(int(confirmed))
                continue
            step.append(int(axis.current_step))

        pending_nd = ledger.latest_pending_value('view', 'main', 'ndisplay')
        if pending_nd is not None:
            ndisplay = int(pending_nd)
        else:
            confirmed_nd = ledger.confirmed_value('view', 'main', 'ndisplay')
            ndisplay = int(confirmed_nd) if confirmed_nd is not None else int(spec.ndisplay)

    primary_axis = int(spec.order[0]) if spec.order else 0
    return DimsProjection(step=tuple(step), ndisplay=ndisplay, primary_axis=primary_axis)


def viewer_update_from_spec(spec: DimsSpec, projection: DimsProjection) -> dict[str, Any]:
    if 0 <= spec.current_level < len(spec.level_shapes):
        active_shape = spec.level_shapes[spec.current_level]
    else:
        active_shape = ()

    dims_range = tuple(
        (0.0, float(max(0, int(dim) - 1)), 1.0)
        for dim in active_shape
    )

    return {
        'current_step': projection.step,
        'ndisplay': projection.ndisplay,
        'ndim': int(spec.ndim),
        'dims_range': dims_range,
        'order': tuple(int(idx) for idx in spec.order),
        'axis_labels': tuple(axis.label for axis in spec.axes),
        'displayed': tuple(int(idx) for idx in spec.displayed),
    }


def current_ndisplay(state, ledger: ClientStateLedger) -> int:
    spec = state.dims_spec
    assert spec is not None, 'dims_spec must be available'
    return project_dims(spec, ledger).ndisplay


def is_volume_mode(state, ledger: ClientStateLedger) -> bool:
    spec = state.dims_spec
    if spec is None:
        return False
    projection = project_dims(spec, ledger)
    return (not spec.plane_mode) and projection.ndisplay >= 3


def project_step(spec: DimsSpec, ledger: ClientStateLedger) -> tuple[int, ...]:
    return project_dims(spec, ledger).step


__all__ = [
    'DimsProjection',
    'project_dims',
    'project_step',
    'viewer_update_from_spec',
    'current_ndisplay',
    'is_volume_mode',
]
