"""DimsSpec (formerly AxesSpec): canonical applied dims document.

Provides DimsSpec/DimsSpecAxis and compatibility aliases for AxesSpec.
"""

from __future__ import annotations

from typing import Any, Mapping

# Import the existing implementation and expose new names that mirror it.
from .axis_spec import (  # noqa: F401
    AxisExtent as AxisExtent,
    AxesSpec as _AxesSpec,
    AxesSpecAxis as _AxesSpecAxis,
    build_axes_spec_from_ledger as _build_axes_spec_from_ledger,
    axes_spec_to_payload as _axes_spec_to_payload,
    axes_spec_from_payload as _axes_spec_from_payload,
    axes_spec_from_notify_payload as _axes_spec_from_notify_payload,
    validate_ledger_against_spec as _validate_ledger_against_spec,
)


# New names (preferred)
DimsSpecAxis = _AxesSpecAxis
DimsSpec = _AxesSpec


def build_dims_spec_from_ledger(snapshot):
    return _build_axes_spec_from_ledger(snapshot)


def dims_spec_to_payload(spec: DimsSpec | None) -> dict[str, Any] | None:  # type: ignore[name-defined]
    return _axes_spec_to_payload(spec)


def dims_spec_from_payload(data: Mapping[str, Any] | None):
    return _axes_spec_from_payload(data)


def dims_spec_from_notify_payload(payload: Any):
    return _axes_spec_from_notify_payload(payload)


def validate_ledger_against_dims_spec(spec: DimsSpec, snapshot) -> None:  # type: ignore[name-defined]
    _validate_ledger_against_spec(spec, snapshot)


# Legacy aliases (compat)
AxesSpecAxis = _AxesSpecAxis
AxesSpec = _AxesSpec
build_axes_spec_from_ledger = _build_axes_spec_from_ledger
axes_spec_to_payload = _axes_spec_to_payload
axes_spec_from_payload = _axes_spec_from_payload
axes_spec_from_notify_payload = _axes_spec_from_notify_payload
validate_ledger_against_spec = _validate_ledger_against_spec


__all__ = [
    # New
    "DimsSpec",
    "DimsSpecAxis",
    "build_dims_spec_from_ledger",
    "dims_spec_from_notify_payload",
    "dims_spec_from_payload",
    "dims_spec_to_payload",
    "validate_ledger_against_dims_spec",
    # Legacy
    "AxesSpec",
    "AxesSpecAxis",
    "build_axes_spec_from_ledger",
    "axes_spec_from_notify_payload",
    "axes_spec_from_payload",
    "axes_spec_to_payload",
    "validate_ledger_against_spec",
    "AxisExtent",
]

