"""Shared helpers for client/server coordination."""

# Prefer dims_spec (new names), but re-export legacy axes_spec for compatibility
from .dims_spec import (  # noqa: F401
    AxisExtent,
    AxesSpec,  # legacy alias
    AxesSpecAxis,  # legacy alias
    DimsSpec,
    DimsSpecAxis,
    axes_spec_from_notify_payload,  # legacy alias
    axes_spec_from_payload,  # legacy alias
    axes_spec_to_payload,  # legacy alias
    build_axes_spec_from_ledger,  # legacy alias
    build_dims_spec_from_ledger,
    dims_spec_from_notify_payload,
    dims_spec_from_payload,
    dims_spec_to_payload,
    validate_ledger_against_spec,  # legacy alias
    validate_ledger_against_dims_spec,
)

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
    "AxisExtent",
    "AxesSpec",
    "AxesSpecAxis",
    "axes_spec_from_notify_payload",
    "axes_spec_from_payload",
    "axes_spec_to_payload",
    "build_axes_spec_from_ledger",
    "validate_ledger_against_spec",
]
