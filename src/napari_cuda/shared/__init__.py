"""Shared helpers for client/server coordination."""

# Prefer dims_spec (new names), but re-export legacy axes_spec for compatibility
from .dims_spec import (  # noqa: F401
    AxisExtent,
    DimsSpec,
    DimsSpecAxis,
    build_dims_spec_from_ledger,
    dims_spec_from_notify_payload,
    dims_spec_from_payload,
    dims_spec_to_payload,
    validate_ledger_against_dims_spec,
)

__all__ = [
    "DimsSpec",
    "DimsSpecAxis",
    "build_dims_spec_from_ledger",
    "dims_spec_from_notify_payload",
    "dims_spec_from_payload",
    "dims_spec_to_payload",
    "validate_ledger_against_dims_spec",
    "AxisExtent",
]
