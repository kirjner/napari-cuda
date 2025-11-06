from __future__ import annotations

import pytest

from napari_cuda.server.state_ledger import LedgerEntry
from napari_cuda.shared.dims_spec import (
    dims_spec_from_payload as axes_spec_from_payload,
    dims_spec_to_payload as axes_spec_to_payload,
    build_dims_spec_from_ledger as build_axes_spec_from_ledger,
    validate_ledger_against_dims_spec as validate_ledger_against_spec,
)


def _spec_payload() -> dict[str, object]:
    return {
        "version": 1,
        "ndim": 3,
        "ndisplay": 3,
        "order": [0, 1, 2],
        "displayed": [0, 1, 2],
        "current_level": 1,
        "current_step": [5, 1, 0],
        "level_shapes": [[10, 20, 30], [4, 12, 15]],
        "plane_mode": False,
        "axes": [
            {
                "index": 0,
                "label": "z",
                "role": "z",
                "displayed": True,
                "order_pos": 0,
                "current_step": 5,
                "margin_left_steps": 0.0,
                "margin_right_steps": 0.0,
                "margin_left_world": 0.0,
                "margin_right_world": 0.0,
                "per_level_steps": [10, 4],
                "per_level_world": [
                    {"start": 0.0, "stop": 9.0, "step": 1.0},
                    {"start": 0.0, "stop": 3.0, "step": 1.0},
                ],
            },
            {
                "index": 1,
                "label": "y",
                "role": "y",
                "displayed": True,
                "order_pos": 1,
                "current_step": 1,
                "margin_left_steps": 0.0,
                "margin_right_steps": 0.0,
                "margin_left_world": 0.0,
                "margin_right_world": 0.0,
                "per_level_steps": [20, 12],
                "per_level_world": [
                    {"start": 0.0, "stop": 19.0, "step": 1.0},
                    {"start": 0.0, "stop": 11.0, "step": 1.0},
                ],
            },
            {
                "index": 2,
                "label": "x",
                "role": "x",
                "displayed": True,
                "order_pos": 2,
                "current_step": 0,
                "margin_left_steps": 0.0,
                "margin_right_steps": 0.0,
                "margin_left_world": 0.0,
                "margin_right_world": 0.0,
                "per_level_steps": [30, 15],
                "per_level_world": [
                    {"start": 0.0, "stop": 29.0, "step": 1.0},
                    {"start": 0.0, "stop": 14.0, "step": 1.0},
                ],
            },
        ],
        "levels": [
            {"index": 0, "shape": [10, 20, 30]},
            {"index": 1, "shape": [4, 12, 15]},
        ],
        "downgraded": False,
        "labels": ["z", "y", "x"],
    }


def _snapshot() -> dict[tuple[str, str, str], LedgerEntry]:
    return {
        ("dims", "main", "dims_spec"): LedgerEntry(
            value=_spec_payload(),
            timestamp=0.0,
            origin="test",
            metadata=None,
            version=None,
        ),
    }


def test_build_axes_spec_from_ledger() -> None:
    snapshot = _snapshot()
    spec = build_axes_spec_from_ledger(snapshot)

    assert spec.version == 1
    assert spec.ndim == 3
    assert spec.order == (0, 1, 2)
    assert spec.displayed == (0, 1, 2)
    assert spec.current_level == 1
    assert spec.ndisplay == 3
    assert spec.axes[0].label == "z"
    assert spec.axes[1].label == "y"
    assert spec.axes[2].label == "x"
    assert spec.axes[0].per_level_steps == (10, 4)
    assert spec.axes[0].per_level_world[0].stop == pytest.approx(9.0)


def test_axes_spec_serialization_round_trip() -> None:
    snapshot = _snapshot()
    spec = build_axes_spec_from_ledger(snapshot)

    payload = axes_spec_to_payload(spec)
    assert payload is not None

    restored = axes_spec_from_payload(payload)
    assert restored == spec


def test_validate_ledger_against_spec_detects_mismatch() -> None:
    snapshot = _snapshot()
    spec = build_axes_spec_from_ledger(snapshot)

    validate_ledger_against_spec(spec, snapshot)

    mismatched = dict(_spec_payload())
    mismatched["ndisplay"] = 2
    snapshot[("dims", "main", "dims_spec")] = LedgerEntry(
        value=mismatched,
        timestamp=0.0,
        origin="test",
        metadata=None,
        version=None,
    )
    with pytest.raises(AssertionError):
        validate_ledger_against_spec(spec, snapshot)
