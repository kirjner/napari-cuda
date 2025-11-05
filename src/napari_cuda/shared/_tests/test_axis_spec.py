from __future__ import annotations

import pytest

from napari_cuda.shared.axis_spec import (
    axes_spec_from_payload,
    axes_spec_to_payload,
    build_axes_spec_from_ledger,
    validate_ledger_against_spec,
)


class _Entry:
    __slots__ = ("value",)

    def __init__(self, value) -> None:
        self.value = value


def _snapshot() -> dict[tuple[str, str, str], _Entry]:
    return {
        ("dims", "main", "current_step"): _Entry((5, 1, 0)),
        ("multiscale", "main", "level_shapes"): _Entry(((10, 20, 30), (4, 12, 15))),
        ("multiscale", "main", "level"): _Entry(1),
        ("view", "main", "ndisplay"): _Entry(3),
        ("dims", "main", "order"): _Entry((0, 1, 2)),
        ("view", "main", "displayed"): _Entry((0, 1, 2)),
        ("dims", "main", "axis_labels"): _Entry(("z", "y", "x")),
        ("dims", "main", "margin_left"): _Entry((0.0, 0.0, 0.0)),
        ("dims", "main", "margin_right"): _Entry((0.0, 0.0, 0.0)),
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

    snapshot[("view", "main", "ndisplay")] = _Entry(2)
    with pytest.raises(AssertionError):
        validate_ledger_against_spec(spec, snapshot)
