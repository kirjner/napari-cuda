from __future__ import annotations

import pytest

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    LedgerEvent,
    ServerStateLedger,
)


def test_record_confirmed_stores_value_and_notifies() -> None:
    ledger = ServerStateLedger(clock=lambda: 42.0)
    seen: list[LedgerEvent] = []
    ledger.subscribe("dims", "main", "current_step", seen.append)

    entry = ledger.record_confirmed(
        "dims",
        "main",
        "current_step",
        (5, 0, 0),
        origin="worker",
    )

    assert entry.value == (5, 0, 0)
    assert entry.timestamp == pytest.approx(42.0)
    assert len(seen) == 1
    event = seen[0]
    assert event.value == entry.value
    assert event.timestamp == entry.timestamp

    stored = ledger.get("dims", "main", "current_step")
    assert isinstance(stored, LedgerEntry)
    assert stored.value == (5, 0, 0)
    assert stored.origin == "worker"


def test_record_confirmed_dedupes_identical_value() -> None:
    clock_steps = iter([1.0, 2.0])
    ledger = ServerStateLedger(clock=lambda: float(next(clock_steps)))

    first = ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")
    assert first.value == 2

    deduped = ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")
    assert deduped is first

    stored = ledger.get("view", "main", "ndisplay")
    assert stored is not None
    assert stored.timestamp == pytest.approx(1.0)


def test_batch_record_confirmed_updates_multiple_keys() -> None:
    ledger = ServerStateLedger(clock=lambda: 99.0)
    captured: list[LedgerEvent] = []
    ledger.subscribe_all(captured.append)

    stored = ledger.batch_record_confirmed(
        [
            ("dims", "main", "current_step", (10, 0, 0), {"level": 1}),
            ("multiscale", "main", "level", 1, None),
        ],
        origin="worker",
    )

    assert len(stored) == 2
    assert len(captured) == 2
    assert ("multiscale", "main", "level") in stored
    assert ledger.get("multiscale", "main", "level") is stored[("multiscale", "main", "level")]


def test_batch_record_confirmed_rejects_invalid_entry_length() -> None:
    ledger = ServerStateLedger()
    with pytest.raises(ValueError):
        ledger.batch_record_confirmed(
            [("dims", "main", "current_step")],  # type: ignore[arg-type]
            origin="worker",
        )


def test_snapshot_returns_copy() -> None:
    ledger = ServerStateLedger(clock=lambda: 5.0)
    ledger.record_confirmed("view", "main", "ndisplay", 3, origin="worker")

    snap = ledger.snapshot()
    assert ("view", "main", "ndisplay") in snap

    ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")
    assert snap[("view", "main", "ndisplay")].value == 3


def test_metadata_requires_mapping() -> None:
    ledger = ServerStateLedger()
    with pytest.raises(TypeError):
        ledger.record_confirmed(
            "dims",
            "main",
            "current_step",
            (0, 0, 0),
            origin="worker",
            metadata=42,  # type: ignore[arg-type]
        )


def test_record_confirmed_assigns_monotonic_versions() -> None:
    ledger = ServerStateLedger()

    first = ledger.record_confirmed("dims", "main", "current_step", (1, 0, 0), origin="worker")
    assert first.version == 1

    stored = ledger.get("dims", "main", "current_step")
    assert stored is not None
    assert stored.version == 1
    assert ledger.current_version("dims", "main", "current_step") == 1

    deduped = ledger.record_confirmed("dims", "main", "current_step", (1, 0, 0), origin="worker")
    assert deduped is first
    assert ledger.current_version("dims", "main", "current_step") == 1

    second = ledger.record_confirmed("dims", "main", "current_step", (2, 0, 0), origin="worker")
    assert second.version == 2
    assert ledger.current_version("dims", "main", "current_step") == 2


def test_record_confirmed_accepts_version_override() -> None:
    ledger = ServerStateLedger()
    entry = ledger.record_confirmed(
        "view",
        "main",
        "ndisplay",
        3,
        origin="worker",
        version=99,
    )
    assert entry.version == 99

    stored = ledger.get("view", "main", "ndisplay")
    assert stored is not None
    assert stored.version == 99
    assert ledger.current_version("view", "main", "ndisplay") == 99

    next_entry = ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")
    assert next_entry.version == 100


def test_batch_record_confirmed_assigns_versions_per_key() -> None:
    ledger = ServerStateLedger()
    stored = ledger.batch_record_confirmed(
        [
            ("dims", "main", "current_step", (0, 0, 0)),
            ("multiscale", "main", "level", 0),
        ],
        origin="worker",
    )
    assert len(stored) == 2
    assert stored[("dims", "main", "current_step")].version == 1
    assert stored[("multiscale", "main", "level")].version == 1

    follow_up = ledger.batch_record_confirmed(
        [
            ("dims", "main", "current_step", (1, 0, 0)),
            ("multiscale", "main", "level", 1, None, 5),
        ],
        origin="worker",
    )
    assert len(follow_up) == 2
    assert follow_up[("dims", "main", "current_step")].version == 2
    assert follow_up[("multiscale", "main", "level")].version == 5


def test_clear_scope_removes_entries_and_versions() -> None:
    ledger = ServerStateLedger()
    ledger.record_confirmed("layer", "layer-1", "visible", True, origin="worker")
    ledger.record_confirmed("layer", "layer-1", "opacity", 0.5, origin="worker")
    ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")

    removed = ledger.clear_scope("layer", target="layer-1")
    assert removed == 2

    assert ledger.get("layer", "layer-1", "visible") is None
    assert ledger.get("layer", "layer-1", "opacity") is None
    assert ledger.current_version("layer", "layer-1", "visible") is None
    assert ledger.get("view", "main", "ndisplay") is not None


def test_clear_scope_supports_target_prefix() -> None:
    ledger = ServerStateLedger()
    ledger.record_confirmed("layer", "layer-1", "visible", True, origin="worker")
    ledger.record_confirmed("layer", "layer-1", "opacity", 0.5, origin="worker")
    ledger.record_confirmed("layer", "layer-2", "visible", False, origin="worker")
    ledger.record_confirmed("layer", "aux-3", "visible", True, origin="worker")

    removed = ledger.clear_scope("layer", target_prefix="layer-")
    assert removed == 3

    assert ledger.get("layer", "layer-1", "visible") is None
    assert ledger.get("layer", "layer-2", "visible") is None
    assert ledger.get("layer", "aux-3", "visible") is not None


def test_clear_scope_rejects_conflicting_filters() -> None:
    ledger = ServerStateLedger()
    with pytest.raises(ValueError):
        ledger.clear_scope("layer", target="layer-1", target_prefix="layer-")
