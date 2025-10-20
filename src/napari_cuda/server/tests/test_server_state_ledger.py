from __future__ import annotations

import pytest

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    LedgerEvent,
    ServerStateLedger,
)


def test_record_confirmed_stores_value_and_notifies() -> None:
    ledger = ServerStateLedger(clock=lambda: 42.0)
    seen: list[LedgerEvent] = []
    ledger.subscribe("dims", "main", "current_step", seen.append)

    event = ledger.record_confirmed(
        "dims",
        "main",
        "current_step",
        (5, 0, 0),
        origin="worker",
    )

    assert event is not None
    assert event.value == (5, 0, 0)
    assert event.timestamp == pytest.approx(42.0)
    assert seen == [event]

    stored = ledger.get("dims", "main", "current_step")
    assert isinstance(stored, LedgerEntry)
    assert stored.value == (5, 0, 0)
    assert stored.origin == "worker"


def test_record_confirmed_dedupes_identical_value() -> None:
    clock_steps = iter([1.0, 2.0])
    ledger = ServerStateLedger(clock=lambda: float(next(clock_steps)))

    first = ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")
    assert first is not None

    deduped = ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")
    assert deduped is None

    stored = ledger.get("view", "main", "ndisplay")
    assert stored is not None
    assert stored.timestamp == pytest.approx(1.0)


def test_batch_record_confirmed_updates_multiple_keys() -> None:
    ledger = ServerStateLedger(clock=lambda: 99.0)
    captured: list[LedgerEvent] = []
    ledger.subscribe_all(captured.append)

    events = ledger.batch_record_confirmed(
        [
            ("dims", "main", "current_step", (10, 0, 0), {"level": 1}),
            ("multiscale", "main", "level", 1, None),
        ],
        origin="worker",
    )

    assert len(events) == 2
    assert captured == events
    assert ledger.get("multiscale", "main", "level") is not None


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
    assert first is not None
    assert first.version == 1

    stored = ledger.get("dims", "main", "current_step")
    assert stored is not None
    assert stored.version == 1
    assert ledger.current_version("dims", "main", "current_step") == 1

    deduped = ledger.record_confirmed("dims", "main", "current_step", (1, 0, 0), origin="worker")
    assert deduped is None
    assert ledger.current_version("dims", "main", "current_step") == 1

    second = ledger.record_confirmed("dims", "main", "current_step", (2, 0, 0), origin="worker")
    assert second is not None
    assert second.version == 2
    assert ledger.current_version("dims", "main", "current_step") == 2


def test_record_confirmed_accepts_version_override() -> None:
    ledger = ServerStateLedger()
    event = ledger.record_confirmed(
        "view",
        "main",
        "ndisplay",
        3,
        origin="worker",
        version=99,
    )
    assert event is not None
    assert event.version == 99

    stored = ledger.get("view", "main", "ndisplay")
    assert stored is not None
    assert stored.version == 99
    assert ledger.current_version("view", "main", "ndisplay") == 99

    next_event = ledger.record_confirmed("view", "main", "ndisplay", 2, origin="worker")
    assert next_event is not None
    assert next_event.version == 100


def test_batch_record_confirmed_assigns_versions_per_key() -> None:
    ledger = ServerStateLedger()
    events = ledger.batch_record_confirmed(
        [
            ("dims", "main", "current_step", (0, 0, 0)),
            ("multiscale", "main", "level", 0),
        ],
        origin="worker",
    )
    assert len(events) == 2
    assert events[0].version == 1
    assert events[1].version == 1

    follow_up = ledger.batch_record_confirmed(
        [
            ("dims", "main", "current_step", (1, 0, 0)),
            ("multiscale", "main", "level", 1, None, 5),
        ],
        origin="worker",
    )
    assert len(follow_up) == 2
    assert follow_up[0].version == 2
    assert follow_up[1].version == 5
