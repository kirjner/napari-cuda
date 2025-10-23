from __future__ import annotations

from collections import deque
from itertools import count
from typing import Any

import pytest

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.protocol import build_ack_state


def _id_pairs():
    counter = count(1)
    while True:
        idx = next(counter)
        yield f"intent-{idx}", f"state-{idx}"


@pytest.fixture
def store() -> ClientStateLedger:
    clock_steps = deque(float(x) for x in range(100, 200))
    return ClientStateLedger(clock=lambda: clock_steps.popleft())


def test_apply_local_tracks_pending_and_projection(store: ClientStateLedger) -> None:
    ids = _id_pairs()

    intent1, frame1 = next(ids)
    pending_start = store.apply_local(
        "layer",
        "layer-0",
        "gamma",
        0.5,
        "start",
        intent_id=intent1,
        frame_id=frame1,
    )
    assert pending_start is not None

    assert pending_start.update_phase == "start"
    assert pending_start.projection_value == pytest.approx(0.5)

    debug = store.dump_debug()["layer:layer-0:gamma"]
    assert [entry["update_phase"] for entry in debug["pending"]] == ["start"]

    intent2, frame2 = next(ids)
    pending_update = store.apply_local(
        "layer",
        "layer-0",
        "gamma",
        0.7,
        "update",
        intent_id=intent2,
        frame_id=frame2,
    )
    assert pending_update is not None

    assert pending_update.update_phase == "update"
    assert pending_update.projection_value == pytest.approx(0.7)

    debug = store.dump_debug()["layer:layer-0:gamma"]
    assert [entry["update_phase"] for entry in debug["pending"]] == ["update"]
    assert debug["pending"][0]["value"] == pytest.approx(0.7)


def test_apply_ack_accepted_updates_confirmed_state(store: ClientStateLedger) -> None:
    ids = _id_pairs()

    intent1, frame1 = next(ids)
    store.apply_local(
        "layer",
        "layer-1",
        "opacity",
        0.3,
        "start",
        intent_id=intent1,
        frame_id=frame1,
    )

    intent2, frame2 = next(ids)
    pending = store.apply_local(
        "layer",
        "layer-1",
        "opacity",
        0.45,
        "update",
        intent_id=intent2,
        frame_id=frame2,
    )
    assert pending is not None

    ack = build_ack_state(
        session_id="session-1",
        frame_id="ack-1",
        payload={
            "intent_id": pending.intent_id,
            "in_reply_to": pending.frame_id,
            "status": "accepted",
            "applied_value": 0.5,
            "version": 11,
        },
        timestamp=10.0,
    )

    outcome = store.apply_ack(ack)

    assert outcome.status == "accepted"
    assert outcome.pending_len == 0
    assert outcome.confirmed_value == pytest.approx(0.5)
    assert outcome.applied_value == pytest.approx(0.5)
    assert outcome.was_pending is True
    assert outcome.version == 11

    debug = store.dump_debug()["layer:layer-1:opacity"]
    assert debug["pending"] == []
    assert debug["confirmed"]["value"] == pytest.approx(0.5)
    assert debug["confirmed"]["version"] == 11


def test_apply_ack_rejected_reverts_to_confirmed(store: ClientStateLedger) -> None:
    ids = _id_pairs()
    store.record_confirmed("layer", "layer-2", "gamma", 0.9, timestamp=5.0)

    intent, frame = next(ids)
    pending = store.apply_local(
        "layer",
        "layer-2",
        "gamma",
        1.2,
        "start",
        intent_id=intent,
        frame_id=frame,
    )
    assert pending is not None

    ack = build_ack_state(
        session_id="session-1",
        frame_id="ack-2",
        payload={
            "intent_id": pending.intent_id,
            "in_reply_to": pending.frame_id,
            "status": "rejected",
            "error": {"code": "state.rejected", "message": "out of range"},
        },
        timestamp=11.0,
    )

    outcome = store.apply_ack(ack)

    assert outcome.status == "rejected"
    assert outcome.pending_len == 0
    assert outcome.error == {"code": "state.rejected", "message": "out of range"}
    assert outcome.confirmed_value == pytest.approx(0.9)

    debug = store.dump_debug()["layer:layer-2:gamma"]
    assert debug["pending"] == []
    assert debug["confirmed"]["value"] == pytest.approx(0.9)


def test_clear_pending_on_reconnect_resets_index(store: ClientStateLedger) -> None:
    ids = _id_pairs()

    intent1, frame1 = next(ids)
    store.apply_local(
        "layer",
        "layer-3",
        "gamma",
        0.1,
        "start",
        intent_id=intent1,
        frame_id=frame1,
    )

    intent2, frame2 = next(ids)
    pending = store.apply_local(
        "layer",
        "layer-3",
        "gamma",
        0.2,
        "update",
        intent_id=intent2,
        frame_id=frame2,
    )
    assert pending is not None

    store.clear_pending_on_reconnect()
    debug = store.dump_debug()["layer:layer-3:gamma"]
    assert debug["pending"] == []


def test_subscribe_all_receives_confirmed_updates(store: ClientStateLedger) -> None:
    events: list[Any] = []

    def _capture(update: Any) -> None:
        events.append(update)

    store.subscribe_all(_capture)
    store.record_confirmed('dims', 'z', 'index', 5, metadata={'axis_index': 0})

    assert len(events) == 1
    update = events[0]
    assert update.scope == 'dims'
    assert update.target == 'z'
    assert update.key == 'index'
    assert update.value == 5
    assert update.metadata == {'axis_index': 0}


def test_batch_record_confirmed_promotes_multiple_entries(store: ClientStateLedger) -> None:
    events: list[Any] = []

    def _capture(update: Any) -> None:
        events.append(update)

    store.subscribe_all(_capture)

    store.batch_record_confirmed(
        [
            ('dims', 'z', 'index', 12, {'axis_index': 0}),
            ('view', 'main', 'ndisplay', 3, None),
        ],
        origin='worker_refresh',
    )

    assert store.confirmed_value('dims', 'z', 'index') == 12
    assert store.confirmed_value('view', 'main', 'ndisplay') == 3

    assert len(events) == 2
    scopes = {event.scope for event in events}
    assert scopes == {'dims', 'view'}
    dims_event = next(event for event in events if event.scope == 'dims')
    assert dims_event.metadata == {'axis_index': 0}
    assert dims_event.origin == 'worker_refresh'


def test_pending_state_snapshot_reports_confirmed_value(store: ClientStateLedger) -> None:
    store.record_confirmed('dims', '0', 'index', 7, metadata={'axis_index': 0})

    snapshot = store.pending_state_snapshot('dims', '0', 'index')

    assert snapshot is not None
    projection, pending_len, origin, confirmed = snapshot
    assert projection == 7
    assert pending_len == 0
    assert origin == 'remote'
    assert confirmed == 7


def test_apply_local_duplicate_absolute_returns_none(store: ClientStateLedger) -> None:
    store.record_confirmed('layer', 'layer-4', 'gamma', 1.0)

    result = store.apply_local(
        'layer',
        'layer-4',
        'gamma',
        1.0,
        'start',
        intent_id='intent-dup-1',
        frame_id='state-dup-1',
    )

    assert result is None


def test_apply_local_delta_update_not_suppressed(store: ClientStateLedger) -> None:
    store.record_confirmed('dims', '0', 'step', 1, metadata={'axis_index': 0})

    pending = store.apply_local(
        'dims',
        '0',
        'step',
        1,
        'start',
        intent_id='intent-delta-1',
        frame_id='state-delta-1',
        metadata={'update_kind': 'step'},
    )

    assert pending is not None
