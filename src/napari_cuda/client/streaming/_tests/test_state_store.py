from __future__ import annotations

from collections import deque
from itertools import count

import pytest

from napari_cuda.client.control.pending_update_store import StateStore
from napari_cuda.protocol import build_ack_state


def _id_pairs():
    counter = count(1)
    while True:
        idx = next(counter)
        yield f"intent-{idx}", f"state-{idx}"


@pytest.fixture
def store() -> StateStore:
    clock_steps = deque(float(x) for x in range(100, 200))
    return StateStore(clock=lambda: clock_steps.popleft())


def test_apply_local_tracks_pending_and_projection(store: StateStore) -> None:
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

    assert pending_update.update_phase == "update"
    assert pending_update.projection_value == pytest.approx(0.7)

    debug = store.dump_debug()["layer:layer-0:gamma"]
    assert [entry["update_phase"] for entry in debug["pending"]] == ["update"]
    assert debug["pending"][0]["value"] == pytest.approx(0.7)


def test_apply_ack_accepted_updates_confirmed_state(store: StateStore) -> None:
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

    ack = build_ack_state(
        session_id="session-1",
        frame_id="ack-1",
        payload={
            "intent_id": pending.intent_id,
            "in_reply_to": pending.frame_id,
            "status": "accepted",
            "applied_value": 0.5,
        },
        timestamp=10.0,
    )

    outcome = store.apply_ack(ack)

    assert outcome.status == "accepted"
    assert outcome.pending_len == 0
    assert outcome.confirmed_value == pytest.approx(0.5)
    assert outcome.applied_value == pytest.approx(0.5)
    assert outcome.was_pending is True

    debug = store.dump_debug()["layer:layer-1:opacity"]
    assert debug["pending"] == []
    assert debug["confirmed"]["value"] == pytest.approx(0.5)


def test_apply_ack_rejected_reverts_to_confirmed(store: StateStore) -> None:
    ids = _id_pairs()
    store.seed_confirmed("layer", "layer-2", "gamma", 0.9, timestamp=5.0)

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


def test_clear_pending_on_reconnect_resets_index(store: StateStore) -> None:
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
    store.apply_local(
        "layer",
        "layer-3",
        "gamma",
        0.2,
        "update",
        intent_id=intent2,
        frame_id=frame2,
    )

    store.clear_pending_on_reconnect()
    debug = store.dump_debug()["layer:layer-3:gamma"]
    assert debug["pending"] == []
