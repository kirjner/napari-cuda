from __future__ import annotations

from itertools import count

import pytest

from napari_cuda.client.control.pending_update_store import StateStore
from napari_cuda.protocol.messages import StateUpdateMessage


@pytest.fixture
def store() -> StateStore:
    seq_counter = count(1)
    clock_counter = count(100)
    return StateStore(
        client_id="client-a",
        next_client_seq=lambda: next(seq_counter),
        clock=lambda: float(next(clock_counter)),
    )


def _ack(message: StateUpdateMessage, *, value=None, client_id=None, server_seq=1) -> StateUpdateMessage:
    return StateUpdateMessage(
        scope=message.scope,
        target=message.target,
        key=message.key,
        value=message.value if value is None else value,
        client_id=message.client_id if client_id is None else client_id,
        client_seq=message.client_seq,
        interaction_id=message.interaction_id,
        phase=message.phase,
        timestamp=message.timestamp,
        server_seq=server_seq,
    )


def test_apply_local_sequences(store: StateStore) -> None:
    start_msg, projection = store.apply_local(
        "layer", "layer-0", "gamma", 0.5, "start", interaction_id="drag-1"
    )
    assert start_msg.client_seq == 1
    assert projection == pytest.approx(0.5)

    update_msg, projection = store.apply_local(
        "layer", "layer-0", "gamma", 0.6, "update", interaction_id="drag-1"
    )
    assert update_msg.client_seq == 2
    assert projection == pytest.approx(0.6)

    commit_msg, projection = store.apply_local(
        "layer", "layer-0", "gamma", 0.7, "commit", interaction_id="drag-1"
    )
    assert commit_msg.client_seq == 3
    assert projection == pytest.approx(0.7)

    debug = store.dump_debug()["layer:layer-0:gamma"]
    assert [entry["client_seq"] for entry in debug["pending"]] == [2, 3]

    result = store.apply_remote(_ack(update_msg, server_seq=10))
    assert result.is_self is True
    assert result.pending_len == 1
    assert result.projection_value == pytest.approx(0.7)
    assert result.overridden is False

    result = store.apply_remote(_ack(commit_msg, server_seq=11))
    assert result.is_self is True
    assert result.pending_len == 0
    assert result.projection_value == pytest.approx(0.7)
    assert result.overridden is False

    debug = store.dump_debug()["layer:layer-0:gamma"]
    assert debug["pending"] == []
    assert debug["confirmed"]["value"] == pytest.approx(0.7)
    assert debug["confirmed"]["server_seq"] == 11


def test_update_replaces_last_entry(store: StateStore) -> None:
    store.apply_local("layer", "layer-1", "gamma", 0.2, "start")
    store.apply_local("layer", "layer-1", "gamma", 0.3, "update")
    debug = store.dump_debug()["layer:layer-1:gamma"]
    assert [entry["client_seq"] for entry in debug["pending"]] == [2]

    store.apply_local("layer", "layer-1", "gamma", 0.4, "reset")
    debug = store.dump_debug()["layer:layer-1:gamma"]
    assert [entry["client_seq"] for entry in debug["pending"]] == [3]


def test_foreign_update_clears_pending(store: StateStore) -> None:
    start_msg, _ = store.apply_local("layer", "layer-2", "gamma", 0.9, "start")
    store.apply_local("layer", "layer-2", "gamma", 1.1, "update")

    foreign = _ack(start_msg, client_id="other-client", value=1.4, server_seq=20)
    result = store.apply_remote(foreign)
    assert result.is_self is False
    assert result.pending_len == 0
    assert result.projection_value == pytest.approx(1.4)
    assert result.overridden is True

    debug = store.dump_debug()["layer:layer-2:gamma"]
    assert debug["pending"] == []
    assert debug["confirmed"]["value"] == pytest.approx(1.4)


def test_seed_and_reconnect(store: StateStore) -> None:
    store.seed_confirmed("layer", "layer-3", "opacity", 0.42, server_seq=5)
    store.apply_local("layer", "layer-3", "opacity", 0.55, "start")
    store.apply_local("layer", "layer-3", "opacity", 0.65, "update")

    store.clear_pending_on_reconnect()
    debug = store.dump_debug()["layer:layer-3:opacity"]
    assert debug["pending"] == []
    assert debug["confirmed"]["value"] == pytest.approx(0.42)
    assert debug["confirmed"]["server_seq"] == 5
