from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from napari_cuda.server.app import scene_publisher


def test_build_scene_payload_delegates(monkeypatch) -> None:
    called = {}

    def fake_builder(*, scene_snapshot, ledger_snapshot, viewer_settings):
        called["scene_snapshot"] = scene_snapshot
        called["ledger_snapshot"] = ledger_snapshot
        called["viewer_settings"] = viewer_settings
        return "payload"

    monkeypatch.setattr(scene_publisher, "build_notify_scene_payload", fake_builder)

    snapshot = SimpleNamespace()
    ledger = {"layers": []}
    viewer = {"fps": 60}

    result = scene_publisher.build_scene_payload(snapshot, ledger, viewer)

    assert result == "payload"
    assert called == {
        "scene_snapshot": snapshot,
        "ledger_snapshot": ledger,
        "viewer_settings": viewer,
    }


def test_cache_scene_history_updates_store_and_clears_clients(monkeypatch) -> None:
    recorded = {"envelopes": [], "resets": []}

    class DummyStore:
        def snapshot_envelope(self, kind, payload, timestamp):
            recorded["envelopes"].append((kind, payload, timestamp))

        def reset_epoch(self, kind, timestamp):
            recorded["resets"].append((kind, timestamp))

    class DummySequencer:
        def __init__(self, key):
            self.key = key
            self.clears = 0

        def clear(self):
            self.clears += 1

    sequencers: dict[tuple[int, str], DummySequencer] = {}

    def fake_state_sequencer(ws, kind):
        key = (id(ws), kind)
        sequencers.setdefault(key, DummySequencer(key))
        return sequencers[key]

    monkeypatch.setattr(scene_publisher, "state_sequencer", fake_state_sequencer)

    store = DummyStore()
    payload = SimpleNamespace(to_dict=lambda: {"scene": "payload"})
    clients = [object(), object()]

    scene_publisher.cache_scene_history(
        store,
        clients,
        payload,
        now=lambda: 123.0,
    )

    assert recorded["envelopes"][0][0] == scene_publisher.NOTIFY_SCENE_TYPE
    assert recorded["envelopes"][0][1] == {"scene": "payload"}
    assert recorded["envelopes"][0][2] == pytest.approx(123.0)
    assert recorded["resets"] == [
        (scene_publisher.NOTIFY_LAYERS_TYPE, 123.0),
        (scene_publisher.NOTIFY_STREAM_TYPE, 123.0),
    ]

    for kind in (
        scene_publisher.NOTIFY_SCENE_TYPE,
        scene_publisher.NOTIFY_LAYERS_TYPE,
        scene_publisher.NOTIFY_STREAM_TYPE,
    ):
        for client in clients:
            seq = sequencers[(id(client), kind)]
            assert seq.clears == 1


def test_broadcast_state_baseline_schedules_tasks(monkeypatch) -> None:
    async def fake_orchestrate(server, ws, resume):
        await asyncio.sleep(0)
        return (server, ws, resume)

    monkeypatch.setattr(scene_publisher, "orchestrate_connect", fake_orchestrate)

    scheduled: list[tuple[asyncio.Future, str]] = []

    def schedule_coro(coro, label: str):
        scheduled.append((coro, label))

    class Client:
        def __init__(self, resume):
            self._napari_cuda_resume_plan = resume

    clients = [Client({"foo": "bar"}), Client({})]

    server = object()

    scene_publisher.broadcast_state_baseline(
        server,
        clients,
        schedule_coro=schedule_coro,
        reason="dataset-load",
    )

    assert len(scheduled) == 2
    for _, label in scheduled:
        assert label.startswith("baseline-dataset-load")
    for coro, _ in scheduled:
        coro.close()
