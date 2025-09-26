from __future__ import annotations

import math
from dataclasses import replace

import pytest

from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.server_scene_queue import ServerSceneQueue


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def advance(self, delta: float) -> None:
        self._now += float(delta)

    def __call__(self) -> float:
        return self._now


def test_drain_pending_updates_returns_last_values() -> None:
    clock = _FakeClock()
    machine = ServerSceneQueue(time_fn=clock)

    machine.queue_display_mode(2)
    machine.queue_display_mode(3)
    machine.queue_multiscale_level(1, "coarse")
    machine.queue_multiscale_level(2, None)
    snapshot = ServerSceneState(center=(1.0, 2.0, 3.0), zoom=1.25)
    machine.queue_scene_state(snapshot)

    updates = machine.drain_pending_updates()
    assert updates.display_mode == 3
    assert updates.multiscale is not None
    assert updates.multiscale.level == 2
    assert updates.multiscale.path is None
    assert updates.scene_state == snapshot

    # Second drain should be empty (one-shot semantics)
    updates_empty = machine.drain_pending_updates()
    assert updates_empty.display_mode is None
    assert updates_empty.multiscale is None
    assert updates_empty.scene_state is None


def test_zoom_intent_recent_then_stale() -> None:
    clock = _FakeClock()
    machine = ServerSceneQueue(time_fn=clock)

    machine.record_zoom_intent(1.2)
    zoom = machine.consume_zoom_intent(max_age=0.5)
    assert zoom is not None
    assert math.isclose(zoom.ratio, 1.2)

    # No intent left after consumption
    assert machine.consume_zoom_intent(max_age=0.5) is None

    with pytest.raises(ValueError):
        machine.record_zoom_intent(0.0)

    machine.record_zoom_intent(0.8)
    clock.advance(1.0)
    # Intent is now stale
    assert machine.consume_zoom_intent(max_age=0.5) is None


def test_update_state_signature_detects_changes() -> None:
    machine = ServerSceneQueue()

    base = ServerSceneState(center=(1.0, 2.0, 3.0), zoom=1.0, current_step=(0,))
    assert machine.update_state_signature(base)
    # Identical state should not trigger a change
    assert not machine.update_state_signature(base)

    moved = replace(base, center=(1.0, 2.5, 3.0))
    assert machine.update_state_signature(moved)

    # Invalid numeric inputs should raise so invariant violations surface immediately
    with pytest.raises((ValueError, TypeError)):
        machine.update_state_signature(replace(base, zoom="invalid"))  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    pytest.main([__file__])
