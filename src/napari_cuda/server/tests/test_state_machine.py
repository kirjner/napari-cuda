from __future__ import annotations

import math
from dataclasses import replace

import pytest

from napari_cuda.server.state.scene_state import ServerSceneState
from napari_cuda.server.runtime.runtime_mailbox import RenderMailbox


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def advance(self, delta: float) -> None:
        self._now += float(delta)

    def __call__(self) -> float:
        return self._now


def test_drain_pending_updates_returns_last_values() -> None:
    clock = _FakeClock()
    mailbox = RenderMailbox(time_fn=clock)

    mailbox.enqueue_display_mode(2)
    mailbox.enqueue_display_mode(3)
    mailbox.enqueue_multiscale(1, "coarse")
    mailbox.enqueue_multiscale(2, None)
    snapshot = ServerSceneState(center=(1.0, 2.0, 3.0), zoom=1.25)
    mailbox.enqueue_scene_state(snapshot)

    updates = mailbox.drain()
    assert updates.display_mode == 3
    assert updates.multiscale is not None
    assert updates.multiscale.level == 2
    assert updates.multiscale.path is None
    assert updates.scene_state == snapshot

    # Second drain should be empty (one-shot semantics)
    updates_empty = mailbox.drain()
    assert updates_empty.display_mode is None
    assert updates_empty.multiscale is None
    assert updates_empty.scene_state is None


def test_zoom_hint_recent_then_stale() -> None:
    clock = _FakeClock()
    mailbox = RenderMailbox(time_fn=clock)

    mailbox.record_zoom_hint(1.2)
    zoom = mailbox.consume_zoom_hint(max_age=0.5)
    assert zoom is not None
    assert math.isclose(zoom.ratio, 1.2)

    # No intent left after consumption
    assert mailbox.consume_zoom_hint(max_age=0.5) is None

    with pytest.raises(ValueError):
        mailbox.record_zoom_hint(0.0)

    mailbox.record_zoom_hint(0.8)
    clock.advance(1.0)
    # Hint is now stale
    assert mailbox.consume_zoom_hint(max_age=0.5) is None


def test_update_state_signature_detects_changes() -> None:
    mailbox = RenderMailbox()

    base = ServerSceneState(center=(1.0, 2.0, 3.0), zoom=1.0, current_step=(0,))
    assert mailbox.update_state_signature(base)
    # Identical state should not trigger a change
    assert not mailbox.update_state_signature(base)

    moved = replace(base, center=(1.0, 2.5, 3.0))
    assert mailbox.update_state_signature(moved)

    # Invalid numeric inputs should raise so invariant violations surface immediately
    with pytest.raises((ValueError, TypeError)):
        mailbox.update_state_signature(replace(base, zoom="invalid"))  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    pytest.main([__file__])
