from __future__ import annotations

import math

import pytest

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.render_update_mailbox import RenderUpdateMailbox


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def advance(self, delta: float) -> None:
        self._now += float(delta)

    def __call__(self) -> float:
        return self._now


def test_drain_pending_updates_returns_last_values() -> None:
    clock = _FakeClock()
    mailbox = RenderUpdateMailbox(time_fn=clock)

    snapshot = RenderLedgerSnapshot(center=(1.0, 2.0, 3.0), zoom=1.25)
    mailbox.set_scene_state(snapshot)

    updates = mailbox.drain()
    assert updates.scene_state == snapshot

    # Second drain should be empty (one-shot semantics)
    updates_empty = mailbox.drain()
    assert updates_empty.scene_state is None


def test_zoom_hint_recent_then_stale() -> None:
    clock = _FakeClock()
    mailbox = RenderUpdateMailbox(time_fn=clock)

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



if __name__ == "__main__":  # pragma: no cover - direct execution helper
    pytest.main([__file__])
