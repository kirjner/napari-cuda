from __future__ import annotations

import math

import pytest

from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdateMailbox
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.shared.axis_spec import (
    derive_axis_labels,
    derive_margins,
    fabricate_axis_spec,
)


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

    snapshot = _snapshot(plane_center=(1.0, 2.0), plane_zoom=1.25, op_seq=1)
    mailbox.set_scene_state(snapshot)

    updates = mailbox.drain()
    assert updates.scene_state == snapshot
    assert updates.op_seq == 1

    # Second drain should be empty (one-shot semantics)
    updates_empty = mailbox.drain()
    assert updates_empty.scene_state is None
    assert updates_empty.op_seq == 1


def test_mailbox_drops_stale_sequence() -> None:
    clock = _FakeClock()
    mailbox = RenderUpdateMailbox(time_fn=clock)

    fresh = _snapshot(current_step=(1, 0, 0), op_seq=5)
    stale = _snapshot(current_step=(2, 0, 0), op_seq=3)

    mailbox.set_scene_state(fresh)
    mailbox.set_scene_state(stale)

    update = mailbox.drain()
    assert update.scene_state == fresh
    assert update.op_seq == 5


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
def _snapshot(**kwargs) -> RenderLedgerSnapshot:
    current_step = tuple(int(v) for v in kwargs.get("current_step", (0, 0)))
    ndisplay = int(kwargs.get("ndisplay", 2))
    current_level = int(kwargs.get("current_level", 0))
    ndim = len(current_step) if current_step else 2
    level_shapes = kwargs.get("level_shapes")
    if level_shapes is None:
        level_shapes = [tuple(max(1, abs(int(value)) + 1) for value in current_step or (0,) * ndim)]
    else:
        level_shapes = [tuple(int(dim) for dim in shape) for shape in level_shapes]

    spec = fabricate_axis_spec(
        ndim=len(level_shapes[0]),
        ndisplay=ndisplay,
        current_level=current_level,
        level_shapes=level_shapes,
        order=kwargs.get("order"),
        displayed=kwargs.get("displayed"),
        labels=kwargs.get("axis_labels"),
        current_step=current_step,
    )
    margins = derive_margins(spec, prefer_world=True)

    kwargs.setdefault("current_step", current_step)
    kwargs.setdefault("current_level", current_level)
    kwargs.setdefault("ndisplay", ndisplay)
    kwargs.setdefault("level_shapes", tuple(tuple(int(dim) for dim in shape) for shape in level_shapes))
    kwargs.setdefault("axis_labels", tuple(derive_axis_labels(spec)))
    kwargs.setdefault("order", spec.order)
    kwargs.setdefault("displayed", spec.displayed)
    kwargs.setdefault("margin_left", margins[0])
    kwargs.setdefault("margin_right", margins[1])
    kwargs.setdefault("axes", spec)

    return RenderLedgerSnapshot(**kwargs)
