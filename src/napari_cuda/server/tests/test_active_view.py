from __future__ import annotations

from napari_cuda.server.control.state_reducers import (
    reduce_bootstrap_state,
    reduce_level_update,
    reduce_plane_restore,
    reduce_view_update,
)
from napari_cuda.server.ledger import ServerStateLedger


def _bootstrap_minimal_ledger() -> ServerStateLedger:
    ledger = ServerStateLedger()
    reduce_bootstrap_state(
        ledger,
        step=(0, 0),
        axis_labels=("y", "x"),
        order=(1, 0),
        level_shapes=((256, 256), (128, 128)),
        levels=(
            {"shape": (256, 256), "downsample": (1.0, 1.0)},
            {"shape": (128, 128), "downsample": (2.0, 2.0)},
        ),
        current_level=0,
        ndisplay=2,
    )
    return ledger


def test_active_view_written_on_view_toggle() -> None:
    ledger = _bootstrap_minimal_ledger()

    reduce_view_update(ledger, ndisplay=3)

    entry = ledger.get("viewport", "active", "state")
    assert entry is not None
    assert isinstance(entry.value, dict)
    payload = entry.value
    assert payload.get("mode") == "volume"
    assert int(payload.get("level", -1)) == 0


def test_active_view_written_on_level_update() -> None:
    ledger = _bootstrap_minimal_ledger()

    reduce_level_update(ledger, level=1, step=(0, 0))

    entry = ledger.get("viewport", "active", "state")
    assert entry is not None
    payload = entry.value
    assert payload.get("mode") == "plane"
    assert int(payload.get("level", -1)) == 1


def test_active_view_written_on_plane_restore() -> None:
    ledger = _bootstrap_minimal_ledger()

    reduce_plane_restore(
        ledger,
        level=0,
        step=(0, 0),
        center=(64.0, 64.0),
        zoom=1.0,
        rect=(0.0, 0.0, 128.0, 128.0),
    )

    entry = ledger.get("viewport", "active", "state")
    assert entry is not None
    payload = entry.value
    assert payload.get("mode") == "plane"
    assert int(payload.get("level", -1)) == 0

