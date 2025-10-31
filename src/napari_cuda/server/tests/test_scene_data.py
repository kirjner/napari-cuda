from __future__ import annotations

from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.viewstate import (
    build_ledger_snapshot,
)
from napari_cuda.server.viewstate import snapshot_render_state


def test_snapshot_render_state_preserves_dims_metadata() -> None:
    ledger = ServerStateLedger()
    ledger.batch_record_confirmed(
        [
            ("scene", "main", "op_seq", 4),
            ("dims", "main", "current_step", (5, 0, 0)),
            ("dims", "main", "order", (0, 1, 2)),
            ("view", "main", "ndisplay", 2),
            ("view", "main", "displayed", (1, 2)),
            ("multiscale", "main", "level", 1),
            ("multiscale", "main", "level_shapes", ((10, 20, 30), (5, 10, 15))),
        ],
        origin="test",
    )

    base = build_ledger_snapshot(ledger)
    result = snapshot_render_state(ledger, plane_center=(1, 2))

    assert result.order == base.order
    assert result.displayed == base.displayed
    assert result.ndisplay == base.ndisplay
    assert result.dims_version == base.dims_version
    assert result.multiscale_level_version == base.multiscale_level_version
    assert result.level_shapes == base.level_shapes
    assert result.op_seq == base.op_seq
    assert result.plane_center == (1.0, 2.0)
