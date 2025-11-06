from __future__ import annotations

from napari_cuda.server.scene import build_ledger_snapshot, snapshot_render_state
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import (
    AxisExtent,
    DimsSpec,
    DimsSpecAxis,
    dims_spec_to_payload,
)


def test_snapshot_render_state_preserves_dims_metadata() -> None:
    ledger = ServerStateLedger()
    ledger.batch_record_confirmed(
        [
            ("scene", "main", "op_seq", 4),
            ("dims", "main", "current_step", (5, 0, 0)),
        ],
        origin="test",
    )
    level_shapes = ((10, 20, 30), (5, 10, 15))
    axes = []
    for idx, label in enumerate(("z", "y", "x")):
        per_steps = tuple(shape[idx] for shape in level_shapes)
        per_world = tuple(
            AxisExtent(start=0.0, stop=float(max(count - 1, 0)), step=1.0) for count in per_steps
        )
        axes.append(
            DimsSpecAxis(
                index=idx,
                label=label,
                role=label,
                displayed=idx in (1, 2),
                order_position=idx,
                current_step=(5, 0, 0)[idx],
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=per_steps,
                per_level_world=per_world,
            )
        )
    spec = DimsSpec(
        version=1,
        ndim=3,
        ndisplay=2,
        order=(0, 1, 2),
        displayed=(1, 2),
        current_level=1,
        current_step=(5, 0, 0),
        level_shapes=level_shapes,
        plane_mode=True,
        axes=tuple(axes),
        levels=(
            {"index": 0, "shape": [10, 20, 30]},
            {"index": 1, "shape": [5, 10, 15]},
        ),
        downgraded=False,
        labels=None,
    )
    ledger.record_confirmed(
        "dims",
        "main",
        "dims_spec",
        dims_spec_to_payload(spec),
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
