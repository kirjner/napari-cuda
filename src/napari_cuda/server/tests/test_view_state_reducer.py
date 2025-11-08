from __future__ import annotations

from napari_cuda.server.control.state_reducers import reduce_view_update
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import AxisExtent, DimsSpec, DimsSpecAxis, dims_spec_to_payload


def _make_spec() -> DimsSpec:
    axis_template = (
        ("z", False),
        ("y", True),
        ("x", True),
    )
    axes: list[DimsSpecAxis] = []
    per_level = ((16, 16, 16), (8, 8, 8), (4, 4, 4))
    for index, (label, displayed) in enumerate(axis_template):
        axes.append(
            DimsSpecAxis(
                index=index,
                label=label,
                role=label,
                displayed=displayed,
                order_position=index,
                current_step=0,
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=tuple(shape[index] for shape in per_level),
                per_level_world=tuple(
                    AxisExtent(0.0, float(shape[index] - 1), 1.0)
                    for shape in per_level
                ),
            )
        )
    levels = tuple({"index": idx, "shape": list(shape)} for idx, shape in enumerate(per_level))
    return DimsSpec(
        version=1,
        ndim=3,
        ndisplay=2,
        order=(0, 1, 2),
        displayed=(1, 2),
        current_level=0,
        current_step=(0, 0, 0),
        level_shapes=per_level,
        plane_mode=True,
        axes=tuple(axes),
        levels=levels,
        labels=None,
    )


def test_reduce_view_update_sets_volume_level_to_coarsest() -> None:
    ledger = ServerStateLedger()
    spec = _make_spec()
    ledger.record_confirmed("dims", "main", "dims_spec", dims_spec_to_payload(spec), origin="test")
    ledger.record_confirmed("dims", "main", "current_step", spec.current_step, origin="test")

    result = reduce_view_update(ledger, ndisplay=3, intent_id="test-intent")

    assert result.dims_spec is not None
    assert int(result.dims_spec.current_level) == 2  # coarsest level
