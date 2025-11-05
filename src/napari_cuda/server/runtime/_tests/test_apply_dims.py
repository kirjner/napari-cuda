from __future__ import annotations

from types import SimpleNamespace

from napari.components.dims import Dims

from napari_cuda.server.runtime.render_loop.applying.plane import (
    apply_dims_from_snapshot,
)
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.shared.axis_spec import (
    derive_axis_labels,
    derive_margins,
    derive_order,
    fabricate_axis_spec,
    with_updated_margins,
)


class _DummyViewer:
    def __init__(self) -> None:
        self.dims = Dims(
            ndim=3,
            ndisplay=2,
            order=(0, 1, 2),
            axis_labels=("z", "y", "x"),
            margin_left=(0.0, 0.0, 0.0),
            margin_right=(0.0, 0.0, 0.0),
        )


class _DummyWorker:
    def __init__(self) -> None:
        self._viewer = _DummyViewer()
        self.viewport_state = SimpleNamespace()
        self._current_level = 0

    def _current_level_index(self) -> int:
        return int(self._current_level)

    def _set_current_level_index(self, level: int) -> None:
        self._current_level = int(level)


def test_apply_dims_from_snapshot_updates_margins() -> None:
    worker = _DummyWorker()
    iface = RenderApplyInterface(worker)  # type: ignore[arg-type]

    spec = fabricate_axis_spec(
        ndim=3,
        ndisplay=2,
        current_level=1,
        level_shapes=[(20, 10, 5)],
        order=(0, 1, 2),
        displayed=(1, 2),
        labels=("z", "y", "x"),
        current_step=(5, 0, 0),
    )
    for idx, (left_val, right_val) in enumerate(zip((24.0, 1.0, 2.0), (12.0, 3.0, 4.0), strict=False)):
        spec = with_updated_margins(
            spec,
            idx,
            margin_left_world=left_val,
            margin_right_world=right_val,
            margin_left_steps=left_val,
            margin_right_steps=right_val,
        )
    margins = derive_margins(spec, prefer_world=True)

    snapshot = RenderLedgerSnapshot(
        current_step=(5, 0, 0),
        order=spec.order,
        displayed=spec.displayed,
        axis_labels=tuple(derive_axis_labels(spec)),
        ndisplay=spec.ndisplay,
        margin_left=margins[0],
        margin_right=margins[1],
        current_level=1,
        level_shapes=spec.level_shapes,
        axes=spec,
    )

    apply_dims_from_snapshot(iface, snapshot)

    dims = worker._viewer.dims
    assert dims.margin_left == (24.0, 1.0, 2.0)
    assert dims.margin_right == (12.0, 3.0, 4.0)
    assert worker._current_level == 1
