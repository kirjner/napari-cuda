from __future__ import annotations

import pytest

from napari_cuda.shared.dims_spec import (
    AxisExtent,
    DimsSpec,
    DimsSpecAxis,
    dims_spec_axis_index_for_target,
    dims_spec_axis_labels,
    dims_spec_clamp_step,
    dims_spec_displayed,
    dims_spec_level_shape,
    dims_spec_order,
    dims_spec_remap_step_for_level,
    dims_spec_primary_axis,
)


def _build_sample_spec() -> DimsSpec:
    axes = (
        DimsSpecAxis(
            index=0,
            label="z",
            role="z",
            displayed=False,
            order_position=0,
            current_step=5,
            margin_left_steps=0.0,
            margin_right_steps=0.0,
            margin_left_world=0.0,
            margin_right_world=0.0,
            per_level_steps=(10, 6),
            per_level_world=(
                AxisExtent(0.0, 9.0, 1.0),
                AxisExtent(0.0, 5.0, 1.0),
            ),
        ),
        DimsSpecAxis(
            index=1,
            label="y",
            role="y",
            displayed=True,
            order_position=1,
            current_step=3,
            margin_left_steps=0.0,
            margin_right_steps=0.0,
            margin_left_world=0.0,
            margin_right_world=0.0,
            per_level_steps=(256, 128),
            per_level_world=(
                AxisExtent(0.0, 255.0, 1.0),
                AxisExtent(0.0, 127.0, 1.0),
            ),
        ),
        DimsSpecAxis(
            index=2,
            label="x",
            role="x",
            displayed=True,
            order_position=2,
            current_step=7,
            margin_left_steps=0.0,
            margin_right_steps=0.0,
            margin_left_world=0.0,
            margin_right_world=0.0,
            per_level_steps=(512, 256),
            per_level_world=(
                AxisExtent(0.0, 511.0, 1.0),
                AxisExtent(0.0, 255.0, 1.0),
            ),
        ),
    )
    return DimsSpec(
        version=1,
        ndim=3,
        ndisplay=2,
        order=(0, 1, 2),
        displayed=(1, 2),
        current_level=0,
        current_step=(5, 3, 7),
        level_shapes=((10, 256, 512), (6, 128, 256)),
        plane_mode=True,
        axes=axes,
        levels=({"index": 0}, {"index": 1}),
        labels=("a", "b", "c"),
    )


def test_dims_spec_axis_label_helpers() -> None:
    spec = _build_sample_spec()
    assert dims_spec_axis_labels(spec) == ("z", "y", "x")
    assert dims_spec_order(spec) == (0, 1, 2)
    assert dims_spec_displayed(spec) == (1, 2)


def test_dims_spec_level_shape() -> None:
    spec = _build_sample_spec()
    assert dims_spec_level_shape(spec, 0) == (10, 256, 512)
    assert dims_spec_level_shape(spec, 1) == (6, 128, 256)
    with pytest.raises(IndexError):
        _ = dims_spec_level_shape(spec, 2)


def test_dims_spec_clamp_step() -> None:
    spec = _build_sample_spec()
    clamped = dims_spec_clamp_step(spec, 0, (12, -5, 900))
    assert clamped == (9, 0, 511)
    clamped = dims_spec_clamp_step(spec, 1, (4,))
    assert clamped == (4, 0, 0)


def test_dims_spec_primary_axis_and_lookup() -> None:
    spec = _build_sample_spec()
    assert dims_spec_primary_axis(spec) == 0
    assert dims_spec_axis_index_for_target(spec, "x") == 2
    assert dims_spec_axis_index_for_target(spec, "axis-1") == 1
    assert dims_spec_axis_index_for_target(spec, "7") is None


def test_dims_spec_remap_step_for_level() -> None:
    spec = _build_sample_spec()
    remapped = dims_spec_remap_step_for_level(spec, step=(5, 3, 7), prev_level=0, next_level=1)
    assert remapped[0] == 3  # proportional z mapping from 10 -> 6
    assert remapped[1:] == (3, 7)
