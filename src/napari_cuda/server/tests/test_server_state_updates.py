"""Tests for server control state reducers."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.control.state_reducers import (
    StateUpdateResult,
    _dims_entries_from_payload,
    clamp_level,
    clamp_opacity,
    clamp_sample_step,
    normalize_clim,
    reduce_camera_state,
    reduce_dims_update,
    reduce_layer_property,
    reduce_multiscale_level,
    reduce_multiscale_policy,
    reduce_volume_colormap,
    reduce_volume_contrast_limits,
    reduce_volume_opacity,
    reduce_volume_render_mode,
    reduce_volume_sample_step,
)
from napari_cuda.server.scene import create_server_scene_data


def _lock() -> threading.RLock:
    return threading.RLock()


def _scene_and_ledger() -> tuple[Any, ServerStateLedger, threading.RLock]:
    ledger = ServerStateLedger()
    scene = create_server_scene_data()
    lock = _lock()
    return scene, ledger, lock


def test_reduce_layer_property_records_ledger() -> None:
    scene, ledger, lock = _scene_and_ledger()

    result = reduce_layer_property(
        scene,
        ledger,
        lock,
        layer_id="layer-0",
        prop="opacity",
        value=0.4,
    )

    assert isinstance(result, StateUpdateResult)
    assert result.scope == "layer"
    assert result.value == 0.4
    entry = ledger.get("layer", "layer-0", "opacity")
    assert entry is not None
    assert entry.value == 0.4
    assert scene.layer_controls["layer-0"].opacity == 0.4
    assert scene.pending_layer_updates["layer-0"]["opacity"] == 0.4
    volume_entry = ledger.get("volume", "main", "opacity")
    assert volume_entry is not None
    assert volume_entry.value == 0.4
    assert scene.volume_state["opacity"] == 0.4


def test_reduce_layer_property_syncs_volume_contrast_and_colormap() -> None:
    scene, ledger, lock = _scene_and_ledger()

    result_clim = reduce_layer_property(
        scene,
        ledger,
        lock,
        layer_id="layer-0",
        prop="contrast_limits",
        value=(2.0, 8.0),
    )
    assert result_clim.value == (2.0, 8.0)
    volume_clim = ledger.get("volume", "main", "contrast_limits")
    assert volume_clim is not None
    assert volume_clim.value == (2.0, 8.0)
    assert scene.volume_state["clim"] == [2.0, 8.0]

    result_cmap = reduce_layer_property(
        scene,
        ledger,
        lock,
        layer_id="layer-0",
        prop="colormap",
        value="green",
    )
    assert result_cmap.value == "green"
    volume_cmap = ledger.get("volume", "main", "colormap")
    assert volume_cmap is not None
    assert volume_cmap.value == "green"
    assert scene.volume_state["colormap"] == "green"


def test_reduce_volume_helpers_write_ledger() -> None:
    scene, ledger, lock = _scene_and_ledger()

    mode = reduce_volume_render_mode(scene, ledger, lock, mode="mip")
    assert ledger.get("volume", "main", "render_mode").value == "mip"
    assert mode.value == "mip"

    limits = reduce_volume_contrast_limits(scene, ledger, lock, lo=1.0, hi=5.0)
    assert limits.value == (1.0, 5.0)
    assert ledger.get("volume", "main", "contrast_limits").value == (1.0, 5.0)

    cmap = reduce_volume_colormap(scene, ledger, lock, name="viridis")
    assert cmap.value == "viridis"
    assert ledger.get("volume", "main", "colormap").value == "viridis"

    opacity = reduce_volume_opacity(scene, ledger, lock, alpha=0.25)
    assert opacity.value == 0.25
    assert ledger.get("volume", "main", "opacity").value == 0.25

    sample = reduce_volume_sample_step(scene, ledger, lock, sample_step=0.75)
    assert sample.value == 0.75
    assert ledger.get("volume", "main", "sample_step").value == 0.75


@pytest.mark.parametrize(
    "alpha,expected",
    [(0.5, 0.5), (-1.0, 0.0), (5.0, 1.0), ("0.2", 0.2)],
)
def test_clamp_opacity(alpha: object, expected: float) -> None:
    assert clamp_opacity(alpha) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,expected",
    [(0.5, 0.5), (0.05, 0.1), (8.0, 4.0), ("1.5", 1.5)],
)
def test_clamp_sample_step(value: object, expected: float) -> None:
    assert clamp_sample_step(value) == pytest.approx(expected)


def test_clamp_level_validates_input() -> None:
    levels = [{"path": "a"}, {"path": "b"}]
    assert clamp_level(5, levels) == 1
    assert clamp_level(-2, levels) == 0
    assert clamp_level("1", levels) == 1
    with pytest.raises(ValueError):
        clamp_level("bad", levels)


def test_reduce_dims_update_batches_ledger_entries() -> None:
    scene, ledger, lock = _scene_and_ledger()
    baseline = NotifyDimsPayload.from_dict(
        {
            "ndisplay": 2,
            "mode": "plane",
            "current_step": [0, 0],
            "step": [0, 0],
            "order": [0, 1],
            "axis_labels": ["z", "t"],
            "displayed": [0, 1],
            "current_level": 0,
            "levels": [{"path": "lvl0"}],
            "level_shapes": [[8, 8]],
        }
    )
    ledger.batch_record_confirmed(
        _dims_entries_from_payload(baseline, axis_index=0, axis_target="z"),
        origin="test.bootstrap",
    )

    result = reduce_dims_update(
        scene,
        ledger,
        lock,
        axis="z",
        prop="index",
        value=3,
    )

    assert result.scope == "dims"
    assert result.value == 3
    step_entry = ledger.get("dims", "main", "current_step")
    assert step_entry is not None
    assert tuple(step_entry.value) == (3, 0)
    assert step_entry.metadata["axis_index"] == 0
    assert ledger.get("view", "main", "ndisplay") is not None


def test_reduce_multiscale_writes_ledger() -> None:
    scene, ledger, lock = _scene_and_ledger()

    policy = reduce_multiscale_policy(scene, ledger, lock, policy="oversampling")
    assert policy.value == "oversampling"
    assert ledger.get("multiscale", "main", "policy").value == "oversampling"

    level = reduce_multiscale_level(scene, ledger, lock, level=2)
    assert level.value == 2
    assert ledger.get("multiscale", "main", "level").value == 2


def test_reduce_camera_state_records_entries() -> None:
    ledger = ServerStateLedger()

    payload = reduce_camera_state(
        ledger,
        center=(1.0, 2.0, 3.0),
        zoom=5.0,
        angles=(0.0, 45.0, 90.0),
        timestamp=123.0,
    )

    assert payload["center"] == [1.0, 2.0, 3.0]
    assert payload["zoom"] == 5.0
    assert payload["angles"] == [0.0, 45.0, 90.0]

    assert ledger.get("camera", "main", "center").value == (1.0, 2.0, 3.0)
    assert ledger.get("camera", "main", "zoom").value == 5.0
    assert ledger.get("camera", "main", "angles").value == (0.0, 45.0, 90.0)


def test_normalize_clim_orders_min_max() -> None:
    lo, hi = normalize_clim(10, 2)
    assert (lo, hi) == (2.0, 10.0)


def test_reduce_functions_stamp_server_sequence() -> None:
    scene, ledger, lock = _scene_and_ledger()
    start = time.time()

    result = reduce_layer_property(
        scene,
        ledger,
        lock,
        layer_id="layer-10",
        prop="gamma",
        value=1.2,
    )

    assert result.server_seq > 0
    assert result.timestamp >= start
    assert ledger.get("layer", "layer-10", "gamma") is not None
