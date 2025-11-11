from __future__ import annotations

import pytest

from napari_cuda.server.control import state_reducers as reducers
from napari_cuda.server.scene import blocks as scene_blocks
from napari_cuda.server.ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import dims_spec_from_payload


def _enable_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reducers, "ENABLE_VIEW_AXES_INDEX_BLOCKS", True, raising=False)
    monkeypatch.setattr(scene_blocks, "ENABLE_VIEW_AXES_INDEX_BLOCKS", True, raising=False)


def _spec(ledger: ServerStateLedger):
    entry = ledger.get("dims", "main", "dims_spec")
    assert entry is not None and entry.value is not None
    spec = dims_spec_from_payload(entry.value)
    assert spec is not None
    return spec


def test_view_toggle_consumes_restore_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_blocks(monkeypatch)
    ledger = ServerStateLedger()
    # Bootstrap baseline dims: 2D, level 0, some shapes
    reducers.reduce_bootstrap_state(
        ledger,
        step=(2, 1, 0),
        axis_labels=("z", "y", "x"),
        order=(0, 1, 2),
        level_shapes=((64, 32, 16), (32, 16, 8), (16, 8, 4)),
        levels=(
            {"index": 0, "shape": [64, 32, 16]},
            {"index": 1, "shape": [32, 16, 8]},
            {"index": 2, "shape": [16, 8, 4]},
        ),
        current_level=0,
        ndisplay=2,
        origin="test.bootstrap",
    )

    # Seed a plane restore cache via reducer (also writes restore_cache.plane.state)
    reducers.reduce_plane_restore(
        ledger,
        level=1,
        step=(5, 3, 1),
        center=(12.0, 24.0),
        zoom=2.5,
        rect=(0.0, 0.0, 100.0, 80.0),
        origin="test.seed.plane",
    )

    # Change dims step, then seed a volume restore cache (index derived from current dims step)
    reducers.reduce_dims_update(ledger, axis=2, prop="index", value=2, origin="test.dims")
    reducers.reduce_volume_restore(
        ledger,
        level=2,
        center=(1.0, 2.0, 3.0),
        angles=(30.0, 10.0, 5.0),
        distance=15.0,
        fov=45.0,
        origin="test.seed.volume",
    )

    # Toggle to volume: expect dims/lod/index/camera to match volume cache
    reducers.reduce_view_update(ledger, ndisplay=3, origin="test.toggle.to_volume")
    spec = _spec(ledger)
    assert int(spec.ndisplay) == 3
    assert int(spec.current_level) == 2
    assert tuple(spec.current_step) == (5, 3, 2)

    cam_entry = ledger.get("camera", "main", "state")
    assert cam_entry is not None and cam_entry.value is not None
    cam_block = cam_entry.value
    assert tuple(cam_block["volume"]["center"]) == (1.0, 2.0, 3.0)
    assert tuple(cam_block["volume"]["angles"]) == (30.0, 10.0, 5.0)
    assert float(cam_block["volume"]["distance"]) == 15.0
    assert float(cam_block["volume"]["fov"]) == 45.0

    lod_entry = ledger.get("lod", "main", "state")
    assert lod_entry is not None and lod_entry.value is not None
    assert int(lod_entry.value["level"]) == 2

    index_entry = ledger.get("index", "main", "cursor")
    assert index_entry is not None and index_entry.value is not None
    assert tuple(index_entry.value["value"]) == tuple(int(v) for v in spec.current_step)

    # Toggle back to plane: expect dims/lod/index/camera to match plane cache
    reducers.reduce_view_update(ledger, ndisplay=2, origin="test.toggle.to_plane")
    spec = _spec(ledger)
    assert int(spec.ndisplay) == 2
    assert int(spec.current_level) == 1
    assert tuple(spec.current_step) == (5, 3, 2)

    cam_entry = ledger.get("camera", "main", "state")
    assert cam_entry is not None and cam_entry.value is not None
    cam_block = cam_entry.value
    assert tuple(cam_block["plane"]["center"]) == (12.0, 24.0)
    assert float(cam_block["plane"]["zoom"]) == 2.5
    assert tuple(cam_block["plane"]["rect"]) == (0.0, 0.0, 100.0, 80.0)
