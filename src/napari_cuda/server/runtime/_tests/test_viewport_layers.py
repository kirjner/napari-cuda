from __future__ import annotations

import numpy as np
import pytest

from napari.layers.base._base_constants import Blending as NapariBlending
from napari.layers.image._image_constants import (
    ImageRendering as NapariImageRendering,
)
from napari_cuda.server.data import SliceROI
from napari_cuda.server.runtime.viewport.layers import (
    apply_slice_layer_data,
    apply_volume_layer_data,
)


class _FakeLayer:
    def __init__(self) -> None:
        self.visible = False
        self.opacity = 0.5
        self.blending = ""
        self.contrast_limits: list[float] | None = None
        self.depiction = ""
        self.rendering = ""
        self.scale: tuple[float, ...] = ()
        self.data = None
        self.translate = ()


def test_apply_slice_layer_data_updates_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeApplier:
        def __init__(self, layer: object) -> None:
            calls["layer"] = layer

        def apply(self, *, slab: object, roi: object, scale: tuple[float, float]) -> None:
            calls["slab"] = slab
            calls["roi"] = roi
            calls["scale"] = scale

    monkeypatch.setattr(
        "napari_cuda.server.runtime.viewport.layers.SliceDataApplier",
        _FakeApplier,
    )
    monkeypatch.setattr(
        "napari_cuda.server.runtime.viewport.layers.plane_scale_for_level",
        lambda _source, _level: (2.0, 3.0),
    )

    layer = _FakeLayer()
    slab = np.ones((4, 6), dtype=np.float32)
    roi = SliceROI(0, 4, 1, 7)

    sy, sx = apply_slice_layer_data(
        layer=layer,
        source=object(),
        level=2,
        slab=slab,
        roi=roi,
        update_contrast=True,
    )

    assert (sy, sx) == (2.0, 3.0)
    assert layer.visible is True
    assert layer.opacity == 1.0
    assert layer.blending == NapariBlending.OPAQUE.value
    assert layer.contrast_limits == [0.0, 1.0]

    assert calls["layer"] is layer
    assert calls["slab"] is slab
    assert calls["roi"] == roi
    assert calls["scale"] == (2.0, 3.0)


def test_apply_volume_layer_data_updates_layer() -> None:
    layer = _FakeLayer()
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))

    ensure_calls: list[str] = []

    def _ensure_volume_visual() -> None:
        ensure_calls.append("called")

    data_wh, depth = apply_volume_layer_data(
        layer=layer,
        volume=volume,
        contrast=(0.0, 10.0),
        scale=(0.5, 1.0, 2.0),
        ensure_volume_visual=_ensure_volume_visual,
    )

    assert ensure_calls == ["called"]
    assert layer.depiction == "volume"
    assert layer.rendering == NapariImageRendering.MIP.value
    assert layer.data is volume
    assert layer.translate == (0.0, 0.0, 0.0)
    assert layer.contrast_limits == [0.0, 10.0]
    assert layer.scale == (0.5, 1.0, 2.0)
    assert data_wh == (4, 3)
    assert depth == 2
