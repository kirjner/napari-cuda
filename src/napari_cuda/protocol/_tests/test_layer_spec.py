from __future__ import annotations

import pytest

from napari_cuda.protocol.messages import LayerSpec


def test_layer_spec_normalizes_axis_labels_length() -> None:
    payload = {
        "layer_id": "layer-0",
        "layer_type": "image",
        "name": "demo",
        "ndim": 3,
        "shape": [8, 16, 32],
        "axis_labels": ["axis -2", "axis -1"],
    }

    spec = LayerSpec.from_dict(payload)

    assert spec.axis_labels is not None
    assert len(spec.axis_labels) == spec.ndim == 3
    assert spec.axis_labels == ["z", "y", "x"]
