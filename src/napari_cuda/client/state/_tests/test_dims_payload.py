from __future__ import annotations

from napari_cuda.client.state.dims_payload import inflate_current_step, normalize_meta


def test_normalize_meta_axes_sequence():
    meta_raw = {
        'axes': [
            {'label': 'z', 'index': 0},
            {'label': 'y', 'index': 1},
            {'label': 'x', 'index': 2},
        ],
        'displayed_axes': [1, 2],
    }
    meta = normalize_meta(meta_raw)
    assert meta['axis_labels'] == ['z', 'y', 'x']
    assert meta['order'] == [0, 1, 2]
    assert meta['displayed'] == [1, 2]


def test_normalize_meta_axes_mapping():
    meta_raw = {'axes': {'label': 'time', 'index': 3}}
    meta = normalize_meta(meta_raw)
    assert meta['axis_labels'] == ['time']
    assert meta['order'] == [3]


def test_inflate_current_step_pads_and_clamps():
    meta = {
        'ndim': 3,
        'range': [(0, 10), (0, 5), (0, 7)],
    }
    current = [8, 9]
    inflated = inflate_current_step(current, meta)
    assert inflated == [8, 5, 0]


def test_inflate_current_step_respects_order():
    meta = {
        'ndim': 3,
        'order': [2, 0, 1],
        'range': [(0, 99), (0, 50), (0, 50)],
    }
    current = [40, -3, 12]
    inflated = inflate_current_step(current[:2], meta)
    # First value targets axis 2, second targets axis 0 (per order mapping)
    assert inflated == [0, 0, 40]


def test_inflate_current_step_ignores_scalar_string():
    assert inflate_current_step('not-a-sequence', {'ndim': 3}) == 'not-a-sequence'

