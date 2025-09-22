from __future__ import annotations

from napari_cuda.client.streaming.client_stream_loop import LoopState, ClientStreamLoop


def _make_loop() -> ClientStreamLoop:
    loop = ClientStreamLoop.__new__(ClientStreamLoop)
    loop._dims_meta = {}
    loop._dims_ready = False
    loop._log_dims_info = False
    loop._primary_axis_index = None
    loop._loop_state = LoopState()
    loop._viewer_mirror = lambda: None
    loop._mirror_dims_to_viewer = lambda *args, **kwargs: None
    loop._first_dims_ready_cb = None
    loop._first_dims_notified = False
    loop._ui_call = None
    loop._loop_state.last_dims_payload = None
    loop._compute_primary_axis_index = lambda: 0
    return loop


def test_handle_dims_update_clears_ack_and_caches_payload() -> None:
    loop = _make_loop()
    loop._loop_state.pending_intents = {41: {'kind': 'dims'}}

    payload = {
        'seq': 7,
        'current_step': [1, 2],
        'ndim': 3,
        'ndisplay': 2,
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'range': [(0, 10), (0, 5), (0, 3)],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
        'ack': True,
        'intent_seq': 41,
    }

    loop._handle_dims_update(payload)

    assert loop._loop_state.pending_intents == {}
    assert loop._loop_state.last_dims_seq == 7
    assert loop._loop_state.last_dims_payload == {
        'current_step': [1, 2],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 10), (0, 5), (0, 3)],
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
    }
    assert loop._dims_ready is True


def test_replay_last_dims_payload_forwards_to_viewer() -> None:
    mirror_calls: list[tuple] = []
    viewer = object()
    loop = _make_loop()
    loop._viewer_mirror = lambda: viewer

    def _mirror_dims_to_viewer(vm, cur, ndisp, ndim, dims_range, order, axis_labels, sizes, displayed):
        mirror_calls.append((vm, cur, ndisp, ndim, dims_range, order, axis_labels, sizes, displayed))

    loop._mirror_dims_to_viewer = _mirror_dims_to_viewer
    loop._loop_state.last_dims_payload = {
        'current_step': [3, 1],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 8), (0, 4), (0, 2)],
        'order': [0, 2, 1],
        'axis_labels': ['z', 'x', 'y'],
        'sizes': [9, 5, 3],
        'displayed': [2, 1],
    }

    loop._replay_last_dims_payload()

    assert mirror_calls == [
        (
            viewer,
            [3, 1],
            2,
            3,
            [(0, 8), (0, 4), (0, 2)],
            [0, 2, 1],
            ['z', 'x', 'y'],
            [9, 5, 3],
            [2, 1],
        )
    ]
