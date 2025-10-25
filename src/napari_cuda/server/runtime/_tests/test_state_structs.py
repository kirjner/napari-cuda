from napari_cuda.server.runtime.state_structs import ViewportState


def test_viewport_state_defaults() -> None:
    state = ViewportState()
    assert state.op_seq == -1
