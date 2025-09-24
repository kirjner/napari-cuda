from __future__ import annotations

import asyncio
from types import SimpleNamespace

from napari_cuda.server.egl_headless_server import EGLHeadlessServer
from napari_cuda.server.scene_state import ServerSceneState


class _StubWorker:
    def __init__(self) -> None:
        self.use_volume = False
        self._data_d = 8
        self._napari_layer = SimpleNamespace(ndim=3)
        self._scene_source = SimpleNamespace(current_step=(0, 0, 0))
        dims = SimpleNamespace(ndisplay=3, current_step=(0, 0, 0))
        self._viewer = SimpleNamespace(dims=dims)
        self.requested = None
        self.force_idr_called = False

    def request_ndisplay(self, ndisplay: int) -> None:
        self.requested = int(ndisplay)
        self.use_volume = bool(ndisplay == 3)

    def force_idr(self) -> None:
        self.force_idr_called = True

    def viewer_model(self):  # type: ignore[no-untyped-def]
        return self._viewer


def test_set_ndisplay_seeds_volume_step(monkeypatch) -> None:
    server = EGLHeadlessServer()
    server._latest_state = ServerSceneState(current_step=(136, 0, 0))
    worker = _StubWorker()
    server._worker = worker

    recorded_step = {}

    async def _fake_broadcast(step, *, last_client_id=None, ack=False):  # type: ignore[no-untyped-def]
        recorded_step['step'] = tuple(step)
        recorded_step['client'] = last_client_id
        recorded_step['ack'] = ack

    async def _noop(*args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(server, '_broadcast_dims_update', _fake_broadcast)
    monkeypatch.setattr(server, '_broadcast_state_json', _noop)
    monkeypatch.setattr(server, '_broadcast_state_binary', _noop, raising=False)

    asyncio.run(server._handle_set_ndisplay(3, client_id='c1', client_seq=1))

    assert server.use_volume is True
    assert server._latest_state.current_step == (0, 0, 0)
    assert worker.requested == 3
    assert worker.force_idr_called is True
    assert recorded_step['step'] == (0, 0, 0)
    assert recorded_step['client'] == 'c1'
    assert recorded_step['ack'] is True
