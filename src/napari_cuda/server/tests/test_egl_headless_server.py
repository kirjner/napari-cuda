from __future__ import annotations

import asyncio
from types import SimpleNamespace

from napari_cuda.server.app.egl_headless_server import EGLHeadlessServer
from napari_cuda.server.runtime.scene_ingest import RenderSceneSnapshot
from napari_cuda.server.control import control_channel_server
from napari_cuda.server.rendering.viewer_builder import CanonicalAxes


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
        self._canonical_axes = CanonicalAxes(
            ndim=3,
            axis_labels=("z", "y", "x"),
            order=(0, 1, 2),
            ndisplay=3,
            current_step=(0, 0, 0),
            ranges=((0.0, 7.0, 1.0), (0.0, 7.0, 1.0), (0.0, 7.0, 1.0)),
            sizes=(8, 8, 8),
        )

    def request_ndisplay(self, ndisplay: int) -> None:
        self.requested = int(ndisplay)
        self.use_volume = bool(ndisplay == 3)
        self.force_idr_called = True

    def force_idr(self) -> None:
        self.force_idr_called = True

    def viewer_model(self):  # type: ignore[no-untyped-def]
        return self._viewer

    def snapshot_dims_metadata(self) -> dict[str, object]:
        return {
            "ndim": 3,
            "ndisplay": int(self._viewer.dims.ndisplay),
            "mode": "volume" if self._viewer.dims.ndisplay == 3 else "plane",
            "current_step": [int(x) for x in self._viewer.dims.current_step],
            "axis_labels": ["z", "y", "x"],
            "order": [0, 1, 2],
            "level_shapes": [[8, 8, 8]],
            "current_level": 0,
        }


def test_set_ndisplay_switches_without_immediate_broadcast(monkeypatch) -> None:
    server = EGLHeadlessServer()
    server._scene.latest_state = RenderSceneSnapshot(current_step=(136, 0, 0))
    worker = _StubWorker()
    server._worker = worker

    broadcast_calls: list[tuple] = []

    async def _fake_broadcast(server_obj, *, payload, intent_id=None, timestamp=None, targets=None):  # type: ignore[no-untyped-def]
        broadcast_calls.append(payload)

    async def _noop(*args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(control_channel_server, '_broadcast_dims_state', _fake_broadcast)
    monkeypatch.setattr(server, '_broadcast_stream_config', _noop)
    monkeypatch.setattr(server, '_broadcast_state_binary', _noop, raising=False)

    result = asyncio.run(server._ingest_set_ndisplay(3, intent_id="test-view"))

    assert result.scope == "view"
    assert result.value == 3
    assert server.use_volume is True
    assert server._scene.latest_state.current_step == (136, 0, 0)
    assert worker.requested is None
    assert worker.force_idr_called is False
    assert broadcast_calls == []
