from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from napari_cuda.client.state import DimsBridge


class DummySignal:
    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, event=None) -> None:
        for callback in list(self._callbacks):
            callback(event)


class DummyDims:
    def __init__(self, ndim: int = 3) -> None:
        self.ndim = ndim
        self.current_step = tuple(0 for _ in range(ndim))
        self.ndisplay = 2
        self.range = tuple((0.0, 9.0, 1.0) for _ in range(ndim))
        self.order = tuple(range(ndim))
        self.axis_labels = tuple(f"axis-{idx}" for idx in range(ndim))
        self.point = tuple(float(x) for x in self.current_step)
        self.events = SimpleNamespace(
            current_step=DummySignal(),
            ndisplay=DummySignal(),
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper only
        return f"DummyDims(step={self.current_step}, ndisplay={self.ndisplay})"


class DummyQtDims:
    def __init__(self) -> None:
        self.is_playing = False


class DummyViewer:
    def __init__(self, ndim: int = 3) -> None:
        self.dims = DummyDims(ndim)
        self.window = SimpleNamespace(_qt_viewer=SimpleNamespace(dims=DummyQtDims()))


class DummySender:
    def __init__(self) -> None:
        self.index_calls: list[tuple[int, int, str]] = []
        self.step_calls: list[tuple[int, int, str]] = []
        self.ndisplay_calls: list[tuple[int, str]] = []
        self._primary_axis_index = 0

    def dims_set_index(self, axis: int, value: int, *, origin: str) -> bool:
        self.index_calls.append((axis, value, origin))
        return True

    def dims_step(self, axis: int, delta: int, *, origin: str) -> bool:
        self.step_calls.append((axis, delta, origin))
        return True

    def view_set_ndisplay(self, value: int, *, origin: str) -> bool:
        self.ndisplay_calls.append((value, origin))
        return True


class DummyTimer:
    def __init__(self) -> None:
        self.started_with: int | None = None

    def setSingleShot(self, *_args, **_kwargs) -> None:  # pragma: no cover - compatibility stub
        pass

    def setTimerType(self, *_args, **_kwargs) -> None:  # pragma: no cover - compatibility stub
        pass

    def timeout(self, *_args, **_kwargs) -> None:  # pragma: no cover - compatibility stub
        pass

    def start(self, interval: int) -> None:
        self.started_with = interval

    def stop(self) -> None:  # pragma: no cover - not needed for tests
        pass

    def deleteLater(self) -> None:  # pragma: no cover - not needed for tests
        pass


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("napari_cuda.tests.dims_bridge")


def test_handle_dims_change_sends_index(logger: logging.Logger) -> None:
    viewer = DummyViewer(ndim=3)
    sender = DummySender()
    sender._primary_axis_index = 2

    bridge = DimsBridge(viewer, logger=logger, tx_interval_ms=0)
    bridge.attach_state_sender(sender)

    viewer.dims.current_step = (0, 0, 5)
    bridge.handle_dims_change()

    assert sender.index_calls == [(2, 5, "ui")]
    assert bridge._last_step_ui == (0, 0, 5)


def test_handle_dims_change_in_play_mode_uses_delta(logger: logging.Logger) -> None:
    viewer = DummyViewer(ndim=2)
    sender = DummySender()
    sender._primary_axis_index = 1

    bridge = DimsBridge(viewer, logger=logger, tx_interval_ms=0)
    bridge.attach_state_sender(sender)

    bridge._last_step_ui = (0, 0)
    viewer.dims.current_step = (0, 3)
    viewer.window._qt_viewer.dims.is_playing = True

    bridge.handle_dims_change()

    assert sender.step_calls == [(1, 3, "play")]
    assert sender.index_calls == []
    assert bridge._last_step_ui == (0, 3)


def test_coalesced_updates_flush_to_sender(logger: logging.Logger) -> None:
    viewer = DummyViewer(ndim=1)
    sender = DummySender()

    bridge = DimsBridge(viewer, logger=logger, tx_interval_ms=5)
    bridge._state_sender = sender
    bridge._dims_tx_timer = DummyTimer()

    bridge._last_step_ui = (0,)
    viewer.dims.current_step = (4,)

    bridge.handle_dims_change()
    assert sender.index_calls == []  # still queued

    bridge._flush_pending()
    assert sender.index_calls == [(0, 4, "ui")]


def test_apply_remote_updates_viewer(logger: logging.Logger) -> None:
    viewer = DummyViewer(ndim=3)
    sender = DummySender()

    bridge = DimsBridge(viewer, logger=logger)
    bridge.attach_state_sender(sender)

    bridge.apply_remote(
        current_step=(2, 1, 0),
        ndisplay=3,
        ndim=3,
        dims_range=[(0, 9, 1), (0, 4, 1), (0, 7, 2)],
        order=[2, 1, 0],
        axis_labels=["z", "y", "x"],
        displayed=[1, 2],
    )

    assert viewer.dims.current_step == (2, 1, 0)
    assert viewer.dims.ndisplay == 3
    assert viewer.dims.axis_labels == ("z", "y", "x")
    assert viewer.dims.order == (0, 1, 2)
    assert bridge.suppress_forward is False
