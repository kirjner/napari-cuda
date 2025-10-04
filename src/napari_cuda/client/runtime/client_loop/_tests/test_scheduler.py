"""Tests for the Qt scheduler proxies used by the client stream loop."""

from __future__ import annotations

import threading
import time

from qtpy import QtCore

from napari_cuda.client.runtime.client_loop.scheduler import CallProxy, WakeProxy


def _ensure_app() -> QtCore.QCoreApplication:
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtCore.QCoreApplication([])
    return app


def _wait_until(predicate, timeout_ms: int = 1000) -> None:
    app = _ensure_app()
    deadline = time.monotonic() + timeout_ms / 1000.0
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition not met before timeout")
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)


def test_call_proxy_runs_callable_on_gui_thread() -> None:
    """Emitting call() schedules the callable on the GUI thread."""

    gui_thread = QtCore.QThread.currentThread()
    observed: list[QtCore.QThread] = []

    proxy = CallProxy()

    proxy.call.emit(lambda: observed.append(QtCore.QThread.currentThread()))

    _wait_until(lambda: len(observed) == 1)

    assert observed[0] is gui_thread


def test_wake_proxy_relay_from_worker_thread() -> None:
    """WakeProxy relays emits from worker threads back onto the GUI thread."""

    gui_thread = QtCore.QThread.currentThread()
    observed: list[QtCore.QThread] = []
    wake_seen = threading.Event()

    def slot() -> None:
        observed.append(QtCore.QThread.currentThread())
        wake_seen.set()

    proxy = WakeProxy(slot)

    worker_done = threading.Event()

    def emit_from_worker() -> None:
        try:
            proxy.trigger.emit()
        finally:
            worker_done.set()

    thread = threading.Thread(target=emit_from_worker)
    thread.start()

    _wait_until(wake_seen.is_set)
    worker_done.wait(timeout=1.0)
    thread.join(timeout=1.0)

    assert observed[0] is gui_thread
