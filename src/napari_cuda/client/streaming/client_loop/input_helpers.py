"""Input sender and shortcut binding helpers for ClientStreamLoop."""

from __future__ import annotations

import logging
from typing import Callable, TYPE_CHECKING

from napari_cuda.client.streaming.input import InputSender


if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop


logger = logging.getLogger(__name__)


def attach_input_sender(loop: "ClientStreamLoop") -> None:
    state_channel = loop._loop_state.state_channel  # noqa: SLF001
    canvas_native = loop._canvas_native  # noqa: SLF001
    assert state_channel is not None, "State channel must be ready before InputSender attaches"
    assert canvas_native is not None, "InputSender requires a native canvas"

    sender = InputSender(
        widget=canvas_native,
        post=state_channel.post,
        max_rate_hz=loop._env_cfg.input_max_rate_hz,  # noqa: SLF001
        resize_debounce_ms=loop._env_cfg.resize_debounce_ms,  # noqa: SLF001
        on_wheel=loop._on_wheel,  # noqa: SLF001
        on_pointer=loop._on_pointer,  # noqa: SLF001
        on_key=loop._on_key_event,  # noqa: SLF001
        log_info=loop._env_cfg.input_log,  # noqa: SLF001
    )
    sender.start()
    loop._input_sender = sender  # noqa: SLF001
    logger.info("InputSender attached (wheel+resize+pointer)")


def bind_shortcuts(loop: "ClientStreamLoop") -> None:
    try:
        from qtpy import QtWidgets, QtGui, QtCore  # type: ignore
    except Exception:
        logger.debug("Shortcut binding skipped: Qt imports failed", exc_info=True)
        return

    parent = loop._canvas_native  # noqa: SLF001
    if parent is not None and hasattr(parent, 'window') and callable(parent.window):
        candidate = parent.window()
        parent = candidate if candidate is not None else parent
    if parent is None:
        parent = QtWidgets.QApplication.activeWindow()

    if parent is None:
        logger.warning("InputSender shortcuts skipped: no Qt window available")
        return

    shortcuts = [
        (QtCore.Qt.Key_Plus, lambda: loop._zoom_steps_at_center(+1)),  # noqa: SLF001
        (QtCore.Qt.Key_Equal, lambda: loop._zoom_steps_at_center(+1)),  # noqa: SLF001
        (QtCore.Qt.Key_Minus, lambda: loop._zoom_steps_at_center(-1)),  # noqa: SLF001
        (QtCore.Qt.Key_Home, lambda: (logger.info("shortcut: Home -> camera.reset"), loop._reset_camera())),  # noqa: SLF001
    ]

    created = []
    for key, cb in shortcuts:
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), parent)  # type: ignore
        sc.setContext(QtCore.Qt.ApplicationShortcut)  # type: ignore[attr-defined]
        sc.setAutoRepeat(True)
        sc.activated.connect(cb)  # type: ignore
        created.append(sc)

    loop._shortcuts = created  # noqa: SLF001
    logger.info("Shortcuts bound: +/-/=→zoom, Home→reset (arrows via keycb)")

