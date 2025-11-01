"""Input sender and shortcut binding helpers for ClientStreamLoop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari_cuda.client.runtime.input import InputSender

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.runtime.stream_runtime import ClientStreamLoop


logger = logging.getLogger(__name__)


def attach_input_sender(loop: ClientStreamLoop) -> None:
    state_channel = loop._loop_state.state_channel
    canvas_native = loop._canvas_native
    assert state_channel is not None, "State channel must be ready before InputSender attaches"
    assert canvas_native is not None, "InputSender requires a native canvas"

    sender = InputSender(
        widget=canvas_native,
        post=state_channel.post,
        max_rate_hz=loop._env_cfg.input_max_rate_hz,
        resize_debounce_ms=loop._env_cfg.resize_debounce_ms,
        on_wheel=loop._on_wheel,
        on_pointer=loop._on_pointer,
        on_key=loop._on_key_event,
        log_info=loop._env_cfg.input_log,
    )
    sender.start()
    loop._loop_state.input_sender = sender
    logger.info("InputSender attached (wheel+resize+pointer)")


def bind_shortcuts(loop: ClientStreamLoop) -> None:
    try:
        from qtpy import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception:
        logger.debug("Shortcut binding skipped: Qt imports failed", exc_info=True)
        return

    parent = loop._canvas_native
    if parent is not None and hasattr(parent, 'window') and callable(parent.window):
        candidate = parent.window()
        parent = candidate if candidate is not None else parent
    if parent is None:
        parent = QtWidgets.QApplication.activeWindow()

    if parent is None:
        logger.warning("InputSender shortcuts skipped: no Qt window available")
        return

    shortcuts = [
        (QtCore.Qt.Key_Plus, lambda: loop._zoom_steps_at_center(+1)),
        (QtCore.Qt.Key_Equal, lambda: loop._zoom_steps_at_center(+1)),
        (QtCore.Qt.Key_Minus, lambda: loop._zoom_steps_at_center(-1)),
        (QtCore.Qt.Key_Home, lambda: (logger.info("shortcut: Home -> camera.reset"), loop._reset_camera())),
    ]

    created = []
    for key, cb in shortcuts:
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), parent)  # type: ignore
        sc.setContext(QtCore.Qt.ApplicationShortcut)  # type: ignore[attr-defined]
        sc.setAutoRepeat(True)
        sc.activated.connect(cb)  # type: ignore
        created.append(sc)

    loop._loop_state.shortcuts = created
    logger.info("Shortcuts bound: +/-/=→zoom, Home→reset (arrows via keycb)")

