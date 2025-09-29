"""Qt bridge helpers for wake scheduling in ClientStreamLoop."""

from __future__ import annotations

from qtpy import QtCore


class WakeProxy(QtCore.QObject):
    """Qt signal proxy to safely schedule wakes from any thread."""

    trigger = QtCore.Signal()

    def __init__(self, slot, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.trigger.connect(slot)


class CallProxy(QtCore.QObject):
    """Post callables to the GUI thread via queued signal delivery."""

    call = QtCore.Signal(object)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.call.connect(self._on_call)

    def _on_call(self, fn) -> None:  # type: ignore[no-untyped-def]
        if callable(fn):
            fn()
