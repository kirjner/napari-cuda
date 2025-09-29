"""Qt helpers used by the client stream loop."""

from __future__ import annotations

from qtpy import QtCore


class WakeProxy(QtCore.QObject):
    """Signal proxy to invoke a slot on the GUI thread from any thread."""

    trigger = QtCore.Signal()

    def __init__(self, slot, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.trigger.connect(slot)


class CallProxy(QtCore.QObject):
    """Queue callables onto the GUI thread via Qt signals."""

    call = QtCore.Signal(object)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.call.connect(self._on_call)

    def _on_call(self, fn) -> None:  # type: ignore[no-untyped-def]
        if callable(fn):
            fn()
