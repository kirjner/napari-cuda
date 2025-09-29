"""Wake scheduling helpers for ClientStreamLoop."""

from __future__ import annotations

import logging
import time
from typing import Callable, TYPE_CHECKING

from qtpy import QtCore

from napari_cuda.client.streaming.client_loop.scheduler import WakeProxy


if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop


logger = logging.getLogger(__name__)


def init_wake_scheduler(loop: "ClientStreamLoop") -> Callable[[], None]:
    """Configure the QTimer-based wake scheduler and return the scheduling closure."""

    if loop._env_cfg.use_display_loop:  # noqa: SLF001
        loop._loop_state.wake_timer = None  # noqa: SLF001
        loop._loop_state.wake_proxy = None  # noqa: SLF001

        def _noop() -> None:
            return

        return _noop

    assert loop._canvas_native is not None, "Precise scheduling requires a native canvas"  # noqa: SLF001

    wake_timer = QtCore.QTimer(loop._canvas_native)  # noqa: SLF001
    wake_timer.setTimerType(QtCore.Qt.PreciseTimer)
    wake_timer.setSingleShot(True)
    wake_timer.timeout.connect(loop._on_present_timer)  # noqa: SLF001
    loop._loop_state.wake_timer = wake_timer  # noqa: SLF001

    wake_proxy = WakeProxy(lambda: schedule_next_wake(loop), loop._canvas_native)  # noqa: SLF001
    loop._loop_state.wake_proxy = wake_proxy  # noqa: SLF001

    loop._loop_state.gui_thread = loop._canvas_native.thread() if loop._canvas_native is not None else None  # noqa: SLF001

    def _schedule() -> None:
        schedule_next_wake(loop)

    return _schedule


def schedule_next_wake(loop: "ClientStreamLoop") -> None:
    if loop._env_cfg.use_display_loop:  # noqa: SLF001
        return
    earliest = loop._presenter.peek_next_due(loop._source_mux.active)  # noqa: SLF001
    if earliest is None:
        return
    now_mono = time.perf_counter()
    delta_ms = (float(earliest) - now_mono) * 1000.0 + float(loop._wake_fudge_ms or 0.0)  # noqa: SLF001
    if loop._loop_state.next_due_pending_until > now_mono:  # noqa: SLF001
        pending_ms = (loop._loop_state.next_due_pending_until - now_mono) * 1000.0  # noqa: SLF001
        if delta_ms >= (pending_ms - 0.5):
            return
    else:
        loop._loop_state.next_due_pending_until = 0.0  # noqa: SLF001
    wake_timer = loop._loop_state.wake_timer  # noqa: SLF001
    if wake_timer is not None and wake_timer.isActive():
        wake_timer.stop()
    loop._loop_state.next_due_pending_until = now_mono + max(0.0, delta_ms) / 1000.0  # noqa: SLF001
    if wake_timer is not None:
        wake_timer.start(max(0, int(delta_ms)))
    else:
        loop._scene_canvas_update()  # noqa: SLF001

