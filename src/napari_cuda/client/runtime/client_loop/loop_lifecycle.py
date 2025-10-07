"""Lifecycle helpers for ``ClientStreamLoop`` startup and shutdown."""

from __future__ import annotations

import logging
import queue
import time
from contextlib import suppress
from typing import TYPE_CHECKING

from qtpy import QtCore

from napari_cuda.client.runtime.channel_threads import ReceiveController, StateController
from napari_cuda.client.runtime.eventloop_monitor import EventLoopMonitor
from napari_cuda.client.rendering.presenter import Source

from .input_helpers import attach_input_sender, bind_shortcuts
from .scheduler_helpers import init_wake_scheduler
from .telemetry import start_metrics_timer, start_stats_timer

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from napari_cuda.client.runtime.stream_runtime import ClientStreamLoop


def start_loop(loop: "ClientStreamLoop") -> None:
    """Bring the streaming loop online (state/pixel threads, timers, hooks)."""

    loop._stopped = False  # noqa: SLF001
    assert (
        loop._dims_mirror is not None and loop._dims_emitter is not None
    ), "dims mirror/emitter must be initialised"

    # State channel thread
    state_controller = StateController(
        loop.server_host,
        loop.state_port,
        ingest_notify_stream=loop._ingest_notify_stream,  # noqa: SLF001
        ingest_dims_notify=loop._ingest_notify_dims,  # noqa: SLF001
        ingest_notify_scene_snapshot=loop._ingest_notify_scene_snapshot,  # noqa: SLF001
        ingest_notify_layers=loop._ingest_notify_layers,  # noqa: SLF001
        ingest_notify_camera=loop._ingest_notify_camera,  # noqa: SLF001
        ingest_ack_state=loop._ingest_ack_state,  # noqa: SLF001
        ingest_reply_command=loop._ingest_reply_command,  # noqa: SLF001
        ingest_error_command=loop._ingest_error_command,  # noqa: SLF001
        on_session_ready=loop._on_state_session_ready,  # noqa: SLF001
        on_connected=loop._on_state_connected,  # noqa: SLF001
        on_disconnect=loop._on_state_disconnect,  # noqa: SLF001
    )
    state_channel, t_state = state_controller.start()
    loop._loop_state.state_channel = state_channel  # noqa: SLF001
    loop._loop_state.state_thread = t_state  # noqa: SLF001
    loop._loop_state.threads.append(t_state)  # noqa: SLF001

    # Input wiring now that state channel exists
    loop._log_dims_info = bool(loop._env_cfg.input_log)  # type: ignore[attr-defined]  # noqa: SLF001
    loop._dims_mirror.set_logging(loop._log_dims_info)  # noqa: SLF001
    loop._dims_emitter.set_logging(loop._log_dims_info)  # noqa: SLF001
    if getattr(loop, "_layer_mirror", None) is not None:  # noqa: SLF001
        loop._layer_mirror.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    if getattr(loop, "_layer_emitter", None) is not None:  # noqa: SLF001
        loop._layer_emitter.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    if getattr(loop, "_camera_mirror", None) is not None:  # noqa: SLF001
        loop._camera_mirror.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    if getattr(loop, "_camera_emitter", None) is not None:  # noqa: SLF001
        loop._camera_emitter.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    attach_input_sender(loop)
    bind_shortcuts(loop)

    # Pipelines ready to prime
    loop._schedule_next_wake = init_wake_scheduler(loop)  # noqa: SLF001
    loop._loop_state.vt_pipeline.start()  # noqa: SLF001
    loop._loop_state.pyav_pipeline.start()  # noqa: SLF001

    # Receiver thread
    receiver_controller = ReceiveController(
        loop.server_host,
        loop.server_port,
        on_connected=loop._handle_connected,  # noqa: SLF001
        on_frame=loop._on_frame,  # noqa: SLF001
        on_disconnect=loop._handle_disconnect,  # noqa: SLF001
    )
    receiver, t_rx = receiver_controller.start()
    loop._loop_state.pixel_receiver = receiver  # noqa: SLF001
    loop._loop_state.pixel_thread = t_rx  # noqa: SLF001
    loop._loop_state.threads.append(t_rx)  # noqa: SLF001

    # Telemetry timers
    loop._loop_state.stats_timer = start_stats_timer(  # noqa: SLF001
        loop._scene_canvas.native,  # noqa: SLF001
        stats_level=loop._stats_level,  # noqa: SLF001
        callback=loop._log_stats,  # noqa: SLF001
        logger=logger,
    )
    loop._loop_state.metrics_timer = start_metrics_timer(  # noqa: SLF001
        loop._scene_canvas.native,  # noqa: SLF001
        config=loop._telemetry_cfg,  # noqa: SLF001
        metrics=loop._loop_state.metrics,  # noqa: SLF001
        logger=logger,
    )

    # Optional draw watchdog
    if loop._watchdog_ms > 0:  # noqa: SLF001
        native_canvas = loop._scene_canvas.native  # noqa: SLF001
        assert native_canvas is not None, "Draw watchdog requires a native canvas"
        timer = QtCore.QTimer(native_canvas)
        timer.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]
        timer.setInterval(max(100, int(loop._watchdog_ms // 2) or 100))  # noqa: SLF001

        def _wd_tick() -> None:
            last = float(loop._loop_state.last_draw_pc or 0.0)  # noqa: SLF001
            if last <= 0.0:
                return
            now = time.perf_counter()
            if (now - last) * 1000.0 < float(loop._watchdog_ms):  # noqa: SLF001
                return
            if loop._presenter.peek_next_due(loop._source_mux.active) is None:  # noqa: SLF001
                return
            loop._scene_canvas_update()  # noqa: SLF001
            if loop._loop_state.metrics is not None:  # noqa: SLF001
                loop._loop_state.metrics.inc('napari_cuda_client_draw_watchdog_kicks', 1.0)

        timer.timeout.connect(_wd_tick)  # type: ignore[attr-defined]
        timer.start()
        loop._loop_state.watchdog_timer = timer  # noqa: SLF001
        logger.info("Draw watchdog enabled: threshold=%d ms", loop._watchdog_ms)  # noqa: SLF001

    # Optional event loop monitor
    if loop._evloop_stall_ms > 0:  # noqa: SLF001
        loop._loop_state.evloop_monitor = EventLoopMonitor(  # noqa: SLF001
            parent=loop._scene_canvas.native,  # noqa: SLF001
            metrics=loop._loop_state.metrics,  # noqa: SLF001
            stall_threshold_ms=int(loop._evloop_stall_ms),  # noqa: SLF001
            sample_interval_ms=int(loop._evloop_sample_ms),  # noqa: SLF001
            on_stall_kick=loop._scene_canvas_update,  # noqa: SLF001
        )

    # Ensure app quit hook stops the loop cleanly
    if not loop._quit_hook_connected:  # noqa: SLF001
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(loop.stop)  # type: ignore[attr-defined]
            loop._quit_hook_connected = True  # noqa: SLF001


def stop_loop(loop: "ClientStreamLoop") -> None:
    """Tear down timers, pipelines, and cached frames for the loop."""

    # Timers + monitors
    if loop._loop_state.stats_timer is not None:  # noqa: SLF001
        loop._loop_state.stats_timer.stop()
        loop._loop_state.stats_timer = None
    if loop._loop_state.metrics_timer is not None:  # noqa: SLF001
        loop._loop_state.metrics_timer.stop()
        loop._loop_state.metrics_timer = None
    if loop._loop_state.warmup_reset_timer is not None:  # noqa: SLF001
        loop._loop_state.warmup_reset_timer.stop()
        loop._loop_state.warmup_reset_timer = None
    if loop._loop_state.watchdog_timer is not None:  # noqa: SLF001
        loop._loop_state.watchdog_timer.stop()
        loop._loop_state.watchdog_timer = None
    if loop._loop_state.evloop_monitor is not None:  # noqa: SLF001
        loop._loop_state.evloop_monitor.stop()
    loop._loop_state.evloop_monitor = None

    # Presenter + fallback cleanup
    if loop._warmup_policy is not None:  # noqa: SLF001
        from . import warmup  # local import to avoid cycle

        warmup.cancel(loop._loop_state, loop._presenter, loop._vt_latency_s)  # noqa: SLF001
    loop._presenter.clear()  # noqa: SLF001
    loop._presenter_facade.shutdown()  # noqa: SLF001

    # Drain any queued frames to release leases
    frame_queue = loop._loop_state.frame_queue  # noqa: SLF001
    if frame_queue is not None:
        while True:
            with suppress(queue.Empty):
                frame = frame_queue.get_nowait()
                if isinstance(frame, tuple) and len(frame) == 2:
                    payload, release_cb = frame  # type: ignore[assignment]
                    if release_cb is not None:
                        release_cb(payload)  # type: ignore[misc]
                continue
            break

    lease = loop._loop_state.fallbacks.pop_vt_cache()  # noqa: SLF001
    if lease is not None:
        lease.close()
    loop._loop_state.fallbacks.clear_pyav()  # noqa: SLF001

    loop._loop_state.vt_pipeline.stop()  # noqa: SLF001
    loop._loop_state.pyav_pipeline.stop()  # noqa: SLF001

    if loop._vt_decoder is not None:  # noqa: SLF001
        loop._vt_decoder.flush()
        close_decoder = getattr(loop._vt_decoder, 'close', None)
        if callable(close_decoder):
            close_decoder()
        loop._vt_decoder = None  # noqa: SLF001

    # Clear handles but leave thread join to controller shutdown
    loop._loop_state.threads.clear()  # noqa: SLF001
    loop._loop_state.state_channel = None  # noqa: SLF001
    loop._loop_state.pixel_receiver = None  # noqa: SLF001
    loop._loop_state.state_session_metadata = None  # noqa: SLF001
    loop._loop_state.state_thread = None  # noqa: SLF001
    loop._loop_state.pixel_thread = None  # noqa: SLF001

    # Clear presenter fallbacks after pipelines halt
    loop._source_mux.set_active(Source.PYAV)  # noqa: SLF001
