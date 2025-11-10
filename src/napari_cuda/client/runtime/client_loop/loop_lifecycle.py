"""Lifecycle helpers for ``ClientStreamLoop`` startup and shutdown."""

from __future__ import annotations

import logging
import queue
import time
from contextlib import suppress
from typing import TYPE_CHECKING

from qtpy import QtCore

from napari_cuda.client.rendering.presenter import Source
from napari_cuda.client.runtime.channel_threads import (
    ReceiveController,
    StateController,
)
from napari_cuda.client.runtime.eventloop_monitor import EventLoopMonitor

from .input_helpers import attach_input_sender, bind_shortcuts
from .scheduler_helpers import init_wake_scheduler
from .telemetry import start_metrics_timer, start_stats_timer

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from napari_cuda.client.runtime.stream_runtime import ClientStreamLoop


def start_loop(loop: ClientStreamLoop) -> None:
    """Bring the streaming loop online (state/pixel threads, timers, hooks)."""

    loop._stopped = False
    assert (
        loop._dims_mirror is not None and loop._dims_emitter is not None
    ), "dims mirror/emitter must be initialised"

    # State channel thread
    state_controller = StateController(
        loop.server_host,
        loop.state_port,
        ingest_notify_stream=loop._ingest_notify_stream,
        ingest_dims_notify=loop._ingest_notify_dims,
        ingest_notify_scene_snapshot=loop._ingest_notify_scene_snapshot,
        ingest_notify_layers=loop._ingest_notify_layers,
        ingest_notify_level=loop._ingest_notify_level,
        ingest_notify_camera=loop._ingest_notify_camera,
        ingest_ack_state=loop._ingest_ack_state,
        ingest_reply_command=loop._ingest_reply_command,
        ingest_error_command=loop._ingest_error_command,
        on_session_ready=loop._on_state_session_ready,
        on_connected=loop._on_state_connected,
        on_disconnect=loop._on_state_disconnect,
    )
    state_channel, t_state = state_controller.start()
    loop._loop_state.state_channel = state_channel
    loop._loop_state.state_thread = t_state
    loop._loop_state.threads.append(t_state)

    # Input wiring now that state channel exists
    loop._log_dims_info = bool(loop._env_cfg.input_log)  # type: ignore[attr-defined]
    loop._dims_mirror.set_logging(loop._log_dims_info)
    loop._dims_emitter.set_logging(loop._log_dims_info)
    if getattr(loop, "_layer_mirror", None) is not None:
        loop._layer_mirror.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    if getattr(loop, "_layer_emitter", None) is not None:
        loop._layer_emitter.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    if getattr(loop, "_camera_mirror", None) is not None:
        loop._camera_mirror.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    if getattr(loop, "_camera_emitter", None) is not None:
        loop._camera_emitter.set_logging(loop._log_dims_info)  # type: ignore[attr-defined]
    attach_input_sender(loop)
    bind_shortcuts(loop)

    # Pipelines ready to prime
    loop._schedule_next_wake = init_wake_scheduler(loop)
    loop._loop_state.vt_pipeline.start()
    loop._loop_state.pyav_pipeline.start()

    # Receiver thread
    receiver_controller = ReceiveController(
        loop.server_host,
        loop.server_port,
        on_connected=loop._handle_connected,
        on_frame=loop._on_frame,
        on_disconnect=loop._handle_disconnect,
    )
    receiver, t_rx = receiver_controller.start()
    loop._loop_state.pixel_receiver = receiver
    loop._loop_state.pixel_thread = t_rx
    loop._loop_state.threads.append(t_rx)

    # Telemetry timers
    loop._loop_state.stats_timer = start_stats_timer(
        loop._scene_canvas.native,
        stats_level=loop._stats_level,
        callback=loop._log_stats,
        logger=logger,
    )
    loop._loop_state.metrics_timer = start_metrics_timer(
        loop._scene_canvas.native,
        config=loop._telemetry_cfg,
        metrics=loop._loop_state.metrics,
        logger=logger,
    )

    # Optional draw watchdog
    if loop._watchdog_ms > 0:
        native_canvas = loop._scene_canvas.native
        assert native_canvas is not None, "Draw watchdog requires a native canvas"
        timer = QtCore.QTimer(native_canvas)
        timer.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]
        timer.setInterval(max(100, int(loop._watchdog_ms // 2) or 100))

        def _wd_tick() -> None:
            last = float(loop._loop_state.last_draw_pc or 0.0)
            if last <= 0.0:
                return
            now = time.perf_counter()
            if (now - last) * 1000.0 < float(loop._watchdog_ms):
                return
            if loop._presenter.peek_next_due(loop._source_mux.active) is None:
                return
            loop._scene_canvas_update()
            if loop._loop_state.metrics is not None:
                loop._loop_state.metrics.inc('napari_cuda_client_draw_watchdog_kicks', 1.0)

        timer.timeout.connect(_wd_tick)  # type: ignore[attr-defined]
        timer.start()
        loop._loop_state.watchdog_timer = timer
        logger.info("Draw watchdog enabled: threshold=%d ms", loop._watchdog_ms)

    # Optional event loop monitor
    if loop._evloop_stall_ms > 0:
        loop._loop_state.evloop_monitor = EventLoopMonitor(
            parent=loop._scene_canvas.native,
            metrics=loop._loop_state.metrics,
            stall_threshold_ms=int(loop._evloop_stall_ms),
            sample_interval_ms=int(loop._evloop_sample_ms),
            on_stall_kick=loop._scene_canvas_update,
        )

    # Ensure app quit hook stops the loop cleanly
    if not loop._quit_hook_connected:
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(loop.stop)  # type: ignore[attr-defined]
            loop._quit_hook_connected = True


def stop_loop(loop: ClientStreamLoop) -> None:
    """Tear down timers, pipelines, and cached frames for the loop."""

    # Timers + monitors
    if loop._loop_state.stats_timer is not None:
        loop._loop_state.stats_timer.stop()
        loop._loop_state.stats_timer = None
    if loop._loop_state.metrics_timer is not None:
        loop._loop_state.metrics_timer.stop()
        loop._loop_state.metrics_timer = None
    if loop._loop_state.warmup_reset_timer is not None:
        loop._loop_state.warmup_reset_timer.stop()
        loop._loop_state.warmup_reset_timer = None
    if loop._loop_state.watchdog_timer is not None:
        loop._loop_state.watchdog_timer.stop()
        loop._loop_state.watchdog_timer = None
    if loop._loop_state.evloop_monitor is not None:
        loop._loop_state.evloop_monitor.stop()
    loop._loop_state.evloop_monitor = None

    # Presenter + fallback cleanup
    if loop._warmup_policy is not None:
        from . import warmup  # local import to avoid cycle

        warmup.cancel(loop._loop_state, loop._presenter, loop._vt_latency_s)
    loop._presenter.clear()
    loop._presenter_facade.shutdown()

    # Drain any queued frames to release leases
    frame_queue = loop._loop_state.frame_queue
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

    lease = loop._loop_state.fallbacks.pop_vt_cache()
    if lease is not None:
        lease.close()
    loop._loop_state.fallbacks.clear_pyav()

    loop._loop_state.vt_pipeline.stop()
    loop._loop_state.pyav_pipeline.stop()

    if loop._vt_decoder is not None:
        loop._vt_decoder.flush()
        close_decoder = getattr(loop._vt_decoder, 'close', None)
        if callable(close_decoder):
            close_decoder()
        loop._vt_decoder = None

    # Clear handles but leave thread join to controller shutdown
    loop._loop_state.threads.clear()
    loop._loop_state.state_channel = None
    loop._loop_state.pixel_receiver = None
    loop._loop_state.state_session_metadata = None
    loop._loop_state.state_thread = None
    loop._loop_state.pixel_thread = None

    # Clear presenter fallbacks after pipelines halt
    loop._source_mux.set_active(Source.PYAV)
