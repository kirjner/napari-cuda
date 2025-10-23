"""Worker lifecycle orchestration for the EGL headless server."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from napari_cuda.server.control import pixel_channel
from napari_cuda.server.rendering.bitstream import build_avcc_config, pack_to_avcc
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.render_ledger_snapshot import pull_render_snapshot
from napari_cuda.server.runtime.egl_worker import EGLRendererWorker
from napari_cuda.server.runtime.camera_pose import CameraPoseApplied
from napari_cuda.server.runtime.intents import LevelSwitchIntent
from napari_cuda.server.rendering.debug_tools import DebugDumper


logger = logging.getLogger(__name__)
@dataclass
class WorkerLifecycleState:
    """Track the worker thread and its stop signal."""

    worker: Optional[EGLRendererWorker] = None
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    ready_event: threading.Event = field(default_factory=threading.Event)
    ready_async_event: Optional[asyncio.Event] = None


def start_worker(server: object, loop: asyncio.AbstractEventLoop, state: WorkerLifecycleState) -> None:
    """Launch the EGL render worker on its dedicated thread."""

    if state.thread and state.thread.is_alive():
        raise RuntimeError("worker thread already running")

    state.stop_event.clear()
    state.ready_event.clear()
    state.ready_async_event = asyncio.Event()

    def on_frame(payload_obj, _flags: int, capture_wall_ts: Optional[float] = None, seq: Optional[int] = None) -> None:
        worker = state.worker
        if server._ctx.debug_policy.encoder.log_nals:
            from .bitstream import parse_nals

            if isinstance(payload_obj, (bytes, bytearray, memoryview)):
                raw_bytes = bytes(payload_obj)
            elif isinstance(payload_obj, (list, tuple)):
                raw_bytes = b"".join(bytes(x) for x in payload_obj if x is not None)
            else:
                raw_bytes = bytes(payload_obj) if payload_obj is not None else b""
            nals = parse_nals(raw_bytes)
            has_sps = any(((n[0] & 0x1F) == 7) or (((n[0] >> 1) & 0x3F) == 33) for n in nals if n)
            has_pps = any(((n[0] & 0x1F) == 8) or (((n[0] >> 1) & 0x3F) == 34) for n in nals if n)
            has_idr = any(((n[0] & 0x1F) == 5) or (((n[0] >> 1) & 0x3F) in (19, 20, 21)) for n in nals if n)
            logger.debug("Raw NALs: count=%d sps=%s pps=%s idr=%s", len(nals), has_sps, has_pps, has_idr)

        t_pack_start = time.perf_counter()
        avcc_pkt, is_key = pack_to_avcc(
            payload_obj,
            server._param_cache,
            encoder_logging=server._ctx.debug_policy.encoder,
        )
        pack_ms = (time.perf_counter() - t_pack_start) * 1000.0
        server.metrics.observe_ms("napari_cuda_pack_ms", pack_ms)

        if server._ctx.debug_policy.encoder.log_nals and avcc_pkt is not None:
            from .bitstream import parse_nals

            nals_after = parse_nals(avcc_pkt)
            has_idr_after = any(((n[0] & 0x1F) == 5) or (((n[0] >> 1) & 0x3F) in (19, 20, 21)) for n in nals_after if n)
            if bool(has_idr_after) != bool(is_key):
                logger.warning(
                    "Keyframe detect mismatch: parse=%s is_key=%s nals_after=%d",
                    has_idr_after,
                    is_key,
                    len(nals_after),
                )

        if not avcc_pkt:
            return

        avcc_cfg = build_avcc_config(server._param_cache)
        if avcc_cfg is not None:
            def _publish_config(avcc_bytes: bytes) -> None:
                server._schedule_coro(
                    pixel_channel.publish_avcc(
                        server._pixel_channel,
                        config=server._pixel_config,
                        metrics=server.metrics,
                        avcc=avcc_bytes,
                        send_stream=server._broadcast_stream_config,
                    ),
                    "notify_stream",
                )

            loop.call_soon_threadsafe(_publish_config, avcc_cfg)

        if server._dump_remaining > 0:
            try:
                os.makedirs(server._dump_dir, exist_ok=True)
                if not server._dump_path:
                    stamp = int(time.time())
                    server._dump_path = os.path.join(
                        server._dump_dir,
                        f"dump_{server.width}x{server.height}_{stamp}.h264",
                    )
                with open(server._dump_path, "ab") as fh:
                    fh.write(avcc_pkt)
                server._dump_remaining -= 1
                if server._dump_remaining == 0:
                    logger.info("Bitstream dump complete: %s", server._dump_path)
            except OSError as exc:
                logger.debug("Bitstream dump error: %s", exc)

        stamp_ts = time.time()
        seq_val = server._seq & 0xFFFFFFFF
        server._seq = (server._seq + 1) & 0xFFFFFFFF
        flags = 0x01 if is_key else 0
        packet = (avcc_pkt, flags, seq_val, stamp_ts)

        def _enqueue() -> None:
            pixel_channel.enqueue_frame(
                server._pixel_channel,
                packet,
                metrics=server.metrics,
            )

        loop.call_soon_threadsafe(_enqueue)

    def worker_loop() -> None:
        try:
            control_loop = loop

            def _forward_level_intent(intent: LevelSwitchIntent) -> None:
                server._worker_intents.enqueue_level_switch(intent)
                control_loop.call_soon_threadsafe(  # type: ignore[attr-defined]
                    server._handle_worker_level_intents,  # type: ignore[attr-defined]
                )

            def _forward_camera_pose(pose: CameraPoseApplied) -> None:
                control_loop.call_soon_threadsafe(  # type: ignore[attr-defined]
                    server._commit_applied_camera,  # type: ignore[attr-defined]
                    pose,
                )

            worker = EGLRendererWorker(
                width=server.width,
                height=server.height,
                use_volume=server.use_volume,
                fps=server.cfg.fps,
                animate=server._animate,
                animate_dps=server._animate_dps,
                zarr_path=server._zarr_path,
                zarr_level=server._zarr_level,
                zarr_axes=server._zarr_axes,
                zarr_z=server._zarr_z,
                policy_name=server._scene.multiscale_state.get("policy"),
                camera_pose_cb=_forward_camera_pose,
                level_intent_cb=_forward_level_intent,
                camera_queue=server._camera_queue,
                ctx=server._ctx,
                env=server._ctx_env,
            )
            state.worker = worker
            worker.attach_ledger(server._state_ledger)  # type: ignore[attr-defined]

            worker._init_cuda()
            worker._init_vispy_scene()
            worker._init_egl()
            worker._init_capture()
            worker._init_cuda_interop()
            worker._init_encoder()

            worker._debug = DebugDumper(worker._debug_config)
            if worker._debug.cfg.enabled:
                worker._debug.log_env_once()
                worker._debug.ensure_out_dir()
            worker._capture.pipeline.set_debug(worker._debug)
            worker._capture.pipeline.set_raw_dump_budget(worker._raw_dump_budget)
            worker._capture.cuda.set_force_tight_pitch(worker._debug_policy.worker.force_tight_pitch)
            worker._log_debug_policy_once()
            logger.info(
                "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s, zarr=%s",
                worker.width,
                worker.height,
                worker._capture.pipeline.enc_input_format,
                worker.fps,
                worker._animate,
                bool(worker._zarr_path),
            )

            initial_snapshot = getattr(server, "_bootstrap_snapshot", None)
            if initial_snapshot is None:
                initial_snapshot = pull_render_snapshot(server)
            else:
                server._bootstrap_snapshot = None  # type: ignore[attr-defined]
            worker._consume_render_snapshot(initial_snapshot)
            worker.drain_scene_updates()

            # Mark server-ready AFTER metadata is available, BEFORE worker is_ready/refresh
            state.ready_event.set()
            ready_async = state.ready_async_event
            if ready_async is not None:
                control_loop.call_soon_threadsafe(ready_async.set)

            # Publish initial avcC if available, then flip worker ready and push first refresh
            avcc_cfg = build_avcc_config(server._param_cache)
            if avcc_cfg is not None:
                def _publish_start_config(avcc_bytes: bytes) -> None:
                    server._schedule_coro(
                        pixel_channel.publish_avcc(
                            server._pixel_channel,
                            config=server._pixel_config,
                            metrics=server.metrics,
                            avcc=avcc_bytes,
                            send_stream=server._broadcast_stream_config,
                        ),
                        "notify_stream_bootstrap",
                    )
                control_loop.call_soon_threadsafe(_publish_start_config, avcc_cfg)

            # Now flip worker ready
            worker._is_ready = True

            tick = 1.0 / max(1, server.cfg.fps)
            next_tick = time.perf_counter()

            while not state.stop_event.is_set():
                snapshot = pull_render_snapshot(server)
                has_camera_deltas = len(server._camera_queue) > 0

                # Request view ndisplay if provided by staged intents
                if has_camera_deltas and server._log_state_traces:
                    logger.info("frame camera deltas snapshot pending")

                frame_state = snapshot

                if has_camera_deltas and (server._log_cam_info or server._log_cam_debug):
                    message = "apply: cam deltas pending"
                    if server._log_cam_info:
                        logger.info(message)
                    else:
                        logger.debug(message)

                # Skip entire pipeline if no new desires, no animation/commands, and no explicit render tick
                broadcast_state = getattr(server, "_pixel_channel", None)
                broadcast = getattr(broadcast_state, "broadcast", None)
                clients_connected = bool(getattr(broadcast, "clients", set()))
                waiting_for_keyframe = bool(getattr(broadcast, "waiting_for_keyframe", False))
                logger.debug(
                    "frame state clients=%d waiting_for_key=%s",
                    len(getattr(broadcast, "clients", set())),
                    waiting_for_keyframe,
                )
                if (
                    not clients_connected
                    and not has_camera_deltas
                    and not server._animate
                    and not getattr(worker, "_render_tick_required", False)
                    and not waiting_for_keyframe
                ):
                    logger.debug("frame skip tick: idle (no clients)")
                    next_tick += tick
                    sleep_duration = next_tick - time.perf_counter()
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                    else:
                        next_tick = time.perf_counter()
                    continue

                logger.debug(
                    "frame apply proceeding camera_deltas_pending=%s animate=%s",
                    bool(has_camera_deltas),
                    server._animate,
                )
                worker._consume_render_snapshot(
                    frame_state,
                    apply_camera_pose=not has_camera_deltas,
                )

                timings, packet, flags, seq = worker.capture_and_encode_packet()

                server.metrics.observe_ms("napari_cuda_render_ms", timings.render_ms)
                if timings.blit_gpu_ns is not None:
                    server.metrics.observe_ms(
                        "napari_cuda_capture_blit_ms",
                        timings.blit_gpu_ns / 1e6,
                    )
                server.metrics.observe_ms(
                    "napari_cuda_capture_blit_cpu_ms",
                    timings.blit_cpu_ms,
                )
                server.metrics.observe_ms("napari_cuda_map_ms", timings.map_ms)
                server.metrics.observe_ms("napari_cuda_copy_ms", timings.copy_ms)
                server.metrics.observe_ms("napari_cuda_convert_ms", timings.convert_ms)
                server.metrics.observe_ms("napari_cuda_encode_ms", timings.encode_ms)
                server.metrics.observe_ms("napari_cuda_pack_ms", timings.pack_ms)
                server.metrics.observe_ms("napari_cuda_total_ms", timings.total_ms)

                # Clear one-shot render tick requirement if set
                if getattr(worker, "_render_tick_required", False):
                    worker._mark_render_tick_complete()

                on_frame(packet, flags, timings.capture_wall_ts, seq)

                next_tick += tick
                sleep_duration = next_tick - time.perf_counter()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                else:
                    next_tick = time.perf_counter()
        except Exception as exc:
            logger.exception("Render worker error: %s", exc)
        finally:
            worker = state.worker
            if worker is not None:
                try:
                    worker.cleanup()
                except Exception as cleanup_exc:  # pragma: no cover - defensive cleanup
                    logger.debug("Worker cleanup error: %s", cleanup_exc)
            state.worker = None
            state.ready_event.clear()
            state.ready_async_event = None

    thread = threading.Thread(target=worker_loop, name="egl-render", daemon=True)
    state.thread = thread
    thread.start()



def stop_worker(state: WorkerLifecycleState) -> None:
    """Signal the render worker to stop and wait for the thread to exit."""

    state.stop_event.set()
    state.ready_event.clear()
    state.ready_async_event = None
    thread = state.thread
    if thread and thread.is_alive():
        thread.join(timeout=3.0)
    state.thread = None
    state.worker = None
