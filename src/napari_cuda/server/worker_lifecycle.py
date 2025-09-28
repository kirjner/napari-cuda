"""Worker lifecycle orchestration for the EGL headless server."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, List

from . import pixel_channel
from .bitstream import build_avcc_config, pack_to_avcc
from .scene_state import ServerSceneState
from .state_channel_handler import build_dims_update_message
from .worker_notifications import WorkerSceneNotification
from .egl_worker import EGLRendererWorker

logger = logging.getLogger(__name__)


@dataclass
class WorkerLifecycleState:
    """Track the worker thread and its stop signal."""

    worker: Optional[EGLRendererWorker] = None
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)


def start_worker(server: object, loop: asyncio.AbstractEventLoop, state: WorkerLifecycleState) -> None:
    """Launch the EGL render worker on its dedicated thread."""

    if state.thread and state.thread.is_alive():
        raise RuntimeError("worker thread already running")

    state.stop_event.clear()

    def on_frame(payload_obj, _flags: int, capture_wall_ts: Optional[float] = None, seq: Optional[int] = None) -> None:
        worker = state.worker
        if worker is not None and worker._orientation_ready is False:
            return

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
            def _send_config(avcc_bytes: bytes) -> None:
                server._schedule_coro(
                    pixel_channel.maybe_send_video_config(
                        server._pixel_channel,
                        config=server._pixel_config,
                        metrics=server.metrics,
                        avcc=avcc_bytes,
                        send_state_json=server._broadcast_state_json,
                    ),
                    "video_config",
                )

            loop.call_soon_threadsafe(_send_config, avcc_cfg)

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
            def _on_scene_refresh(step: object = None) -> None:
                if isinstance(step, Sequence):
                    chosen = tuple(int(x) for x in step)
                    with server._state_lock:
                        snapshot = server._scene.latest_state
                        server._scene.latest_state = ServerSceneState(
                            center=snapshot.center,
                            zoom=snapshot.zoom,
                            angles=snapshot.angles,
                            current_step=chosen,
                        )
                    server._worker_notifications.push(
                        WorkerSceneNotification(
                            kind="dims_update",
                            step=chosen,
                            last_client_id=None,
                            ack=True,
                            intent_seq=None,
                        )
                    )
                else:
                    server._worker_notifications.push(
                        WorkerSceneNotification(
                            kind="meta_refresh",
                            last_client_id=None,
                        )
                    )
                loop.call_soon_threadsafe(server._process_worker_notifications)

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
                scene_refresh_cb=_on_scene_refresh,
                ctx=server._ctx,
                env=server._ctx_env,
            )
            state.worker = worker

            z_index = worker._z_index
            if z_index is not None:
                with server._state_lock:
                    latest = server._scene.latest_state
                    server._scene.latest_state = ServerSceneState(
                        center=latest.center,
                        zoom=latest.zoom,
                        angles=latest.angles,
                        current_step=(int(z_index),),
                    )
                baseline = build_dims_update_message(
                    server,
                    step_list=[int(z_index)],
                    last_client_id=None,
                    ack=False,
                    intent_seq=None,
                )
                loop.call_soon_threadsafe(
                    lambda obj=baseline: asyncio.create_task(server._broadcast_state_json(obj))
                )
                if server._log_dims_info:
                    logger.info("init: dims.update current_step=%s", baseline.get("current_step"))
                else:
                    logger.debug("init: dims.update current_step=%s", baseline.get("current_step"))
            else:
                meta = server._dims_metadata() or {}
                ndim = int(meta.get("ndim") or 3)
                baseline = build_dims_update_message(
                    server,
                    step_list=[0 for _ in range(max(1, ndim))],
                    last_client_id=None,
                    ack=False,
                    intent_seq=None,
                )
                loop.call_soon_threadsafe(
                    lambda obj=baseline: asyncio.create_task(server._broadcast_state_json(obj))
                )
                if server._log_dims_info:
                    logger.info(
                        "init: dims.update current_step=%s (baseline volume)",
                        baseline.get("current_step"),
                    )
                else:
                    logger.debug(
                        "init: dims.update current_step=%s (baseline volume)",
                        baseline.get("current_step"),
                    )

            tick = 1.0 / max(1, server.cfg.fps)
            next_tick = time.perf_counter()

            while not state.stop_event.is_set():
                with server._state_lock:
                    queued = server._scene.latest_state
                    commands = list(server._scene.camera_commands)
                    server._scene.camera_commands.clear()
                    server._scene.latest_state = ServerSceneState(
                        center=queued.center,
                        zoom=queued.zoom,
                        angles=queued.angles,
                        current_step=queued.current_step,
                        layer_updates=None,
                    )

                if commands and server._log_state_traces:
                    logger.info("frame commands snapshot count=%d", len(commands))

                frame_state = ServerSceneState(
                    center=queued.center,
                    zoom=queued.zoom,
                    angles=queued.angles,
                    current_step=queued.current_step,
                    volume_mode=queued.volume_mode,
                    volume_colormap=queued.volume_colormap,
                    volume_clim=queued.volume_clim,
                    volume_opacity=queued.volume_opacity,
                    volume_sample_step=queued.volume_sample_step,
                    layer_updates=queued.layer_updates,
                )

                if commands and (server._log_cam_info or server._log_cam_debug):
                    summaries: List[str] = []
                    for cmd in commands:
                        if cmd.kind == "zoom":
                            factor = cmd.factor if cmd.factor is not None else 0.0
                            if cmd.anchor_px is not None:
                                ax, ay = cmd.anchor_px
                                summaries.append(
                                    f"zoom factor={factor:.4f} anchor=({ax:.1f},{ay:.1f})"
                                )
                            else:
                                summaries.append(f"zoom factor={factor:.4f}")
                        elif cmd.kind == "pan":
                            summaries.append(f"pan dx={cmd.dx_px:.2f} dy={cmd.dy_px:.2f}")
                        elif cmd.kind == "orbit":
                            summaries.append(
                                f"orbit daz={cmd.d_az_deg:.2f} del={cmd.d_el_deg:.2f}"
                            )
                        elif cmd.kind == "reset":
                            summaries.append("reset")
                        else:
                            summaries.append(cmd.kind)
                    message = "apply: cam cmds=" + "; ".join(summaries)
                    if server._log_cam_info:
                        logger.info(message)
                    else:
                        logger.debug(message)

                worker.apply_state(frame_state)
                if commands:
                    worker.process_camera_commands(commands)

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

                on_frame(packet, flags, timings.capture_wall_ts, seq)

                server._publish_policy_metrics()
                snapshot = server._scene.policy_metrics_snapshot
                last_decision = snapshot.get("last_decision") if snapshot else None
                if isinstance(last_decision, dict):
                    seq_value = int(last_decision.get("seq") or 0)
                    if seq_value > server._scene.last_written_decision_seq:
                        try:
                            server._scene.policy_event_path.parent.mkdir(parents=True, exist_ok=True)
                            with server._scene.policy_event_path.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(last_decision) + "\n")
                        except OSError as exc:
                            logger.debug("Policy event write failed: %s", exc)
                        else:
                            server._scene.last_written_decision_seq = seq_value

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

    thread = threading.Thread(target=worker_loop, name="egl-render", daemon=True)
    state.thread = thread
    thread.start()


def stop_worker(state: WorkerLifecycleState) -> None:
    """Signal the render worker to stop and wait for the thread to exit."""

    state.stop_event.set()
    thread = state.thread
    if thread and thread.is_alive():
        thread.join(timeout=3.0)
    state.thread = None
    state.worker = None
