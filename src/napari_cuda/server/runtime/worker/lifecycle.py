"""Worker lifecycle orchestration for the EGL headless server."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from napari_cuda.server.engine.api import (
    DebugDumper,
    build_avcc_config,
    enqueue_frame,
    pack_to_avcc,
    send_cached_stream_snapshot,
)
from napari_cuda.server.runtime.bootstrap.runtime_driver import (
    init_egl as core_init_egl,
    init_vispy_scene as core_init_vispy_scene,
)
from napari_cuda.server.runtime.camera import CameraPoseApplied
from napari_cuda.server.runtime.ipc import LevelSwitchIntent
from napari_cuda.server.runtime.render_loop.planning.staging import (
    consume_render_snapshot,
    drain_scene_updates,
)
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import (
    pull_render_snapshot,
    snapshot_multiscale_state,
)
import numpy as np
from napari_cuda.server.runtime.ipc.mailboxes.worker_intent import ThumbnailCapture
from napari_cuda.server.utils.signatures import layer_inputs_signature
from napari_cuda.shared.dims_spec import dims_spec_from_payload

from .egl import EGLRendererWorker

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
                    send_cached_stream_snapshot(
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
            enqueue_frame(
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
                    server._apply_worker_camera_pose,  # type: ignore[attr-defined]
                    pose,
                )

            policy_state = snapshot_multiscale_state(server._state_ledger.snapshot())

            spec_entry = server._state_ledger.get("dims", "main", "dims_spec")
            if spec_entry is not None:
                spec = dims_spec_from_payload(getattr(spec_entry, "value", None))
            else:
                spec = None
            if spec is not None:
                use_volume_flag = int(spec.ndisplay) >= 3
            else:
                use_volume_flag = server._initial_mode is RenderMode.VOLUME

            worker = EGLRendererWorker(
                width=server.width,
                height=server.height,
                use_volume=use_volume_flag,
                fps=server.cfg.fps,
                animate=server._animate,
                animate_dps=server._animate_dps,
                zarr_path=server._zarr_path,
                zarr_level=server._zarr_level,
                zarr_axes=server._zarr_axes,
                zarr_z=server._zarr_z,
                policy_name=policy_state.get("policy"),
                camera_pose_cb=_forward_camera_pose,
                level_intent_cb=_forward_level_intent,
                camera_queue=server._camera_queue,
                ctx=server._ctx,
                env=server._ctx_env,
            )
            state.worker = worker
            worker.attach_ledger(server._state_ledger)  # type: ignore[attr-defined]

            core_init_vispy_scene(worker)
            core_init_egl(worker)

            worker._debug = DebugDumper(worker._debug_config)
            if worker._debug.cfg.enabled:
                worker._debug.log_env_once()
                worker._debug.ensure_out_dir()
            worker.resources.capture.pipeline.set_debug(worker._debug)
            worker.resources.capture.pipeline.set_raw_dump_budget(worker._raw_dump_budget)
            worker.resources.capture.cuda.set_force_tight_pitch(worker._debug_policy.worker.force_tight_pitch)
            worker._log_debug_policy_once()
            logger.info(
                "EGL renderer initialized: %dx%d, GL fmt=RGBA8, NVENC fmt=%s, fps=%d, animate=%s, zarr=%s",
                worker.width,
                worker.height,
                worker.resources.capture.pipeline.enc_input_format,
                worker.fps,
                worker._animate,
                bool(worker._zarr_path),
            )

            initial_snapshot = getattr(server, "_bootstrap_snapshot", None)
            if initial_snapshot is None:
                initial_snapshot = pull_render_snapshot(server)
            else:
                server._bootstrap_snapshot = None  # type: ignore[attr-defined]
            consume_render_snapshot(worker, initial_snapshot)
            drain_scene_updates(worker)
            frame_state = pull_render_snapshot(server)

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
                        send_cached_stream_snapshot(
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

            # No explicit thumbnail queue; post-frame logic will emit automatically

            tick = 1.0 / max(1, server.cfg.fps)
            next_tick = time.perf_counter()

            while not state.stop_event.is_set():
                has_camera_deltas = len(server._camera_queue) > 0

                # Request view ndisplay if provided by staged intents
                if has_camera_deltas and server._log_state_traces:
                    logger.info("frame camera deltas snapshot pending")

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
                    frame_state = pull_render_snapshot(server)
                    continue

                logger.debug(
                    "frame apply proceeding camera_deltas_pending=%s animate=%s",
                    bool(has_camera_deltas),
                    server._animate,
                )
                consume_render_snapshot(worker, frame_state)
                server._ack_scene_op_if_open(
                    frame_state=frame_state,
                    origin="worker.render.apply",
                )

                timings, packet, flags, seq = worker.capture_and_encode_packet()
                thumbnail_state = frame_state
                frame_state = pull_render_snapshot(server)
                # Post-frame: capture thumbnail on worker and hand off via mailbox
                viewer = worker.viewer_model()
                if viewer is not None and viewer.layers:
                    # Choose target layer id: prefer first from snapshot.layer_values
                    target_layer_id: Optional[str] = None
                    if frame_state.layer_values:
                        keys = list(frame_state.layer_values.keys())
                        if keys:
                            target_layer_id = str(keys[0])
                    # Resolve layer object
                    if len(viewer.layers) == 1:
                        layer_obj = viewer.layers[0]
                    else:
                        layer_obj = None
                        if target_layer_id is not None:
                            for candidate in viewer.layers:
                                if candidate.name == target_layer_id:
                                    layer_obj = candidate
                                    break
                        if layer_obj is None:
                            layer_obj = viewer.layers[0]
                    if layer_obj is not None:
                        layer_obj._set_view_slice()
                        layer_obj._update_thumbnail()
                        thumb = layer_obj.thumbnail
                        if thumb is not None:
                            arr = np.asarray(thumb)
                            layer_state = None
                            layer_values = thumbnail_state.layer_values if thumbnail_state.layer_values else {}
                            if layer_values:
                                if target_layer_id is not None and target_layer_id in layer_values:
                                    layer_state = layer_values[target_layer_id]
                                else:
                                    values = list(layer_values.values())
                                    if values:
                                        layer_state = values[0]
                            if layer_state is not None:
                                # Build inputs-only token from the same frame_state we just rendered
                                lid = str(target_layer_id or "layer-0")
                                token = layer_inputs_signature(thumbnail_state, lid).value
                                payload = ThumbnailCapture(
                                    layer_id=lid,
                                    array=arr,
                                    frame_token=token,
                                )
                                server._worker_intents.enqueue_thumbnail_capture(payload)
                                control_loop.call_soon_threadsafe(server._handle_worker_thumbnails)

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
