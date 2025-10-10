"""Worker lifecycle orchestration for the EGL headless server."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import logging
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Optional, Sequence, List, Dict, Any, Tuple

from napari_cuda.server.control import pixel_channel
from napari_cuda.server.rendering.bitstream import build_avcc_config, pack_to_avcc
from napari_cuda.server.control.intent_queue import ReducerIntentQueue
from napari_cuda.server.control.state_models import (
    BootstrapSceneMetadata,
    WorkerStateUpdateConfirmation,
)
from napari_cuda.server.runtime.scene_ingest import RenderSceneSnapshot
from napari_cuda.server.runtime.frame_input import build_frame_input
from napari_cuda.server.runtime.egl_worker import EGLRendererWorker
from napari_cuda.server.rendering.debug_tools import DebugDumper


def _ingest_scene_refresh(
    server: object,
    state: WorkerLifecycleState,
    loop: asyncio.AbstractEventLoop,
    step: object = None,
) -> None:
    worker_ref = state.worker
    assert worker_ref is not None, "render worker missing during refresh"
    assert worker_ref.is_ready, "render worker refresh fired before bootstrap"
    del loop

    head = server._reducer_intents.peek()
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "ingest_scene_refresh: head_intent=%s step_arg=%s worker_last=%s active_level=%s",
            (head.intent_id if head else None),
            step,
            tuple(int(v) for v in worker_ref._last_step) if worker_ref._last_step is not None else None,
            int(worker_ref._active_ms_level),
        )
    # Allow confirmations even without a head intent for continuous dims

    step_hint: Optional[tuple[int, ...]] = None
    if isinstance(step, (list, tuple)):
        step_hint = tuple(int(value) for value in step)

    confirmation = _build_applied_confirmation(worker_ref, step_hint, server._reducer_intents)
    if confirmation is None:
        logger.info("ingest_scene_refresh: no confirmation generated (step_hint=%s)", step_hint)
        return

    state.scene_seq = int(state.scene_seq) + 1

    server._submit_worker_confirmation(confirmation)  # type: ignore[attr-defined]
    logger.info(
        "ingest_scene_refresh: submitted confirmation scope=%s intent_id=%s step=%s",
        confirmation.scope,
        confirmation.intent_id,
        confirmation.step,
    )

logger = logging.getLogger(__name__)


def _build_applied_confirmation(
    worker: "EGLRendererWorker",
    step_hint: Optional[Sequence[int]],
    intents: ReducerIntentQueue,
) -> Optional[WorkerStateUpdateConfirmation]:
    pending_intent = intents.peek()
    intent_id: Optional[str] = pending_intent.intent_id if pending_intent is not None else None
    # Ignore intent payload for continuous dims/view confirmations (applied-first)
    intent_payload: Dict[str, Any] = {}

    canonical = worker._canonical_axes
    assert canonical is not None, "worker missing canonical axes"

    current_step: Tuple[int, ...]
    if step_hint is not None:
        current_step = tuple(int(v) for v in step_hint)
    elif worker._last_step is not None:
        current_step = tuple(int(v) for v in worker._last_step)
    else:
        current_step = tuple(int(v) for v in canonical.current_step)

    axis_labels = tuple(str(label) for label in canonical.axis_labels)
    order = tuple(int(idx) for idx in canonical.order)

    # Applied-first: always use canonical ndisplay for confirmations
    ndisplay = int(canonical.ndisplay)

    displayed = order[-ndisplay:] if order else tuple()

    source = worker._scene_source
    levels: List[Dict[str, Any]] = []
    level_shapes: List[tuple[int, ...]] = []
    if source is not None and source.level_descriptors:
        for descriptor in source.level_descriptors:
            shape = tuple(int(s) for s in descriptor.shape)
            level_shapes.append(shape)
            entry: Dict[str, Any] = {
                "index": int(descriptor.index),
                "shape": [int(s) for s in descriptor.shape],
            }
            entry["downsample"] = [float(x) for x in descriptor.downsample]
            if descriptor.path:
                entry["path"] = str(descriptor.path)
            levels.append(entry)
    if not level_shapes:
        fallback_shape = tuple(int(s) for s in canonical.sizes)
        level_shapes.append(fallback_shape)
        levels.append(
            {
                "index": int(worker._active_ms_level),
                "shape": [int(s) for s in fallback_shape],
            }
        )

    mode = "volume" if ndisplay >= 3 else "plane"
    downgraded_attr = worker._level_downgraded

    plane_state = worker._plane_restore_state
    metadata: Dict[str, Any] = {}
    if plane_state is not None:
        metadata["plane_state"] = plane_state

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "build_confirmation: step_hint=%s worker_last=%s chosen_step=%s level=%d",
            tuple(int(v) for v in step_hint) if isinstance(step_hint, Sequence) else None,
            tuple(int(v) for v in worker._last_step) if worker._last_step is not None else None,
            current_step,
            int(worker._active_ms_level),
        )

    confirmation = WorkerStateUpdateConfirmation(
        scope="dims" if intent_id is None else pending_intent.scope,  # type: ignore[union-attr]
        target="main",
        key="snapshot",
        step=current_step,
        ndisplay=ndisplay,
        mode=mode,
        displayed=tuple(displayed) if displayed else None,
        order=order if order else None,
        axis_labels=axis_labels if axis_labels else None,
        labels=None,
        current_level=int(worker._active_ms_level),
        levels=tuple(levels),
        level_shapes=tuple(level_shapes),
        downgraded=bool(downgraded_attr) if downgraded_attr is not None else None,
        timestamp=time.time(),
        metadata=metadata or None,
        intent_id=intent_id,
    )
    worker._last_step = current_step
    return confirmation


def capture_bootstrap_state(worker: "EGLRendererWorker") -> BootstrapSceneMetadata:
    canonical = worker._canonical_axes
    assert canonical is not None, "worker missing canonical axes for bootstrap capture"

    axis_labels = tuple(str(label) for label in canonical.axis_labels)
    order = tuple(int(idx) for idx in canonical.order)
    if worker._last_step is not None:
        step_values = tuple(int(value) for value in worker._last_step)
    else:
        step_values = tuple(int(value) for value in canonical.current_step)

    source = worker._scene_source
    level_shapes_list: List[tuple[int, ...]] = []
    levels_payload: List[Dict[str, Any]] = []
    if source is not None:
        descriptors = source.level_descriptors
        for descriptor in descriptors:
            shape_tuple = tuple(int(dim) for dim in descriptor.shape)
            level_shapes_list.append(shape_tuple)
            entry: Dict[str, Any] = {
                "index": int(descriptor.index),
                "shape": [int(dim) for dim in descriptor.shape],
                "downsample": [float(value) for value in descriptor.downsample],
            }
            if descriptor.path:
                entry["path"] = str(descriptor.path)
            levels_payload.append(entry)
    if not level_shapes_list:
        fallback_shape = tuple(int(value) for value in canonical.sizes)
        level_shapes_list.append(fallback_shape)
        levels_payload.append(
            {
                "index": int(worker._active_ms_level),
                "shape": [int(value) for value in fallback_shape],
                "downsample": [1.0 for _ in fallback_shape],
            }
        )

    ndisplay_value = int(canonical.ndisplay)
    current_level = int(worker._active_ms_level)

    return BootstrapSceneMetadata(
        step=step_values,
        axis_labels=axis_labels,
        order=order,
        level_shapes=tuple(level_shapes_list),
        levels=tuple(levels_payload),
        current_level=current_level,
        ndisplay=ndisplay_value,
    )


@dataclass
class WorkerLifecycleState:
    """Track the worker thread and its stop signal."""

    worker: Optional[EGLRendererWorker] = None
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    ready_event: threading.Event = field(default_factory=threading.Event)
    scene_seq: int = 0
    bootstrap_meta: Optional[BootstrapSceneMetadata] = None
    bootstrap_event: threading.Event = field(default_factory=threading.Event)
    ready_async_event: Optional[asyncio.Event] = None
    bootstrap_async_event: Optional[asyncio.Event] = None


def start_worker(server: object, loop: asyncio.AbstractEventLoop, state: WorkerLifecycleState) -> None:
    """Launch the EGL render worker on its dedicated thread."""

    if state.thread and state.thread.is_alive():
        raise RuntimeError("worker thread already running")

    state.stop_event.clear()
    state.ready_event.clear()
    state.scene_seq = int(server._scene.last_scene_seq)
    state.bootstrap_meta = None
    state.bootstrap_event.clear()
    state.ready_async_event = asyncio.Event()
    state.bootstrap_async_event = asyncio.Event()

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
                scene_refresh_cb=None,
                ctx=server._ctx,
                env=server._ctx_env,
            )
            state.worker = worker
            worker.attach_ledger(server._state_ledger)  # type: ignore[attr-defined]
            # Provide a server-side intent enqueue hook for worker policy events
            worker._enqueue_server_intent = lambda intent: server._reducer_intents.push(intent)

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

            worker.set_scene_refresh_callback(
                partial(
                    _ingest_scene_refresh,
                    server,
                    state,
                    loop,
                )
            )

            state.bootstrap_meta = capture_bootstrap_state(worker)
            state.bootstrap_event.set()
            bootstrap_async = state.bootstrap_async_event
            if bootstrap_async is not None:
                loop.call_soon_threadsafe(bootstrap_async.set)

            # Mark server-ready AFTER metadata is available, BEFORE worker is_ready/refresh
            state.ready_event.set()
            ready_async = state.ready_async_event
            if ready_async is not None:
                loop.call_soon_threadsafe(ready_async.set)

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
                loop.call_soon_threadsafe(_publish_start_config, avcc_cfg)

            # Now flip worker ready and push first refresh
            worker._is_ready = True
            _ingest_scene_refresh(server, state, loop)

            tick = 1.0 / max(1, server.cfg.fps)
            next_tick = time.perf_counter()

            while not state.stop_event.is_set():
                snapshot, desired_seqs, desired_ndisplay = build_frame_input(server)
                with server._state_lock:
                    server._scene.latest_state = snapshot
                    commands = list(server._scene.camera_commands)
                    server._scene.camera_commands.clear()
                frame_input_step = snapshot.current_step

                # Request view ndisplay if provided by LatestIntent
                if desired_ndisplay is not None:
                    worker.request_ndisplay(int(desired_ndisplay))

                if commands and server._log_state_traces:
                    logger.info("frame commands snapshot count=%d", len(commands))

                frame_state = snapshot

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

                # Skip entire pipeline if no new desires, no animation/commands, and no explicit render tick
                applied_dims_seq = int(getattr(server, "_applied_seqs", {}).get("dims", -1))
                applied_view_seq = int(getattr(server, "_applied_seqs", {}).get("view", -1))
                desired_dims_seq = int(desired_seqs.get("dims", -1))
                desired_view_seq = int(desired_seqs.get("view", -1))
                dims_satisfied = (desired_dims_seq >= 0 and desired_dims_seq <= applied_dims_seq)
                view_satisfied = (desired_view_seq >= 0 and desired_view_seq <= applied_view_seq)
                if (
                    dims_satisfied
                    and view_satisfied
                    and not commands
                    and not server._animate
                    and not getattr(worker, "_render_tick_required", False)
                ):
                    next_tick += tick
                    sleep_duration = next_tick - time.perf_counter()
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                    else:
                        next_tick = time.perf_counter()
                    continue

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

                # Emit confirmation when applied differs from frame input (applied-first for dims)
                if worker._last_step is not None:
                    applied_step = tuple(int(value) for value in worker._last_step)
                    if frame_input_step is None or tuple(int(v) for v in frame_input_step) != applied_step:
                        worker._notify_scene_refresh(applied_step)

                # Clear one-shot render tick requirement if set
                if getattr(worker, "_render_tick_required", False):
                    worker._mark_render_tick_complete()

                # No local seq tracking; server._applied_seqs updated on confirmation write

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
            state.ready_event.clear()
            state.bootstrap_meta = None
            state.bootstrap_event.clear()
            state.ready_async_event = None
            state.bootstrap_async_event = None

    thread = threading.Thread(target=worker_loop, name="egl-render", daemon=True)
    state.thread = thread
    thread.start()



def stop_worker(state: WorkerLifecycleState) -> None:
    """Signal the render worker to stop and wait for the thread to exit."""

    state.stop_event.set()
    state.ready_event.clear()
    state.bootstrap_event.clear()
    state.ready_async_event = None
    state.bootstrap_async_event = None
    thread = state.thread
    if thread and thread.is_alive():
        thread.join(timeout=3.0)
    state.thread = None
    state.worker = None
