"""Worker lifecycle orchestration for the EGL headless server."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Sequence, List, Mapping, Dict, Any

from . import pixel_channel
from .bitstream import build_avcc_config, pack_to_avcc
from napari_cuda.server.state.scene_state import ServerSceneState
from .worker_notifications import WorkerSceneNotification
from .render_worker import EGLRendererWorker
from .debug_tools import DebugDumper


def _handle_scene_refresh(
    server: object,
    state: WorkerLifecycleState,
    loop: asyncio.AbstractEventLoop,
    step: object = None,
) -> None:
    level_payload: Optional[Mapping[str, object]] = None
    if step and type(step) is dict and "scene_level" in step:
        raw_level = step["scene_level"]
        assert type(raw_level) is dict, "scene level payload must be a dict"
        level_payload = dict(raw_level)

    worker_ref = state.worker
    assert worker_ref is not None, "render worker missing during refresh"
    assert worker_ref.is_ready, "render worker refresh fired before bootstrap"

    meta = dict(worker_ref.snapshot_dims_metadata())
    assert meta, "render worker returned empty dims metadata"
    meta = _normalize_dims_meta(meta)

    assert worker_ref._last_step is not None, "worker dims step missing"
    step_tuple = tuple(int(value) for value in worker_ref._last_step)
    plane_state = worker_ref._plane_restore_state
    mode = str(meta.get("mode"))
    if mode == "volume" and plane_state is not None:
        step_tuple = plane_state.step
        meta["level"] = plane_state.level
    else:
        meta["level"] = int(worker_ref._active_ms_level)
    meta_copy = dict(meta)
    meta_copy["current_step"] = [int(value) for value in step_tuple]

    notifications: list[WorkerSceneNotification] = []

    if level_payload:
        state.scene_seq = int(state.scene_seq) + 1
        notifications.append(
            WorkerSceneNotification(
                kind="scene_level",
                seq=state.scene_seq,
                level=level_payload,
            )
        )

    state.scene_seq = int(state.scene_seq) + 1
    with server._state_lock:  # type: ignore[attr-defined]
        snapshot = server._scene.latest_state  # type: ignore[attr-defined]
        server._scene.latest_state = ServerSceneState(  # type: ignore[attr-defined]
            center=snapshot.center,
            zoom=snapshot.zoom,
            angles=snapshot.angles,
            current_step=step_tuple,
        )
        server._scene.last_dims_payload = meta_copy  # type: ignore[attr-defined]

    if plane_state is not None:
        server._scene.plane_restore_state = plane_state

    server._worker_notifications.discard_kind("dims_update")
    notifications.append(
        WorkerSceneNotification(
            kind="dims_update",
            seq=state.scene_seq,
            step=step_tuple,
            meta=meta,
        )
    )

    if not notifications:
        return

    for note in notifications:
        server._worker_notifications.push(note)  # type: ignore[attr-defined]

    loop.call_soon_threadsafe(server._process_worker_notifications)  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@dataclass
class WorkerLifecycleState:
    """Track the worker thread and its stop signal."""

    worker: Optional[EGLRendererWorker] = None
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    scene_seq: int = 0


def start_worker(server: object, loop: asyncio.AbstractEventLoop, state: WorkerLifecycleState) -> None:
    """Launch the EGL render worker on its dedicated thread."""

    if state.thread and state.thread.is_alive():
        raise RuntimeError("worker thread already running")

    state.stop_event.clear()
    state.scene_seq = int(server._scene.last_scene_seq)

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
            def _send_config(avcc_bytes: bytes) -> None:
                server._schedule_coro(
                    pixel_channel.maybe_send_stream_config(
                        server._pixel_channel,
                        config=server._pixel_config,
                        metrics=server.metrics,
                        avcc=avcc_bytes,
                        send_stream=server._broadcast_stream_config,
                    ),
                    "notify_stream",
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
            worker._is_ready = True
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
                    _handle_scene_refresh,
                    server,
                    state,
                    loop,
                )
            )

            meta_init = worker.snapshot_dims_metadata()
            assert meta_init, "render worker returned empty dims metadata during init"
            meta_init = _normalize_dims_meta(dict(meta_init))
            server._scene.last_dims_payload = dict(meta_init)

            assert worker._last_step is not None, "worker dims step missing during init"
            init_step = tuple(int(value) for value in worker._last_step)

            with server._state_lock:
                latest = server._scene.latest_state
                server._scene.latest_state = ServerSceneState(
                    center=latest.center,
                    zoom=latest.zoom,
                    angles=latest.angles,
                    current_step=init_step,
                )

            state.scene_seq = int(state.scene_seq) + 1
            server._worker_notifications.push(
                WorkerSceneNotification(
                    kind="dims_update",
                    seq=state.scene_seq,
                    step=init_step,
                    meta=meta_init,
                )
            )
            loop.call_soon_threadsafe(server._process_worker_notifications)

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
def _normalize_dims_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in meta.items():
        normalized[key] = value

    if "sizes" in normalized:
        normalized["sizes"] = [int(v) for v in normalized["sizes"]]

    if "range" in normalized:
        ranges: list[list[int]] = []
        for bounds in normalized["range"]:
            ranges.append([int(bounds[0]), int(bounds[1])])
        normalized["range"] = ranges

    if "order" in normalized:
        normalized["order"] = [int(v) for v in normalized["order"]]

    if "displayed" in normalized:
        normalized["displayed"] = [int(v) for v in normalized["displayed"]]

    return normalized
