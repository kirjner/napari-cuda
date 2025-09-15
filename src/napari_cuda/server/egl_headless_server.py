"""
EGLHeadlessServer - Async server harness for headless EGL rendering + NVENC streaming.
"""

from __future__ import annotations

import asyncio
import json
import base64
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Set

import websockets
import importlib.resources as ilr
import socket

from .egl_worker import EGLRendererWorker, ServerSceneState
from .bitstream import ParamCache, pack_to_avcc, build_avcc_config
from .metrics import Metrics
from napari_cuda.utils.env import env_bool

logger = logging.getLogger(__name__)


@dataclass
class EncodeConfig:
    fps: int = 60
    codec: int = 1  # 1=h264, 2=hevc, 3=av1
    bitrate: int = 10_000_000
    keyint: int = 120


 


class EGLHeadlessServer:
    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False,
                 host: str = '0.0.0.0', state_port: int = 8081, pixel_port: int = 8082, fps: int = 60,
                 animate: bool = False, animate_dps: float = 30.0, log_sends: bool = False,
                 zarr_path: str | None = None, zarr_level: str | None = None,
                 zarr_axes: str | None = None, zarr_z: int | None = None) -> None:
        self.width = width
        self.height = height
        self.use_volume = use_volume
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        self.cfg = EncodeConfig(fps=fps)
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception:
            self._animate_dps = 30.0

        self.metrics = Metrics()
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._state_clients: Set[websockets.WebSocketServerProtocol] = set()
        try:
            # Keep queue size at 1 for latest-wins, never-block behavior
            qsize = int(os.getenv('NAPARI_CUDA_FRAME_QUEUE', '1'))
        except Exception:
            qsize = 1
        # Queue holds tuples of (payload_bytes, flags, seq, stamp_ts)
        # stamp_ts is minted post-pack in on_frame to keep encode/pack jitter common-mode
        self._frame_q: asyncio.Queue[tuple[bytes, int, int, float]] = asyncio.Queue(maxsize=max(1, qsize))
        self._seq = 0
        self._stop = threading.Event()
        

        self._worker: Optional[EGLRendererWorker] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._latest_state = ServerSceneState()
        # Bitstream parameter cache and config tracking (server-side)
        self._param_cache = ParamCache()
        self._needs_config = True
        self._last_avcc: Optional[bytes] = None
        # Optional bitstream dump for validation
        try:
            self._dump_remaining = int(os.getenv('NAPARI_CUDA_DUMP_BITSTREAM', '0'))
        except Exception:
            self._dump_remaining = 0
        self._dump_dir = os.getenv('NAPARI_CUDA_DUMP_DIR', 'benchmarks/bitstreams')
        self._dump_path: Optional[str] = None
        # Track last keyframe for metrics only
        self._last_key_seq: Optional[int] = None
        self._last_key_ts: Optional[float] = None
        # Watchdog task handle (cancel when keyframe arrives)
        self._kf_watchdog_task: Optional[asyncio.Task] = None
        # Broadcaster pacing: bypass once until keyframe for immediate start
        self._bypass_until_key: bool = False
        # State access synchronization for latest-wins camera op coalescing
        self._state_lock = threading.Lock()
        # Logging controls for camera ops
        self._log_cam_info: bool = env_bool('NAPARI_CUDA_LOG_CAMERA_INFO', False)
        self._log_cam_debug: bool = env_bool('NAPARI_CUDA_LOG_CAMERA_DEBUG', False)
        
        # Drop/send tracking
        self._drops_total: int = 0
        self._last_send_ts: Optional[float] = None
        self._send_count: int = 0
        # Optional detailed per-send logging (seq, send_ts, stamp_ts, delta)
        try:
            self._log_sends = bool(log_sends or int(os.getenv('NAPARI_CUDA_LOG_SENDS', '0') or '0'))
        except Exception:
            self._log_sends = bool(log_sends)
        
        # Data configuration (optional OME-Zarr dataset for real data)
        self._zarr_path = zarr_path or os.getenv('NAPARI_CUDA_ZARR_PATH') or None
        self._zarr_level = zarr_level or os.getenv('NAPARI_CUDA_ZARR_LEVEL') or None
        self._zarr_axes = zarr_axes or os.getenv('NAPARI_CUDA_ZARR_AXES') or None
        try:
            _z = zarr_z if zarr_z is not None else int(os.getenv('NAPARI_CUDA_ZARR_Z', '-1'))
            self._zarr_z = _z if _z >= 0 else None
        except Exception:
            self._zarr_z = None
        # Verbose dims logging control: default debug, upgrade to info with flag
        self._log_dims_info: bool = env_bool('NAPARI_CUDA_LOG_DIMS_INFO', False)

    async def start(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
        logger.info("Starting EGLHeadlessServer %dx%d @ %dfps", self.width, self.height, self.cfg.fps)
        loop = asyncio.get_running_loop()
        self._start_worker(loop)
        # Start websocket servers; disable permessage-deflate to avoid CPU and latency on large frames
        state_server = await websockets.serve(
            self._handle_state, self.host, self.state_port, compression=None
        )
        pixel_server = await websockets.serve(
            self._handle_pixel, self.host, self.pixel_port, compression=None
        )
        metrics_server = await self._start_metrics_server()
        logger.info(
            "WS listening on %s:%d (state), %s:%d (pixel) | Dashboard: http://%s:%s/dash/ JSON: http://%s:%s/metrics.json",
            self.host,
            self.state_port,
            self.host,
            self.pixel_port,
            self.host,
            os.getenv('NAPARI_CUDA_METRICS_PORT', '8083'),
            self.host,
            os.getenv('NAPARI_CUDA_METRICS_PORT', '8083'),
        )
        broadcaster = asyncio.create_task(self._broadcast_loop())
        try:
            await asyncio.Future()
        finally:
            broadcaster.cancel()
            state_server.close(); await state_server.wait_closed()
            pixel_server.close(); await pixel_server.wait_closed()
            await self._stop_metrics_server(metrics_server)
            self._stop_worker()

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        def on_frame(payload_obj, _flags: int, capture_wall_ts: Optional[float] = None, seq: Optional[int] = None) -> None:
            # Convert encoder AU (Annex B or AVCC) to AVCC and detect keyframe, and record pack time
            # Optional: raw NAL summary before packing for diagnostics
            try:
                if int(os.getenv('NAPARI_CUDA_LOG_NALS', '0')):
                    from .bitstream import parse_nals
                    raw_bytes: bytes
                    if isinstance(payload_obj, (bytes, bytearray, memoryview)):
                        raw_bytes = bytes(payload_obj)
                    elif isinstance(payload_obj, (list, tuple)):
                        raw_bytes = b''.join([bytes(x) for x in payload_obj if x is not None])
                    else:
                        raw_bytes = bytes(payload_obj) if payload_obj is not None else b''
                    nals = parse_nals(raw_bytes)
                    has_sps = any(((n[0] & 0x1F) == 7) or (((n[0] >> 1) & 0x3F) == 33) for n in nals if n)
                    has_pps = any(((n[0] & 0x1F) == 8) or (((n[0] >> 1) & 0x3F) == 34) for n in nals if n)
                    has_idr = any(((n[0] & 0x1F) == 5) or (((n[0] >> 1) & 0x3F) in (19, 20, 21)) for n in nals if n)
                    logger.debug("Raw NALs: count=%d sps=%s pps=%s idr=%s", len(nals), has_sps, has_pps, has_idr)
            except Exception as e:
                logger.debug("Raw NAL summary failed: %s", e)
            t_p0 = time.perf_counter()
            avcc_pkt, is_key = pack_to_avcc(payload_obj, self._param_cache)
            t_p1 = time.perf_counter()
            try:
                self.metrics.observe_ms('napari_cuda_pack_ms', (t_p1 - t_p0) * 1000.0)
            except Exception:
                pass
            try:
                if int(os.getenv('NAPARI_CUDA_LOG_NALS', '0')) and avcc_pkt is not None:
                    from .bitstream import parse_nals
                    nals2 = parse_nals(avcc_pkt)
                    has_idr2 = any(((n[0] & 0x1F) == 5) or (((n[0] >> 1) & 0x3F) in (19, 20, 21)) for n in nals2 if n)
                    if bool(has_idr2) != bool(is_key):
                        logger.warning("Keyframe detect mismatch: parse=%s is_key=%s nals_after=%d", has_idr2, is_key, len(nals2))
            except Exception as e:
                logger.debug("Post-pack NAL summary failed: %s", e)
            if not avcc_pkt:
                return
            # Build and send video_config if needed or changed
            avcc_cfg = build_avcc_config(self._param_cache)
            if avcc_cfg is not None and (self._needs_config or self._last_avcc != avcc_cfg):
                try:
                    msg = {
                        'type': 'video_config',
                        'codec': 'h264',
                        'format': 'avcc',
                        'data': base64.b64encode(avcc_cfg).decode('ascii'),
                        'width': self.width,
                        'height': self.height,
                        'fps': self.cfg.fps,
                    }
                    loop.call_soon_threadsafe(lambda: asyncio.create_task(self._broadcast_state_json(msg)))
                    self._last_avcc = avcc_cfg
                    self._needs_config = False
                    try:
                        self.metrics.inc('napari_cuda_video_config_sends')
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug("Failed to schedule video_config broadcast: %s", e)
            # Optional payload dump (AVCC payload)
            if self._dump_remaining > 0:
                try:
                    os.makedirs(self._dump_dir, exist_ok=True)
                    if not self._dump_path:
                        ts = int(time.time())
                        self._dump_path = os.path.join(self._dump_dir, f"dump_{self.width}x{self.height}_{ts}.h264")
                    with open(self._dump_path, 'ab') as f:
                        f.write(avcc_pkt)
                    self._dump_remaining -= 1
                    if self._dump_remaining == 0:
                        logger.info("Bitstream dump complete: %s", self._dump_path)
                except Exception as e:
                    logger.debug("Bitstream dump error: %s", e)
            # Mint the header timestamp at post-pack time to keep encode/pack jitter common-mode
            stamp_ts = time.time()
            flags = 0x01 if is_key else 0
            # Mint sequence at pack/enqueue time to match previous semantics
            seq_val = self._seq & 0xFFFFFFFF
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            # Enqueue via callback that handles QueueFull inside the event loop thread
            def _enqueue():
                try:
                    self._frame_q.put_nowait((avcc_pkt, flags, seq_val, stamp_ts))
                    try:
                        self.metrics.set('napari_cuda_frame_queue_depth', float(self._frame_q.qsize()))
                    except Exception:
                        pass
                except asyncio.QueueFull:
                    try:
                        self._drain_and_put((avcc_pkt, flags, seq_val, stamp_ts))
                        try:
                            self.metrics.inc('napari_cuda_frames_dropped')
                            self._drops_total += 1
                            if (self._drops_total % 100) == 1:
                                logger.info("Pixel queue full: dropped oldest (total drops=%d)", self._drops_total)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug("Failed to drain and enqueue frame: %s", e)
            loop.call_soon_threadsafe(_enqueue)

        def worker_loop() -> None:
            try:
                self._worker = EGLRendererWorker(
                    width=self.width,
                    height=self.height,
                    use_volume=self.use_volume,
                    fps=self.cfg.fps,
                    animate=self._animate,
                    animate_dps=self._animate_dps,
                    zarr_path=self._zarr_path,
                    zarr_level=self._zarr_level,
                    zarr_axes=self._zarr_axes,
                    zarr_z=self._zarr_z,
                )
                tick = 1.0 / max(1, self.cfg.fps)
                next_t = time.perf_counter()
                
                while not self._stop.is_set():
                    # Snapshot and clear one-shot camera ops atomically
                    with self._state_lock:
                        state = self._latest_state
                        # Clear coalesced ops in the shared state so they are one-shot
                        self._latest_state = ServerSceneState(
                            center=state.center,
                            zoom=state.zoom,
                            angles=state.angles,
                            current_step=state.current_step,
                        )
                    # Optional frame-level log of applied camera ops
                    try:
                        has_cam_ops = bool(getattr(state, 'reset_view', False)) or \
                                      (getattr(state, 'zoom_factor', None) is not None) or \
                                      (abs(float(getattr(state, 'pan_dx_px', 0.0) or 0.0)) > 1e-6) or \
                                      (abs(float(getattr(state, 'pan_dy_px', 0.0) or 0.0)) > 1e-6)
                        if has_cam_ops and (self._log_cam_info or self._log_cam_debug):
                            zf = getattr(state, 'zoom_factor', None)
                            anc = getattr(state, 'zoom_anchor_px', None)
                            dx = float(getattr(state, 'pan_dx_px', 0.0) or 0.0)
                            dy = float(getattr(state, 'pan_dy_px', 0.0) or 0.0)
                            msg = f"apply: cam reset={bool(getattr(state,'reset_view', False))} zf={zf} anc={anc} pan=({dx:.2f},{dy:.2f})"
                            if self._log_cam_info:
                                logger.info(msg)
                            else:
                                logger.debug(msg)
                    except Exception:
                        pass
                    self._worker.apply_state(state)
                    
                    timings, pkt, flags, seq = self._worker.capture_and_encode_packet()
                    # Observe timings (ms)
                    try:
                        self.metrics.observe_ms('napari_cuda_render_ms', timings.render_ms)
                        if timings.blit_gpu_ns is not None:
                            self.metrics.observe_ms('napari_cuda_capture_blit_ms', timings.blit_gpu_ns / 1e6)
                        # CPU wall time for blit (adds to additivity)
                        self.metrics.observe_ms('napari_cuda_capture_blit_cpu_ms', getattr(timings, 'blit_cpu_ms', 0.0))
                        self.metrics.observe_ms('napari_cuda_map_ms', timings.map_ms)
                        self.metrics.observe_ms('napari_cuda_copy_ms', timings.copy_ms)
                        self.metrics.observe_ms('napari_cuda_convert_ms', getattr(timings, 'convert_ms', 0.0))
                        self.metrics.observe_ms('napari_cuda_encode_ms', timings.encode_ms)
                        self.metrics.observe_ms('napari_cuda_pack_ms', getattr(timings, 'pack_ms', 0.0))
                        self.metrics.observe_ms('napari_cuda_total_ms', timings.total_ms)
                    except Exception:
                        pass
                    # Track last keyframe timestamp/seq for metrics if needed
                    # Keyframe tracking is handled after AVCC packing in on_frame
                    # Forward the encoder output along with the capture wall timestamp
                    cap_ts = getattr(timings, 'capture_wall_ts', None)
                    on_frame(pkt, flags, cap_ts, seq)
                    next_t += tick
                    sleep = next_t - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_t = time.perf_counter()
            except Exception as e:
                logger.exception("Render worker error: %s", e)
            finally:
                try:
                    if self._worker:
                        self._worker.cleanup()
                except Exception as e:
                    logger.debug("Worker cleanup error: %s", e)

        # Start render thread outside of worker_loop definition
        self._worker_thread = threading.Thread(target=worker_loop, name="egl-render", daemon=True)
        self._worker_thread.start()

    # No per-frame header color hints; reserved set to 0

    def _stop_worker(self) -> None:
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

    def _drain_and_put(self, data: tuple[bytes, int, int, float]) -> None:
        try:
            while not self._frame_q.empty():
                self._frame_q.get_nowait()
        except Exception as e:
            logger.debug("Queue drain error: %s", e)
        try:
            self._frame_q.put_nowait(data)
        except Exception as e:
            logger.debug("Frame enqueue error: %s", e)

    async def _handle_state(self, ws: websockets.WebSocketServerProtocol):
        self._state_clients.add(ws)
        self.metrics.inc('napari_cuda_state_connects')
        try:
            self._update_client_gauges()
            # Reduce latency: disable Nagle for control channel
            try:
                sock = ws.transport.get_extra_info('socket')  # type: ignore[attr-defined]
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            # Send latest video config if available
            try:
                if self._last_avcc is not None:
                    msg = {
                        'type': 'video_config',
                        'codec': 'h264',
                        'format': 'avcc',
                        'data': base64.b64encode(self._last_avcc).decode('ascii'),
                        'width': self.width,
                        'height': self.height,
                        'fps': self.cfg.fps,
                    }
                    await ws.send(json.dumps(msg))
            except Exception as e:
                logger.debug("Initial state config send failed: %s", e)
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                t = data.get('type')
                if t == 'set_camera':
                    center = data.get('center')
                    zoom = data.get('zoom')
                    angles = data.get('angles')
                    try:
                        if self._log_cam_info:
                            logger.info("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
                        elif self._log_cam_debug:
                            logger.debug("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
                    except Exception:
                        pass
                    with self._state_lock:
                        self._latest_state = ServerSceneState(
                            center=tuple(center) if center else None,
                            zoom=float(zoom) if zoom is not None else None,
                            angles=tuple(angles) if angles else None,
                            current_step=self._latest_state.current_step,
                        )
                elif t == 'dims.set':
                    step = data.get('current_step')
                    if self._log_dims_info:
                        logger.info("state: dims.set -> current_step=%s", step)
                    else:
                        logger.debug("state: dims.set -> current_step=%s", step)
                    # Preserve any coalesced camera ops already queued; dims is latest-wins
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=tuple(step) if step else None,
                            zoom_factor=getattr(s, 'zoom_factor', None),
                            zoom_anchor_px=getattr(s, 'zoom_anchor_px', None),
                            pan_dx_px=float(getattr(s, 'pan_dx_px', 0.0) or 0.0),
                            pan_dy_px=float(getattr(s, 'pan_dy_px', 0.0) or 0.0),
                            reset_view=bool(getattr(s, 'reset_view', False)),
                        )
                elif t == 'camera.zoom_at':
                    try:
                        factor = float(data.get('factor') or 0.0)
                    except Exception:
                        factor = 0.0
                    anchor = data.get('anchor_px')
                    anc_t = tuple(anchor) if isinstance(anchor, (list, tuple)) and len(anchor) >= 2 else None
                    if factor > 0.0 and anc_t is not None:
                        if self._log_cam_info:
                            logger.info("state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)", factor, float(anc_t[0]), float(anc_t[1]))
                        elif self._log_cam_debug:
                            logger.debug("state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)", factor, float(anc_t[0]), float(anc_t[1]))
                        with self._state_lock:
                            s = self._latest_state
                            nf = (s.zoom_factor if getattr(s, 'zoom_factor', None) else 1.0) * float(factor)
                            self._latest_state = ServerSceneState(
                                center=s.center,
                                zoom=s.zoom,
                                angles=s.angles,
                                current_step=s.current_step,
                                zoom_factor=float(nf),
                                zoom_anchor_px=(float(anc_t[0]), float(anc_t[1])),
                                pan_dx_px=getattr(s, 'pan_dx_px', 0.0),
                                pan_dy_px=getattr(s, 'pan_dy_px', 0.0),
                                reset_view=bool(getattr(s, 'reset_view', False)),
                            )
                elif t == 'camera.pan_px':
                    try:
                        dx = float(data.get('dx_px') or 0.0)
                        dy = float(data.get('dy_px') or 0.0)
                    except Exception:
                        dx = 0.0; dy = 0.0
                    if dx != 0.0 or dy != 0.0:
                        if self._log_cam_info:
                            logger.info("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
                        elif self._log_cam_debug:
                            logger.debug("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
                        with self._state_lock:
                            s = self._latest_state
                            self._latest_state = ServerSceneState(
                                center=s.center,
                                zoom=s.zoom,
                                angles=s.angles,
                                current_step=s.current_step,
                                zoom_factor=getattr(s, 'zoom_factor', None),
                                zoom_anchor_px=getattr(s, 'zoom_anchor_px', None),
                                pan_dx_px=float(getattr(s, 'pan_dx_px', 0.0) + dx),
                                pan_dy_px=float(getattr(s, 'pan_dy_px', 0.0) + dy),
                                reset_view=bool(getattr(s, 'reset_view', False)),
                            )
                elif t == 'camera.reset':
                    if self._log_cam_info:
                        logger.info("state: camera.reset")
                    elif self._log_cam_debug:
                        logger.debug("state: camera.reset")
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=s.current_step,
                            zoom_factor=getattr(s, 'zoom_factor', None),
                            zoom_anchor_px=getattr(s, 'zoom_anchor_px', None),
                            pan_dx_px=float(getattr(s, 'pan_dx_px', 0.0)),
                            pan_dy_px=float(getattr(s, 'pan_dy_px', 0.0)),
                            reset_view=True,
                        )
                elif t == 'ping':
                    await ws.send(json.dumps({'type': 'pong'}))
                elif t in ('request_keyframe', 'force_idr'):
                    try:
                        if self._worker is not None:
                            try:
                                # Try lightweight IDR request first
                                self._worker.force_idr()
                            except Exception:
                                # Fallback to full encoder reset
                                self._worker.reset_encoder()
                            # Bypass pacing once to deliver next keyframe immediately
                            self._bypass_until_key = True
                            # Count encoder resets/force events for diagnostics
                            try:
                                self.metrics.inc('napari_cuda_encoder_resets')
                            except Exception:
                                pass
                            # Watchdog: if no keyframe arrives within 300 ms, reset encoder
                            async def _kf_watchdog(last_key_seq: Optional[int]):
                                try:
                                    await asyncio.sleep(0.30)
                                    # If no new keyframe observed, hard reset the encoder
                                    if self._last_key_seq == last_key_seq and self._worker is not None:
                                        logger.warning("Keyframe watchdog fired; resetting encoder")
                                        self._worker.reset_encoder()
                                        self._bypass_until_key = True
                                except Exception as e:
                                    logger.debug("Keyframe watchdog error: %s", e)
                            try:
                                # Cancel previous watchdog before starting a new one
                                if self._kf_watchdog_task is not None and not self._kf_watchdog_task.done():
                                    self._kf_watchdog_task.cancel()
                                self._kf_watchdog_task = asyncio.create_task(_kf_watchdog(self._last_key_seq))
                            except Exception:
                                pass
                            # Re-broadcast current video config if known to tighten init window
                            if self._last_avcc is not None:
                                msg = {
                                    'type': 'video_config',
                                    'codec': 'h264',
                                    'format': 'avcc',
                                    'data': base64.b64encode(self._last_avcc).decode('ascii'),
                                    'width': self.width,
                                    'height': self.height,
                                    'fps': self.cfg.fps,
                                }
                                await self._broadcast_state_json(msg)
                            else:
                                self._needs_config = True
                    except Exception as e:
                        logger.debug("request_keyframe handling failed: %s", e)
        finally:
            try:
                await ws.close()
            except Exception as e:
                logger.debug("State WS close error: %s", e)
            self._state_clients.discard(ws)
            self._update_client_gauges()

    async def _handle_pixel(self, ws: websockets.WebSocketServerProtocol):
        self._clients.add(ws)
        self._update_client_gauges()
        # Reduce latency: disable Nagle for binary pixel stream
        try:
            sock = ws.transport.get_extra_info('socket')  # type: ignore[attr-defined]
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        # Reset encoder on new client to guarantee an immediate keyframe
        try:
            if self._worker is not None:
                logger.info("Resetting encoder for new pixel client to force keyframe")
                self._worker.reset_encoder()
                # Allow immediate send of the first keyframe when pacing is enabled
                self._bypass_until_key = True
                # Request re-send of video configuration for new clients
                self._needs_config = True
                # If we already have a cached avcC, proactively re-broadcast on state channel
                if self._last_avcc is not None:
                    try:
                        msg = {
                            'type': 'video_config',
                            'codec': 'h264',
                            'format': 'avcc',
                            'data': base64.b64encode(self._last_avcc).decode('ascii'),
                            'width': self.width,
                            'height': self.height,
                            'fps': self.cfg.fps,
                        }
                        await self._broadcast_state_json(msg)
                    except Exception as e:
                        logger.debug("Proactive video_config broadcast failed: %s", e)
                try:
                    self.metrics.inc('napari_cuda_encoder_resets')
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Encoder reset on client connect failed: %s", e)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)
            self._update_client_gauges()

    async def _broadcast_loop(self) -> None:
        # Always use paced broadcasting with latest-wins coalescing for smooth delivery
        try:
            target_fps = float(os.getenv('NAPARI_CUDA_BROADCAST_FPS', str(self.cfg.fps)))
        except Exception:
            target_fps = float(self.cfg.fps)
        loop = asyncio.get_running_loop()
        tick = 1.0 / max(1.0, target_fps)
        next_t = loop.time()
        # Map loop.time() monotonic clock to wall time for stable header timestamps
        mono0 = loop.time()
        wall0 = time.time()
        latest: Optional[tuple[bytes, int, int, float]] = None

        async def _fill_until(deadline: float) -> None:
            nonlocal latest
            remaining = max(0.0, deadline - loop.time())
            try:
                item = await asyncio.wait_for(self._frame_q.get(), timeout=remaining if remaining > 0 else 1e-6)
                latest = item
                # Drain rest without waiting; keep the newest only
                while True:
                    try:
                        latest = self._frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                return

        async def _fill_and_find_key(deadline: float) -> Optional[tuple[bytes, int, int, float]]:
            """Drain queue up to deadline and return the FIRST keyframe seen (if any).

            If no keyframe found, returns None and leaves `latest` pointing to the newest item.
            """
            nonlocal latest
            found: Optional[tuple[bytes, int, int, float]] = None
            remaining = max(0.0, deadline - loop.time())
            try:
                item = await asyncio.wait_for(self._frame_q.get(), timeout=remaining if remaining > 0 else 1e-6)
            except asyncio.TimeoutError:
                return None
            latest = item
            payload, flags, _seq, _stamp_ts = item
            if (flags & 0x01) != 0:
                found = item
            # Drain the rest without waiting, but capture the first keyframe encountered
            while True:
                try:
                    it2 = self._frame_q.get_nowait()
                    latest = it2
                    if found is None:
                        _payload2, flags2, _seq2, _stamp_ts2 = it2
                        if (flags2 & 0x01) != 0:
                            found = it2
                except asyncio.QueueEmpty:
                    break
            return found

        while True:
            # Immediate keyframe bypass for new clients or after keyframe request
            if self._bypass_until_key:
                key_item = await _fill_and_find_key(loop.time() + 0.050)
                if key_item is not None:
                    if self._clients:
                        # Build header timestamp from stamp minted post-pack (common-mode with encode/pack)
                        payload, flags, seq_cap, stamp_ts = key_item
                        send_ts_mono = loop.time()
                        send_ts_wall = wall0 + (send_ts_mono - mono0)
                        seq32 = int(seq_cap) & 0xFFFFFFFF
                        header = struct.pack('!IdIIBBH', seq32, float(stamp_ts), self.width, self.height, self.cfg.codec & 0xFF, flags & 0xFF, 0)
                        to_send = header + payload
                        await asyncio.gather(*(self._safe_send(c, to_send) for c in list(self._clients)), return_exceptions=True)
                        # Update counters/metrics and log
                        try:
                            self.metrics.inc('napari_cuda_frames_total')
                            self.metrics.inc('napari_cuda_bytes_total', len(payload))
                            if flags & 0x01:
                                self.metrics.inc('napari_cuda_keyframes_total')
                                self._last_key_seq = seq32
                                # Track last keyframe timestamp based on stamped header time
                                self._last_key_ts = float(stamp_ts)
                                try:
                                    self.metrics.set('napari_cuda_last_key_seq', float(self._last_key_seq))
                                    self.metrics.set('napari_cuda_last_key_ts', float(self._last_key_ts))
                                except Exception:
                                    pass
                                # Cancel any pending keyframe watchdog once a keyframe is observed
                                try:
                                    if self._kf_watchdog_task is not None and not self._kf_watchdog_task.done():
                                        self._kf_watchdog_task.cancel()
                                except Exception:
                                    pass
                            if self._log_sends:
                                # Keep broadcaster-to-stamp delta for observability
                                stamp_to_send_ms = (send_ts_wall - float(stamp_ts)) * 1000.0
                                logger.info("Send frame seq=%d send_ts=%.6f stamp_ts=%.6f delta=%.3f ms (bypass)", seq32, send_ts_mono, float(stamp_ts), stamp_to_send_ms)
                        except Exception:
                            pass
                    latest = None
                    self._bypass_until_key = False
                    next_t = loop.time() + tick
                    continue

            now = loop.time()
            if now < next_t:
                await _fill_until(next_t)
            else:
                missed = int((now - next_t) // tick) + 1
                next_t += missed * tick

            if latest is not None:
                if self._clients:
                    payload, flags, seq_cap, stamp_ts = latest
                    send_ts_mono = loop.time()
                    send_ts_wall = wall0 + (send_ts_mono - mono0)
                    seq32 = int(seq_cap) & 0xFFFFFFFF
                    # Use stamped post-pack timestamp in header for paced sends
                    header = struct.pack('!IdIIBBH', seq32, float(stamp_ts), self.width, self.height, self.cfg.codec & 0xFF, flags & 0xFF, 0)
                    to_send = header + payload
                    await asyncio.gather(*(self._safe_send(c, to_send) for c in list(self._clients)), return_exceptions=True)
                    # Update counters/metrics and log
                    try:
                        self.metrics.inc('napari_cuda_frames_total')
                        self.metrics.inc('napari_cuda_bytes_total', len(payload))
                        if flags & 0x01:
                            self.metrics.inc('napari_cuda_keyframes_total')
                            self._last_key_seq = seq32
                            # Track last keyframe timestamp based on stamped header time
                            self._last_key_ts = float(stamp_ts)
                            try:
                                self.metrics.set('napari_cuda_last_key_seq', float(self._last_key_seq))
                                self.metrics.set('napari_cuda_last_key_ts', float(self._last_key_ts))
                            except Exception:
                                pass
                            # Cancel any pending keyframe watchdog once a keyframe is observed
                            try:
                                if self._kf_watchdog_task is not None and not self._kf_watchdog_task.done():
                                    self._kf_watchdog_task.cancel()
                            except Exception:
                                pass
                        if self._log_sends:
                            # Keep broadcaster-to-stamp delta for observability
                            stamp_to_send_ms = (send_ts_wall - float(stamp_ts)) * 1000.0
                            logger.info("Send frame seq=%d send_ts=%.6f stamp_ts=%.6f delta=%.3f ms", seq32, send_ts_mono, float(stamp_ts), stamp_to_send_ms)
                    except Exception:
                        pass
                    # Simple send timing log for smoothing diagnostics
                    try:
                        now2 = loop.time()
                        if self._last_send_ts is not None:
                            dt = now2 - self._last_send_ts
                            self._send_count += 1
                            if (self._send_count % int(max(1, self.cfg.fps))) == 0:
                                logger.debug("Pixel send dt=%.3f s (target=%.3f), drops=%d", dt, 1.0/max(1, self.cfg.fps), self._drops_total)
                        self._last_send_ts = now2
                    except Exception:
                        pass
                latest = None
            next_t += tick

    async def _safe_send(self, ws: websockets.WebSocketServerProtocol, data: bytes) -> None:
        try:
            await ws.send(data)
        except Exception as e:
            logger.debug("Pixel send error: %s", e)
            try:
                await ws.close()
            except Exception as e2:
                logger.debug("Pixel WS close error: %s", e2)
            self._clients.discard(ws)

    async def _broadcast_state_json(self, obj: dict) -> None:
        data = json.dumps(obj)
        if not self._state_clients:
            return
        coros = []
        for c in list(self._state_clients):
            coros.append(self._safe_state_send(c, data))
        try:
            await asyncio.gather(*coros, return_exceptions=True)
        except Exception as e:
            logger.debug("State broadcast error: %s", e)

    

    async def _safe_state_send(self, ws: websockets.WebSocketServerProtocol, text: str) -> None:
        try:
            await ws.send(text)
        except Exception as e:
            logger.debug("State send error: %s", e)
            try:
                await ws.close()
            except Exception as e2:
                logger.debug("State WS close error: %s", e2)
            self._state_clients.discard(ws)

    def _update_client_gauges(self) -> None:
        try:
            self.metrics.set('napari_cuda_pixel_clients', float(len(self._clients)))
            # We could track state clients separately if desired; here we reuse pixel_clients for demo
        except Exception:
            pass

    async def _start_metrics_server(self):
        # Start Dash/Plotly dashboard on a background thread with a Flask server.
        try:
            port = int(os.getenv('NAPARI_CUDA_METRICS_PORT', '8083'))
        except Exception:
            port = 8083
        try:
            refresh_ms = int(os.getenv('NAPARI_CUDA_METRICS_REFRESH_MS', '1000'))
        except Exception:
            refresh_ms = 1000
        try:
            # Import here to allow running without dash installed (graceful fallback)
            from .dash_dashboard import start_dash_dashboard  # type: ignore
            th = start_dash_dashboard(self.host, port, self.metrics, refresh_ms)
            return th
        except Exception as e:
            logger.error("Dashboard init failed; continuing without UI: %s", e)
            return None

    async def _stop_metrics_server(self, runner):
        # Dash thread is daemonized; nothing to stop cleanly at shutdown.
        return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='napari-cuda EGL headless server')
    parser.add_argument('--host', default=os.getenv('NAPARI_CUDA_HOST', '0.0.0.0'))
    parser.add_argument('--state-port', type=int, default=int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')))
    parser.add_argument('--pixel-port', type=int, default=int(os.getenv('NAPARI_CUDA_PIXEL_PORT', '8082')))
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--animate', action='store_true', help='Enable simple turntable camera animation')
    parser.add_argument('--animate-dps', type=float, default=float(os.getenv('NAPARI_CUDA_TURNTABLE_DPS', '30.0')),
                        help='Turntable speed in degrees per second (default 30)')
    parser.add_argument('--volume', action='store_true', help='Use 3D volume visual')
    parser.add_argument('--zarr', dest='zarr_path', default=os.getenv('NAPARI_CUDA_ZARR_PATH'), help='Path to OME-Zarr root (enables 2D slice from ZYX volume)')
    parser.add_argument('--zarr-level', dest='zarr_level', default=os.getenv('NAPARI_CUDA_ZARR_LEVEL'), help='Dataset level path inside OME-Zarr (e.g., level_02). If omitted, inferred from multiscales.')
    parser.add_argument('--zarr-axes', dest='zarr_axes', default=os.getenv('NAPARI_CUDA_ZARR_AXES', 'zyx'), help='Axes order of the dataset (default: zyx)')
    parser.add_argument('--zarr-z', dest='zarr_z', type=int, default=int(os.getenv('NAPARI_CUDA_ZARR_Z', '-1')), help='Initial Z index for 2D slice (default: mid-slice)')
    parser.add_argument('--log-sends', action='store_true', help='Log per-send timing (seq, send_ts, stamp_ts, delta)')
    args = parser.parse_args()

    async def run():
        srv = EGLHeadlessServer(width=args.width, height=args.height, use_volume=args.volume,
                                host=args.host, state_port=args.state_port, pixel_port=args.pixel_port, fps=args.fps,
                                animate=args.animate, animate_dps=args.animate_dps, log_sends=bool(args.log_sends),
                                zarr_path=args.zarr_path, zarr_level=args.zarr_level,
                                zarr_axes=args.zarr_axes, zarr_z=(None if int(args.zarr_z) < 0 else int(args.zarr_z)))
        await srv.start()

    asyncio.run(run())


if __name__ == '__main__':
    main()
