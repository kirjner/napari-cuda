"""
EGLHeadlessServer - Async server harness for headless EGL rendering + NVENC streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Set

import websockets

from .egl_worker import EGLRendererWorker, ServerSceneState
from .metrics import Metrics

logger = logging.getLogger(__name__)


@dataclass
class EncodeConfig:
    fps: int = 60
    codec: int = 1  # 1=h264, 2=hevc, 3=av1
    bitrate: int = 10_000_000
    keyint: int = 120


def pack_header(seq: int, ts: float, w: int, h: int, codec: int, flags: int) -> bytes:
    return struct.pack('!IdIIBBH', seq, ts, w, h, codec & 0xFF, flags & 0xFF, 0)


class EGLHeadlessServer:
    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False,
                 host: str = '0.0.0.0', state_port: int = 8081, pixel_port: int = 8082, fps: int = 60) -> None:
        self.width = width
        self.height = height
        self.use_volume = use_volume
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        self.cfg = EncodeConfig(fps=fps)

        self.metrics = Metrics()
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._frame_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=4)
        self._seq = 0
        self._stop = threading.Event()

        self._worker: Optional[EGLRendererWorker] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._latest_state = ServerSceneState()

    async def start(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
        logger.info("Starting EGLHeadlessServer %dx%d @ %dfps", self.width, self.height, self.cfg.fps)
        loop = asyncio.get_running_loop()
        self._start_worker(loop)
        state_server = await websockets.serve(self._handle_state, self.host, self.state_port)
        pixel_server = await websockets.serve(self._handle_pixel, self.host, self.pixel_port)
        logger.info("WS listening on %s:%d (state), %s:%d (pixel)", self.host, self.state_port, self.host, self.pixel_port)
        broadcaster = asyncio.create_task(self._broadcast_loop())
        try:
            await asyncio.Future()
        finally:
            broadcaster.cancel()
            state_server.close(); await state_server.wait_closed()
            pixel_server.close(); await pixel_server.wait_closed()
            self._stop_worker()

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        def on_frame(pkt: Optional[bytes]) -> None:
            if not pkt:
                return
            ts = time.time()
            header = pack_header(self._seq, ts, self.width, self.height, self.cfg.codec, 0)
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            data = header + pkt
            try:
                loop.call_soon_threadsafe(self._frame_q.put_nowait, data)
            except asyncio.QueueFull:
                try:
                    loop.call_soon_threadsafe(self._drain_and_put, data)
                except Exception as e:
                    logger.debug("Failed to drain and enqueue frame: %s", e)

        def worker_loop() -> None:
            try:
                self._worker = EGLRendererWorker(width=self.width, height=self.height, use_volume=self.use_volume)
                tick = 1.0 / max(1, self.cfg.fps)
                next_t = time.perf_counter()
                while not self._stop.is_set():
                    self._worker.apply_state(self._latest_state)
                    timings, pkt = self._worker.capture_and_encode_packet()
                    on_frame(pkt)
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

        self._worker_thread = threading.Thread(target=worker_loop, name="egl-render", daemon=True)
        self._worker_thread.start()

    def _stop_worker(self) -> None:
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

    def _drain_and_put(self, data: bytes) -> None:
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
        self.metrics.inc('napari_cuda_state_connects')
        try:
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
                    self._latest_state = ServerSceneState(
                        center=tuple(center) if center else None,
                        zoom=float(zoom) if zoom is not None else None,
                        angles=tuple(angles) if angles else None,
                        current_step=self._latest_state.current_step,
                    )
                elif t == 'set_dims':
                    step = data.get('current_step')
                    self._latest_state = ServerSceneState(
                        center=self._latest_state.center,
                        zoom=self._latest_state.zoom,
                        angles=self._latest_state.angles,
                        current_step=tuple(step) if step else None,
                    )
                elif t == 'ping':
                    await ws.send(json.dumps({'type': 'pong'}))
        finally:
            try:
                await ws.close()
            except Exception as e:
                logger.debug("State WS close error: %s", e)

    async def _handle_pixel(self, ws: websockets.WebSocketServerProtocol):
        self._clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    async def _broadcast_loop(self) -> None:
        while True:
            data = await self._frame_q.get()
            if not self._clients:
                continue
            await asyncio.gather(*(self._safe_send(c, data) for c in list(self._clients)), return_exceptions=True)

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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='napari-cuda EGL headless server')
    parser.add_argument('--host', default=os.getenv('NAPARI_CUDA_HOST', '0.0.0.0'))
    parser.add_argument('--state-port', type=int, default=int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')))
    parser.add_argument('--pixel-port', type=int, default=int(os.getenv('NAPARI_CUDA_PIXEL_PORT', '8082')))
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--volume', action='store_true', help='Use 3D volume visual')
    args = parser.parse_args()

    async def run():
        srv = EGLHeadlessServer(width=args.width, height=args.height, use_volume=args.volume,
                                host=args.host, state_port=args.state_port, pixel_port=args.pixel_port, fps=args.fps)
        await srv.start()

    asyncio.run(run())


if __name__ == '__main__':
    main()
