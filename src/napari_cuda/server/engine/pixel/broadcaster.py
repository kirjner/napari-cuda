"""Pixel stream broadcaster helpers.

Provides a small data bag plus procedural helpers for managing the pixel
websocket channel. The broadcaster owns the client set, pacing counters, and
keyframe bookkeeping while delegating lifecycle orchestration to the enclosing
server.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import websockets

if TYPE_CHECKING:  # pragma: no cover - metrics interface is runtime provided
    from napari_cuda.server.metrics import Metrics


FramePacket = tuple[bytes, int, int, float]


logger = logging.getLogger(__name__)


@dataclass
class PixelBroadcastConfig:
    """Static configuration for the pixel broadcaster."""

    width: int
    height: int
    codec: int
    fps: float


@dataclass
class PixelBroadcastState:
    """Mutable broadcaster state carried by the server."""

    frame_queue: asyncio.Queue[FramePacket]
    clients: set[websockets.WebSocketServerProtocol]
    log_sends: bool
    bypass_until_key: bool = False
    last_key_seq: Optional[int] = None
    last_key_ts: Optional[float] = None
    last_send_ts: Optional[float] = None
    send_count: int = 0
    drops_total: int = 0
    kf_watchdog_task: Optional[asyncio.Task] = None
    kf_last_reset_ts: Optional[float] = None
    waiting_for_keyframe: bool = False


def configure_socket(ws: websockets.WebSocketServerProtocol, *, label: str = "pixel ws") -> None:
    """Disable Nagle’s algorithm on the websocket transport if available."""

    try:
        sock = ws.transport.get_extra_info("socket")  # type: ignore[attr-defined]
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        logger.debug("%s: TCP_NODELAY toggle failed", label, exc_info=True)


async def safe_send(state: PixelBroadcastState, ws: websockets.WebSocketServerProtocol, data: bytes) -> None:
    """Send `data` to a single websocket, pruning dead connections on failure."""

    try:
        await ws.send(data)
    except Exception as exc:
        logger.debug("Pixel send error: %s", exc)
        try:
            await ws.close()
        except Exception as exc_close:
            logger.debug("Pixel WS close error: %s", exc_close)
        state.clients.discard(ws)


async def broadcast_loop(
    state: PixelBroadcastState,
    config: PixelBroadcastConfig,
    metrics: Metrics,
) -> None:
    """Continuously drain the frame queue and deliver packets to connected clients."""

    target_fps = float(max(1.0, config.fps))
    log_mod = max(1, int(round(config.fps)))
    loop = asyncio.get_running_loop()
    tick = 1.0 / target_fps
    next_tick = loop.time()
    mono_origin = loop.time()
    wall_origin = time.time()
    latest: Optional[FramePacket] = None

    async def _fill_until(deadline: float) -> None:
        nonlocal latest
        remaining = max(0.0, deadline - loop.time())
        try:
            item = await asyncio.wait_for(
                state.frame_queue.get(),
                timeout=remaining if remaining > 0 else 1e-6,
            )
        except TimeoutError:
            return
        latest = item
        while True:
            try:
                latest = state.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _fill_and_find_key(deadline: float) -> Optional[FramePacket]:
        nonlocal latest
        found: Optional[FramePacket] = None
        remaining = max(0.0, deadline - loop.time())
        try:
            item = await asyncio.wait_for(
                state.frame_queue.get(),
                timeout=remaining if remaining > 0 else 1e-6,
            )
        except TimeoutError:
            return None
        latest = item
        payload, flags, _seq, _stamp_ts = item
        if (flags & 0x01) != 0:
            found = item
        while True:
            try:
                candidate = state.frame_queue.get_nowait()
                latest = candidate
                if found is None and (candidate[1] & 0x01) != 0:
                    found = candidate
            except asyncio.QueueEmpty:
                break
        return found

    async def send_frame_packet(packet: FramePacket, *, bypass: bool) -> None:
        payload, flags, seq_cap, stamp_ts = packet
        if not state.clients:
            return
        send_ts_mono = loop.time()
        send_ts_wall = wall_origin + (send_ts_mono - mono_origin)
        seq32 = int(seq_cap) & 0xFFFFFFFF
        header = struct.pack(
            "!IdIIBBH",
            seq32,
            float(stamp_ts),
            config.width,
            config.height,
            config.codec & 0xFF,
            flags & 0xFF,
            0,
        )
        payload_bytes = header + payload
        await asyncio.gather(
            *(safe_send(state, client, payload_bytes) for client in list(state.clients)),
            return_exceptions=True,
        )
        try:
            metrics.inc("napari_cuda_frames_total")
            metrics.inc("napari_cuda_bytes_total", len(payload))
            if flags & 0x01:
                metrics.inc("napari_cuda_keyframes_total")
                state.last_key_seq = seq32
                state.last_key_ts = float(stamp_ts)
                state.waiting_for_keyframe = False
                try:
                    metrics.set("napari_cuda_last_key_seq", float(state.last_key_seq))
                    metrics.set("napari_cuda_last_key_ts", float(state.last_key_ts))
                except Exception:
                    logger.debug("metrics set last_key_* failed", exc_info=True)
                try:
                    task = state.kf_watchdog_task
                    if task is not None and not task.done():
                        task.cancel()
                    state.kf_watchdog_task = None
                except Exception:
                    logger.debug("keyframe watchdog cancel failed", exc_info=True)
            if state.log_sends:
                delta_ms = (send_ts_wall - float(stamp_ts)) * 1000.0
                if bypass:
                    logger.info(
                        "Send frame seq=%d send_ts=%.6f stamp_ts=%.6f delta=%.3f ms (bypass)",
                        seq32,
                        send_ts_mono,
                        float(stamp_ts),
                        delta_ms,
                    )
                else:
                    logger.info(
                        "Send frame seq=%d send_ts=%.6f stamp_ts=%.6f delta=%.3f ms",
                        seq32,
                        send_ts_mono,
                        float(stamp_ts),
                        delta_ms,
                    )
        except Exception:
            logger.debug(
                "metrics update (%s) failed",
                "bypass" if bypass else "paced",
                exc_info=True,
            )
        try:
            now2 = loop.time()
            if state.last_send_ts is not None:
                dt = now2 - state.last_send_ts
                state.send_count += 1
                if state.log_sends and (state.send_count % log_mod) == 0:
                    logger.debug(
                        "Pixel send dt=%.3f s (target=%.3f), drops=%d",
                        dt,
                        1.0 / log_mod,
                        state.drops_total,
                    )
            state.last_send_ts = now2
        except Exception:
            logger.debug("send timing update failed", exc_info=True)

    while True:
        if state.bypass_until_key:
            key_packet = await _fill_and_find_key(loop.time() + 0.050)
            if key_packet is not None:
                await send_frame_packet(key_packet, bypass=True)
                latest = None
                state.bypass_until_key = False
                next_tick = loop.time() + tick
                continue

        now = loop.time()
        if now < next_tick:
            await _fill_until(next_tick)
        else:
            missed = int((now - next_tick) // tick) + 1
            next_tick += missed * tick

        if latest is not None:
            await send_frame_packet(latest, bypass=False)
            latest = None
        next_tick += tick
