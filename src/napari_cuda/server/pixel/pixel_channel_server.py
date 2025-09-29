"""Pixel channel orchestration helpers.

Encapsulates websocket lifecycle, keyframe coordination, and queue mutation for
serving encoded pixel frames. The broadcaster loop itself lives in
``pixel_broadcaster``; this module provides the higher-level control surface
  that the headless server uses to drive it without reimplementing the state
bookkeeping inline.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

import websockets

from napari_cuda.server import pixel_broadcaster

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from napari_cuda.server.metrics import Metrics  # circular guard at runtime

FramePacket = pixel_broadcaster.FramePacket
logger = logging.getLogger(__name__)


@dataclass
class PixelChannelConfig:
    """Static description of the pixel channel."""

    width: int
    height: int
    fps: float
    codec_id: int
    codec_name: str
    kf_watchdog_cooldown_s: float


@dataclass
class PixelChannelState:
    """Mutable state for the pixel channel."""

    broadcast: pixel_broadcaster.PixelBroadcastState
    needs_config: bool = True
    last_avcc: Optional[bytes] = None
    kf_watchdog_cooldown_s: float = 0.30


def build_video_config_payload(config: PixelChannelConfig, avcc: bytes) -> dict:
    """Return a ``video_config`` payload for the state channel."""

    return {
        "type": "video_config",
        "codec": config.codec_name,
        "format": "avcc",
        "data": base64.b64encode(avcc).decode("ascii"),
        "width": config.width,
        "height": config.height,
        "fps": config.fps,
    }


async def handle_client(
    state: PixelChannelState,
    ws: websockets.WebSocketServerProtocol,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
    reset_encoder: Callable[[], bool],
    send_state_json: Callable[[dict], Awaitable[None]],
    on_clients_change: Callable[[], None],
) -> None:
    """Handle a pixel-channel websocket connection."""

    state.broadcast.clients.add(ws)
    on_clients_change()
    pixel_broadcaster.configure_socket(ws, label="pixel ws")
    state.needs_config = True

    try:
        if reset_encoder():
            logger.info("Resetting encoder for new pixel client to force keyframe")
            state.broadcast.bypass_until_key = True
            state.broadcast.kf_last_reset_ts = time.time()
            try:
                metrics.inc("napari_cuda_encoder_resets")
            except Exception:  # pragma: no cover - defensive metrics guard
                logger.debug("metrics inc encoder_resets failed", exc_info=True)
            if state.last_avcc is not None:
                try:
                    await send_state_json(build_video_config_payload(config, state.last_avcc))
                    state.needs_config = False
                except Exception:
                    logger.debug("Proactive video_config broadcast failed", exc_info=True)
        else:
            logger.debug("Pixel client connected but encoder reset unavailable")
    except Exception:
        logger.debug("Encoder reset on client connect failed", exc_info=True)

    try:
        await ws.wait_closed()
    finally:
        state.broadcast.clients.discard(ws)
        on_clients_change()


async def ensure_keyframe(
    state: PixelChannelState,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
    try_force_idr: Callable[[], bool],
    reset_encoder: Callable[[], bool],
    send_state_json: Callable[[dict], Awaitable[None]],
) -> None:
    """Force the next frame to be a keyframe and schedule a watchdog."""

    forced = False
    try:
        forced = try_force_idr()
    except Exception:
        logger.debug("force_idr request failed; will reset encoder", exc_info=True)

    if not forced:
        try:
            if not reset_encoder():
                logger.debug("Encoder reset unavailable during keyframe ensure")
                return
        except Exception:
            logger.exception("Encoder reset failed in ensure_keyframe")
            return
        state.broadcast.kf_last_reset_ts = time.time()

    state.broadcast.bypass_until_key = True
    state.needs_config = True

    try:
        metrics.inc("napari_cuda_encoder_resets")
    except Exception:  # pragma: no cover - defensive metrics guard
        logger.debug("metrics inc encoder_resets failed", exc_info=True)

    start_watchdog(state, reset_encoder=reset_encoder)

    if state.last_avcc is not None:
        try:
            await send_state_json(build_video_config_payload(config, state.last_avcc))
            state.needs_config = False
        except Exception:
            logger.debug("ensure_keyframe video_config broadcast failed", exc_info=True)


def start_watchdog(
    state: PixelChannelState,
    *,
    reset_encoder: Callable[[], bool],
    watchdog_delay_s: float = 0.30,
) -> None:
    """Schedule a watchdog to reset the encoder if a keyframe does not arrive."""

    broadcast = state.broadcast

    async def _watchdog(last_key_seq: Optional[int]) -> None:
        try:
            await asyncio.sleep(watchdog_delay_s)
            if broadcast.last_key_seq == last_key_seq:
                now = time.time()
                last_reset = broadcast.kf_last_reset_ts
                cooldown = float(state.kf_watchdog_cooldown_s)
                if last_reset is not None and (now - last_reset) < cooldown:
                    remaining = cooldown - (now - last_reset)
                    logger.debug(
                        "Keyframe watchdog cooldown active (%.2fs remaining); skip reset",
                        remaining,
                    )
                    return
                logger.warning("Keyframe watchdog fired; resetting encoder")
                try:
                    if not reset_encoder():
                        logger.debug("Watchdog reset no-op: encoder unavailable")
                        return
                except Exception:
                    logger.exception("Encoder reset failed during keyframe watchdog")
                    return
                broadcast.bypass_until_key = True
                broadcast.kf_last_reset_ts = now
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            return

    try:
        task = broadcast.kf_watchdog_task
        if task is not None and not task.done():
            task.cancel()
        broadcast.kf_watchdog_task = asyncio.create_task(_watchdog(broadcast.last_key_seq))
    except Exception:  # pragma: no cover - scheduler failure is logged for diagnosis
        logger.debug("start watchdog failed", exc_info=True)


def mark_config_dirty(state: PixelChannelState) -> None:
    """Flag that the next encoder configuration must be broadcast."""

    state.needs_config = True


async def maybe_send_video_config(
    state: PixelChannelState,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
    avcc: Optional[bytes],
    send_state_json: Callable[[dict], Awaitable[None]],
) -> None:
    """Send the latest avcC block if it changed or the channel requested it."""

    if avcc is None:
        state.needs_config = True
        return

    if not state.needs_config and state.last_avcc == avcc:
        return

    payload = build_video_config_payload(config, avcc)
    try:
        await send_state_json(payload)
    except Exception:
        logger.debug("video_config broadcast failed", exc_info=True)
        return

    state.last_avcc = avcc
    state.needs_config = False
    try:
        metrics.inc("napari_cuda_video_config_sends")
    except Exception:  # pragma: no cover - defensive metrics guard
        logger.debug("metrics inc napari_cuda_video_config_sends failed", exc_info=True)


def enqueue_frame(
    state: PixelChannelState,
    packet: FramePacket,
    *,
    metrics: "Metrics",
) -> None:
    """Attempt to queue a frame for broadcast, dropping oldest on overflow."""

    queue = state.broadcast.frame_queue
    try:
        queue.put_nowait(packet)
        try:
            metrics.set("napari_cuda_frame_queue_depth", float(queue.qsize()))
        except Exception:  # pragma: no cover - defensive metrics guard
            logger.debug("metrics set frame_queue_depth failed", exc_info=True)
        return
    except asyncio.QueueFull:
        pass

    drain_and_put(state, packet)
    try:
        metrics.inc("napari_cuda_frames_dropped")
    except Exception:  # pragma: no cover
        logger.debug("metrics inc frames_dropped failed", exc_info=True)


def drain_and_put(state: PixelChannelState, packet: FramePacket) -> None:
    """Drop the oldest frame and enqueue the replacement."""

    queue = state.broadcast.frame_queue
    try:
        while not queue.empty():
            queue.get_nowait()
    except Exception:
        logger.debug("Queue drain error", exc_info=True)
    try:
        queue.put_nowait(packet)
    except Exception:
        logger.debug("Frame enqueue error after drain", exc_info=True)
        return

    broadcast = state.broadcast
    broadcast.drops_total += 1
    if (broadcast.drops_total % 100) == 1:
        logger.info(
            "Pixel queue full: dropped oldest (total drops=%d)",
            broadcast.drops_total,
        )


async def run_channel_loop(
    state: PixelChannelState,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
) -> None:
    """Run the broadcaster loop using the stored configuration."""

    broadcast_cfg = pixel_broadcaster.PixelBroadcastConfig(
        width=config.width,
        height=config.height,
        codec=config.codec_id,
        fps=config.fps,
    )
    await pixel_broadcaster.broadcast_loop(state.broadcast, broadcast_cfg, metrics)
