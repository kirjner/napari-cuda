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

from napari_cuda.server.rendering import pixel_broadcaster
from napari_cuda.protocol import NotifyStreamPayload

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from napari_cuda.server.app.metrics_core import Metrics  # circular guard at runtime

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
    needs_stream_config: bool = True
    last_avcc: Optional[bytes] = None
    kf_watchdog_cooldown_s: float = 0.30


def build_notify_stream_payload(config: PixelChannelConfig, avcc: bytes) -> NotifyStreamPayload:
    """Build a ``notify.stream`` payload from the pixel channel config."""

    latency_policy = {
        "max_buffer_ms": float(getattr(config, "max_buffer_ms", 120.0)),
        "grace_keyframe_ms": float(config.kf_watchdog_cooldown_s * 1000.0),
    }

    return NotifyStreamPayload(
        codec=str(config.codec_name),
        format="avcc",
        fps=float(config.fps),
        frame_size=(int(config.width), int(config.height)),
        nal_length_size=4,
        avcc=base64.b64encode(avcc).decode("ascii"),
        latency_policy=latency_policy,
        vt_hint=None,
    )


def prepare_client_attach(state: PixelChannelState) -> None:
    """Flush encoder buffers so the next client starts from a clean slate."""

    # If another client is already connected, leave the queue alone; the stream is live.
    if state.broadcast.clients:
        return

    queue = state.broadcast.frame_queue
    try:
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    except Exception:
        logger.debug("Pixel queue flush failed during client attach prep", exc_info=True)

    broadcast = state.broadcast
    broadcast.bypass_until_key = True
    broadcast.waiting_for_keyframe = True
    broadcast.last_key_seq = None
    broadcast.last_key_ts = None
    broadcast.last_send_ts = None
    broadcast.send_count = 0
    state.needs_stream_config = True


async def ingest_client(
    state: PixelChannelState,
    ws: websockets.WebSocketServerProtocol,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
    reset_encoder: Callable[[], bool],
    send_stream: Callable[[NotifyStreamPayload], Awaitable[None]],
    on_clients_change: Callable[[], None],
    on_client_join: Optional[Callable[[], None]] = None,
) -> None:
    """Ingest a pixel-channel websocket connection."""

    state.broadcast.clients.add(ws)
    on_clients_change()
    pixel_broadcaster.configure_socket(ws, label="pixel ws")
    state.needs_stream_config = True

    state.broadcast.bypass_until_key = True
    state.broadcast.waiting_for_keyframe = True
    state.needs_stream_config = True
    if on_client_join is not None:
        try:
            on_client_join()
        except Exception:
            logger.debug("pixel client join hook failed", exc_info=True)

    try:
        await ws.wait_closed()
    finally:
        state.broadcast.clients.discard(ws)
        on_clients_change()
        if not state.broadcast.clients:
            prepare_client_attach(state)


async def ensure_keyframe(
    state: PixelChannelState,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
    reset_encoder: Callable[[], bool],
    send_stream: Callable[[NotifyStreamPayload], Awaitable[None]],
    capture_avcc: Optional[Callable[[], Optional[bytes]]] = None,
) -> None:
    """Force the next frame to be a keyframe and schedule a watchdog."""

    # Emit the latest stream configuration before we gate on the new keyframe.
    avcc_bytes = state.last_avcc
    if avcc_bytes is None and capture_avcc is not None:
        avcc_bytes = capture_avcc()
        if avcc_bytes is not None:
            state.last_avcc = avcc_bytes

    if avcc_bytes is not None:
        if isinstance(avcc_bytes, memoryview):
            avcc_bytes = avcc_bytes.tobytes()
        elif isinstance(avcc_bytes, bytearray):
            avcc_bytes = bytes(avcc_bytes)
        elif not isinstance(avcc_bytes, bytes):
            raise AssertionError("capture_avcc must return bytes-like avcc payload")

        state.last_avcc = avcc_bytes
        await send_stream(build_notify_stream_payload(config, avcc_bytes))
        state.needs_stream_config = False

    now = time.time()
    if state.broadcast.waiting_for_keyframe:
        last_reset = state.broadcast.kf_last_reset_ts
        if last_reset is not None and (now - last_reset) < state.kf_watchdog_cooldown_s:
            logger.debug("ensure_keyframe skipped: awaiting previous keyframe")
            return

    # TODO(encoder-idr): Re-introduce `force_idr` handling once NVENC reliably
    # produces an IDR on demand. For now we reset the encoder so the next frame
    # is guaranteed to be a keyframe, but gate repeated resets while the caller
    # is still waiting on the previous one.
    if not reset_encoder():
        logger.debug("Encoder reset unavailable during keyframe ensure")
        return
    state.broadcast.kf_last_reset_ts = now

    state.broadcast.bypass_until_key = True
    state.broadcast.waiting_for_keyframe = True
    state.needs_stream_config = True
    try:
        metrics.inc("napari_cuda_encoder_resets")
    except Exception:  # pragma: no cover - defensive metrics guard
        logger.debug("metrics inc encoder_resets failed", exc_info=True)

    if state.last_avcc is not None and state.needs_stream_config:
        try:
            await send_stream(build_notify_stream_payload(config, state.last_avcc))
            state.needs_stream_config = False
        except Exception:
            logger.debug("ensure_keyframe post-reset notify.stream failed", exc_info=True)


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
                broadcast.waiting_for_keyframe = True
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            return

    try:
        task = broadcast.kf_watchdog_task
        if task is not None and not task.done():
            task.cancel()
        broadcast.kf_watchdog_task = asyncio.create_task(_watchdog(broadcast.last_key_seq))
    except Exception:  # pragma: no cover - scheduler failure is logged for diagnosis
        logger.debug("start watchdog failed", exc_info=True)


def mark_stream_config_dirty(state: PixelChannelState) -> None:
    """Flag that the next encoder configuration must be broadcast."""

    state.needs_stream_config = True


async def maybe_send_stream_config(
    state: PixelChannelState,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
    avcc: Optional[bytes],
    send_stream: Callable[[NotifyStreamPayload], Awaitable[None]],
) -> None:
    """Send a ``notify.stream`` snapshot when the avcC block changes."""

    if avcc is None:
        state.needs_stream_config = True
        return

    if not state.needs_stream_config and state.last_avcc == avcc:
        return

    payload = build_notify_stream_payload(config, avcc)
    try:
        await send_stream(payload)
    except Exception:
        logger.debug("notify.stream broadcast failed", exc_info=True)
        return

    state.last_avcc = avcc
    state.needs_stream_config = False
    try:
        metrics.inc("napari_cuda_stream_config_sends")
    except Exception:  # pragma: no cover - defensive metrics guard
        logger.debug("metrics inc napari_cuda_stream_config_sends failed", exc_info=True)


async def publish_avcc(
    state: PixelChannelState,
    *,
    config: PixelChannelConfig,
    metrics: "Metrics",
    avcc: Optional[bytes],
    send_stream: Callable[[NotifyStreamPayload], Awaitable[None]],
) -> None:
    """Publish a cached avcC block and broadcast ``notify.stream`` when needed."""

    if avcc is None:
        state.needs_stream_config = True
        return
    if isinstance(avcc, memoryview):
        avcc = avcc.tobytes()
    elif isinstance(avcc, bytearray):
        avcc = bytes(avcc)
    elif not isinstance(avcc, bytes):
        raise AssertionError("publish_avcc requires bytes-like avcc payload")

    if state.last_avcc == avcc and not state.needs_stream_config:
        return

    state.last_avcc = avcc
    if state.needs_stream_config:
        payload = build_notify_stream_payload(config, avcc)
        try:
            await send_stream(payload)
        except Exception:
            logger.debug("publish_avcc notify.stream failed", exc_info=True)
            return
        state.needs_stream_config = False
        try:
            metrics.inc("napari_cuda_stream_config_sends")
        except Exception:  # pragma: no cover - defensive metrics guard
            logger.debug("metrics inc napari_cuda_stream_config_sends failed", exc_info=True)


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
