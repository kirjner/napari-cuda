import asyncio
import time
from collections import Counter

import pytest

from napari_cuda.server import pixel_broadcaster, pixel_channel


class DummyMetrics:
    def __init__(self) -> None:
        self.counts = Counter()
        self.values: dict[str, float] = {}

    def inc(self, name: str, value: float | int = 1) -> None:
        self.counts[name] += value

    def set(self, name: str, value: float) -> None:
        self.values[name] = value


def test_enqueue_frame_drops_oldest() -> None:
    async def runner() -> None:
        queue: asyncio.Queue[pixel_broadcaster.FramePacket] = asyncio.Queue(maxsize=1)
        broadcast = pixel_broadcaster.PixelBroadcastState(
            frame_queue=queue,
            clients=set(),
            log_sends=False,
        )
        state = pixel_channel.PixelChannelState(broadcast=broadcast, kf_watchdog_cooldown_s=0.3)
        metrics = DummyMetrics()

        packet1: pixel_broadcaster.FramePacket = (b"a", 0, 0, 0.0)
        packet2: pixel_broadcaster.FramePacket = (b"b", 0, 1, 0.0)

        pixel_channel.enqueue_frame(state, packet1, metrics=metrics)
        assert queue.qsize() == 1
        assert metrics.values["napari_cuda_frame_queue_depth"] == pytest.approx(1.0)

        pixel_channel.enqueue_frame(state, packet2, metrics=metrics)
        assert queue.qsize() == 1
        queued = queue.get_nowait()
        assert queued == packet2
        assert state.broadcast.drops_total == 1
        assert metrics.counts["napari_cuda_frames_dropped"] == 1

    asyncio.run(runner())


def test_maybe_send_video_config_tracks_cache() -> None:
    async def runner() -> None:
        queue: asyncio.Queue[pixel_broadcaster.FramePacket] = asyncio.Queue()
        broadcast = pixel_broadcaster.PixelBroadcastState(
            frame_queue=queue,
            clients=set(),
            log_sends=False,
        )
        state = pixel_channel.PixelChannelState(broadcast=broadcast, kf_watchdog_cooldown_s=0.3)
        metrics = DummyMetrics()
        config = pixel_channel.PixelChannelConfig(
            width=640,
            height=480,
            fps=60.0,
            codec_id=1,
            codec_name="h264",
            kf_watchdog_cooldown_s=0.3,
        )

        sent: list[dict] = []

        async def sender(payload: dict) -> None:
            sent.append(payload)

        avcc_blob = b"test-avcc"

        await pixel_channel.maybe_send_video_config(
            state,
            config=config,
            metrics=metrics,
            avcc=avcc_blob,
            send_state_json=sender,
        )
        assert sent, "expected config broadcast"
        assert state.last_avcc == avcc_blob
        assert not state.needs_config
        assert metrics.counts["napari_cuda_video_config_sends"] == 1

        # Second call with unchanged avcC should be a no-op
        sent.clear()
        await pixel_channel.maybe_send_video_config(
            state,
            config=config,
            metrics=metrics,
            avcc=avcc_blob,
            send_state_json=sender,
        )
        assert sent == []

    asyncio.run(runner())


def test_start_watchdog_respects_cooldown() -> None:
    async def runner() -> None:
        queue: asyncio.Queue[pixel_broadcaster.FramePacket] = asyncio.Queue()
        broadcast = pixel_broadcaster.PixelBroadcastState(
            frame_queue=queue,
            clients=set(),
            log_sends=False,
        )
        state = pixel_channel.PixelChannelState(broadcast=broadcast, kf_watchdog_cooldown_s=0.05)

        calls: list[float] = []

        def reset_encoder() -> bool:
            calls.append(asyncio.get_running_loop().time())
            return True

        pixel_channel.start_watchdog(state, reset_encoder=reset_encoder, watchdog_delay_s=0.01)
        await asyncio.sleep(0.02)
        assert calls, "watchdog should trigger reset"
        first_call = calls[-1]

        # Set recent reset time to trigger cooldown and ensure second watchdog does nothing
        state.broadcast.kf_last_reset_ts = time.time()
        pixel_channel.start_watchdog(state, reset_encoder=reset_encoder, watchdog_delay_s=0.01)
        await asyncio.sleep(0.02)
        assert calls[-1] == pytest.approx(first_call)

    asyncio.run(runner())
