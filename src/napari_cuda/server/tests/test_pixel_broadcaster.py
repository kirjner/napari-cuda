from __future__ import annotations

import asyncio
import time

import pytest

from napari_cuda.server.rendering import pixel_broadcaster


class DummyMetrics:
    def __init__(self) -> None:
        self.inc_calls: list[tuple[str, int]] = []
        self.set_calls: list[tuple[str, float]] = []

    def inc(self, name: str, value: int = 1) -> None:
        self.inc_calls.append((name, value))

    def set(self, name: str, value: float) -> None:
        self.set_calls.append((name, value))


class StubWebSocket:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[bytes] = []
        self.closed = False

    async def send(self, data: bytes) -> None:
        if self.fail:
            raise RuntimeError("send failure")
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    # Minimal transport shim for TCP_NODELAY helper
    @property
    def transport(self):  # pragma: no cover - only used in configure_socket
        class _Dummy:
            def get_extra_info(self, name: str):  # pragma: no cover
                return None

        return _Dummy()


def test_safe_send_removes_dead_client() -> None:
    async def _run() -> None:
        queue: asyncio.Queue[pixel_broadcaster.FramePacket] = asyncio.Queue()
        stub = StubWebSocket(fail=True)
        state = pixel_broadcaster.PixelBroadcastState(frame_queue=queue, clients={stub}, log_sends=False)

        await pixel_broadcaster.safe_send(state, stub, b"payload")

        assert stub.closed is True
        assert stub not in state.clients

    asyncio.run(_run())


def test_broadcast_loop_sends_keyframe_on_bypass() -> None:
    async def _run() -> None:
        queue: asyncio.Queue[pixel_broadcaster.FramePacket] = asyncio.Queue()
        stub = StubWebSocket()
        state = pixel_broadcaster.PixelBroadcastState(frame_queue=queue, clients={stub}, log_sends=False)
        state.bypass_until_key = True
        metrics = DummyMetrics()
        cfg = pixel_broadcaster.PixelBroadcastConfig(width=4, height=4, codec=1, fps=60.0)

        stamp = time.time()
        await queue.put((b"frame-bytes", 0x01, 123, stamp))

        task = asyncio.create_task(pixel_broadcaster.broadcast_loop(state, cfg, metrics))
        await asyncio.sleep(0.05)

        assert stub.sent, "Expected frame payload to be sent to the client"
        assert state.bypass_until_key is False
        assert state.last_key_seq == (123 & 0xFFFFFFFF)
        assert pytest.approx(state.last_key_ts or 0.0, rel=1e-6) == stamp
        assert any(name == "napari_cuda_keyframes_total" for name, _ in metrics.inc_calls)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
