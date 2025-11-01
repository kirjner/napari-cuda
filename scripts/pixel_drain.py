#!/usr/bin/env python
"""Consume frames from the napari-cuda pixel stream without touching state."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Final

import websockets

_LOG = logging.getLogger(__name__)
_HEADER_BYTES: Final[int] = 24  # size of napari-cuda stream header (ignored here)


async def _drain_once(uri: str, frames: int, idle_timeout: float, max_runtime: float) -> tuple[int, float]:
    received = 0
    start_ts = asyncio.get_event_loop().time()
    async with websockets.connect(uri) as ws:
        _LOG.info("Pixel drain connected to %s", uri)
        while received < frames:
            elapsed = asyncio.get_event_loop().time() - start_ts
            if max_runtime > 0 and elapsed >= max_runtime:
                _LOG.info("Pixel drain reached max runtime %.2fs", elapsed)
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
            except TimeoutError:
                _LOG.warning("Pixel drain timeout after %d frames", received)
                break
            except Exception:
                _LOG.exception("Pixel drain receive failed")
                break
            if not isinstance(msg, (bytes, bytearray, memoryview)):
                continue
            if len(msg) <= _HEADER_BYTES:
                continue
            received += 1
    return received, asyncio.get_event_loop().time() - start_ts


async def _drain(uri: str, frames: int, idle_timeout: float, reconnect_delay: float, max_runtime: float) -> tuple[int, float]:
    total = 0
    total_time = 0.0
    backoff = reconnect_delay
    start_wall = time.monotonic()
    while total < frames:
        if max_runtime > 0 and (time.monotonic() - start_wall) >= max_runtime:
            break
        try:
            remaining_time = max_runtime - total_time if max_runtime > 0 else 0.0
            taken, dt = await _drain_once(uri, frames - total, idle_timeout, remaining_time)
            total += taken
            total_time += dt
            if taken == 0:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)
            else:
                backoff = reconnect_delay
        except TimeoutError:
            _LOG.info("Pixel drain handshake timed out; stopping after %d frames", total)
            break
        except Exception:
            _LOG.exception("Pixel drain connection failure")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 10.0)
    return total, max(total_time, time.monotonic() - start_wall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read frames from the napari-cuda pixel channel")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--pixel-port", type=int, default=8082, help="Pixel WebSocket port")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to capture before exiting")
    parser.add_argument("--idle-timeout", type=float, default=2.0, help="Timeout waiting for the next frame (seconds)")
    parser.add_argument("--reconnect-delay", type=float, default=1.0, help="Initial reconnect backoff in seconds")
    parser.add_argument("--max-runtime", type=float, default=45.0, help="Ceiling on total drain runtime (seconds; 0=unbounded)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    uri = f"ws://{args.host}:{args.pixel_port}"
    total, elapsed = asyncio.run(
        _drain(
            uri,
            max(1, args.frames),
            max(0.1, args.idle_timeout),
            max(0.1, args.reconnect_delay),
            max(0.0, args.max_runtime),
        )
    )
    fps = total / elapsed if elapsed > 0 else 0.0
    _LOG.info("Pixel drain complete: %d frames in %.2fs (%.2f fps)", total, elapsed, fps)


if __name__ == "__main__":
    main()
