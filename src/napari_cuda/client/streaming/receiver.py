from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import websockets
import struct

logger = logging.getLogger(__name__)


# Server prepends a fixed header before Annex B bitstream
HEADER_STRUCT = struct.Struct('!IdIIBBH')  # seq:uint32, ts:double, w:uint32, h:uint32, codec:uint8, flags:uint8, reserved:uint16


@dataclass
class Packet:
    seq: int
    ts: float
    width: int
    height: int
    codec: int
    flags: int
    payload: memoryview


class PixelReceiver:
    """Receives pixel stream frames over WebSocket and forwards via callbacks.

    This component owns the network loop and reconnection policy only. All
    decode/gating/scheduling decisions happen in the consumer callback.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_connected: Optional[Callable[[], None]] = None,
        on_frame: Optional[Callable[[Packet], None]] = None,
        on_disconnect: Optional[Callable[[Exception | None], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.on_connected = on_connected
        self.on_frame = on_frame
        self.on_disconnect = on_disconnect

    def run(self) -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(self._run())

    async def _run(self) -> None:
        url = f"ws://{self.host}:{self.port}"
        logger.info("Connecting to pixel stream at %s", url)
        while True:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Connected to pixel stream")
                    if self.on_connected:
                        try:
                            self.on_connected()
                        except Exception:
                            logger.debug("on_connected callback failed", exc_info=True)
                    async for message in ws:
                        if not isinstance(message, (bytes, bytearray, memoryview)):
                            continue
                        b = bytes(message)
                        if len(b) < HEADER_STRUCT.size:
                            continue
                        try:
                            hdr = b[: HEADER_STRUCT.size]
                            # Use memoryview to avoid copying payload on the hot path
                            payload = memoryview(b)[HEADER_STRUCT.size :]
                            seq, ts, w, h, codec, flags, _ = HEADER_STRUCT.unpack(hdr)
                        except Exception:
                            logger.debug("Invalid stream header", exc_info=True)
                            continue
                        if self.on_frame:
                            try:
                                self.on_frame(
                                    Packet(
                                        seq=int(seq),
                                        ts=float(ts),
                                        width=int(w),
                                        height=int(h),
                                        codec=int(codec),
                                        flags=int(flags),
                                        payload=payload,
                                    )
                                )
                            except Exception:
                                logger.debug("on_frame callback failed", exc_info=True)
            except Exception as e:
                logger.exception("Stream connection lost")
                if self.on_disconnect:
                    try:
                        self.on_disconnect(e)
                    except Exception:
                        logger.debug("on_disconnect callback failed", exc_info=True)
                await asyncio.sleep(5)
                logger.info("Reconnecting to stream...")
