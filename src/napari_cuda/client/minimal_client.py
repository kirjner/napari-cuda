"""
Minimal Qt client for napari-cuda EGL headless server.

Receives binary NVENC (H.264) frames over WebSocket, decodes with PyAV (CPU),
and displays frames in a QLabel. Sends optional control messages (ping).
"""

from __future__ import annotations

import asyncio
import struct
import logging
import os
from dataclasses import dataclass
from typing import Optional

import av  # PyAV for decoding
import websockets
from qtpy import QtWidgets, QtGui, QtCore
import qasync

logger = logging.getLogger(__name__)


HEADER_STRUCT = struct.Struct('!IdIIBBH')  # seq:uint32, ts:double, w:uint32, h:uint32, codec:uint8, flags:uint8, reserved:uint16


@dataclass
class FrameHeader:
    seq: int
    ts: float
    width: int
    height: int
    codec: int
    flags: int
    reserved: int

    # Derived color hints (packed in reserved)
    color_space: int = 0   # 0=unspec, 1=BT601, 2=BT709
    color_range: int = 0   # 0=unspec, 1=limited, 2=full
    chroma: int = 0        # 0=unspec, 1=NV12, 2=YUV444, 3=RGB


def parse_header(b: bytes) -> FrameHeader:
    seq, ts, w, h, codec, flags, reserved = HEADER_STRUCT.unpack(b)
    cs = reserved & 0xF
    cr = (reserved >> 4) & 0xF
    ch = (reserved >> 8) & 0x3
    return FrameHeader(seq=seq, ts=ts, width=w, height=h, codec=codec, flags=flags,
                       reserved=reserved, color_space=cs, color_range=cr, chroma=ch)


class VideoWidget(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)

    @QtCore.Slot(object)
    def set_frame(self, image: QtGui.QImage) -> None:
        self.setPixmap(QtGui.QPixmap.fromImage(image))


class MinimalClient(QtCore.QObject):
    frame_ready = QtCore.Signal(object)

    def __init__(self, host: str, state_port: int, pixel_port: int) -> None:
        super().__init__()
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        self._decoder = av.codec.CodecContext.create('h264', 'r')  # default to H.264
        self._running = True
        # Log a few header fields on startup for sanity (seq/flags/size)
        try:
            import os as _os
            self._log_headers_remaining = int(_os.getenv('NAPARI_CUDA_LOG_HEADERS', '5'))
        except Exception:
            self._log_headers_remaining = 5
        # Log client pixel format env at startup
        try:
            p = os.getenv('NAPARI_CUDA_CLIENT_PIXEL_FMT', 'rgb24').lower()
            s = os.getenv('NAPARI_CUDA_CLIENT_SWAP_RB', '0')
            logger.info("Client env: NAPARI_CUDA_CLIENT_PIXEL_FMT=%s, NAPARI_CUDA_CLIENT_SWAP_RB=%s", p, s)
        except Exception as e:
            logger.debug("Client env log failed: %s", e)
        # No client-side color matrix forcing; rely on bitstream VUI

    async def run(self) -> None:
        # Launch both connections
        pixel_uri = f"ws://{self.host}:{self.pixel_port}"
        state_uri = f"ws://{self.host}:{self.state_port}"
        try:
            async with websockets.connect(pixel_uri, max_size=None) as pixel_ws:
                # Optional: keep a lightweight state WS for ping; don't fail if not available
                state_task = asyncio.create_task(self._state_pinger(state_uri, kick_keyframe=True))
                try:
                    seen_keyframe = False
                    while self._running:
                        data = await pixel_ws.recv()
                        if not isinstance(data, (bytes, bytearray)):
                            continue
                        if len(data) <= HEADER_STRUCT.size:
                            continue
                        header = parse_header(data[:HEADER_STRUCT.size])
                        # Optional brief header logging for startup sanity
                        if self._log_headers_remaining > 0:
                            logger.info(
                                "Pixel header: seq=%d flags=0x%02x size=%dx%d",
                                header.seq, header.flags, header.width, header.height,
                            )
                            self._log_headers_remaining -= 1
                        payload = memoryview(data)[HEADER_STRUCT.size:]
                        # Wait for keyframe before feeding decoder to avoid mid-GOP starts
                        if not seen_keyframe:
                            if header.flags & 0x01:
                                seen_keyframe = True
                            else:
                                if self._log_headers_remaining > 0:
                                    logger.info("Waiting for keyframe before decoding...")
                                continue
                        # Decode payload
                        packet = av.Packet(bytes(payload))
                        frames = self._decoder.decode(packet)
                        for f in frames:
                            # Prefer a deterministic ndarray conversion to control channel order
                            pixfmt = os.getenv('NAPARI_CUDA_CLIENT_PIXEL_FMT', 'rgb24').lower()
                            if pixfmt not in {'rgb24', 'bgr24'}:
                                pixfmt = 'rgb24'
                            # Log frame color metadata once if available
                            if not hasattr(self, '_logged_colorspace'):
                                try:
                                    cs = getattr(f, 'color_space', None)
                                    cr = getattr(f, 'color_range', None)
                                    cp = getattr(f, 'color_primaries', None)
                                    ct = getattr(f, 'color_trc', None)
                                    logger.info("Decoder frame color: space=%s range=%s primaries=%s trc=%s", cs, cr, cp, ct)
                                except Exception:
                                    pass
                                setattr(self, '_logged_colorspace', True)
                            try:
                                # If decoder did not set colors and server provided hints via header,
                                # decode to YUV444 and apply RGB matrix using header values.
                                use_header_matrix = False
                                if getattr(f, 'color_space', None) is None:
                                    # Look back at the last parsed header from this packet
                                    # We still have 'header' in scope above; safe in loop body
                                    if header.chroma in (1, 2) and header.color_space in (1, 2) and header.color_range in (1, 2):
                                        use_header_matrix = True
                                if use_header_matrix:
                                    yuv = f.to_ndarray(format='yuv444p')
                                    matrix = 'BT709' if header.color_space == 2 else 'BT601'
                                    rng = 'FULL' if header.color_range == 2 else 'LIMITED'
                                    arr = self._yuv444_to_rgb(yuv, matrix=matrix, rng=rng)
                                    # Force RGB path for QImage
                                    pixfmt_eff = 'rgb24'
                                else:
                                    arr = f.to_ndarray(format=pixfmt)
                                    pixfmt_eff = pixfmt
                                h, w, c = arr.shape
                                if c != 3:
                                    raise ValueError(f'Unexpected channels in {pixfmt}: {c}')
                                if pixfmt_eff == 'rgb24':
                                    qfmt = QtGui.QImage.Format_RGB888
                                else:
                                    qfmt = QtGui.QImage.Format_BGR888
                                # Log once to confirm active pixel format branch
                                if not hasattr(self, '_logged_pixfmt'):
                                    logger.info("Client decode pixfmt=%s -> QImage fmt=%s (%dx%d)", pixfmt, qfmt, w, h)
                                    setattr(self, '_logged_pixfmt', True)
                                qimg = QtGui.QImage(arr.data, w, h, 3 * w, qfmt)
                                # Deep copy because arr will be released after loop
                                self.frame_ready.emit(qimg.copy())
                            except Exception:
                                # Fallback: use PIL conversion
                                img = f.to_image()
                                qimg = self._pil_to_qimage(img)
                                self.frame_ready.emit(qimg)
                finally:
                    state_task.cancel()
        except Exception as e:
            logger.exception("Client error: %s", e)

    async def _state_pinger(self, uri: str, kick_keyframe: bool = False) -> None:
        try:
            async with websockets.connect(uri) as ws:
                if kick_keyframe:
                    try:
                        await ws.send('{"type":"request_keyframe"}')
                    except Exception as e:
                        logger.debug("request_keyframe send failed: %s", e)
                while self._running:
                    await ws.send('{"type":"ping"}')
                    await asyncio.sleep(2.0)
        except Exception as e:
            logger.debug("State pinger ended: %s", e)

    @staticmethod
    def _pil_to_qimage(img) -> QtGui.QImage:
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        data = img.tobytes()
        qimg = QtGui.QImage(data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        return qimg.copy()  # deep copy since data is ephemeral

    @staticmethod
    def _yuv444_to_rgb(yuv, matrix: str, rng: str):
        import numpy as np
        # yuv444p from PyAV is typically shape (3, H, W)
        if yuv.ndim == 3 and yuv.shape[0] == 3:
            Y = yuv[0].astype(np.float32)
            U = yuv[1].astype(np.float32)
            V = yuv[2].astype(np.float32)
        else:
            Y = yuv[..., 0].astype(np.float32)
            U = yuv[..., 1].astype(np.float32)
            V = yuv[..., 2].astype(np.float32)
        if rng.upper() in ('FULL', 'JPEG'):
            y = Y / 255.0
            cb = (U - 128.0) / 255.0
            cr = (V - 128.0) / 255.0
        else:
            y = (Y - 16.0) / 219.0
            cb = (U - 128.0) / 224.0
            cr = (V - 128.0) / 224.0
        if matrix.upper() == 'BT709':
            r = y + 1.5748 * cr
            g = y - 0.1873 * cb - 0.4681 * cr
            b = y + 1.8556 * cb
        else:  # BT601
            r = y + 1.4020 * cr
            g = y - 0.344136 * cb - 0.714136 * cr
            b = y + 1.7720 * cb
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb

    # No client-side YUV->RGB override; rely on decoder defaults


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='napari-cuda minimal streaming client')
    parser.add_argument('--host', default=os.getenv('NAPARI_CUDA_HOST', '127.0.0.1'))
    parser.add_argument('--state-port', type=int, default=int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')))
    parser.add_argument('--pixel-port', type=int, default=int(os.getenv('NAPARI_CUDA_PIXEL_PORT', '8082')))
    args = parser.parse_args()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = QtWidgets.QMainWindow()
    window.setWindowTitle('napari-cuda client (PyAV decode)')
    video = VideoWidget()
    window.setCentralWidget(video)
    window.resize(960, 540)
    window.show()

    client = MinimalClient(args.host, args.state_port, args.pixel_port)
    client.frame_ready.connect(video.set_frame)

    async def runner():
        await client.run()

    with loop:
        loop.create_task(runner())
        loop.run_forever()


if __name__ == '__main__':
    main()
