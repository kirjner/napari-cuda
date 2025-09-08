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
from typing import Optional, Deque, Tuple
from collections import deque

import av  # PyAV for decoding
import websockets
from qtpy import QtWidgets, QtGui, QtCore
import qasync
import sys

logger = logging.getLogger(__name__)

try:
    # Lightweight AVCC → AnnexB shim for PyAV
    from napari_cuda.client.avcc_shim import normalize_to_annexb  # type: ignore
except Exception:  # pragma: no cover - optional import for older snapshots
    normalize_to_annexb = None  # type: ignore

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


def parse_header(b: bytes) -> FrameHeader:
    seq, ts, w, h, codec, flags, reserved = HEADER_STRUCT.unpack(b)
    return FrameHeader(seq=seq, ts=ts, width=w, height=h, codec=codec, flags=flags, reserved=reserved)


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
        self._vt_decoder = None  # VideoToolbox decoder instance when enabled/initialized
        self._vt_enabled = False
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
        # No client-side color correction; rely on decoder
        # No client-side color matrix forcing; rely on bitstream VUI
        # Jitter buffer and display pacing
        try:
            self._display_fps = float(os.getenv('NAPARI_CUDA_CLIENT_DISPLAY_FPS', '60'))
        except Exception:
            self._display_fps = 60.0
        # VT toggle (opt-in)
        try:
            self._vt_enabled = (sys.platform == 'darwin') and bool(int(os.getenv('NAPARI_CUDA_USE_VT', '0')))
            if self._vt_enabled:
                logger.info("Client: VideoToolbox path requested (NAPARI_CUDA_USE_VT=1)")
        except Exception:
            self._vt_enabled = False
        try:
            buf_n = int(os.getenv('NAPARI_CUDA_CLIENT_BUFFER_FRAMES', '3'))
        except Exception:
            buf_n = 3
        self._buffer: Deque[Tuple[QtGui.QImage, float]] = deque(maxlen=max(1, buf_n))
        self._last_qimg: Optional[QtGui.QImage] = None
        self._display_timer = QtCore.QTimer(self)
        self._display_timer.setTimerType(QtCore.Qt.PreciseTimer)
        interval_ms = max(1, int(round(1000.0 / max(1.0, self._display_fps))))
        self._display_timer.setInterval(interval_ms)
        self._display_timer.timeout.connect(self._on_display_tick)
        self._display_timer.start()

    async def run(self) -> None:
        # Launch both connections
        pixel_uri = f"ws://{self.host}:{self.pixel_port}"
        state_uri = f"ws://{self.host}:{self.state_port}"
        try:
            async with websockets.connect(pixel_uri, max_size=None) as pixel_ws:
                # State channel: receive video_config (avcC/fps) and send ping
                state_task = asyncio.create_task(self._state_channel(state_uri, kick_keyframe=True))
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
                                # Keep quiet; rely on first-frame header log above
                                continue
                        # Decode via VideoToolbox if enabled and initialized
                        if self._vt_enabled and self._vt_decoder is not None:
                            try:
                                qimg = self._decode_vt_to_qimage(bytes(payload))
                                if qimg is not None:
                                    self._buffer.append((qimg, header.ts))
                                    continue
                            except Exception as e:
                                logger.error("VT decode error; falling back to PyAV: %s", e, exc_info=True)
                                self._vt_enabled = False
                                self._vt_decoder = None
                        # PyAV path (AVCC→AnnexB shim if available)
                        if normalize_to_annexb is not None:
                            annexb, converted = normalize_to_annexb(payload)
                            if not hasattr(self, '_logged_payload_format'):
                                fmt = 'AVCC' if converted else 'AnnexB'
                                logger.info("Client bitstream detected: %s", fmt)
                                setattr(self, '_logged_payload_format', True)
                            packet = av.Packet(annexb)
                        else:
                            packet = av.Packet(bytes(payload))
                        frames = self._decoder.decode(packet)
                        for f in frames:
                            # Prefer a deterministic ndarray conversion to control channel order
                            pixfmt = os.getenv('NAPARI_CUDA_CLIENT_PIXEL_FMT', 'rgb24').lower()
                            if pixfmt not in {'rgb24', 'bgr24'}:
                                pixfmt = 'rgb24'
                            try:
                                arr = f.to_ndarray(format=pixfmt)
                                pixfmt_eff = pixfmt
                                h, w, c = arr.shape
                                if c != 3:
                                    raise ValueError(f'Unexpected channels in {pixfmt}: {c}')
                                qfmt = QtGui.QImage.Format_RGB888 if pixfmt_eff == 'rgb24' else QtGui.QImage.Format_BGR888
                                if not hasattr(self, '_logged_pixfmt'):
                                    logger.info("Client decode pixfmt=%s -> QImage fmt=%s (%dx%d)", pixfmt, qfmt, w, h)
                                    setattr(self, '_logged_pixfmt', True)
                                qimg = QtGui.QImage(arr.data, w, h, 3 * w, qfmt).copy()
                                self._buffer.append((qimg, header.ts))
                            except Exception:
                                img = f.to_image()
                                qimg = self._pil_to_qimage(img)
                                self._buffer.append((qimg, header.ts))
                finally:
                    state_task.cancel()
        except Exception as e:
            logger.exception("Client error: %s", e)

    async def _state_channel(self, uri: str, kick_keyframe: bool = False) -> None:
        try:
            async with websockets.connect(uri) as ws:
                # periodic ping
                async def _pinger():
                    while self._running:
                        try:
                            await ws.send('{"type":"ping"}')
                        except Exception:
                            break
                        await asyncio.sleep(2.0)
                ping_task = asyncio.create_task(_pinger())
                if kick_keyframe:
                    try:
                        await ws.send('{"type":"request_keyframe"}')
                    except Exception as e:
                        logger.debug("request_keyframe send failed: %s", e)
                try:
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        t = data.get('type')
                        if t == 'video_config':
                            try:
                                fps = float(data.get('fps') or 0.0)
                                if fps > 0:
                                    interval_ms = max(1, int(round(1000.0 / max(1.0, fps))))
                                    self._display_timer.setInterval(interval_ms)
                                    logger.info("Client video_config: fps=%.3f interval=%dms", fps, interval_ms)
                                width = int(data.get('width') or 0)
                                height = int(data.get('height') or 0)
                                avcc_b64 = data.get('data')
                                if width > 0 and height > 0:
                                    self._init_vt_if_possible(avcc_b64, width, height)
                            except Exception as e:
                                logger.debug("video_config parse failed: %s", e)
                finally:
                    ping_task.cancel()
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

    # No client-side YUV->RGB override

    # No client-side YUV->RGB override; rely on decoder defaults

    @QtCore.Slot()
    def _on_display_tick(self) -> None:
        # Drain buffer and present the newest frame
        try:
            latest: Optional[Tuple[QtGui.QImage, float]] = None
            while self._buffer:
                latest = self._buffer.pop()
            if latest is not None:
                self._last_qimg = latest[0]
            if self._last_qimg is not None:
                self.frame_ready.emit(self._last_qimg)
        except Exception as e:
            logger.debug("display tick error: %s", e)

    def _init_vt_if_possible(self, avcc_b64: Optional[str], width: int, height: int) -> None:
        if not self._vt_enabled:
            return
        if not avcc_b64:
            logger.warning("VT requested but missing avcC in video_config; skipping VT init")
            return
        try:
            import base64
            avcc = base64.b64decode(avcc_b64)
            from napari_cuda.client.vt_decoder import VideoToolboxDecoder, is_vt_available  # type: ignore
            if not is_vt_available():
                logger.warning("VideoToolbox frameworks not available; install PyObjC frameworks to enable VT")
                return
            self._vt_decoder = VideoToolboxDecoder(avcc, width, height)
            logger.info("VideoToolbox decoder initialized: %dx%d BGRA output", width, height)
        except Exception as e:
            logger.error("VT init failed; falling back to PyAV: %s", e, exc_info=True)
            self._vt_decoder = None
            self._vt_enabled = False

    def _decode_vt_to_qimage(self, avcc_au: bytes) -> Optional[QtGui.QImage]:
        if not self._vt_decoder:
            return None
        try:
            img_buf = self._vt_decoder.decode(avcc_au)
            if img_buf is None:
                return None
            # Map CVPixelBuffer to QImage with a copy (safer for first integration)
            import CoreVideo  # type: ignore
            CoreVideo.CVPixelBufferLockBaseAddress(img_buf, 0)
            try:
                w = CoreVideo.CVPixelBufferGetWidth(img_buf)
                h = CoreVideo.CVPixelBufferGetHeight(img_buf)
                bpr = CoreVideo.CVPixelBufferGetBytesPerRow(img_buf)
                base = CoreVideo.CVPixelBufferGetBaseAddress(img_buf)
                import ctypes
                buf_len = int(bpr) * int(h)
                raw = ctypes.string_at(base, buf_len)
                qimg = QtGui.QImage(raw, int(w), int(h), int(bpr), QtGui.QImage.Format_ARGB32).copy()
                return qimg
            finally:
                CoreVideo.CVPixelBufferUnlockBaseAddress(img_buf, 0)
        except Exception as e:
            logger.debug("VT map/QImage failed: %s", e)
            return None


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
