"""
StreamingCanvas - Displays video stream from remote server.

This replaces the normal VispyCanvas with one that shows decoded video frames
instead of locally rendered content.
"""

import asyncio
import logging
import queue
import numpy as np
import os
import io
import ctypes
import json
import base64
from fractions import Fraction
from qtpy import QtCore
from threading import Thread

import websockets
import struct
from napari._vispy.canvas import VispyCanvas
from vispy.gloo import Texture2D, Program

logger = logging.getLogger(__name__)

# Simple shaders for displaying video texture
VERTEX_SHADER = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

FRAGMENT_SHADER = """
uniform sampler2D texture;
varying vec2 v_texcoord;

void main() {
    gl_FragColor = texture2D(texture, v_texcoord);
}
"""

# Server prepends a fixed header before Annex B bitstream
HEADER_STRUCT = struct.Struct('!IdIIBBH')  # seq:uint32, ts:double, w:uint32, h:uint32, codec:uint8, flags:uint8, reserved:uint16


class StreamingCanvas(VispyCanvas):
    """
    Canvas that displays video stream from remote server instead of
    rendering local content.
    """
    
    def __init__(self, viewer, server_host='localhost', server_port=8082, vt_smoke: bool = False, key_map_handler=None, state_port: int | None = None, **kwargs):
        """
        Initialize streaming canvas.
        
        Parameters
        ----------
        viewer : ProxyViewer
            The proxy viewer instance
        server_host : str
            Remote server hostname
        server_port : int
            Remote server pixel stream port
        """
        # Ensure we have a KeymapHandler; create a minimal one if not provided
        if key_map_handler is None:
            try:
                from napari.utils.key_bindings import KeymapHandler  # type: ignore
                key_map_handler = KeymapHandler()
                key_map_handler.keymap_providers = [viewer]
            except Exception:
                # Fallback to a dummy object with required attributes
                class _DummyKM:
                    def on_key_press(self, *a, **k):
                        pass
                    def on_key_release(self, *a, **k):
                        pass
                key_map_handler = _DummyKM()  # type: ignore
        # Forward to base canvas
        super().__init__(viewer, key_map_handler, **kwargs)
        
        self.server_host = server_host
        self.server_port = server_port
        self.state_port = int(state_port or int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')))
        # Offline VT smoke test mode (no server required)
        try:
            env_smoke = bool(int(os.getenv('NAPARI_CUDA_VT_SMOKE', '0')))
        except Exception:
            env_smoke = False
        self._vt_smoke = bool(vt_smoke or env_smoke)
        
        # Queue for decoded frames (latest-wins draining in draw)
        try:
            buf_n = int(os.getenv('NAPARI_CUDA_CLIENT_BUFFER_FRAMES', '3'))
        except Exception:
            buf_n = 3
        self.frame_queue = queue.Queue(maxsize=max(1, buf_n))
        
        # Video display resources
        self._video_texture = None
        self._video_program = None
        self._fullscreen_quad = None
        
        # Decoder (will be initialized based on stream type)
        self.decoder = None
        # VT live-decoder state
        self._vt_decoder = None
        self._vt_cfg_key = None
        self._vt_errors = 0
        
        # Start streaming or offline VT smoke thread
        if self._vt_smoke:
            self._streaming_thread = Thread(target=self._vt_smoke_worker, daemon=True)
            logger.info("StreamingCanvas in VT smoke test mode (offline)")
        else:
            self._streaming_thread = Thread(target=self._stream_worker, daemon=True)
        self._streaming_thread.start()

        # State channel thread: receive avcC (video_config) for VT init
        self._state_thread = Thread(target=self._state_worker, daemon=True)
        self._state_thread.start()
        
        # Override draw to show video instead
        self._scene_canvas.events.draw.disconnect()
        self._scene_canvas.events.draw.connect(self._draw_video_frame)
        # Timer-driven display at target fps
        try:
            fps = float(os.getenv('NAPARI_CUDA_CLIENT_DISPLAY_FPS', '60'))
        except Exception:
            fps = 60.0
        self._display_timer = QtCore.QTimer()
        self._display_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._display_timer.setInterval(max(1, int(round(1000.0 / max(1.0, fps)))))
        self._display_timer.timeout.connect(lambda: self._scene_canvas.update())
        self._display_timer.start()
        
        logger.info(f"StreamingCanvas initialized for {server_host}:{server_port}")

    def _state_worker(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(self._receive_state())

    async def _receive_state(self):
        url = f"ws://{self.server_host}:{self.state_port}"
        try:
            async with websockets.connect(url) as ws:
                # periodic ping
                async def _pinger():
                    while True:
                        try:
                            await ws.send('{"type":"ping"}')
                        except Exception:
                            break
                        await asyncio.sleep(2.0)
                ping_task = asyncio.create_task(_pinger())
                try:
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        if data.get('type') == 'video_config':
                            try:
                                width = int(data.get('width') or 0)
                                height = int(data.get('height') or 0)
                                fps = float(data.get('fps') or 0.0)
                                avcc_b64 = data.get('data')
                                if fps > 0:
                                    logger.info("Client video_config: fps=%.3f", fps)
                                if width > 0 and height > 0 and avcc_b64:
                                    self._init_vt_from_avcc(avcc_b64, width, height)
                            except Exception as e:
                                logger.warning("video_config parse failed: %s", e)
                finally:
                    ping_task.cancel()
        except Exception as e:
            logger.debug("State channel ended: %s", e)

    def _init_vt_from_avcc(self, avcc_b64: str, width: int, height: int) -> None:
        try:
            avcc = base64.b64decode(avcc_b64)
            cfg_key = (int(width), int(height), avcc)
            if self._vt_decoder is not None and self._vt_cfg_key == cfg_key:
                logger.info("VT already initialized; ignoring duplicate video_config")
                return
            from napari_cuda.client.vt_decoder import VideoToolboxDecoder, is_vt_available
            if not is_vt_available():
                logger.warning("VideoToolbox not available; cannot init VT")
                return
            self._vt_decoder = VideoToolboxDecoder(avcc, width, height)
            self._vt_cfg_key = cfg_key
            logger.info("VideoToolbox live decoder initialized: %dx%d", width, height)
        except Exception as e:
            logger.error("VT live init failed: %s", e)

    def _vt_smoke_worker(self):
        """Generate a local H.264 stream, decode via VideoToolbox, and display.

        This exercises the VT decode path without any server or QImage usage.
        """
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(self._run_vt_smoke_test())

    async def _run_vt_smoke_test(self):
        width = int(os.getenv('NAPARI_CUDA_VT_SMOKE_W', '1280'))
        height = int(os.getenv('NAPARI_CUDA_VT_SMOKE_H', '720'))
        seconds = float(os.getenv('NAPARI_CUDA_VT_SMOKE_SECS', '3'))
        fps = float(os.getenv('NAPARI_CUDA_VT_SMOKE_FPS', '60'))
        nframes = max(1, int(seconds * fps))
        logger.info("VT smoke: generating %d frames at %dx%d @ %.1ffps", nframes, width, height, fps)

        # Encode synthetic frames to raw AnnexB and feed VT by converting to AVCC
        try:
            import av
        except Exception as e:
            logger.error("VT smoke requires PyAV: %s", e)
            return

        enc = av.CodecContext.create('h264', 'w')
        enc.width = width
        enc.height = height
        enc.pix_fmt = 'yuv420p'
        enc.time_base = Fraction(1, int(round(fps)))
        enc.options = {
            'tune': 'zerolatency',
            'preset': 'veryfast',
            'bf': '0',
            'keyint': str(int(round(fps))),
            'sc_threshold': '0',
            'annexb': '1',
        }

        def split_annexb(data: bytes) -> list[bytes]:
            out: list[bytes] = []
            i = 0
            n = len(data)
            idx: list[int] = []
            while i + 3 <= n:
                if data[i:i+3] == b'\x00\x00\x01':
                    idx.append(i); i += 3
                elif i + 4 <= n and data[i:i+4] == b'\x00\x00\x00\x01':
                    idx.append(i); i += 4
                else:
                    i += 1
            idx.append(n)
            for a, b in zip(idx, idx[1:]):
                j = a
                while j < b and data[j] == 0:
                    j += 1
                if j + 3 <= b and data[j:j+3] == b'\x00\x00\x01':
                    j += 3
                elif j + 4 <= b and data[j:j+4] == b'\x00\x00\x00\x01':
                    j += 4
                nal = data[j:b]
                if nal:
                    out.append(nal)
            return out

        def parse_nals_any(data: bytes) -> list[bytes]:
            # Try AnnexB start codes first
            nals = split_annexb(data)
            if nals:
                return nals
            # Try AVCC length-prefixed
            nals = []
            i = 0
            n = len(data)
            while i + 4 <= n:
                ln = int.from_bytes(data[i:i+4], 'big')
                i += 4
                if ln <= 0 or i + ln > n:
                    nals = []
                    break
                nals.append(data[i:i+ln])
                i += ln
            return nals

        def annexb_to_avcc(data: bytes) -> bytes:
            nals = split_annexb(data)
            out = bytearray()
            for n in nals:
                out.extend(len(n).to_bytes(4, 'big'))
                out.extend(n)
            return bytes(out)

        def build_avcc(sps: bytes, pps: bytes) -> bytes:
            if len(sps) < 4:
                raise ValueError('SPS too short for avcC')
            profile = sps[1]
            compat = sps[2]
            level = sps[3]
            avcc = bytearray()
            avcc.append(1)
            avcc.append(profile)
            avcc.append(compat)
            avcc.append(level)
            avcc.append(0xFF)
            avcc.append(0xE1 | 1)
            avcc.extend(len(sps).to_bytes(2, 'big'))
            avcc.extend(sps)
            avcc.append(1)
            avcc.extend(len(pps).to_bytes(2, 'big'))
            avcc.extend(pps)
            return bytes(avcc)

        # Initialize VT when SPS/PPS observed
        try:
            from napari_cuda.client.vt_decoder import VideoToolboxDecoder, is_vt_available
        except Exception as e:
            logger.error("VT frameworks missing: %s", e)
            return
        if not is_vt_available():
            logger.error("VideoToolbox not available on this system")
            return

        avcc_bytes = None
        vt = None
        decoded = 0

        for i in range(nframes):
            # RGB gradient test pattern
            x = np.linspace(0, 1, width, dtype=np.float32)
            y = np.linspace(0, 1, height, dtype=np.float32)
            xv, yv = np.meshgrid(x, y)
            r = (xv * 255).astype(np.uint8)
            g = (yv * 255).astype(np.uint8)
            b = ((xv * 0.5 + yv * 0.5) * 255).astype(np.uint8)
            rgb = np.dstack([r, g, b])
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts = i
            frame.time_base = enc.time_base
            for pkt in enc.encode(frame):
                try:
                    data = pkt.to_bytes()  # newer PyAV
                except Exception as e:
                    try:
                        data = bytes(pkt)  # fallback via buffer protocol
                    except Exception as e2:
                        # Last resort: use memoryview
                        try:
                            data = memoryview(pkt).tobytes()
                        except Exception as e3:
                            logger.error("Failed to extract packet bytes: %s | %s | %s", e, e2, e3)
                            raise
                if avcc_bytes is None:
                    # Prefer encoder extradata if available (avcC or AnnexB param sets)
                    try:
                        extradata = bytes(getattr(enc, 'extradata', b'') or b'')
                    except Exception as e:
                        logger.debug("No encoder extradata: %s", e)
                        extradata = b''
                    if extradata and avcc_bytes is None:
                        if extradata[:1] == b'\x01':
                            avcc_bytes = extradata
                            vt = VideoToolboxDecoder(avcc_bytes, width, height)
                            logger.info("VideoToolbox decoder initialized for smoke test (extradata)")
                        else:
                            nals_ed = parse_nals_any(extradata)
                            sps_ed = next((n for n in nals_ed if (n[0] & 0x1F) == 7), None)
                            pps_ed = next((n for n in nals_ed if (n[0] & 0x1F) == 8), None)
                            if sps_ed and pps_ed:
                                avcc_bytes = build_avcc(sps_ed, pps_ed)
                                vt = VideoToolboxDecoder(avcc_bytes, width, height)
                                logger.info("VideoToolbox decoder initialized for smoke test (extradata SPS/PPS)")
                    nals = parse_nals_any(data)
                    sps = next((n for n in nals if (n[0] & 0x1F) == 7), None)
                    pps = next((n for n in nals if (n[0] & 0x1F) == 8), None)
                    if i < 3:
                        logger.info(
                            "Smoke pkt %d: NALs=%d SPS=%s PPS=%s first_types=%s lens=%s",
                            i,
                            len(nals),
                            bool(sps),
                            bool(pps),
                            [(n[0] & 0x1F) for n in nals[:5]],
                            [len(n) for n in nals[:5]],
                        )
                    if sps and pps:
                        avcc_bytes = build_avcc(sps, pps)
                        vt = VideoToolboxDecoder(avcc_bytes, width, height)
                        logger.info("VideoToolbox decoder initialized for smoke test (in-band SPS/PPS)")
                if vt is None:
                    # Log first few misses to help diagnose
                    if i < 3:
                        logger.info("VT not initialized yet at frame %d; waiting for SPS/PPS", i)
                    continue
                # If data already looks like AVCC, use as-is; otherwise convert
                if split_annexb(data):
                    avcc_au = annexb_to_avcc(data)
                else:
                    avcc_au = data
                img_buf = vt.decode(avcc_au)
                if img_buf is None:
                    if decoded == 0 and i < 5:
                        logger.info("VT decode returned None at frame %d (len=%d)", i, len(avcc_au))
                    continue
                try:
                    from Quartz import CoreVideo as CV  # type: ignore
                    CV.CVPixelBufferLockBaseAddress(img_buf, 0)
                    w = CV.CVPixelBufferGetWidth(img_buf)
                    h = CV.CVPixelBufferGetHeight(img_buf)
                    bpr = CV.CVPixelBufferGetBytesPerRow(img_buf)
                    base = CV.CVPixelBufferGetBaseAddress(img_buf)
                    size = int(bpr) * int(h)
                    ctype_arr = ctypes.cast(int(base), ctypes.POINTER(ctypes.c_ubyte * size)).contents
                    bgra = np.frombuffer(ctype_arr, dtype=np.uint8).reshape((int(h), int(bpr)//4, 4))
                    rgb = bgra[:, :int(w), [2, 1, 0]].copy()
                finally:
                    try:
                        CV.CVPixelBufferUnlockBaseAddress(img_buf, 0)
                    except Exception as e:
                        logger.warning("CVPixelBufferUnlockBaseAddress failed: %s", e)
                self._decoded_to_queue(rgb)
                decoded += 1
        logger.info("VT smoke: decoded %d frames", decoded)
        # Fallback: if nothing decoded, try container-based encode to ensure SPS/PPS
        if decoded == 0:
            try:
                import av
                logger.info("VT smoke: fallback to container-based H.264 encoding")
                buf = io.BytesIO()
                rate = Fraction(int(round(fps)), 1)
                out = av.open(buf, mode='w', format='h264')
                stream = out.add_stream('libx264', rate=rate)
                stream.width = width
                stream.height = height
                stream.pix_fmt = 'yuv420p'
                # Try to force in-band headers and AnnexB
                stream.options = {
                    'tune': 'zerolatency',
                    'preset': 'veryfast',
                    'bf': '0',
                    'keyint': str(int(rate)),
                    'sc_threshold': '0',
                    'x264-params': 'repeat-headers=1:annexb=1',
                }
                for i in range(nframes):
                    x = np.linspace(0, 1, width, dtype=np.float32)
                    y = np.linspace(0, 1, height, dtype=np.float32)
                    xv, yv = np.meshgrid(x, y)
                    r = (xv * 255).astype(np.uint8)
                    g = (yv * 255).astype(np.uint8)
                    b = ((xv * 0.5 + yv * 0.5) * 255).astype(np.uint8)
                    rgb = np.dstack([r, g, b])
                    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                    for p in stream.encode(frame):
                        out.mux(p)
                for p in stream.encode(None):
                    out.mux(p)
                out.close()
                raw = buf.getvalue()
                inp = av.open(io.BytesIO(raw), mode='r', format='h264')
                vstream = next(s for s in inp.streams if s.type == 'video')
                avcc_bytes2 = None
                vt2 = None
                decoded2 = 0
                for packet in inp.demux(vstream):
                    try:
                        data = packet.to_bytes()
                    except Exception:
                        try:
                            data = bytes(packet)
                        except Exception:
                            data = memoryview(packet).tobytes()
                    if avcc_bytes2 is None:
                        nals = parse_nals_any(data)
                        sps = next((n for n in nals if (n[0] & 0x1F) == 7), None)
                        pps = next((n for n in nals if (n[0] & 0x1F) == 8), None)
                        if sps and pps:
                            avcc_bytes2 = build_avcc(sps, pps)
                            vt2 = VideoToolboxDecoder(avcc_bytes2, width, height)
                            logger.info("VT smoke: VT initialized via container fallback")
                    if vt2 is None:
                        continue
                    avcc_au = annexb_to_avcc(data)
                    img_buf = vt2.decode(avcc_au)
                    if img_buf is None:
                        continue
                    try:
                        from Quartz import CoreVideo as CV  # type: ignore
                        CV.CVPixelBufferLockBaseAddress(img_buf, 0)
                        w = CV.CVPixelBufferGetWidth(img_buf)
                        h = CV.CVPixelBufferGetHeight(img_buf)
                        bpr = CV.CVPixelBufferGetBytesPerRow(img_buf)
                        base = CV.CVPixelBufferGetBaseAddress(img_buf)
                        size = int(bpr) * int(h)
                        ctype_arr = ctypes.cast(int(base), ctypes.POINTER(ctypes.c_ubyte * size)).contents
                        bgra = np.frombuffer(ctype_arr, dtype=np.uint8).reshape((int(h), int(bpr)//4, 4))
                        rgb = bgra[:, :int(w), [2, 1, 0]].copy()
                    finally:
                        try:
                            CV.CVPixelBufferUnlockBaseAddress(img_buf, 0)
                        except Exception:
                            pass
                    self._decoded_to_queue(rgb)
                    decoded2 += 1
                inp.close()
                logger.info("VT smoke: fallback decoded %d frames", decoded2)
            except Exception as e:
                logger.error("VT smoke fallback failed: %s", e)
    
    def _stream_worker(self):
        """Background thread to receive and decode video stream."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(self._receive_stream())
    
    async def _receive_stream(self):
        """Receive video stream from server via WebSocket."""
        url = f"ws://{self.server_host}:{self.server_port}"
        logger.info(f"Connecting to pixel stream at {url}")
        
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info("Connected to pixel stream")
                    
                    # Initialize decoder
                    self._init_decoder()
                    
                    async for message in websocket:
                        # Message is header + Annex B H.264 frame
                        if isinstance(message, bytes):
                            if len(message) > HEADER_STRUCT.size:
                                header = message[:HEADER_STRUCT.size]
                                payload = memoryview(message)[HEADER_STRUCT.size:]
                            else:
                                payload = message
                            # If VT live decoder ready, prefer it
                            if self._vt_decoder is not None:
                                self._decode_vt_live(bytes(payload))
                            else:
                                self._decode_frame(payload)
                            
            except Exception as e:
                logger.exception("Stream connection lost: %s", e)
                await asyncio.sleep(5)
                logger.info("Reconnecting to stream...")
    
    def _init_decoder(self):
        """Initialize H.264 decoder."""
        try:
            # Try PyAV first (better control)
            import av
            
            class PyAVDecoder:
                def __init__(self):
                    self.codec = av.CodecContext.create('h264', 'r')
                    # Optional channel swap to diagnose R/B confusion
                    import os as _os
                    try:
                        self.swap_rb = bool(int(_os.getenv('NAPARI_CUDA_CLIENT_SWAP_RB', '0')))
                    except Exception:
                        self.swap_rb = False
                    # Optional explicit pixel fmt (rgb24 or bgr24); default rgb24
                    try:
                        pf = _os.getenv('NAPARI_CUDA_CLIENT_PIXEL_FMT', 'rgb24').lower()
                    except Exception:
                        pf = 'rgb24'
                    self.pixfmt = pf if pf in {'rgb24', 'bgr24'} else 'rgb24'
                    # Reduce logging noise

                def decode(self, data):
                    packet = av.Packet(data)
                    frames = self.codec.decode(packet)
                    for frame in frames:
                        # Log frame color metadata once if available
                        # No color metadata logs
                        arr = frame.to_ndarray(format=self.pixfmt)
                        # Ensure the GL texture sees RGB ordering
                        if self.pixfmt == 'bgr24':
                            arr = arr[..., ::-1].copy()
                        if self.swap_rb:
                            arr = arr[..., ::-1].copy()
                        if not hasattr(self, '_logged_once'):
                            try:
                                h, w, c = arr.shape
                                logger.info("Client(PyAV) decode to %s -> RGB array (%dx%d)", self.pixfmt, w, h)
                            except Exception:
                                pass
                            setattr(self, '_logged_once', True)
                        return arr
                    return None
            
            self.decoder = PyAVDecoder()
            
        except ImportError:
            # Fallback to OpenCV
            try:
                import cv2
                
                class OpenCVDecoder:
                    def decode(self, data):
                        # This is simplified - real implementation needs more work
                        # OpenCV VideoCapture doesn't easily handle raw H.264 chunks
                        # Would need to use ffmpeg pipe or similar
                        logger.warning("OpenCV decoder not fully implemented")
                        # Return dummy frame for now
                        return np.zeros((1080, 1920, 3), dtype=np.uint8)
                
                self.decoder = OpenCVDecoder()
                logger.warning("Using OpenCV decoder (limited)")
                
            except ImportError:
                logger.error("No video decoder available!")
                
                class DummyDecoder:
                    def decode(self, data):
                        # Generate test pattern
                        frame = np.random.rand(1080, 1920, 3) * 255
                        return frame.astype(np.uint8)
                
                self.decoder = DummyDecoder()

    @staticmethod
    def _annexb_to_avcc(data: bytes) -> bytes:
        """Convert AnnexB stream to AVCC (length-prefixed).
        
        AnnexB uses start codes (00 00 01 or 00 00 00 01).
        AVCC uses 4-byte length prefixes.
        """
        out = bytearray()
        i = 0
        n = len(data)
        nalus_found = 0
        
        while i < n:
            # Find next start code
            next_start = n  # default to end
            j = i
            
            # Skip current start code if at one
            if j + 3 <= n and data[j:j+3] == b'\x00\x00\x01':
                i += 3
            elif j + 4 <= n and data[j:j+4] == b'\x00\x00\x00\x01':
                i += 4
            else:
                # Not at start code, find next one
                while j < n - 2:
                    if data[j:j+3] == b'\x00\x00\x01':
                        next_start = j
                        break
                    if j < n - 3 and data[j:j+4] == b'\x00\x00\x00\x01':
                        next_start = j
                        break
                    j += 1
            
            if i < next_start:
                # Extract NALU between i and next_start
                nalu = data[i:next_start]
                if nalu:  # Skip empty NALUs
                    out.extend(len(nalu).to_bytes(4, 'big'))
                    out.extend(nalu)
                    nalus_found += 1
                    if nalus_found <= 2:
                        logger.debug("NALU #%d: len=%d, type=0x%02x", nalus_found, len(nalu), nalu[0] & 0x1f if nalu else 0)
            
            i = next_start
        
        result = bytes(out)
        logger.debug("AnnexB→AVCC: input %d bytes → %d NALUs → output %d bytes", len(data), nalus_found, len(result))
        return result

    def _decode_vt_live(self, data: bytes) -> None:
        try:
            if not data:
                logger.debug("VT decode skipped: empty data")
                return
            
            # Detect format and convert if needed
            is_annexb = data[:3] == b'\x00\x00\x01' or data[:4] == b'\x00\x00\x00\x01'
            logger.debug("VT input: %d bytes, format=%s, first bytes=%s", 
                        len(data), "AnnexB" if is_annexb else "AVCC", data[:8].hex())
            
            avcc_au = self._annexb_to_avcc(data) if is_annexb else data
            
            # Validate AVCC format
            if len(avcc_au) >= 4:
                first_len = int.from_bytes(avcc_au[:4], 'big')
                logger.debug("AVCC first NALU: len_field=%d, total_au=%d", first_len, len(avcc_au))
                if first_len > len(avcc_au) - 4:
                    logger.error("Invalid AVCC: first NALU length %d exceeds remaining data %d", 
                                first_len, len(avcc_au) - 4)
                    return
            
            img_buf = self._vt_decoder.decode(avcc_au)
            if img_buf is None:
                self._vt_errors += 1
                if self._vt_errors % 10 == 0:
                    logger.warning("VT decode returned None repeatedly (%d)", self._vt_errors)
                return
            # Map BGRA and copy to RGB for GL upload (temporary until zero-copy)
            from Quartz import CoreVideo as CV  # type: ignore
            CV.CVPixelBufferLockBaseAddress(img_buf, 0)
            try:
                w = int(CV.CVPixelBufferGetWidth(img_buf))
                h = int(CV.CVPixelBufferGetHeight(img_buf))
                bpr = int(CV.CVPixelBufferGetBytesPerRow(img_buf))
                base = CV.CVPixelBufferGetBaseAddress(img_buf)
                size = bpr * h
                raw = ctypes.string_at(int(base), size)
                bgra = np.frombuffer(raw, dtype=np.uint8).reshape((h, bpr // 4, 4))
                rgb = bgra[:, :w, [2, 1, 0]].copy()
            finally:
                CV.CVPixelBufferUnlockBaseAddress(img_buf, 0)
            self._decoded_to_queue(rgb)
            self._vt_errors = 0
        except Exception as e:
            self._vt_errors += 1
            logger.exception("VT live decode/map failed (%d): %s", self._vt_errors, e)
            if self._vt_errors >= 3:
                logger.error("Disabling VT after repeated errors; falling back to PyAV")
                self._vt_decoder = None
            return
    
    def _decode_frame(self, h264_data):
        """Decode H.264 frame and add to queue."""
        if self.decoder:
            frame = self.decoder.decode(h264_data)
            
            if frame is not None:
                self._decoded_to_queue(frame)
                # Display is timer-driven; avoid redraw on every arrival

    def _decoded_to_queue(self, frame: np.ndarray) -> None:
        """Enqueue a decoded RGB frame with latest-wins behavior."""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)
    
    def _draw_video_frame(self, event):
        """Draw the latest video frame instead of scene content."""
        try:
            # Get latest frame
            frame = None
            
            # Drain queue to get most recent frame
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if frame is not None:
                self._display_frame(frame)
            elif self._video_texture is not None:
                # Redraw last frame
                self._display_frame(None)
                
        except Exception as e:
            logger.error(f"Error drawing video frame: {e}")
    
    def _display_frame(self, frame):
        """Display frame using OpenGL."""
        with self._scene_canvas.context:
            # Initialize resources if needed
            if self._video_program is None:
                self._init_video_display()
            
            if frame is not None:
                # Update texture with new frame
                self._video_texture.set_data(frame)
            
            # Clear and draw
            self._scene_canvas.context.clear('black')
            self._video_program.draw('triangle_strip')
    
    def _init_video_display(self):
        """Initialize OpenGL resources for video display."""
        # Create texture
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._video_texture = Texture2D(dummy_frame)
        
        # Create shader program
        self._video_program = Program(VERTEX_SHADER, FRAGMENT_SHADER)
        
        # Create fullscreen quad
        vertices = np.array([
            [-1, -1, 0, 1],  # Bottom-left
            [ 1, -1, 1, 1],  # Bottom-right
            [-1,  1, 0, 0],  # Top-left
            [ 1,  1, 1, 0],  # Top-right
        ], dtype=np.float32)
        
        self._video_program['position'] = vertices[:, :2]
        self._video_program['texcoord'] = vertices[:, 2:]
        self._video_program['texture'] = self._video_texture
        
        logger.info("Video display initialized")
    
    def screenshot(self, *args, **kwargs):
        """Get screenshot of current video frame."""
        # Return the last displayed frame
        if not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                self.frame_queue.put(frame)  # Put it back
                return frame
            except:
                pass
        
        return np.zeros((1080, 1920, 3), dtype=np.uint8)
