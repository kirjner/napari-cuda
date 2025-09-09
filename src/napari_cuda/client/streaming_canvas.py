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
import time
import sys
from fractions import Fraction
from qtpy import QtCore
from threading import Thread

import websockets
import struct
from napari._vispy.canvas import VispyCanvas
from vispy.gloo import Texture2D, Program
from vispy import app as vispy_app

# Silence VisPy warnings about copying discontiguous data
vispy_logger = logging.getLogger('vispy')
vispy_logger.setLevel(logging.ERROR)

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
        # Gate VT decode until we see a keyframe after (re)initialization
        self._vt_wait_keyframe = False
        self._vt_gate_lift_time: float | None = None
        # VT presenter jitter buffer + latency target
        try:
            self._vt_latency_s = max(0.0, float(os.getenv('NAPARI_CUDA_CLIENT_VT_LATENCY_MS', '80')) / 1000.0)
        except Exception:
            self._vt_latency_s = 0.08
        try:
            self._vt_buffer_limit = int(os.getenv('NAPARI_CUDA_CLIENT_VT_BUFFER', '3'))
        except Exception:
            self._vt_buffer_limit = 3
        self._vt_present: list[tuple[float, object]] = []
        # VT diagnostics
        self._vt_last_stats_log: float = 0.0
        self._vt_last_submit_count: int = 0
        self._vt_last_out_count: int = 0
        # VT submission decoupling: queue + worker to avoid blocking asyncio recv loop
        # Larger input queue to avoid backpressure on websocket when VT is momentarily slow
        self._vt_in_q: "queue.Queue[tuple[bytes, float|None]]" = queue.Queue(maxsize=64)
        self._vt_enqueued = 0
        # Backlog handling: if queue builds up, resync on next keyframe to avoid smear
        try:
            self._vt_backlog_trigger = int(os.getenv('NAPARI_CUDA_CLIENT_VT_BACKLOG_TRIGGER', '16'))
        except Exception:
            self._vt_backlog_trigger = 16
        def _vt_submit_worker():
            while True:
                try:
                    data, ts = self._vt_in_q.get()
                except Exception:
                    continue
                try:
                    # Submit to VT (may log failures internally)
                    self._decode_vt_live(data, ts)
                except Exception as e:
                    logger.debug("VT submit worker error: %s", e)
        self._vt_submit_thread = Thread(target=_vt_submit_worker, daemon=True)
        self._vt_submit_thread.start()
        
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
        # Timer-driven display at target fps (use VisPy app timer to ensure GUI-thread delivery)
        try:
            fps = float(os.getenv('NAPARI_CUDA_CLIENT_DISPLAY_FPS', '60'))
        except Exception:
            fps = 60.0
        # Default to Qt timer for napari/Qt integration; enable vispy timer only if requested
        use_vispy_timer = True if os.getenv('NAPARI_CUDA_CLIENT_VISPY_TIMER', '0') == '1' else False
        interval = max(1.0 / max(1.0, fps), 1.0 / 120.0)
        if use_vispy_timer:
            self._display_timer = vispy_app.Timer(interval=interval, connect=lambda ev: self._scene_canvas.update(), start=True)
            logger.info("Video display initialized (vispy.Timer @ %.1f fps)", 1.0/interval)
        else:
            # Fallback to Qt timer, ensure it belongs to the canvas' GUI thread
            self._display_timer = QtCore.QTimer(self._scene_canvas.native)
            self._display_timer.setTimerType(QtCore.Qt.PreciseTimer)
            self._display_timer.setInterval(max(1, int(round(1000.0 / max(1.0, fps)))))
            self._display_timer.timeout.connect(self._scene_canvas.native.update)
            self._display_timer.start()
            logger.info("Video display initialized (Qt QTimer @ %.1f fps)", fps)
        
        logger.info(f"StreamingCanvas initialized for {server_host}:{server_port}")

        # Timestamp handling for VT scheduling
        self._vt_ts_mode = (os.getenv('NAPARI_CUDA_CLIENT_VT_TS_MODE') or 'server').lower()
        self._vt_ts_offset = None  # server_ts -> local_now offset (seconds)
        # Keyframe request throttling while VT waits for sync
        self._vt_last_key_req: float | None = None

        # VT stats logging level control (default: disabled)
        stats_env = (os.getenv('NAPARI_CUDA_VT_STATS') or '').lower()
        if stats_env in ('1', 'true', 'yes', 'info'):
            self._vt_stats_level = logging.INFO
        elif stats_env in ('debug', 'dbg'):
            self._vt_stats_level = logging.DEBUG
        else:
            self._vt_stats_level = None

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
                # Request a keyframe on connect to ensure VT has a sync point
                try:
                    await ws.send('{"type":"request_keyframe"}')
                except Exception:
                    pass
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
                                    logger.debug("Client video_config: fps=%.3f", fps)
                                if width > 0 and height > 0 and avcc_b64:
                                    self._init_vt_from_avcc(avcc_b64, width, height)
                            except Exception as e:
                                logger.debug("video_config parse failed: %s", e)
                finally:
                    ping_task.cancel()
        except Exception as e:
            logger.debug("State channel ended: %s", e)

    async def _request_keyframe_once(self):
        """Best-effort request for a keyframe via state channel (throttled)."""
        now = time.time()
        if self._vt_last_key_req is not None and (now - self._vt_last_key_req) < 0.5:
            return
        self._vt_last_key_req = now
        url = f"ws://{self.server_host}:{self.state_port}"
        try:
            async with websockets.connect(url) as ws:
                await ws.send('{"type":"request_keyframe"}')
        except Exception:
            pass

    def _init_vt_from_avcc(self, avcc_b64: str, width: int, height: int) -> None:
        try:
            avcc = base64.b64decode(avcc_b64)
            cfg_key = (int(width), int(height), avcc)
            if self._vt_decoder is not None and self._vt_cfg_key == cfg_key:
                logger.debug("VT already initialized; ignoring duplicate video_config")
                return
            # Prefer native shim on macOS; fallback to PyAV only (PyObjC VT retired)
            backend = (os.getenv('NAPARI_CUDA_VT_BACKEND', 'shim') or 'shim').lower()
            self._vt_backend = None
            if sys.platform == 'darwin' and backend != 'off':
                try:
                    from napari_cuda.client.vt_shim import VTShimDecoder  # type: ignore
                    self._vt_decoder = VTShimDecoder(avcc, width, height)
                    self._vt_backend = 'shim'
                except Exception as e:
                    logger.warning("VT shim unavailable: %s; falling back to PyAV", e)
                    self._vt_decoder = None
            self._vt_cfg_key = cfg_key
            # Require a fresh keyframe before decoding with VT to ensure sync
            self._vt_wait_keyframe = True
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
        logger.debug("VT smoke: generating %d frames at %dx%d @ %.1ffps", nframes, width, height, fps)

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
                logger.debug("VT smoke: fallback decoded %d frames", decoded2)
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
        
        seen_keyframe = False
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
                                hdr = message[:HEADER_STRUCT.size]
                                payload = memoryview(message)[HEADER_STRUCT.size:]
                                try:
                                    # Unpack flags to gate until first keyframe
                                    seq, ts, w, h, codec, flags, _ = HEADER_STRUCT.unpack(hdr)
                                    # Global initial gate (first frame must be keyframe)
                                    if not seen_keyframe:
                                        if flags & 0x01:
                                            seen_keyframe = True
                                        else:
                                            # Skip frames until first keyframe
                                            continue
                                    # VT-specific gate: after VT (re)init, require a fresh keyframe
                                    if self._vt_decoder is not None and self._vt_wait_keyframe:
                                        if flags & 0x01:
                                            self._vt_wait_keyframe = False
                                            self._vt_gate_lift_time = time.time()
                                            # Establish server->client clock offset for scheduling
                                            try:
                                                self._vt_ts_offset = float(self._vt_gate_lift_time - float(ts))
                                            except Exception:
                                                self._vt_ts_offset = None
                                            logger.info("VT gate lifted on keyframe (seq=%d)", seq)
                                        else:
                                            # While waiting for VT keyframe, keep PyAV display alive
                                            try:
                                                self._decode_frame(payload)
                                            except Exception:
                                                pass
                                            # Nudge server for a fresh keyframe (best-effort, throttled)
                                            try:
                                                asyncio.create_task(self._request_keyframe_once())
                                            except Exception:
                                                pass
                                            continue
                                except Exception:
                                    pass
                            else:
                                payload = message
                            # If VT live decoder ready and not waiting for keyframe, prefer it
                            if self._vt_decoder is not None and not self._vt_wait_keyframe:
                                # Pass server timestamp through for scheduling
                                try:
                                    ts_float = float(ts)
                                except Exception:
                                    ts_float = None
                                # Enqueue to VT worker to keep websocket recv non-blocking
                                b = bytes(payload)
                                try:
                                    # If backlog builds, resync: request keyframe and gate until it arrives
                                    if self._vt_in_q.qsize() >= max(2, self._vt_backlog_trigger - 1):
                                        self._vt_wait_keyframe = True
                                        logger.info("VT backlog detected (q=%d); requesting keyframe and resync", self._vt_in_q.qsize())
                                        try:
                                            asyncio.create_task(self._request_keyframe_once())
                                        except Exception:
                                            pass
                                        # Drain queued items; we'll resume on next keyframe
                                        try:
                                            while self._vt_in_q.qsize() > 0:
                                                _ = self._vt_in_q.get_nowait()
                                        except Exception:
                                            pass
                                        # Also drop any pending presented frames
                                        try:
                                            from napari_cuda import _vt as vt  # type: ignore
                                            for _, cap in self._vt_present:
                                                try:
                                                    vt.release_frame(cap)
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                        self._vt_present.clear()
                                    self._vt_in_q.put_nowait((b, ts_float))
                                    self._vt_enqueued += 1
                                    if self._vt_enqueued <= 3:
                                        logger.debug("VT enqueued #%d (seq=%d, %d bytes)", self._vt_enqueued, int(seq), len(b))
                                except queue.Full:
                                    # Queue full: enter resync mode; drop until keyframe
                                    self._vt_wait_keyframe = True
                                    try:
                                        asyncio.create_task(self._request_keyframe_once())
                                    except Exception:
                                        pass
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
                    # Ensure AnnexB for PyAV: convert from AVCC if needed
                    try:
                        from napari_cuda.client.avcc_shim import normalize_to_annexb  # type: ignore
                        annexb, _ = normalize_to_annexb(data)
                        packet = av.Packet(annexb)
                    except Exception:
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
        n = len(data)
        # Collect all start-code indices
        idx: list[int] = []
        i = 0
        while i + 3 <= n:
            if data[i:i+3] == b"\x00\x00\x01":
                idx.append(i)
                i += 3
            elif i + 4 <= n and data[i:i+4] == b"\x00\x00\x00\x01":
                idx.append(i)
                i += 4
            else:
                i += 1
        idx.append(n)

        nalus_found = 0
        for a, b in zip(idx, idx[1:]):
            # Trim leading zeros
            j = a
            while j < b and data[j] == 0:
                j += 1
            # Skip the start code itself
            if j + 3 <= b and data[j:j+3] == b"\x00\x00\x01":
                j += 3
            elif j + 4 <= b and data[j:j+4] == b"\x00\x00\x00\x01":
                j += 4
            # Extract NAL payload
            nal = data[j:b]
            if not nal:
                continue
            out.extend(len(nal).to_bytes(4, "big"))
            out.extend(nal)
            nalus_found += 1
            if nalus_found <= 3:
                try:
                    ntype = nal[0] & 0x1F
                except Exception:
                    ntype = -1
                logger.debug("AnnexB→AVCC NAL #%d: len=%d type=%s", nalus_found, len(nal), ntype)

        result = bytes(out)
        logger.debug(
            "AnnexB→AVCC: input %d bytes → %d NALs → output %d bytes",
            len(data), nalus_found, len(result)
        )
        return result

    def _decode_vt_live(self, data: bytes, ts: float | None) -> None:
        try:
            if not data:
                logger.debug("VT decode skipped: empty data")
                return
            
            # Detect format and convert if needed
            is_annexb = data[:3] == b'\x00\x00\x01' or data[:4] == b'\x00\x00\x00\x01'
            logger.debug("VT input: %d bytes, format=%s, first bytes=%s", 
                        len(data), "AnnexB" if is_annexb else "AVCC", data[:8].hex())
            
            # Normalize to AVCC once per AU and feed as-is (including SPS/PPS/SEI)
            avcc_au = self._annexb_to_avcc(data) if is_annexb else data
            # Log composition of the first couple of AUs for diagnostics
            try:
                if self._vt_enqueued < 2:
                    off = 0
                    total = len(avcc_au)
                    cnt = {1:0,5:0,6:0,7:0,8:0,9:0}
                    while off + 4 <= total:
                        ln = int.from_bytes(avcc_au[off:off+4], 'big'); off += 4
                        if ln <= 0 or off + ln > total:
                            break
                        ntype = avcc_au[off] & 0x1F
                        if ntype in cnt:
                            cnt[ntype] += 1
                        off += ln
                    logger.debug(
                        "VT AU pre-submit composition: AUD=%d SPS=%d PPS=%d SEI=%d IDR=%d NONIDR=%d",
                        cnt.get(9,0), cnt.get(7,0), cnt.get(8,0), cnt.get(6,0), cnt.get(5,0), cnt.get(1,0)
                    )
            except Exception:
                pass
            ok = self._vt_decoder.decode(avcc_au, ts)
            if not ok:
                self._vt_errors += 1
                # Be chatty on the first few failures, then throttle
                if self._vt_errors <= 3 or (self._vt_errors % 50 == 0):
                    logger.warning("VT decode submit failed (errors=%d)", self._vt_errors)
                return
            # Asynchronous output; presenter will map/display when due
            self._vt_errors = 0
            # For the first couple of frames, nudge delivery to confirm callback path
            try:
                if self._vt_enqueued <= 3:
                    self._vt_decoder.flush()
            except Exception:
                pass
        except Exception as e:
            self._vt_errors += 1
            logger.exception("VT live decode/map failed (%d): %s", self._vt_errors, e)
            if self._vt_errors >= 3:
                logger.error("Disabling VT after repeated errors; falling back to PyAV")
                self._vt_decoder = None
            return

    def _pixelbuffer_to_rgb(self, img_buf) -> np.ndarray:
        from Quartz import CoreVideo as CV  # type: ignore
        CV.CVPixelBufferLockBaseAddress(img_buf, 0)
        try:
            w = int(CV.CVPixelBufferGetWidth(img_buf))
            h = int(CV.CVPixelBufferGetHeight(img_buf))
            pf = int(CV.CVPixelBufferGetPixelFormatType(img_buf))
            if pf == getattr(CV, 'kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange', 0):
                y_bpr = int(CV.CVPixelBufferGetBytesPerRowOfPlane(img_buf, 0))
                uv_bpr = int(CV.CVPixelBufferGetBytesPerRowOfPlane(img_buf, 1))
                y_base = CV.CVPixelBufferGetBaseAddressOfPlane(img_buf, 0)
                uv_base = CV.CVPixelBufferGetBaseAddressOfPlane(img_buf, 1)
                y_raw = ctypes.string_at(int(y_base), y_bpr * h)
                uv_raw = ctypes.string_at(int(uv_base), uv_bpr * (h // 2))
                y = np.frombuffer(y_raw, dtype=np.uint8).reshape((h, y_bpr))[:, :w].astype(np.int16)
                uv = np.frombuffer(uv_raw, dtype=np.uint8).reshape((h // 2, uv_bpr))[:, : (w // 1)]
                u = uv[:, :w:2].astype(np.int16)
                v = uv[:, 1:w:2].astype(np.int16)
                u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:h, :w]
                v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:h, :w]
                c = y - 16
                d = u_up - 128
                e = v_up - 128
                r = (298 * c + 409 * e + 128) >> 8
                g = (298 * c - 100 * d - 208 * e + 128) >> 8
                b = (298 * c + 516 * d + 128) >> 8
                rgb = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
            else:
                bpr = int(CV.CVPixelBufferGetBytesPerRow(img_buf))
                base = CV.CVPixelBufferGetBaseAddress(img_buf)
                size = bpr * h
                raw = ctypes.string_at(int(base), size)
                bgra = np.frombuffer(raw, dtype=np.uint8).reshape((h, bpr // 4, 4))
                rgb = bgra[:, :w, [2, 1, 0]].copy()
            return rgb
        finally:
            CV.CVPixelBufferUnlockBaseAddress(img_buf, 0)
    
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
            # Presenter: drain VT frames, schedule by server ts + latency
            if self._vt_decoder is not None and self._vt_backend == 'shim':
                now = time.time()
                # Periodic diagnostics about VT submits/outputs
                try:
                    if now - self._vt_last_stats_log >= 1.0:
                        try:
                            sub, out, qlen = self._vt_decoder.counts()
                        except Exception:
                            sub = out = 0
                            qlen = -1
                        # Emit once-per-second stats if enabled
                        lvl = getattr(self, '_vt_stats_level', None)
                        if lvl is not None:
                            try:
                                logger.log(
                                    lvl,
                                    "VT stats: submit=%d out=%d shim_q=%d present=%d mode=%s fixed_target=%.1fms",
                                    sub, out, qlen, len(self._vt_present),
                                    getattr(self, '_vt_ts_mode', 'server'),
                                    self._vt_latency_s * 1000.0,
                                )
                            except Exception:
                                pass
                        # Warn if we've submitted but not received any outputs shortly after gate lift
                        if (
                            self._vt_gate_lift_time is not None
                            and (now - self._vt_gate_lift_time) > 1.0
                            and sub > 0
                            and out == 0
                        ):
                            logger.warning(
                                "VT stalled? submits=%d outputs=%d latency_ms=%d present_buf=%d shim_q=%d",
                                sub, out, int(round(self._vt_latency_s * 1000.0)), len(self._vt_present), qlen,
                            )
                            # Nudge VT to deliver any pending frames
                            try:
                                self._vt_decoder.flush()
                            except Exception:
                                pass
                        self._vt_last_stats_log = now
                        self._vt_last_submit_count = sub
                        self._vt_last_out_count = out
                except Exception:
                    pass
                while True:
                    item = self._vt_decoder.get_frame_nowait()
                    if not item:
                        break
                    img_buf, pts = item
                    # Compute due time based on configured mode and fixed target
                    now2 = time.time()
                    if self._vt_ts_mode == 'arrival' or pts is None:
                        due = now2 + self._vt_latency_s
                    else:
                        offset = self._vt_ts_offset
                        if offset is None or abs(offset) > 5.0:
                            due = now2 + self._vt_latency_s
                        else:
                            due = (float(pts) + float(offset)) + self._vt_latency_s
                    # Keep a CF retain count from callback; store for later release
                    self._vt_present.append((due, img_buf))
                    # Request an immediate GUI wakeup to reduce perceived stalls
                    try:
                        QtCore.QTimer.singleShot(0, self._scene_canvas.native.update)
                    except Exception:
                        pass
                    # Bound buffer size
                    if len(self._vt_present) > max(1, self._vt_buffer_limit):
                        self._vt_present.sort(key=lambda t: t[0])
                        drop = self._vt_present[:-self._vt_buffer_limit]
                        self._vt_present = self._vt_present[-self._vt_buffer_limit:]
                        # Release dropped buffers via shim
                        try:
                            from napari_cuda import _vt as vt  # type: ignore
                            for _, cap in drop:
                                try:
                                    vt.release_frame(cap)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                if self._vt_present:
                    self._vt_present.sort(key=lambda t: t[0])
                    due_now = [p for p in self._vt_present if p[0] <= now]
                    if due_now:
                        # Display the latest due frame (latest-wins)
                        _, img_buf = due_now[-1]
                        try:
                            # Map capsule to contiguous RGB via shim helper
                            from napari_cuda import _vt as vt  # type: ignore
                            rgb_bytes, w, h = vt.map_to_rgb(img_buf)
                            rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((int(h), int(w), 3))
                            self._decoded_to_queue(rgb)
                            # Release the consumed buffer now
                            try:
                                vt.release_frame(img_buf)
                            except Exception:
                                pass
                        except Exception as e:
                            logger.debug("VT map/display error: %s", e)
                        # Drop all due frames we just presented
                        pending = [p for p in self._vt_present if p[0] > now]
                        # Release all due frames (except the one we just released above)
                        try:
                            from napari_cuda import _vt as vt  # type: ignore
                            for _, cap in due_now[:-1]:
                                try:
                                    vt.release_frame(cap)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        self._vt_present = pending
                    else:
                        # If no frames are due for a while (clock skew), show the latest to avoid visible stalls
                        earliest_due = self._vt_present[0][0]
                        if earliest_due - now > 0.2:  # 200ms guard
                            _, img_buf = self._vt_present[-1]
                            try:
                                from napari_cuda import _vt as vt  # type: ignore
                                rgb_bytes, w, h = vt.map_to_rgb(img_buf)
                                rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((int(h), int(w), 3))
                                self._decoded_to_queue(rgb)
                            except Exception:
                                pass
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
        ctx = self._scene_canvas.context
        # Initialize resources if needed
        if self._video_program is None:
            self._init_video_display()

        if frame is not None:
            # Ensure contiguous uint8 RGB before uploading to GL to avoid VisPy copies
            try:
                if frame.dtype != np.uint8 or not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
            except Exception:
                pass
            # Update texture with new frame
            self._video_texture.set_data(frame)

        # Clear and draw
        ctx.clear('black')
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
        
        # Ensure contiguous attribute arrays to avoid VisPy copying warnings
        self._video_program['position'] = np.ascontiguousarray(vertices[:, :2])
        self._video_program['texcoord'] = np.ascontiguousarray(vertices[:, 2:])
        self._video_program['texture'] = self._video_texture
        
        logger.debug("Video display initialized")
    
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
