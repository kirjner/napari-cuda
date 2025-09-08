"""
StreamingCanvas - Displays video stream from remote server.

This replaces the normal VispyCanvas with one that shows decoded video frames
instead of locally rendered content.
"""

import asyncio
import logging
import queue
import numpy as np
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
    
    def __init__(self, viewer, server_host='localhost', server_port=8082, **kwargs):
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
        super().__init__(viewer, **kwargs)
        
        self.server_host = server_host
        self.server_port = server_port
        
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
        
        # Start streaming thread
        self._streaming_thread = Thread(target=self._stream_worker, daemon=True)
        self._streaming_thread.start()
        
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
                                # Optionally log a few headers for sanity
                                # Strip header; no per-frame header logs
                                payload = memoryview(message)[HEADER_STRUCT.size:]
                            else:
                                payload = message
                            self._decode_frame(payload)
                            
            except Exception as e:
                logger.error(f"Stream connection lost: {e}")
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
    
    def _decode_frame(self, h264_data):
        """Decode H.264 frame and add to queue."""
        if self.decoder:
            frame = self.decoder.decode(h264_data)
            
            if frame is not None:
                # Add to queue (drop oldest if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                # Display is timer-driven; avoid redraw on every arrival
    
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
