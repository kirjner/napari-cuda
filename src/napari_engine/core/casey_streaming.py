"""
The Casey Muratori approach: Stop overthinking, start streaming.

Core principles:
1. The frame IS the state. Stop syncing objects.
2. Input and rendering are separate problems with separate solutions.
3. Measure everything. Assume nothing.
4. If it's not on screen, it doesn't exist.
"""

import asyncio
import struct
import time
import numpy as np
from typing import Optional
from collections import deque


class RingBuffer:
    """Zero-copy ring buffer for frame streaming.
    Casey would approve: no allocations in the hot path.
    """
    def __init__(self, frame_size: int, num_frames: int = 3):
        # Pre-allocate everything
        self.buffer = bytearray(frame_size * num_frames)
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.write_index = 0
        self.read_index = 0
        
    def write_frame(self, frame_data: bytes) -> int:
        """Write frame to next slot. Returns frame number."""
        offset = (self.write_index % self.num_frames) * self.frame_size
        self.buffer[offset:offset + self.frame_size] = frame_data
        frame_num = self.write_index
        self.write_index += 1
        return frame_num
    
    def read_frame(self, frame_num: int) -> memoryview:
        """Read frame by number. Zero-copy via memoryview."""
        offset = (frame_num % self.num_frames) * self.frame_size
        return memoryview(self.buffer)[offset:offset + self.frame_size]


class FrameTimer:
    """Deterministic frame timing. No希望, only reality."""
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.next_frame_time = time.perf_counter()
        self.frame_times = deque(maxlen=100)
        
    def wait_for_next_frame(self):
        """Sleep until next frame time. Returns actual frame time."""
        now = time.perf_counter()
        sleep_time = self.next_frame_time - now
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        actual_time = time.perf_counter()
        if self.frame_times:
            self.frame_times.append(actual_time - self.next_frame_time + self.frame_time)
        
        self.next_frame_time = max(self.next_frame_time + self.frame_time, actual_time)
        
    @property
    def stats(self) -> dict:
        if not self.frame_times:
            return {}
        times = np.array(self.frame_times)
        return {
            'avg_fps': 1.0 / times.mean() if times.mean() > 0 else 0,
            'min_fps': 1.0 / times.max() if times.max() > 0 else 0,
            'max_fps': 1.0 / times.min() if times.min() > 0 else 0,
            'stdev_ms': times.std() * 1000,
        }


class CaseyStreamingServer:
    """
    The Casey approach: Just send frames. Period.
    No state sync, no complex protocols. The frame IS the truth.
    """
    
    def __init__(self, viewer, target_fps: int = 30):
        self.viewer = viewer
        self.timer = FrameTimer(target_fps)
        self.ring_buffer = None  # Created after first frame
        self.frame_counter = 0
        self.clients = set()
        
        # Separate channels for input and frames
        self.input_queue = asyncio.Queue()  # Low latency input
        self.frame_queue = asyncio.Queue()  # Frame streaming
        
    async def render_loop(self):
        """The hot path. This is ALL that matters."""
        while True:
            # Handle all pending inputs first (low latency)
            while not self.input_queue.empty():
                try:
                    event = self.input_queue.get_nowait()
                    self._handle_input(event)
                except asyncio.QueueEmpty:
                    break
            
            # Render frame
            frame_start = time.perf_counter()
            
            # Screenshot is the only truth
            screenshot = self.viewer.screenshot(canvas_only=True)
            frame_bytes = screenshot.tobytes()
            
            # Initialize ring buffer on first frame
            if self.ring_buffer is None:
                self.ring_buffer = RingBuffer(len(frame_bytes))
            
            # Store in ring buffer (zero-copy)
            frame_num = self.ring_buffer.write_frame(frame_bytes)
            
            # Send to all clients (fire and forget)
            if self.clients:
                frame_msg = self._pack_frame_header(frame_num, len(frame_bytes))
                frame_view = self.ring_buffer.read_frame(frame_num)
                
                for client in list(self.clients):
                    try:
                        client.write(frame_msg + frame_view)
                    except:
                        self.clients.discard(client)
            
            # Performance tracking
            render_time = time.perf_counter() - frame_start
            if render_time > self.timer.frame_time:
                print(f"FRAME BUDGET EXCEEDED: {render_time*1000:.1f}ms")
            
            # Wait for next frame time
            self.timer.wait_for_next_frame()
            self.frame_counter += 1
            
            # Print stats every second
            if self.frame_counter % self.timer.target_fps == 0:
                stats = self.timer.stats
                print(f"FPS: {stats.get('avg_fps', 0):.1f} "
                      f"(min: {stats.get('min_fps', 0):.1f}, "
                      f"max: {stats.get('max_fps', 0):.1f})")
    
    def _handle_input(self, event: dict):
        """Apply input immediately. No round trips."""
        event_type = event.get('type')
        
        if event_type == 'mouse_move':
            # This is THE critical path for interactivity
            x, y = event['x'], event['y']
            # Apply to viewer immediately
            # Note: we'd need to translate to viewer coordinates
            pass
            
        elif event_type == 'camera':
            self.viewer.camera.center = event.get('center', self.viewer.camera.center)
            self.viewer.camera.zoom = event.get('zoom', self.viewer.camera.zoom)
            
        elif event_type == 'dims':
            self.viewer.dims.current_step = event.get('step', self.viewer.dims.current_step)
    
    @staticmethod
    def _pack_frame_header(frame_num: int, size: int) -> bytes:
        """Simple binary header: [frame_num:u32][size:u32]"""
        return struct.pack('!II', frame_num, size)


class CaseyStreamingClient:
    """
    Client side: Display frames, send input.
    No state tracking. The frame IS the state.
    """
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.current_frame = 0
        self.input_buffer = []
        
    async def connect(self, host: str, port: int):
        """Two connections: one for frames (TCP), one for input (UDP)."""
        # TCP for reliable frame delivery
        self.frame_reader, self.frame_writer = await asyncio.open_connection(host, port)
        
        # UDP for low-latency input
        self.input_socket = await asyncio.open_datagram_endpoint(
            lambda: InputProtocol(self),
            remote_addr=(host, port + 1)
        )
        
        # Start receiving frames
        asyncio.create_task(self.receive_frames())
        
    async def receive_frames(self):
        """Just display frames as fast as they arrive."""
        while True:
            # Read header
            header = await self.frame_reader.readexactly(8)
            frame_num, size = struct.unpack('!II', header)
            
            # Read frame data
            frame_data = await self.frame_reader.readexactly(size)
            
            # Convert and display immediately
            # No state check, no synchronization, just display
            self.display_frame(frame_data)
            self.current_frame = frame_num
    
    def display_frame(self, frame_bytes: bytes):
        """Display frame on canvas. This is all that matters."""
        # Convert bytes to image and display
        # In practice, this would update the Qt canvas
        pass
    
    def send_input(self, event_type: str, **kwargs):
        """Send input immediately via UDP. Fire and forget."""
        event = {'type': event_type, 'frame': self.current_frame, **kwargs}
        packet = json.dumps(event).encode()
        self.input_socket.sendto(packet)


# The Casey philosophy in action:
if __name__ == "__main__":
    """
    What Casey would say:
    
    "You're overthinking this. The user sees pixels. Send pixels.
    Everything else is engineering masturbation.
    
    Your 'state synchronization' problem? Fake. The frame IS the state.
    Your 'protocol design'? Three integers and a byte array.
    Your 'async complexity'? Two threads. One renders. One handles input.
    
    Stop designing. Start measuring. If it's slower than 16ms, fix it.
    If it's faster than 16ms, ship it."
    """
    
    import napari
    
    # This is the entire server
    viewer = napari.Viewer(show=False)
    viewer.add_image(np.random.random((100, 512, 512)))
    
    server = CaseyStreamingServer(viewer, target_fps=30)
    asyncio.run(server.render_loop())