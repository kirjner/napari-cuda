# napari-cuda MVP Implementation Plan

## Overview
Transform napari into a distributed visualization tool using CUDA-accelerated rendering and streaming from HPC to local client.

## Critical Technical Challenges & Solutions

### 1. Qt/asyncio Event Loop Integration
**Challenge**: Running Qt event loop (`napari.run()`) alongside asyncio for WebSocket server.

**Solution**: Use QThread with asyncio event loop
```python
from qtpy.QtCore import QThread
import asyncio

class AsyncioThread(QThread):
    def __init__(self):
        super().__init__()
        self.loop = None
    
    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)

# In HeadlessServer
def start(self):
    # Create asyncio thread for WebSocket server
    self.asyncio_thread = AsyncioThread()
    self.asyncio_thread.start()
    
    # Run state server in asyncio thread
    asyncio.run_coroutine_threadsafe(
        self._start_state_server(),
        self.asyncio_thread.loop
    )
    
    # Qt runs in main thread
    with napari.gui_qt(visible=False):
        self.viewer = napari.Viewer()
        # ... setup ...
        napari.run()  # Blocks here
```

### 2. Pixel Stream Protocol Choice
**Challenge**: RTMP has high latency, WebRTC is complex.

**Solution**: Start with WebSocket + H.264, migrate to WebRTC later
```python
# Initial MVP: WebSocket binary frames
class PixelStreamServer:
    def __init__(self, port=8082):
        self.clients = set()
        
    async def handle_client(self, websocket, path):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def broadcast_frame(self, encoded_frame):
        # Send to all connected clients
        if self.clients:
            await asyncio.gather(*[
                client.send(encoded_frame) 
                for client in self.clients
            ])

# Future: WebRTC with aiortc
# - Lower latency (50ms vs 200ms)
# - Adaptive bitrate
# - P2P capability
```

### 3. Client-Side Decoding
**Challenge**: cv2.VideoCapture is blocking and has limited control.

**Solution**: Use PyAV for better control
```python
import av
import numpy as np
from threading import Thread, Event

class H264Decoder:
    def __init__(self):
        self.codec = av.CodecContext.create('h264', 'r')
        self.packets_queue = queue.Queue()
        
    def decode_frame(self, h264_data):
        packet = av.Packet(h264_data)
        frames = self.codec.decode(packet)
        for frame in frames:
            return frame.to_ndarray(format='rgb24')
        return None
```

### 4. CUDA-OpenGL Context Management
**Challenge**: Contexts must be on correct thread, OpenGL requires specific initialization.

**Solution**: Dedicated render thread with persistent contexts

**Threading Model Architecture**:
- **Main Thread**: Runs napari Qt event loop (napari.run())
- **CUDARenderThread (QThread)**: Manages CUDA/OpenGL contexts, performs all GPU operations
- **AsyncioThread (QThread)**: Runs asyncio event loop for WebSocket communication
- **Communication**: Thread-safe queues (queue.Queue) pass data between threads
```python
import pycuda.driver as cuda
from pycuda.gl import graphics_map_flags
from OpenGL import GL
from qtpy.QtCore import QThread

class CUDARenderThread(QThread):
    def __init__(self, gl_context):
        super().__init__()
        self.gl_context = gl_context
        self.cuda_context = None
        
    def run(self):
        # Make OpenGL context current on this thread
        self.gl_context.makeCurrent()
        
        # Initialize CUDA on same thread
        cuda.init()
        device = cuda.Device(0)
        self.cuda_context = device.make_context()
        
        # Enable GL interop
        import pycuda.gl
        pycuda.gl.init()
        
        # Keep contexts alive
        self.exec_()  # Qt event loop for this thread
    
    def capture_texture(self, gl_texture_id):
        """Must be called from this thread."""
        # Register texture (cached if already registered)
        if gl_texture_id not in self.registered_textures:
            self.registered_textures[gl_texture_id] = RegisteredImage(
                gl_texture_id, 
                GL.GL_TEXTURE_2D,
                graphics_map_flags.READ_ONLY
            )
        
        reg_image = self.registered_textures[gl_texture_id]
        reg_image.map()
        cuda_array = reg_image.get_mapped_array(0, 0)
        
        # Encode frame (NVENC operations happen on same thread)
        encoded = self.nvenc_encoder.encode_surface(cuda_array)
        
        reg_image.unmap()
        return encoded
```

## Directory Structure
```
napari-cuda/
├── MVP_IMPLEMENTATION_PLAN.md          # This file
├── docs/
│   ├── NVIDIA_FELLOWSHIP_NOTES.md     # For your application
│   └── architecture.md                # Technical architecture
├── src/napari_cuda/
│   ├── server/
│   │   ├── __init__.py
│   │   ├── headless_server.py         # Main server orchestrator
│   │   ├── cuda_streaming_layer.py    # Custom VispyScalarFieldBaseLayer
│   │   ├── render_thread.py           # CUDA/GL context management
│   │   └── nvenc_wrapper.py           # NVENC encoding
│   ├── client/
│   │   ├── __init__.py
│   │   ├── proxy_viewer.py            # ProxyViewer replacing Viewer
│   │   ├── streaming_canvas.py        # Video display canvas
│   │   └── h264_decoder.py            # PyAV-based decoder
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── messages.py                # Protocol message definitions
│   │   ├── state_sync.py              # State synchronization
│   │   └── frame_transport.py         # Frame delivery (WS then WebRTC)
│   └── cuda/
│       ├── __init__.py
│       ├── gl_interop.py              # OpenGL-CUDA utilities
│       └── memory_pool.py             # CUDA memory management
├── tests/
│   ├── test_cuda_capture.py           # Test CUDA-GL interop
│   ├── test_encoding.py               # Test NVENC
│   └── test_streaming.py              # End-to-end test
└── scripts/
    ├── install_cuda_deps.sh           # Install on HPC
    ├── start_server.py                # Launch server
    └── test_client.py                 # Launch test client
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
1. Set up CUDA-OpenGL interop on HPC
2. Test GL texture capture to CUDA memory
3. Verify NVENC encoding works
4. Benchmark capture + encode pipeline

### Phase 2: Server Implementation (Week 2)
1. Create CudaStreamingLayer subclass
2. Implement render thread with context management
3. Set up WebSocket servers (state + pixel streams)
4. Test with headless napari viewer

### Phase 3: Client Implementation (Week 3)
1. Create ProxyViewer with careful initialization handling
2. Implement StreamingCanvas with H.264 decoding
3. Connect state and pixel streams
4. Test round-trip latency

**Critical ProxyViewer Architecture**:
The ProxyViewer must prevent local Window creation. Two approaches:

**Option A: Override __init__ completely**
```python
class ProxyViewer(Viewer):
    def __init__(self, server_address, *args, **kwargs):
        # DON'T call super().__init__() which creates Window
        # Instead, manually initialize only what we need:
        self.layers = LayerList()
        self.dims = Dims()
        self.camera = Camera()
        # ... other components
        self._window = None  # Explicitly None
        self._connect_to_server(server_address)
```

**Option B: Surgical Window replacement**
```python
def launch_streaming_client():
    # Create viewer but immediately replace its window
    viewer = Viewer(show=False)  # Prevents immediate show
    viewer._window.close()  # Close the auto-created window
    
    # Create our custom window with StreamingCanvas
    qt_viewer = QtViewer(
        ProxyViewer(viewer),  # Wrap existing viewer
        canvas_class=StreamingCanvas
    )
    viewer._window = Window(qt_viewer)  # Replace window
```

### Phase 4: Optimization (Week 4)
1. Profile and optimize bottlenecks
2. Implement frame skipping/throttling
3. Add connection resilience
4. Document performance metrics

## HPC Development Setup

### Initial Setup Script (scripts/setup_hpc.sh)
```bash
#!/bin/bash
# Run once on HPC to set up environment

# Load modules
module load cuda/12.4
module load gcc/11.2

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy napari[all] qtpy PyQt5
pip install pycuda
pip install websockets aiohttp
pip install av  # For video encoding/decoding

# Install Video Codec SDK (for NVENC)
cd /tmp
wget https://developer.download.nvidia.com/compute/nvenc/v11.1/Video_Codec_SDK_11.1.5.zip
unzip Video_Codec_SDK_11.1.5.zip
export NVENC_SDK_PATH=/tmp/Video_Codec_SDK_11.1.5

# Build PyNvCodec (Python bindings for NVENC)
git clone https://github.com/NVIDIA/VideoProcessingFramework.git
cd VideoProcessingFramework
pip install .

echo "Setup complete! Activate with: source venv/bin/activate"
```

### Test CUDA-GL Interop

Run the Makefile target instead of a standalone script—it configures the Mesa
runtime and validates CUDA↔EGL interop in one step:

```bash
make verify-gl
```

This relies on `.env` exporting `NAPARI_CUDA_GL_PREFIX` and
`NAPARI_CUDA_GL_DRIVERS` (populated after `make setup-gl`).

## Next Steps for HPC Development

### Immediate Actions (Do First on HPC):
1. **Test Environment**
   ```bash
   cd ~/napari-cuda
   ./scripts/setup_hpc.sh
   source venv/bin/activate
   make verify-gl
   ```

2. **Verify NVENC**
   ```bash
   nvidia-smi --query-gpu=encoder.stats.sessionCount --format=csv
   # Should show NVENC is available
   ```

3. **Create Minimal Texture Capture**
   ```python
   # In src/napari_cuda/server/test_capture.py
   # Minimal test of texture->CUDA->encoded frame
   ```

4. **Benchmark Pipeline**
   ```python
   # Time each step:
   # 1. Render to texture: ?ms
   # 2. CUDA capture: ?ms  
   # 3. NVENC encode: ?ms
   # Target: <10ms total
   ```

### Development Order:
1. Get CUDA-GL interop working (`make verify-gl`)
2. Get NVENC encoding working (standalone test)
3. Integrate into CudaStreamingLayer
4. Add WebSocket streaming
5. Build client components
6. Test end-to-end

### SSH Tunnel for Testing:
```bash
# On local machine, tunnel to HPC
ssh -L 8080:localhost:8080 -L 8081:localhost:8081 -L 8082:localhost:8082 kirjner@node2810
```

## Performance Targets

### Ideal Targets
- Texture capture: <2ms
- NVENC encoding (1080p): <3ms  
- Network transport: <10ms (LAN)
- Client decode: <5ms
- **Total latency: <20ms**
- **Target FPS: 60**

### MVP Acceptance Criteria
- Total latency: <100ms (with clear optimization path)
- Minimum FPS: 30
- Stretch goal: <50ms latency at 60 FPS

## Known Issues to Address

### Critical MVP Issues
1. **Virtual Display**: May need Xvfb or EGL for headless OpenGL
2. **Thread Safety**: All GL operations must happen on render thread
3. **Memory Leaks**: Must properly unregister textures
4. **Frame Pacing**: Need timestamp-based synchronization
5. **Connection Loss**: Implement reconnection logic
6. **Round-trip Pan/Zoom**: Every interaction requires network round-trip (adds latency)

### Post-MVP Enhancements
1. **Client-Side Prediction**: Implement texture caching and local transformation for instant pan/zoom feedback
   - Cache last N frames from server
   - Apply transformations locally during interaction
   - Request new keyframe from server on interaction end
   - Reduces perceived latency from 100ms to <16ms

2. **Adaptive Quality**: Dynamically adjust resolution/quality based on network conditions

3. **Efference Copy**: Sophisticated state prediction on client to hide latency

## Success Criteria
✅ Server runs headless on HPC  
✅ Client connects and shows remote data  
✅ Pan/zoom updates in <100ms  
✅ 30+ FPS for Full HD streaming  
✅ All rendering on GPU (0% CPU for capture)
