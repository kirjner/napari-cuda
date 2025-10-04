# napari-cuda Development Roadmap

## Current Status
We have implemented the core architecture with proper separation of concerns:
- ✅ ProxyViewer (ViewerModel-based) preventing local rendering
- ✅ Dual WebSocket protocol (state + pixels)
- ✅ Clean layer factory registration pattern
- ✅ Thread-safe CUDA context management
- ✅ qasync integration for Qt/asyncio event loops

However, the actual CUDA-OpenGL interop and streaming implementation are still placeholders.

## Immediate Next Steps

### Phase 1: HPC Validation (Days 1-2)
**Goal**: Verify our architecture works with real CUDA hardware

```bash
# On HPC:
git pull
./scripts/setup_hpc.sh
uv run python scripts/test_cuda_gl.py  # Create standalone test
```

**Deliverables**:
- [ ] Confirm CUDA-OpenGL interop works
- [ ] Verify texture registration succeeds
- [ ] Measure baseline texture capture time

### Phase 2: Basic Frame Streaming (Days 3-5)
**Goal**: Prove end-to-end pipeline with simple JPEG encoding

#### 2.1 Implement Real Texture Capture
```python
# src/napari_cuda/server/render_thread.py
def _capture_and_encode(self, request):
    # Map OpenGL texture to CUDA
    reg_image.map()
    cuda_array = reg_image.get_mapped_array(0, 0)
    
    # Copy to host memory (temporary, until NVENC works)
    import cupy as cp
    frame = cp.asnumpy(cuda_array)
    
    # Simple JPEG encoding
    import cv2
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()
```

#### 2.2 Implement Client Canvas
```python
# src/napari_cuda/client/app/streaming_canvas.py
class StreamingCanvas(QWidget):
    async def receive_frames(self):
        async with websockets.connect(self.pixel_url) as ws:
            while True:
                jpeg_data = await ws.recv()
                # Decode and display
                pixmap = QPixmap()
                pixmap.loadFromData(jpeg_data)
                self.update_display(pixmap)
```

**Deliverables**:
- [ ] Working texture capture to numpy/CuPy
- [ ] JPEG encoding on server
- [ ] Client decoding and display
- [ ] Measure end-to-end latency

### Phase 3: Performance Benchmarking (Days 6-7)
**Goal**: Quantify performance vs CPU baseline

Create `scripts/benchmark_streaming.py`:
```python
def benchmark_capture_methods():
    # Method 1: CPU screenshot (baseline)
    # Method 2: CUDA capture + JPEG
    # Method 3: CUDA capture + NVENC (future)
    
    return {
        'capture_time': [],
        'encode_time': [],
        'network_time': [],
        'decode_time': [],
        'total_latency': [],
        'fps': []
    }
```

**Key Metrics**:
- [ ] OpenGL texture capture: Target < 1ms (currently 40ms CPU)
- [ ] JPEG encoding: Target < 5ms
- [ ] Network transmission: Target < 2ms (local)
- [ ] Total latency: Target < 10ms
- [ ] Achieve 60 FPS at 1920x1080

### Phase 4: NVENC Integration (Days 8-10)
**Goal**: Hardware-accelerated H.264 encoding

Create `src/napari_cuda/cuda/nvenc_wrapper.py`:
```python
class NVENCEncoder:
    def __init__(self, width, height, bitrate=5000000, fps=60):
        self.encoder = self._create_encoder()
        self.configure_h264()
    
    def encode_cuda_surface(self, cuda_array):
        # Direct CUDA surface to NVENC
        # No CPU copy needed
        return h264_nal_units
```

**Deliverables**:
- [ ] NVENC encoder wrapper
- [ ] H.264 streaming working
- [ ] Client H.264 decoding (PyAV)
- [ ] Measure encoding time < 1ms

### Phase 5: State Synchronization (Days 11-12)
**Goal**: Complete bidirectional state sync

Implement in protocol:
- [ ] Layer property changes (colormap, contrast, opacity)
- [ ] Layer visibility toggles
- [ ] Layer addition/removal
- [ ] Annotation layers (points, shapes, labels)
- [ ] Viewer settings (axes, scale bar, etc.)

### Phase 6: Production Features (Days 13-15)
**Goal**: Make it robust and usable

#### 6.1 Adaptive Streaming
```python
class AdaptiveController:
    def monitor_performance(self):
        if self.encode_queue.qsize() > 3:
            self.reduce_quality()
        if self.network_latency > 50:
            self.skip_frames()
```

#### 6.2 Multi-client Support
- [ ] Broadcast frames to multiple clients
- [ ] Per-client state management
- [ ] Client authentication

#### 6.3 Error Recovery
- [ ] Auto-reconnect on disconnect
- [ ] Graceful degradation on errors
- [ ] Health checks and monitoring

## Success Criteria

### MVP (2 weeks)
- [ ] 10x faster than CPU screenshots (4ms vs 40ms)
- [ ] 30 FPS streaming at 1920x1080
- [ ] Basic state synchronization
- [ ] Works on HPC with L4 GPU

### Production (1 month)
- [ ] 40x faster with NVENC (1ms capture)
- [ ] 60 FPS at 4K resolution
- [ ] Complete state synchronization
- [ ] Multi-client support
- [ ] Docker deployment

## Technical Risks & Mitigations

| Risk | Impact | Mitigation |
|------|---------|------------|
| CUDA-OpenGL interop fails | High | Test early on HPC, have CPU fallback |
| NVENC not available | Medium | Use JPEG/VP8 software encoding |
| Network latency too high | Medium | Implement frame skipping, reduce quality |
| Qt/asyncio conflicts | Low | Already solved with qasync |

## Testing Strategy

### Unit Tests
```python
# tests/test_cuda_interop.py
@pytest.mark.gpu
def test_texture_registration():
    # Requires GPU
    
# tests/test_protocol.py
def test_message_serialization():
    # Can run locally
```

### Integration Tests
```python
# tests/test_end_to_end.py
async def test_full_pipeline():
    server = HeadlessServer()
    client = ProxyViewer()
    # Verify frames received
```

### Performance Tests
```python
# tests/test_performance.py
def test_capture_latency():
    assert capture_time < 5  # ms
    assert fps >= 30
```

## Documentation Needs

1. **User Guide**
   - How to setup SSH tunnel
   - How to connect client
   - Troubleshooting guide

2. **Developer Guide**
   - Architecture overview
   - How to add new layer types
   - Protocol specification

3. **HPC Admin Guide**
   - Installation requirements
   - CUDA/driver versions
   - Network configuration

## NVIDIA Fellowship Deliverables

For the fellowship application, prioritize:

1. **Performance Numbers**
   - Benchmark showing 40x speedup
   - Graph of latency vs resolution
   - Comparison with existing solutions

2. **Demo Video**
   - Show real microscopy data
   - Demonstrate smooth interaction
   - Multiple clients connecting

3. **Technical Innovation**
   - Novel dual-stream protocol
   - Zero-copy GPU pipeline
   - Clean architecture pattern

## Development Schedule

| Week | Focus | Deliverable |
|------|-------|------------|
| 1 | HPC validation & basic streaming | JPEG streaming working |
| 2 | NVENC & benchmarking | 30 FPS with measurements |
| 3 | State sync & multi-client | Complete MVP |
| 4 | Polish & documentation | Production ready |

## Key Commands for Development

```bash
# Local development (Mac)
uv sync --extra client
uv run napari-cuda-client --host hpc.server.edu

# HPC development
uv sync --extra server --extra cuda
uv run python -m napari_cuda.server.headless_server data.npy

# Testing
uv run pytest tests/ -v -m "not gpu"  # Local tests
uv run pytest tests/ -v -m "gpu"       # HPC tests

# Benchmarking
uv run python scripts/benchmark_streaming.py --resolution 1920x1080
```

## Open Questions

1. **Should we support multiple GPUs?**
   - Could distribute layers across GPUs
   - Parallel encoding for multiple clients

2. **What about volume rendering?**
   - 3D data is much larger
   - May need different streaming strategy

3. **How to handle annotations?**
   - Vector data (points, shapes) could be sent separately
   - Might not need GPU acceleration

4. **Authentication and security?**
   - Currently no auth
   - Need to consider for production

## Contact & Resources

- GitHub: https://github.com/kirjner/napari-cuda
- napari Zulip: https://napari.zulipchat.com
- NVIDIA NVENC SDK: https://developer.nvidia.com/nvidia-video-codec-sdk
- CuPy Docs: https://docs.cupy.dev/
- PyCUDA GL Interop: https://documen.tician.de/pycuda/gl.html