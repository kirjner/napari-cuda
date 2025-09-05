# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

napari-cuda transforms napari into a distributed GPU-accelerated visualization engine. It streams CUDA-accelerated napari sessions from HPC clusters to local clients, enabling interactive visualization of terabyte-scale microscopy datasets.

## Build and Development Commands

This is a **uv-managed project** - all dependency management uses uv, not pip directly.

```bash
# Local development
uv sync                        # Install base dependencies
uv sync --extra server        # Install server dependencies (includes CUDA)
uv sync --extra client        # Install client dependencies

# Run components
uv run napari-cuda-server data.npy    # Start server (on HPC with GPU)
uv run napari-cuda-client             # Start client (local machine)

# Testing
uv run pytest                          # Run tests
uv run pytest -m "not slow"           # Skip slow tests
make typecheck                        # Run mypy type checking

# HPC deployment (from project root)
./scripts/setup_hpc.sh                # Complete HPC setup with CUDA detection
./start_server.sh                     # Launch server with GPU
./test_interop.sh                     # Test CUDA-OpenGL interop

# Development workflow (laptop to HPC)
./dev/sync.sh sync                    # Sync code to HPC
./dev/sync.sh test                    # Run tests on HPC
./dev/sync.sh shell                   # Interactive HPC session
```

## Architecture Overview

### Dual-Stream Protocol
- **State WebSocket (8081)**: JSON commands for camera, dims, layers
- **Pixel WebSocket (8082)**: Binary H.264-encoded frames

### Core Components

#### Server (HPC with GPU)
- `src/napari_cuda/server/headless_server.py`: Main orchestrator, runs napari headless
- `src/napari_cuda/server/cuda_streaming_layer.py`: Captures OpenGL textures on GPU
- `src/napari_cuda/server/render_thread.py`: CUDA/OpenGL contexts, NVENC encoding

#### Client (Local machine)
- `src/napari_cuda/client/proxy_viewer.py`: Thin client preventing local Window
- `src/napari_cuda/client/streaming_canvas.py`: Decodes and displays H.264 streams
- `src/napari_cuda/client/launcher.py`: Entry point for connections

#### Protocol
- `src/napari_cuda/protocol/messages.py`: Type-safe message classes, binary packing

### Threading Model
1. **Qt Main Thread**: napari GUI and events
2. **Asyncio Thread**: WebSocket servers (QThread isolated)
3. **CUDA Thread**: GPU operations and encoding (QThread isolated)

Critical: Each thread manages its own contexts to prevent conflicts.

## CUDA-OpenGL Interoperability

Zero-copy pipeline: OpenGL texture → CUDA registered image → NVENC → H.264

```python
# Key pattern in render_thread.py
from pycuda.gl import RegisteredImage, graphics_map_flags
registered_texture = RegisteredImage(texture_id, GL.GL_TEXTURE_2D, graphics_map_flags.READ_ONLY)
```

## HPC Environment

CUDA drivers come from HPC modules, not Python packages:
- System CUDA loaded via `module load cuda/12.4`
- CuPy version matched dynamically: `cupy-cuda12x` or `cupy-cuda11x`
- Headless OpenGL via EGL: `export PYOPENGL_PLATFORM=egl`

## Development Strategy

**Hybrid workflow**: Develop on laptop, test CUDA features on HPC.

```python
# Auto-fallback pattern used throughout:
try:
    import pycuda.driver as cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

if HAS_CUDA:
    from napari_cuda.cuda.implementation import CUDAClass
else:
    from napari_cuda.cpu.implementation import CPUClass as CUDAClass
```

## Key Constraints

1. **ProxyViewer must prevent Window creation**: Override `_init_without_window()`
2. **CUDA contexts are thread-local**: Never share across threads
3. **Qt and asyncio need isolation**: Use QThreads with dedicated event loops
4. **Frame capture must be async**: Don't block napari's render loop

## Performance Targets

- 60 FPS at 1920x1080
- Sub-10ms latency (local network)
- 40x speedup over CPU (1ms vs 40ms screenshots)

## Common Issues and Solutions

1. **"No CUDA device found"**: Check `CUDA_VISIBLE_DEVICES` and module loading
2. **OpenGL context errors**: Verify `QT_QPA_PLATFORM=offscreen` and `PYOPENGL_PLATFORM=egl`
3. **Import errors on Mac**: Expected - CUDA components have CPU fallbacks
4. **WebSocket connection refused**: Check ports 8081/8082 and SSH tunnel

## Testing Strategy

- Mock CUDA components when testing locally without GPU
- Use `pytest.mark.gpu` for GPU-required tests
- Validate with `scripts/test_cuda_gl.py` before full integration

## NVIDIA Fellowship Context

This project is being developed for an NVIDIA Graduate Fellowship application. Key selling points:
- First open-source CUDA-accelerated napari
- Novel dual-stream architecture
- Real HPC deployment with L4 GPUs
- Uses full NVIDIA stack (CUDA, CuPy, NVENC)