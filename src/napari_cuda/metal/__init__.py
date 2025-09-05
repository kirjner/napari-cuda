"""Metal GPU acceleration for macOS.

The M3 Pro has:
- 18-core GPU (14-core on base M3)
- Hardware ray tracing
- Dynamic Caching
- Mesh shading
- 18GB unified memory (shared between CPU/GPU)

Advantages over CUDA:
- Unified memory architecture (no PCIe transfers!)
- Lower latency for small operations
- Better integration with macOS display pipeline

Disadvantages:
- No NVENC equivalent (must use VideoToolbox)
- Less mature compute ecosystem
- No CuPy equivalent (need to use raw Metal or MLX)
"""

from .screenshot import MetalScreenshot, HAS_METAL

__all__ = ["MetalScreenshot", "HAS_METAL"]