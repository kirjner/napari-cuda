"""
napari-cuda: CUDA acceleration for napari

This module provides CUDA-accelerated functions for napari,
with automatic CPU fallbacks when CUDA is not available.
"""

import os
import warnings
from typing import Optional

# Check CUDA availability
try:
    import cupy as cp
    import pycuda.driver as cuda
    cuda.init()
    HAS_CUDA = True
    CUDA_DEVICE_NAME = cuda.Device(0).name()
    print(f"✅ CUDA available: {CUDA_DEVICE_NAME}")
except (ImportError, Exception) as e:
    HAS_CUDA = False
    CUDA_DEVICE_NAME = None
    if os.environ.get('NAPARI_CUDA_REQUIRED'):
        raise ImportError("CUDA required but not available") from e
    else:
        warnings.warn(f"CUDA not available, using CPU fallbacks: {e}")

# Smart imports based on availability
if HAS_CUDA:
    from napari_cuda.cuda.screenshot import CUDAScreenshot as Screenshot
    from napari_cuda.cuda.memory import GPUMemoryManager
else:
    from napari_cuda.cpu.screenshot import CPUScreenshot as Screenshot
    from napari_cuda.cpu.memory import CPUMemoryManager as GPUMemoryManager

__version__ = "0.1.0"
__all__ = ["Screenshot", "GPUMemoryManager", "HAS_CUDA", "accelerate"]


def accelerate(viewer=None):
    """
    Enable CUDA acceleration for napari.
    
    Parameters
    ----------
    viewer : napari.Viewer, optional
        If provided, accelerate this specific viewer.
        If None, patch napari globally.
    
    Returns
    -------
    bool
        True if CUDA acceleration was enabled, False otherwise.
    """
    if not HAS_CUDA:
        warnings.warn("Cannot enable CUDA acceleration: CUDA not available")
        return False
    
    from napari_cuda.cuda import monkey_patch
    
    if viewer is not None:
        # Accelerate specific viewer
        monkey_patch.patch_viewer(viewer)
    else:
        # Global patch
        monkey_patch.patch_global()
    
    print(f"🚀 CUDA acceleration enabled on {CUDA_DEVICE_NAME}")
    return True


def benchmark():
    """Run benchmarks comparing CPU vs CUDA performance."""
    from napari_cuda.benchmarks import run_all
    return run_all()