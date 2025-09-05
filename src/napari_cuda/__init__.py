"""
napari-cuda: CUDA acceleration for napari

This module provides CUDA-accelerated functions for napari,
with automatic CPU fallbacks when CUDA is not available.
"""

import os
import warnings
from typing import Optional

# Lazy CUDA availability check
HAS_CUDA = None
CUDA_DEVICE_NAME = None

def _check_cuda():
    """Lazy check for CUDA availability."""
    global HAS_CUDA, CUDA_DEVICE_NAME
    
    if HAS_CUDA is not None:
        return HAS_CUDA
        
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
            # Only warn when actually trying to use CUDA
            pass
    
    return HAS_CUDA

# Don't import anything at module level - let individual components check CUDA when needed

__version__ = "0.1.0"
# Export only safe, lazily evaluated API at import time.
# Client-side imports must not force CUDA checks.
__all__ = ["HAS_CUDA", "accelerate", "benchmark", "_check_cuda", "__version__"]


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
    if not _check_cuda():
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
