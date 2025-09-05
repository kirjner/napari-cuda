"""CPU fallback for screenshot operations."""

import numpy as np
from typing import Optional


class CPUScreenshot:
    """CPU-based screenshot capture (fallback when CUDA not available)."""
    
    def __init__(self):
        self.backend = 'cpu'
    
    def capture(self, canvas, size: Optional[tuple] = None) -> np.ndarray:
        """
        Capture screenshot using CPU.
        This is the standard napari method.
        """
        # Standard napari screenshot method
        return canvas.native.grabFramebuffer()
    
    def __repr__(self):
        return "CPUScreenshot(backend='cpu')"