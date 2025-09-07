"""
patterns.py - Utility to generate simple RGBA test patterns for 2D rendering.

Provides structured images that compress well under H.264, avoiding
high-entropy random noise that tends to trigger macroblock artifacts.
"""

from __future__ import annotations

import os
import numpy as np


def make_rgba_image(width: int, height: int, pattern: str | None = None) -> np.ndarray:
    """Return an RGBA uint8 image of shape (H, W, 4).

    Patterns:
    - 'gradient' (default): horizontal RGB ramps blended with vertical luminance ramp
    - 'bars': 8 vertical color bars
    - 'checker': checkerboard tiles
    - 'noise': random noise (not recommended for streaming demos)
    """
    pat = (pattern or os.getenv('NAPARI_CUDA_2D_PATTERN', 'gradient')).lower()
    H, W = int(height), int(width)

    if pat == 'bars':
        colors = [
            (255, 255, 255),  # white
            (255, 255, 0),    # yellow
            (0, 255, 255),    # cyan
            (0, 255, 0),      # green
            (255, 0, 255),    # magenta
            (255, 0, 0),      # red
            (0, 0, 255),      # blue
            (32, 32, 32),     # dark gray
        ]
        image = np.zeros((H, W, 4), dtype=np.uint8)
        bw = max(1, W // len(colors))
        for i, (r, g, b) in enumerate(colors):
            x0 = i * bw
            x1 = W if i == len(colors) - 1 else (i + 1) * bw
            image[:, x0:x1, 0] = r
            image[:, x0:x1, 1] = g
            image[:, x0:x1, 2] = b
        image[..., 3] = 255
        return image

    if pat == 'checker':
        tile = 32
        yy, xx = np.mgrid[0:H, 0:W]
        check = (((yy // tile) + (xx // tile)) % 2).astype(np.uint8)
        image = np.zeros((H, W, 4), dtype=np.uint8)
        image[..., 0] = np.where(check == 1, 220, 50)
        image[..., 1] = np.where(check == 1, 220, 50)
        image[..., 2] = np.where(check == 1, 220, 50)
        image[..., 3] = 255
        return image

    if pat == 'noise':
        return np.random.randint(0, 255, (H, W, 4), dtype=np.uint8)

    # Default: gradient
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    r = x[None, :]
    g = 1.0 - x[None, :]
    b = 0.5 * np.ones_like(r)
    l = y  # luminance ramp
    R = np.clip((0.5 * r + 0.5 * l) * 255.0, 0, 255).astype(np.uint8)
    G = np.clip((0.5 * g + 0.5 * l) * 255.0, 0, 255).astype(np.uint8)
    B = np.clip((0.5 * b + 0.5 * l) * 255.0, 0, 255).astype(np.uint8)
    A = np.full((H, W), 255, dtype=np.uint8)
    return np.stack([R, G, B, A], axis=-1)

