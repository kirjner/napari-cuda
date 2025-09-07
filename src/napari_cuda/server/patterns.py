"""
patterns.py - Utility to generate simple RGBA test patterns for 2D rendering.

Provides structured images that compress well under H.264, avoiding
high-entropy random noise that tends to trigger macroblock artifacts.
"""

from __future__ import annotations

import os
from typing import Optional
import numpy as np


def _load_image_rgba(path: str, width: int, height: int) -> Optional[np.ndarray]:
    """Load an image from disk and return RGBA uint8 resized to (H, W).

    Tries imageio.v3 first; falls back to PIL if available.
    Returns None on failure.
    """
    try:
        import imageio.v3 as iio  # type: ignore
        img = iio.imread(path)
    except Exception:
        try:
            from PIL import Image  # type: ignore
            img = Image.open(path).convert('RGBA').resize((width, height))
            arr = np.array(img, dtype=np.uint8)
            return arr
        except Exception:
            return None
    try:
        # Ensure 3 or 4 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 3:
            alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
            img = np.concatenate([img.astype(np.uint8), alpha], axis=-1)
        elif img.shape[-1] == 4:
            img = img.astype(np.uint8)
        else:
            return None
        # Resize if needed (nearest to avoid import of skimage)
        if img.shape[1] != width or img.shape[0] != height:
            try:
                from PIL import Image  # type: ignore
                img = Image.fromarray(img, mode='RGBA').resize((width, height))
                img = np.array(img, dtype=np.uint8)
            except Exception:
                # Simple crop/pad fallback
                h, w = img.shape[:2]
                out = np.zeros((height, width, 4), dtype=np.uint8)
                out[: min(height, h), : min(width, w)] = img[: min(height, h), : min(width, w)]
                img = out
        return img
    except Exception:
        return None


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

    if pat in ('image', 'cat'):
        path = os.getenv('NAPARI_CUDA_2D_IMAGE')
        if path:
            arr = _load_image_rgba(path, W, H)
            if arr is not None:
                return arr
        # Fall back if not provided or failed
        pat = 'gradient'

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
