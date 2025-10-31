"""
patterns.py - Utility to generate simple RGBA test patterns for 2D rendering.

Provides structured images that compress well under H.264, avoiding
high-entropy random noise that tends to trigger macroblock artifacts.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple
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

    if pat == 'image':
        path = os.getenv('NAPARI_CUDA_2D_IMAGE')
        if path:
            arr = _load_image_rgba(path, W, H)
            if arr is not None:
                return arr
        # Fall back if not provided or failed
        pat = 'gradient'

    if pat in ('cat', 'cat_hd'):
        # Pixel-art cat (16x16) with custom palette; 'cat_hd' applies micro-shading to 32x32
        # Palette: 0=bg, 1=fur, 2=shadow, 3=lines/eyes, 4=inner ear, 5=nose, 6=eye highlight
        pal: Tuple[Tuple[int, int, int, int], ...] = (
            (0, 0, 0, 255),        # 0 bg
            (220, 220, 220, 255),  # 1 fur
            (180, 180, 180, 255),  # 2 shadow
            (20, 20, 20, 255),     # 3 lines/eyes
            (255, 140, 170, 255),  # 4 inner ear
            (255, 90, 150, 255),   # 5 nose
            (255, 255, 255, 255),  # 6 eye highlight
        )
        cat = np.array([
            [0,0,3,3,0,0,0,0,0,0,3,3,0,0,0,0],
            [0,3,1,1,3,0,0,0,0,3,1,1,3,0,0,0],
            [3,1,4,1,1,3,0,0,3,1,1,4,1,3,0,0],
            [3,1,1,1,1,3,0,0,3,1,1,1,1,3,0,0],
            [3,1,1,1,1,1,3,3,1,1,1,1,1,3,0,0],
            [3,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0],
            [0,3,1,1,2,2,1,1,1,1,2,2,1,3,0,0],
            [0,3,1,1,3,6,1,1,1,1,6,3,1,3,0,0],
            [0,3,1,1,3,3,1,5,5,1,3,3,1,3,0,0],
            [0,3,1,1,1,1,3,5,5,3,1,1,1,3,0,0],
            [0,3,1,1,1,1,3,3,3,3,1,1,1,3,0,0],
            [0,3,1,1,1,1,1,3,3,1,1,1,1,3,0,0],
            [0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0],
            [0,0,3,1,1,1,1,1,1,1,1,1,3,0,0,0],
            [0,0,0,3,3,3,0,0,0,3,3,3,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ], dtype=np.uint8)
        # Optional micro-shading upscale (2x) for 'cat_hd'
        scale_env = 1
        try:
            scale_env = max(1, int(os.getenv('NAPARI_CUDA_CAT_SCALE', '1')))
        except Exception:
            scale_env = 1
        use_hd = (pat == 'cat_hd') or (scale_env > 1)
        if use_hd:
            # 2x block expansion with simple shading patterns
            block = {
                0: np.array([[0, 0], [0, 0]], dtype=np.uint8),  # bg
                1: np.array([[1, 1], [1, 2]], dtype=np.uint8),  # fur -> add shadow bottom-right
                2: np.array([[2, 2], [2, 2]], dtype=np.uint8),  # shadow stays
                3: np.array([[3, 3], [3, 3]], dtype=np.uint8),  # lines solid
                4: np.array([[4, 4], [2, 4]], dtype=np.uint8),  # inner ear with slight shadow
                5: np.array([[5, 5], [5, 5]], dtype=np.uint8),  # nose solid
                6: np.array([[6, 6], [6, 6]], dtype=np.uint8),  # highlight solid
            }
            Hs, Ws = cat.shape
            up = np.zeros((Hs * 2, Ws * 2), dtype=np.uint8)
            for r in range(Hs):
                for c in range(Ws):
                    up[r*2:r*2+2, c*2:c*2+2] = block[int(cat[r, c])]
            cat_img = up
        else:
            cat_img = cat
        # Map to RGBA
        Hs, Ws = cat_img.shape
        img_small = np.zeros((Hs, Ws, 4), dtype=np.uint8)
        for idx, color in enumerate(pal):
            mask = (cat_img == idx)
            if np.any(mask):
                img_small[mask] = np.array(color, dtype=np.uint8)
        # Nearest-neighbor upscale
        cell_w = max(1, W // Ws)
        cell_h = max(1, H // Hs)
        img = np.repeat(np.repeat(img_small, cell_h, axis=0), cell_w, axis=1)
        # Center and crop/pad to exact size
        out = np.zeros((H, W, 4), dtype=np.uint8)
        out[..., :] = pal[0]  # bg
        h, w = img.shape[:2]
        x0 = max(0, (W - w) // 2)
        y0 = max(0, (H - h) // 2)
        xs = min(W, w)
        ys = min(H, h)
        out[y0:y0+ys, x0:x0+xs] = img[:ys, :xs]
        return out

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
