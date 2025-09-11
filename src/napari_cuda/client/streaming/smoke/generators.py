from __future__ import annotations

"""
Lightweight RGB frame generators for smoke tests.

Factory returns a callable(frame_idx) -> np.ndarray[H,W,3] (uint8 RGB).
Modes supported: 'checker', 'gradient'.
"""

from typing import Callable

import numpy as np


def make_generator(mode: str, w: int, h: int, fps: float) -> Callable[[int], "np.ndarray"]:
    mode = (mode or "checker").lower()
    w = int(w); h = int(h); fps = float(fps)
    # Precompute bases
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]

    if mode == "gradient":
        def _gen(i: int) -> "np.ndarray":
            r = np.broadcast_to(x, (h, w))
            g = np.broadcast_to(y, (h, w))
            b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
            return np.dstack([r, g, b])
        return _gen

    # default: checker with slight motion
    def _gen_checker(i: int) -> "np.ndarray":
        # Modest drift per frame to avoid strobing; approx sine/cosine at ~2 Hz
        t = float(i) / max(1.0, fps)
        ox = int(((np.sin(t * 2.0) * 0.5 + 0.5) * 64))
        oy = int(((np.cos(t * 1.7) * 0.5 + 0.5) * 64))
        xx = (np.arange(w, dtype=np.int32)[None, :] + ox) // 32
        yy = (np.arange(h, dtype=np.int32)[:, None] + oy) // 32
        mask = ((xx ^ yy) & 1).astype(np.uint8)
        r = (mask * 255).astype(np.uint8, copy=False)
        g = np.broadcast_to(x, (h, w))
        b = np.broadcast_to(y, (h, w))
        return np.dstack([r, g, b])

    return _gen_checker

