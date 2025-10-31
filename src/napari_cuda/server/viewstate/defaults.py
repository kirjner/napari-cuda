"""Shared default state helpers for server scene components."""

from __future__ import annotations

from typing import Dict, Any


def default_volume_state() -> Dict[str, Any]:
    """Return the canonical defaults for volume render hints."""

    return {
        "mode": "mip",
        "colormap": "gray",
        "clim": [0.0, 1.0],
        "opacity": 1.0,
        "sample_step": 1.0,
    }


__all__ = ["default_volume_state"]
