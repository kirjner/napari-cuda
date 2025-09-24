"""Helpers for applying camera animation logic."""

from __future__ import annotations

from typing import Any

from napari_cuda.server import camera_ops as camops


def animate_if_enabled(
    *,
    enabled: bool,
    view: Any,
    width: int,
    height: int,
    animate_dps: float,
    anim_start: float,
) -> None:
    """Animate the camera when animation is enabled and a camera is present."""

    if not enabled or view is None:
        return

    camera = getattr(view, "camera", None)
    if camera is None:
        return

    camops.animate_camera(
        camera=camera,
        width=int(width),
        height=int(height),
        animate_dps=float(animate_dps),
        anim_start=float(anim_start),
    )


__all__ = ["animate_if_enabled"]
