"""Registry of state.update handlers keyed by scope:key."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from .camera import (
    handle_camera_orbit,
    handle_camera_pan,
    handle_camera_reset,
    handle_camera_set,
    handle_camera_zoom,
)
from .dims import handle_dims_update
from .layer import handle_layer_property
from .multiscale import handle_multiscale_level
from .view import handle_view_ndisplay
from .volume import handle_volume_update

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

StateUpdateHandler = Callable[["StateUpdateContext"], Awaitable[bool]]

STATE_UPDATE_HANDLERS: dict[str, StateUpdateHandler] = {
    "view:ndisplay": handle_view_ndisplay,
    "camera:zoom": handle_camera_zoom,
    "camera:pan": handle_camera_pan,
    "camera:orbit": handle_camera_orbit,
    "camera:reset": handle_camera_reset,
    "camera:set": handle_camera_set,
    "layer:*": handle_layer_property,
    "volume:*": handle_volume_update,
    "multiscale:level": handle_multiscale_level,
    "multiscale:policy": handle_multiscale_level,
    "dims:index": handle_dims_update,
    "dims:step": handle_dims_update,
}

__all__ = ["STATE_UPDATE_HANDLERS"]
