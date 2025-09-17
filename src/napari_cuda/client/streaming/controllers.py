from __future__ import annotations

"""
Lightweight controllers to keep the StreamCoordinator slim.

Procedural style with small dataclasses that wrap starting threads for
state and pixel receivers. Behavior mirrors the existing inline closures.
"""

from dataclasses import dataclass
from threading import Thread
from typing import Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from napari_cuda.protocol.messages import (
        LayerRemoveMessage,
        LayerUpdateMessage,
        SceneSpecMessage,
    )

from .receiver import PixelReceiver
from .state import StateChannel


@dataclass
class StateController:
    host: str
    port: int
    on_video_config: Optional[Callable[[dict], None]] = None
    on_dims_update: Optional[Callable[[dict], None]] = None
    on_scene_spec: Optional[Callable[["SceneSpecMessage"], None]] = None
    on_layer_update: Optional[Callable[["LayerUpdateMessage"], None]] = None
    on_layer_remove: Optional[Callable[["LayerRemoveMessage"], None]] = None
    on_connected: Optional[Callable[[], None]] = None
    on_disconnect: Optional[Callable[[Optional[Exception]], None]] = None

    def start(self) -> Tuple[StateChannel, Thread]:
        ch = StateChannel(
            self.host,
            int(self.port),
            on_video_config=self.on_video_config,
            on_dims_update=self.on_dims_update,
            on_scene_spec=self.on_scene_spec,
            on_layer_update=self.on_layer_update,
            on_layer_remove=self.on_layer_remove,
            on_connected=self.on_connected,
            on_disconnect=self.on_disconnect,
        )
        t = Thread(target=ch.run, daemon=True)
        t.start()
        return ch, t


@dataclass
class ReceiveController:
    host: str
    port: int
    on_connected: Optional[Callable[[], None]] = None
    on_frame: Optional[Callable[["object"], None]] = None
    on_disconnect: Optional[Callable[[Optional[Exception]], None]] = None

    def start(self) -> Tuple[PixelReceiver, Thread]:
        rx = PixelReceiver(
            self.host,
            int(self.port),
            on_connected=self.on_connected,
            on_frame=self.on_frame,
            on_disconnect=self.on_disconnect,
        )
        t = Thread(target=rx.run, daemon=True)
        t.start()
        return rx, t
