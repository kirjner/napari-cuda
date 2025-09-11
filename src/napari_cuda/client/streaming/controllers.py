from __future__ import annotations

"""
Lightweight controllers to keep StreamManager slimmer.

Procedural style with small dataclasses that wrap starting threads for
state and pixel receivers. Behavior mirrors the existing inline closures.
"""

from dataclasses import dataclass
from threading import Thread
from typing import Callable, Optional, Tuple

from .receiver import PixelReceiver
from .state import StateChannel


@dataclass
class StateController:
    host: str
    port: int
    on_video_config: Optional[Callable[[dict], None]] = None

    def start(self) -> Tuple[StateChannel, Thread]:
        ch = StateChannel(self.host, int(self.port), on_video_config=self.on_video_config)
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

