from __future__ import annotations

"""Lightweight controllers used by :class:`ClientStreamLoop`.

Provide small dataclasses that wrap starting threads for state and pixel
receivers so the loop can delegate orchestration to pure helpers.
"""

from dataclasses import dataclass
from threading import Thread
from typing import Callable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from napari_cuda.client.control.control_channel_client import StateChannel
    from napari_cuda.protocol.messages import (
        LayerRemoveMessage,
        LayerUpdateMessage,
        NotifyDimsFrame,
        NotifyStreamFrame,
        SceneSpecMessage,
    )
    from napari_cuda.protocol import AckState, ErrorCommand, ReplyCommand
    from napari_cuda.client.control.control_channel_client import SessionMetadata

from napari_cuda.client.streaming.receiver import PixelReceiver


@dataclass
class StateController:
    host: str
    port: int
    handle_notify_stream: Optional[Callable[["NotifyStreamFrame"], None]] = None
    handle_dims_update: Optional[Callable[["NotifyDimsFrame"], None]] = None
    handle_scene_spec: Optional[Callable[["SceneSpecMessage"], None]] = None
    handle_layer_update: Optional[Callable[["LayerUpdateMessage"], None]] = None
    handle_layer_remove: Optional[Callable[["LayerRemoveMessage"], None]] = None
    handle_ack_state: Optional[Callable[["AckState"], None]] = None
    handle_reply_command: Optional[Callable[["ReplyCommand"], None]] = None
    handle_error_command: Optional[Callable[["ErrorCommand"], None]] = None
    handle_session_ready: Optional[Callable[["SessionMetadata"], None]] = None
    handle_connected: Optional[Callable[[], None]] = None
    handle_disconnect: Optional[Callable[[Optional[Exception]], None]] = None

    def start(self) -> Tuple["StateChannel", Thread]:
        from napari_cuda.client.control.control_channel_client import StateChannel

        ch = StateChannel(
            self.host,
            int(self.port),
            handle_notify_stream=self.handle_notify_stream,
            handle_dims_update=self.handle_dims_update,
            handle_scene_spec=self.handle_scene_spec,
            handle_layer_update=self.handle_layer_update,
            handle_layer_remove=self.handle_layer_remove,
            handle_ack_state=self.handle_ack_state,
            handle_reply_command=self.handle_reply_command,
            handle_error_command=self.handle_error_command,
            handle_session_ready=self.handle_session_ready,
            handle_connected=self.handle_connected,
            handle_disconnect=self.handle_disconnect,
        )
        t = Thread(target=ch.run, daemon=True)
        t.start()
        return ch, t

    def stop(self, channel: "StateChannel", thread: Thread, timeout: float = 2.0) -> None:
        try:
            channel.stop()
        except Exception:
            pass
        thread.join(timeout)


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
