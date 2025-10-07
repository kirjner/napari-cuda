from __future__ import annotations

"""Lightweight controllers used by :class:`ClientStreamLoop`.

Provide small dataclasses that wrap starting threads for state and pixel
receivers so the loop can delegate orchestration to pure helpers.
"""

from dataclasses import dataclass
from threading import Thread
from typing import Any, Callable, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from napari_cuda.client.control.control_channel_client import StateChannel
    from napari_cuda.protocol.messages import (
        NotifyDimsFrame,
        NotifyLayersFrame,
        NotifySceneFrame,
        NotifyStreamFrame,
    )
    from napari_cuda.protocol import AckState, ErrorCommand, ReplyCommand
    from napari_cuda.client.control.control_channel_client import SessionMetadata

from napari_cuda.client.runtime.receiver import PixelReceiver


@dataclass
class StateController:
    host: str
    port: int
    ingest_notify_stream: Optional[Callable[["NotifyStreamFrame"], None]] = None
    ingest_dims_notify: Optional[Callable[["NotifyDimsFrame"], None]] = None
    ingest_notify_scene_snapshot: Optional[Callable[["NotifySceneFrame"], None]] = None
    ingest_notify_layers: Optional[Callable[["NotifyLayersFrame"], None]] = None
    ingest_notify_camera: Optional[Callable[[Any], None]] = None
    ingest_ack_state: Optional[Callable[["AckState"], None]] = None
    ingest_reply_command: Optional[Callable[["ReplyCommand"], None]] = None
    ingest_error_command: Optional[Callable[["ErrorCommand"], None]] = None
    on_session_ready: Optional[Callable[["SessionMetadata"], None]] = None
    on_connected: Optional[Callable[[], None]] = None
    on_disconnect: Optional[Callable[[Optional[Exception]], None]] = None

    def start(self) -> Tuple["StateChannel", Thread]:
        from napari_cuda.client.control.control_channel_client import StateChannel

        ch = StateChannel(
            self.host,
            int(self.port),
            ingest_notify_stream=self.ingest_notify_stream,
            ingest_dims_notify=self.ingest_dims_notify,
            ingest_notify_scene_snapshot=self.ingest_notify_scene_snapshot,
            ingest_notify_layers=self.ingest_notify_layers,
            ingest_notify_camera=self.ingest_notify_camera,
            ingest_ack_state=self.ingest_ack_state,
            ingest_reply_command=self.ingest_reply_command,
            ingest_error_command=self.ingest_error_command,
            on_session_ready=self.on_session_ready,
            on_connected=self.on_connected,
            on_disconnect=self.on_disconnect,
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
