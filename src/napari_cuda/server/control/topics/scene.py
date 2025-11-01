from __future__ import annotations

from typing import Any

from napari_cuda.protocol import NOTIFY_SCENE_TYPE
from napari_cuda.protocol.messages import NotifyScenePayload
from napari_cuda.protocol.envelopes import build_notify_scene_snapshot
from napari_cuda.server.control.protocol_runtime import (
    state_session,
    state_sequencer,
)
from napari_cuda.server.control.protocol_io import send_frame
from napari_cuda.server.control.resumable_history_store import EnvelopeSnapshot


async def send_scene_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = state_session(ws)
    if not session_id:
        return
    payload = NotifyScenePayload.from_dict(snapshot.payload)
    sequencer = state_sequencer(ws, NOTIFY_SCENE_TYPE)
    # Resume first, so emitted frame carries the resumed seq/delta state
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    frame = build_notify_scene_snapshot(
        session_id=session_id,
        viewer=payload.viewer,
        layers=payload.layers,
        policies=payload.policies,
        metadata=payload.metadata,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        delta_token=snapshot.delta_token,
        intent_id=snapshot.intent_id,
        sequencer=sequencer,
    )
    await send_frame(server, ws, frame)


__all__ = [
    "send_scene_snapshot",
]
