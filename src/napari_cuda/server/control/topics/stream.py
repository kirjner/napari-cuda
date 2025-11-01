from __future__ import annotations

import asyncio
from typing import Any, Mapping, Optional, Sequence

from napari_cuda.protocol import NOTIFY_STREAM_TYPE
from napari_cuda.protocol.messages import NotifyStreamPayload
from napari_cuda.protocol.envelopes import build_notify_stream
from napari_cuda.server.control.protocol_runtime import (
    feature_enabled,
    history_store,
    state_sequencer,
    state_session,
)
from napari_cuda.server.control.protocol_io import send_frame
from napari_cuda.server.control.resumable_history_store import EnvelopeSnapshot


async def send_stream_frame(
    server: Any,
    ws: Any,
    *,
    payload: NotifyStreamPayload | Mapping[str, Any],
    timestamp: Optional[float],
    snapshot: EnvelopeSnapshot | None = None,
    force_snapshot: bool = False,
) -> EnvelopeSnapshot | None:
    session_id = state_session(ws)
    if not session_id:
        return snapshot
    if not isinstance(payload, NotifyStreamPayload):
        payload = NotifyStreamPayload.from_dict(payload)
    now = __import__("time").time() if timestamp is None else float(timestamp)
    store = history_store(server)
    if snapshot is None and store is not None:
        payload_dict = payload.to_dict()
        if force_snapshot:
            snapshot = store.snapshot_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )
        else:
            snapshot = store.delta_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )

    kwargs: dict[str, Any] = {
        "session_id": session_id,
        "payload": payload,
        "timestamp": snapshot.timestamp if snapshot is not None else now,
    }
    sequencer = state_sequencer(ws, NOTIFY_STREAM_TYPE)
    if snapshot is not None:
        kwargs["seq"] = snapshot.seq
        kwargs["delta_token"] = snapshot.delta_token
        kwargs["frame_id"] = snapshot.frame_id
        sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    else:
        if force_snapshot or sequencer.seq is None:
            cursor = sequencer.snapshot()
            kwargs["seq"] = cursor.seq
            kwargs["delta_token"] = cursor.delta_token
        else:
            kwargs["sequencer"] = sequencer
    frame = build_notify_stream(**kwargs)
    await send_frame(server, ws, frame)
    return snapshot


async def send_stream_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = state_session(ws)
    if not session_id:
        return
    payload = NotifyStreamPayload.from_dict(snapshot.payload)
    frame = build_notify_stream(
        session_id=session_id,
        payload=payload,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        seq=snapshot.seq,
        delta_token=snapshot.delta_token,
    )
    await send_frame(server, ws, frame)
    state_sequencer(ws, NOTIFY_STREAM_TYPE).resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def broadcast_stream_config(
    server: Any,
    *,
    payload: NotifyStreamPayload,
    timestamp: Optional[float] = None,
) -> None:
    clients = list(server._state_clients)
    if not clients:
        store = history_store(server)
        if store is not None:
            now = __import__("time").time() if timestamp is None else float(timestamp)
            if store.current_snapshot(NOTIFY_STREAM_TYPE) is None:
                store.snapshot_envelope(
                    NOTIFY_STREAM_TYPE,
                    payload=payload.to_dict(),
                    timestamp=now,
                )
            else:
                store.delta_envelope(
                    NOTIFY_STREAM_TYPE,
                    payload=payload.to_dict(),
                    timestamp=now,
                )
        return

    now = __import__("time").time() if timestamp is None else float(timestamp)
    store = history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    snapshot_mode = False
    if store is not None:
        payload_dict = payload.to_dict()
        snapshot_mode = store.current_snapshot(NOTIFY_STREAM_TYPE) is None
        if snapshot_mode:
            snapshot = store.snapshot_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )
        else:
            snapshot = store.delta_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )

    tasks = []
    for ws in clients:
        if not feature_enabled(ws, "notify.stream"):
            continue
        if not state_session(ws):
            continue
        tasks.append(
            send_stream_frame(
                server,
                ws,
                payload=payload,
                timestamp=now,
                snapshot=snapshot,
                force_snapshot=snapshot_mode,
            )
        )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "send_stream_frame",
    "send_stream_snapshot",
    "broadcast_stream_config",
]

