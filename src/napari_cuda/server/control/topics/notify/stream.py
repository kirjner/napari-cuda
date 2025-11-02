from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Optional

from napari_cuda.protocol import NOTIFY_STREAM_TYPE
from napari_cuda.protocol.envelopes import build_notify_stream
from napari_cuda.protocol.messages import NotifyStreamPayload
from napari_cuda.server.control.protocol.io import send_frame
from napari_cuda.server.control.protocol.runtime import (
    feature_enabled,
    history_store,
    state_sequencer,
    state_session,
)
from napari_cuda.server.control.resumable_history_store import EnvelopeSnapshot


async def _deliver_stream_frame(
    server: Any,
    *,
    payload: NotifyStreamPayload,
    timestamp: Optional[float],
    targets: Optional[list[Any]] = None,
    force_snapshot: bool = False,
) -> None:
    clients = targets if targets is not None else list(server._state_clients)
    if not clients:
        store = history_store(server)
        if store is not None:
            now = __import__("time").time() if timestamp is None else float(timestamp)
            payload_dict = payload.to_dict()
            if store.current_snapshot(NOTIFY_STREAM_TYPE) is None:
                store.snapshot_envelope(NOTIFY_STREAM_TYPE, payload=payload_dict, timestamp=now)
            else:
                store.delta_envelope(NOTIFY_STREAM_TYPE, payload=payload_dict, timestamp=now)
        return

    now = __import__("time").time() if timestamp is None else float(timestamp)
    store = history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    snapshot_mode = force_snapshot
    if store is not None:
        payload_dict = payload.to_dict()
        if snapshot_mode or store.current_snapshot(NOTIFY_STREAM_TYPE) is None:
            snapshot = store.snapshot_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )
            snapshot_mode = True
        else:
            snapshot = store.delta_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )

    tasks: list[Awaitable[None]] = []
    for ws in clients:
        if not feature_enabled(ws, "notify.stream"):
            continue
        session_id = state_session(ws)
        if not session_id:
            continue
        kwargs: dict[str, Any] = {
            "session_id": session_id,
            "payload": payload,
            "timestamp": snapshot.timestamp if snapshot is not None else now,
        }
        seq = state_sequencer(ws, NOTIFY_STREAM_TYPE)
        if snapshot is not None:
            kwargs["seq"] = snapshot.seq
            kwargs["delta_token"] = snapshot.delta_token
            kwargs["frame_id"] = snapshot.frame_id
            seq.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
        else:
            if snapshot_mode or seq.seq is None:
                cursor = seq.snapshot()
                kwargs["seq"] = cursor.seq
                kwargs["delta_token"] = cursor.delta_token
            else:
                kwargs["sequencer"] = seq
        frame = build_notify_stream(**kwargs)
        tasks.append(send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def send_stream_frame(
    server: Any,
    ws: Any,
    *,
    payload: NotifyStreamPayload | Mapping[str, Any],
    timestamp: Optional[float],
    snapshot: EnvelopeSnapshot | None = None,
    force_snapshot: bool = False,
) -> EnvelopeSnapshot | None:
    if not isinstance(payload, NotifyStreamPayload):
        payload = NotifyStreamPayload.from_dict(payload)
    await _deliver_stream_frame(
        server,
        payload=payload,
        timestamp=timestamp,
        targets=[ws],
        force_snapshot=force_snapshot,
    )
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
    await _deliver_stream_frame(server, payload=payload, timestamp=timestamp, targets=None)


async def send_stream_payload(
    server: Any,
    ws: Any,
    *,
    payload: NotifyStreamPayload,
    timestamp: Optional[float] = None,
) -> None:
    await _deliver_stream_frame(server, payload=payload, timestamp=timestamp, targets=[ws])


__all__ = [
    "broadcast_stream_config",
    "broadcast_stream_config",
    "send_stream_frame",
    "send_stream_payload",
    "send_stream_snapshot",
]
