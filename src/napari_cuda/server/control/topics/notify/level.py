from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Sequence
from typing import Any, Optional

from napari_cuda.protocol import NOTIFY_LEVEL_TYPE
from napari_cuda.protocol.envelopes import build_notify_level
from napari_cuda.protocol.messages import NotifyLevelPayload
from napari_cuda.server.control.protocol.io import send_frame
from napari_cuda.server.control.protocol.runtime import (
    feature_enabled,
    history_store,
    state_sequencer,
    state_session,
)
from napari_cuda.server.control.resumable_history_store import EnvelopeSnapshot

logger = logging.getLogger(__name__)


async def _deliver_level(
    server: Any,
    *,
    payload: NotifyLevelPayload,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    targets: Optional[Sequence[Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(getattr(server, "_state_clients", []))

    # Always record in history if available, even when no clients yet.
    store = history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    now = time.time() if timestamp is None else float(timestamp)

    if store is not None and targets is None:
        # Record as delta; store promotes to snapshot automatically on first write.
        snapshot = store.delta_envelope(
            NOTIFY_LEVEL_TYPE,
            payload=payload.to_dict(),
            timestamp=now,
            intent_id=intent_id,
        )

    if not clients:
        return

    tasks: list[Awaitable[None]] = []
    for ws in clients:
        if not feature_enabled(ws, NOTIFY_LEVEL_TYPE):
            continue
        session_id = state_session(ws)
        if not session_id:
            continue

        kwargs: dict[str, Any] = {
            "session_id": session_id,
            "payload": payload,
            "timestamp": snapshot.timestamp if snapshot is not None else now,
            "intent_id": intent_id,
        }
        if snapshot is not None:
            # When using a store snapshot/delta, carry explicit cursor and avoid sequencer.
            kwargs["seq"] = snapshot.seq
            kwargs["delta_token"] = snapshot.delta_token
            kwargs["frame_id"] = snapshot.frame_id
        else:
            # No store; use per-client sequencer. This will raise if we never snapshotted,
            # so ensure store path handled the first write when retention is configured.
            kwargs["sequencer"] = state_sequencer(ws, NOTIFY_LEVEL_TYPE)

        frame = build_notify_level(**kwargs)
        tasks.append(send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("notify.level broadcast count=%d", len(tasks))

    if snapshot is not None:
        # Bring per-client sequencers in sync with the stored cursor for future deltas.
        for ws in clients:
            sequencer = state_sequencer(ws, NOTIFY_LEVEL_TYPE)
            sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def broadcast_level(
    server: Any,
    *,
    payload: NotifyLevelPayload,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    await _deliver_level(
        server,
        payload=payload,
        intent_id=intent_id,
        timestamp=timestamp,
        targets=None,
    )


async def send_level(
    server: Any,
    ws: Any,
    *,
    payload: NotifyLevelPayload,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    await _deliver_level(
        server,
        payload=payload,
        intent_id=intent_id,
        timestamp=timestamp,
        targets=[ws],
    )


__all__ = ["broadcast_level", "send_level", "send_level_snapshot"]


async def send_level_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    """Replay a stored notify.level frame to a single websocket and sync sequencer."""

    session_id = state_session(ws)
    if not session_id:
        return

    payload = NotifyLevelPayload.from_dict(snapshot.payload)
    frame = build_notify_level(
        session_id=session_id,
        payload=payload,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        intent_id=snapshot.intent_id,
        seq=snapshot.seq,
        delta_token=snapshot.delta_token,
    )
    await send_frame(server, ws, frame)
    sequencer = state_sequencer(ws, NOTIFY_LEVEL_TYPE)
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
