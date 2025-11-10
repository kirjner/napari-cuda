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
    state_sequencer,
    state_session,
)

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
    if not clients:
        return

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not feature_enabled(ws, NOTIFY_LEVEL_TYPE):
            continue
        session_id = state_session(ws)
        if not session_id:
            continue
        frame = build_notify_level(
            session_id=session_id,
            payload=payload,
            timestamp=now,
            intent_id=intent_id,
            sequencer=state_sequencer(ws, NOTIFY_LEVEL_TYPE),
        )
        tasks.append(send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("notify.level broadcast count=%d", len(tasks))


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


__all__ = ["broadcast_level", "send_level"]
