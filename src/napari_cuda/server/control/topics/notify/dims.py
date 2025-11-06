from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Sequence
from typing import Any, Optional

from napari_cuda.protocol import build_notify_dims
from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.server.control.protocol.io import send_frame
from napari_cuda.server.control.protocol.runtime import (
    feature_enabled,
    state_session,
)
from napari_cuda.shared.dims_spec import (
    dims_spec_axis_labels,
    dims_spec_displayed,
    dims_spec_order,
)

logger = logging.getLogger(__name__)


async def _deliver_dims_state(
    server: Any,
    *,
    payload: NotifyDimsPayload,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    targets: Optional[Sequence[Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    if getattr(server, "_log_dims_info", False):
        spec = payload.dims_spec
        logger.info(
            "notify.dims: step=%s order=%s displayed=%s axis_labels=%s level_shapes=%s current_level=%s",
            payload.current_step,
            dims_spec_order(spec),
            dims_spec_displayed(spec),
            dims_spec_axis_labels(spec),
            payload.level_shapes,
            payload.current_level,
        )

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not feature_enabled(ws, "notify.dims"):
            continue
        session_id = state_session(ws)
        if not session_id:
            continue
        frame = build_notify_dims(
            session_id=session_id,
            payload=payload,
            timestamp=now,
            intent_id=intent_id,
        )
        tasks.append(send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def broadcast_dims_state(
    server: Any,
    *,
    payload: NotifyDimsPayload,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    await _deliver_dims_state(
        server,
        payload=payload,
        intent_id=intent_id,
        timestamp=timestamp,
        targets=None,
    )


async def send_dims_state(
    server: Any,
    ws: Any,
    *,
    payload: NotifyDimsPayload,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    await _deliver_dims_state(
        server,
        payload=payload,
        intent_id=intent_id,
        timestamp=timestamp,
        targets=[ws],
    )


__all__ = ["broadcast_dims_state", "send_dims_state"]
