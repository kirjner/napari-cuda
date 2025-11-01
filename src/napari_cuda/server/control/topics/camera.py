from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Optional

from napari_cuda.protocol import build_notify_camera
from napari_cuda.server.control.protocol_io import send_frame
from napari_cuda.server.control.protocol_runtime import (
    feature_enabled,
    state_session,
)


async def broadcast_camera_update(
    server: Any,
    *,
    mode: str,
    delta: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    intent_id: Optional[str],
    origin: str,
    timestamp: Optional[float] = None,
    targets: Optional[Sequence[Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    payload: dict[str, Any] = {
        "mode": str(mode),
        "origin": str(origin),
    }
    if delta is not None:
        payload["delta"] = _normalize_camera_value(delta)
    if state is not None:
        payload["state"] = _normalize_camera_value(state)
    if "delta" not in payload and "state" not in payload:
        return

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not feature_enabled(ws, "notify.camera"):
            continue
        session_id = state_session(ws)
        if not session_id:
            continue
        frame = build_notify_camera(
            session_id=session_id,
            payload=payload,
            timestamp=now,
            intent_id=intent_id,
        )
        tasks.append(send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _normalize_camera_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_camera_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_camera_value(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


__all__ = ["broadcast_camera_update"]

