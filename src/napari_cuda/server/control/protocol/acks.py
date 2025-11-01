"""Control-channel acknowledgement helpers."""

from __future__ import annotations

import time
from numbers import Integral
from typing import Any, Mapping, Optional

from napari_cuda.protocol import (
    build_ack_state,
    build_session_goodbye,
)
from napari_cuda.server.control.protocol.io import send_frame


async def send_state_ack(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    intent_id: str,
    in_reply_to: str,
    status: str,
    applied_value: Any | None = None,
    error: Mapping[str, Any] | Any | None = None,
    version: int | None = None,
) -> None:
    """Build and emit an ``ack.state`` frame mirroring the incoming update."""

    if not intent_id or not in_reply_to:
        raise ValueError("ack.state requires intent_id and in_reply_to identifiers")

    normalized_status = str(status).lower()
    if normalized_status not in {"accepted", "rejected"}:
        raise ValueError("ack.state status must be 'accepted' or 'rejected'")

    payload: dict[str, Any] = {
        "intent_id": str(intent_id),
        "in_reply_to": str(in_reply_to),
        "status": normalized_status,
    }

    if normalized_status == "accepted":
        if error is not None:
            raise ValueError("accepted ack.state payload cannot include error details")
        if version is None:
            raise ValueError("accepted ack.state payload requires version")
        if not isinstance(version, Integral):
            raise ValueError("ack.state version must be integer")
        if applied_value is not None:
            payload["applied_value"] = applied_value
        payload["version"] = int(version)
    else:
        if not isinstance(error, Mapping):
            raise ValueError("rejected ack.state payload requires {code, message}")
        if "code" not in error or "message" not in error:
            raise ValueError("ack.state error payload must include 'code' and 'message'")
        payload["error"] = dict(error)
        if version is not None:
            if not isinstance(version, Integral):
                raise ValueError("ack.state version must be integer")
            payload["version"] = int(version)

    frame = build_ack_state(
        session_id=str(session_id),
        frame_id=None,
        payload=payload,
        timestamp=time.time(),
    )

    await send_frame(server, ws, frame)


async def send_session_goodbye(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    code: str,
    message: str,
    reason: Optional[str] = None,
) -> None:
    """Send ``session.goodbye`` once per websocket session."""

    if ws._napari_cuda_goodbye_sent:
        return
    frame = build_session_goodbye(
        session_id=session_id,
        code=code,
        message=message,
        reason=reason,
        timestamp=time.time(),
    )
    await send_frame(server, ws, frame)
    ws._napari_cuda_goodbye_sent = True


__all__ = ["send_state_ack", "send_session_goodbye"]
