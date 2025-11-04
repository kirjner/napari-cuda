"""Protocol I/O helpers for the control channel.

These functions are thin wrappers around websocket send operations and frame
serialization. Keeping them here avoids repeating boilerplate and makes it
easy to mock in tests.
"""

from __future__ import annotations

import json
from typing import Any

from napari_cuda.server.utils.websocket import safe_send


async def send_text(ws: Any, text: str) -> bool:
    """Send raw text to the websocket, returning True on success."""

    return await safe_send(ws, text)


async def send_frame(server: Any, ws: Any, frame: Any) -> bool:
    """Serialize a frame via ``to_dict()`` and send it on the state channel."""

    payload = frame.to_dict()  # type: ignore[attr-defined]
    text = json.dumps(payload, separators=(",", ":"))
    return await send_text(ws, text)


__all__ = ["send_text", "send_frame"]
