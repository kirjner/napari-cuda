"""Protocol I/O helpers for the control channel.

These functions are thin wrappers around websocket send operations and frame
serialization. Keeping them here avoids repeating boilerplate and makes it
easy to mock in tests.
"""

from __future__ import annotations

import json
from typing import Any


async def await_state_send(server: Any, ws: Any, text: str) -> None:
    """Send serialized text on the state channel."""
    await ws.send(text)


async def send_frame(server: Any, ws: Any, frame: Any) -> None:
    """Serialize a frame via ``to_dict()`` and send it on the state channel."""

    payload = frame.to_dict()  # type: ignore[attr-defined]
    text = json.dumps(payload, separators=(",", ":"))
    await await_state_send(server, ws, text)


__all__ = ["await_state_send", "send_frame"]
