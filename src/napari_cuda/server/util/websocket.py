"""Shared helpers for resilient WebSocket sends."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def safe_send(ws: Any, data: Any) -> bool:
    """Send ``data`` on ``ws`` and close on failure.

    Returns True when the send succeeds, False when the send fails (the helper
    logs the failure, attempts to close the socket, and swallows exceptions).
    """

    try:
        await ws.send(data)
        return True
    except Exception:
        logger.debug("WebSocket send failed", exc_info=True)
        try:
            await ws.close()
        except Exception:
            logger.debug("WebSocket close failed after send failure", exc_info=True)
        return False


__all__ = ["safe_send"]
