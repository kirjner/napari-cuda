from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class Source(str, Enum):
    PYAV = "pyav"
    VT = "vt"


"""Timestamping is simplified to server timestamps only.

Historically there was a TimestampMode with ARRIVAL and SERVER. The client
now always uses server timestamps when available, and falls back to arrival
time only for missing/invalid server timestamps.
"""


@dataclass
class SubmittedFrame:
    """A frame submitted to the presenter.

    - `server_ts` is the server's timestamp in seconds if available.
    - `arrival_ts` is the local monotonic-ish wall time in seconds at receipt.
    - `payload` is the frame data (e.g., VT pixel buffer handle or RGB ndarray).
    - `release_cb` is an optional callable to release resources when the
      presenter drops a frame (e.g., VT retain/release handling).
    """

    source: Source
    server_ts: Optional[float]
    arrival_ts: float
    payload: Any
    release_cb: Optional[Callable[[Any], None]] = None


@dataclass
class ReadyFrame:
    """A frame ready to present.

    If `preview` is True, the frame is returned for display to avoid a visible
    stall but is not consumed from the presenter's buffer. Callers must not
    invoke `release_cb` for preview frames.
    """

    source: Source
    due_ts: float
    payload: Any
    release_cb: Optional[Callable[[Any], None]] = None
    preview: bool = False
