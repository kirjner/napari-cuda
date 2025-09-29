"""Client streaming components (presenter, types, mux).

Step 1 of the refactor extracts a fixed‑latency presenter and source
mux along with common types. This package will grow to include
receiver/state/decoders/renderer in subsequent steps.
"""

from __future__ import annotations

__all__ = [
    "types",
    "presenter",
    "receiver",
    "state",
    "decoders",
    "renderer",
    "client_stream_loop",
    "ClientStreamLoop",
]

from .client_stream_loop import ClientStreamLoop  # noqa: E402,F401
