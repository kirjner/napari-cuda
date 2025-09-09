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
    "orchestrator",
]

# Friendly re-export for cleaner imports
from .orchestrator import StreamManager  # noqa: E402,F401
