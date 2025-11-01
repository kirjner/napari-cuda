"""Small, pure helpers for control-channel protocol runtime state.

These helpers centralize access to websocket/session features, resumable
sequencers, and the optional resumable history store. They deliberately avoid
carrying behavior beyond simple lookups and object creation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from napari_cuda.protocol import FeatureToggle
from napari_cuda.protocol.envelopes import ResumableTopicSequencer
from napari_cuda.server.control.resumable_history_store import (
    ResumableHistoryStore,
)


def state_session(ws: Any) -> Optional[str]:
    """Return the session id bound to a websocket."""
    return ws._napari_cuda_session  # type: ignore[attr-defined]


def history_store(server: Any) -> Optional[ResumableHistoryStore]:
    """Return the configured resumable history store, if any."""
    return server._resumable_store  # type: ignore[attr-defined]


def _state_features(ws: Any) -> Mapping[str, FeatureToggle]:
    return ws._napari_cuda_features  # type: ignore[attr-defined]


def feature_enabled(ws: Any, name: str) -> bool:
    """Return True if a feature toggle is enabled for this websocket."""
    return bool(_state_features(ws)[name].enabled)


def state_sequencer(ws: Any, topic: str) -> ResumableTopicSequencer:
    """Fetch or create a per-topic sequencer for resumable notify lanes."""
    sequencers = ws._napari_cuda_sequencers  # type: ignore[attr-defined]
    if topic not in sequencers:
        sequencers[topic] = ResumableTopicSequencer(topic=topic)
    return sequencers[topic]


__all__ = [
    "feature_enabled",
    "history_store",
    "state_sequencer",
    "state_session",
]
