"""Helpers for dual-emitting greenfield protocol envelopes from the server."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

from napari_cuda.protocol.messages import SCENE_SPEC_TYPE, STATE_UPDATE_TYPE

logger = logging.getLogger(__name__)


def encode_envelope(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return a greenfield envelope dict for *payload*, or ``None`` if unknown."""

    msg_type = str(payload.get("type") or "").lower()

    if msg_type in {STATE_UPDATE_TYPE, SCENE_SPEC_TYPE, "video_config"}:
        logger.debug(
            "Dual emission for %s pending greenfield wiring; legacy payload only", msg_type,
        )
        return None

    return None


def encode_envelope_json(payload: Mapping[str, Any]) -> Optional[str]:
    """Return a JSON string for the envelope built from *payload* if supported."""

    envelope = encode_envelope(payload)
    if envelope is None:
        return None
    try:
        return json.dumps(envelope, separators=(",", ":"))
    except Exception:
        logger.debug("Envelope JSON encode failed; skipping dual emission", exc_info=True)
        return None


__all__ = ["encode_envelope", "encode_envelope_json"]
