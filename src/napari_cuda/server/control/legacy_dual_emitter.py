"""Helpers for dual-emitting greenfield protocol envelopes from the server."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

from napari_cuda.protocol.envelopes import (
    NotifyScene,
    NotifyScenePayload,
    NotifyState,
    NotifyStream,
    NotifyStreamPayload,
)
from napari_cuda.protocol.messages import (
    SCENE_SPEC_TYPE,
    STATE_UPDATE_TYPE,
    SceneSpecMessage,
    StateUpdateMessage,
)

logger = logging.getLogger(__name__)


def encode_envelope(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return a greenfield envelope dict for *payload*, or ``None`` if unknown."""

    msg_type = str(payload.get("type") or "").lower()

    if msg_type == STATE_UPDATE_TYPE:
        try:
            message = StateUpdateMessage.from_dict(dict(payload))
        except Exception:
            logger.debug("Failed to coerce state.update payload for dual emission", exc_info=True)
            return None
        envelope = NotifyState(
            payload=message,
            id=str(payload.get("id")) if payload.get("id") is not None else None,
            timestamp=_coerce_float(payload.get("timestamp")),
        )
        return envelope.to_dict()

    if msg_type == SCENE_SPEC_TYPE:
        try:
            message = SceneSpecMessage.from_dict(dict(payload))
        except Exception:
            logger.debug("Failed to coerce scene.spec payload for dual emission", exc_info=True)
            return None
        state_block: Optional[dict[str, Any]] = None
        if message.capabilities:
            state_block = {"capabilities": list(message.capabilities)}
        envelope = NotifyScene(
            payload=NotifyScenePayload(
                version=message.version,
                scene=message.scene,
                state=state_block,
            ),
            id=str(payload.get("id")) if payload.get("id") is not None else None,
            timestamp=_coerce_float(payload.get("timestamp")),
        )
        return envelope.to_dict()

    if msg_type == "video_config":
        codec = payload.get("codec")
        if not codec:
            logger.debug("Skipping video_config dual emission; codec missing")
            return None
        extras: dict[str, Any] = {}
        for key in ("format", "data"):
            value = payload.get(key)
            if value is not None:
                extras[key] = value
        notify_payload = NotifyStreamPayload(
            codec=str(codec),
            fps=_coerce_float(payload.get("fps")),
            width=_coerce_int(payload.get("width")),
            height=_coerce_int(payload.get("height")),
            bitrate=_coerce_int(payload.get("bitrate")),
            idr_interval=_coerce_int(payload.get("idr_interval")),
            extras=extras or None,
        )
        envelope = NotifyStream(
            payload=notify_payload,
            id=str(payload.get("id")) if payload.get("id") is not None else None,
            timestamp=_coerce_float(payload.get("timestamp")),
        )
        return envelope.to_dict()

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


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


__all__ = ["encode_envelope", "encode_envelope_json"]

