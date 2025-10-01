"""Compatibility shim forwarding to legacy protocol messages with greenfield additions."""

from .legacy.messages import *  # noqa: F401,F403
from .greenfield.messages import (  # noqa: F401
    HelloAuthInfo,
    HelloClientInfo,
    NotifyCameraPayload,
    NotifyDimsFrame,
    NotifyDimsPayload,
    NotifyErrorPayload,
    NotifyLayersFrame,
    NotifyLayersPayload,
    NotifySceneFrame,
    NotifySceneLevelFrame,
    NotifySceneLevelPayload,
    NotifyScenePayload,
    NotifyStreamFrame,
    NotifyStreamPayload,
    NotifyTelemetryPayload,
    WelcomeSessionInfo,
)

__all__ = [
    name
    for name in globals().keys()
    if not name.startswith("_")
]
