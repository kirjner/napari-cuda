"""Rendering/encoding engine components for napari-cuda.

This module provides a lazy façade that re-exports the engine packages'
public surface. Importing anything from here will only load the backing
module when the attribute is first accessed, avoiding heavy GL/EGL work
before the worker thread has established its context.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

__all__ = [
    # Capture façade
    "CaptureFacade",
    "FrameTimings",
    "FrameCapture",
    "EncodedFrame",
    "capture_frame_for_encoder",
    "encode_frame",
    "DebugConfig",
    "DebugDumper",
    "EglContext",
    # Encoding helpers
    "Encoder",
    "EncoderTimings",
    "ParamCache",
    "configure_bitstream",
    "parse_nals",
    "pack_to_annexb",
    "pack_to_avcc",
    "build_avcc_config",
    # Pixel broadcaster / channel
    "PixelBroadcastConfig",
    "PixelBroadcastState",
    "FramePacket",
    "broadcast_loop",
    "configure_socket",
    "safe_send",
    "PixelChannelConfig",
    "PixelChannelState",
    "build_notify_stream_payload",
    "prepare_client_attach",
    "ingest_client",
    "ensure_keyframe",
    "start_watchdog",
    "mark_stream_config_dirty",
    "maybe_send_stream_config",
    "publish_avcc",
    "enqueue_frame",
    "run_channel_loop",
]

_EXPORT_MAP: Dict[str, Tuple[str, str | None]] = {
    # Capture façade
    "CaptureFacade": ("napari_cuda.server.engine.capture.capture", "CaptureFacade"),
    "FrameTimings": ("napari_cuda.server.engine.capture.capture", "FrameTimings"),
    "FrameCapture": ("napari_cuda.server.engine.capture.capture", "FrameCapture"),
    "EncodedFrame": ("napari_cuda.server.engine.capture.capture", "EncodedFrame"),
    "capture_frame_for_encoder": (
        "napari_cuda.server.engine.capture.capture",
        "capture_frame_for_encoder",
    ),
    "encode_frame": ("napari_cuda.server.engine.capture.capture", "encode_frame"),
    "DebugConfig": ("napari_cuda.server.engine.capture.debug", "DebugConfig"),
    "DebugDumper": ("napari_cuda.server.engine.capture.debug", "DebugDumper"),
    "EglContext": ("napari_cuda.server.engine.capture.egl_context", "EglContext"),
    # Encoding helpers
    "Encoder": ("napari_cuda.server.engine.encoding.encoder", "Encoder"),
    "EncoderTimings": ("napari_cuda.server.engine.encoding.encoder", "EncoderTimings"),
    "ParamCache": ("napari_cuda.server.engine.encoding.bitstream", "ParamCache"),
    "configure_bitstream": (
        "napari_cuda.server.engine.encoding.bitstream",
        "configure_bitstream",
    ),
    "parse_nals": ("napari_cuda.server.engine.encoding.bitstream", "parse_nals"),
    "pack_to_annexb": ("napari_cuda.server.engine.encoding.bitstream", "pack_to_annexb"),
    "pack_to_avcc": ("napari_cuda.server.engine.encoding.bitstream", "pack_to_avcc"),
    "build_avcc_config": (
        "napari_cuda.server.engine.encoding.bitstream",
        "build_avcc_config",
    ),
    # Pixel broadcaster / channel
    "PixelBroadcastConfig": (
        "napari_cuda.server.engine.pixel.broadcaster",
        "PixelBroadcastConfig",
    ),
    "PixelBroadcastState": (
        "napari_cuda.server.engine.pixel.broadcaster",
        "PixelBroadcastState",
    ),
    "FramePacket": ("napari_cuda.server.engine.pixel.broadcaster", "FramePacket"),
    "broadcast_loop": (
        "napari_cuda.server.engine.pixel.broadcaster",
        "broadcast_loop",
    ),
    "configure_socket": (
        "napari_cuda.server.engine.pixel.broadcaster",
        "configure_socket",
    ),
    "safe_send": ("napari_cuda.server.engine.pixel.broadcaster", "safe_send"),
    "PixelChannelConfig": (
        "napari_cuda.server.engine.pixel.channel",
        "PixelChannelConfig",
    ),
    "PixelChannelState": (
        "napari_cuda.server.engine.pixel.channel",
        "PixelChannelState",
    ),
    "build_notify_stream_payload": (
        "napari_cuda.server.engine.pixel.channel",
        "build_notify_stream_payload",
    ),
    "prepare_client_attach": (
        "napari_cuda.server.engine.pixel.channel",
        "prepare_client_attach",
    ),
    "ingest_client": ("napari_cuda.server.engine.pixel.channel", "ingest_client"),
    "ensure_keyframe": ("napari_cuda.server.engine.pixel.channel", "ensure_keyframe"),
    "start_watchdog": ("napari_cuda.server.engine.pixel.channel", "start_watchdog"),
    "mark_stream_config_dirty": (
        "napari_cuda.server.engine.pixel.channel",
        "mark_stream_config_dirty",
    ),
    "maybe_send_stream_config": (
        "napari_cuda.server.engine.pixel.channel",
        "maybe_send_stream_config",
    ),
    "publish_avcc": ("napari_cuda.server.engine.pixel.channel", "publish_avcc"),
    "enqueue_frame": ("napari_cuda.server.engine.pixel.channel", "enqueue_frame"),
    "run_channel_loop": ("napari_cuda.server.engine.pixel.channel", "run_channel_loop"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORT_MAP[name]
    except KeyError as exc:  # pragma: no cover - standard attribute error path
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - trivial helper
    return sorted(list(globals().keys()) + list(__all__))


if TYPE_CHECKING:  # pragma: no cover - assist static type checkers
    from napari_cuda.server.engine.capture.capture import (
        CaptureFacade,
        EncodedFrame,
        FrameCapture,
        FrameTimings,
        capture_frame_for_encoder,
        encode_frame,
    )
    from napari_cuda.server.engine.capture.debug import DebugConfig, DebugDumper
    from napari_cuda.server.engine.capture.egl_context import EglContext
    from napari_cuda.server.engine.encoding.bitstream import (
        ParamCache,
        build_avcc_config,
        configure_bitstream,
        pack_to_annexb,
        pack_to_avcc,
        parse_nals,
    )
    from napari_cuda.server.engine.encoding.encoder import Encoder, EncoderTimings
    from napari_cuda.server.engine.pixel.broadcaster import (
        FramePacket,
        PixelBroadcastConfig,
        PixelBroadcastState,
        broadcast_loop,
        configure_socket,
        safe_send,
    )
    from napari_cuda.server.engine.pixel.channel import (
        PixelChannelConfig,
        PixelChannelState,
        build_notify_stream_payload,
        enqueue_frame,
        ensure_keyframe,
        ingest_client,
        mark_stream_config_dirty,
        maybe_send_stream_config,
        prepare_client_attach,
        publish_avcc,
        run_channel_loop,
        start_watchdog,
    )
