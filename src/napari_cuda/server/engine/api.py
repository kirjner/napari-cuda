"""Curated lazy façade for engine consumers (runtime/control/app)."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

__all__ = [
    # Capture / encoding interfaces
    "CaptureFacade",
    "DebugConfig",
    "DebugDumper",
    "EglContext",
    "Encoder",
    "FrameTimings",
    "ParamCache",
    "build_avcc_config",
    "configure_bitstream",
    "encode_frame",
    "pack_to_avcc",
    # Pixel broadcast/channel surfaces
    "PixelBroadcastConfig",
    "PixelBroadcastState",
    "PixelChannelConfig",
    "PixelChannelState",
    "broadcast_loop",
    "build_notify_stream_payload",
    "enqueue_frame",
    "ensure_keyframe",
    "ingest_client",
    "mark_stream_config_dirty",
    "prepare_client_attach",
    "publish_avcc",
    "run_channel_loop",
]

_EXPORT_MAP: Dict[str, Tuple[str, str | None]] = {
    # Capture / encoding
    "CaptureFacade": ("napari_cuda.server.engine.capture.capture", "CaptureFacade"),
    "FrameTimings": ("napari_cuda.server.engine.capture.capture", "FrameTimings"),
    "encode_frame": ("napari_cuda.server.engine.capture.capture", "encode_frame"),
    "DebugConfig": ("napari_cuda.server.engine.capture.debug", "DebugConfig"),
    "DebugDumper": ("napari_cuda.server.engine.capture.debug", "DebugDumper"),
    "EglContext": ("napari_cuda.server.engine.capture.egl_context", "EglContext"),
    "Encoder": ("napari_cuda.server.engine.encoding.encoder", "Encoder"),
    "ParamCache": ("napari_cuda.server.engine.encoding.bitstream", "ParamCache"),
    "build_avcc_config": ("napari_cuda.server.engine.encoding.bitstream", "build_avcc_config"),
    "configure_bitstream": ("napari_cuda.server.engine.encoding.bitstream", "configure_bitstream"),
    "pack_to_avcc": ("napari_cuda.server.engine.encoding.bitstream", "pack_to_avcc"),
    # Pixel broadcast/channel
    "PixelBroadcastConfig": ("napari_cuda.server.engine.pixel.broadcaster", "PixelBroadcastConfig"),
    "PixelBroadcastState": ("napari_cuda.server.engine.pixel.broadcaster", "PixelBroadcastState"),
    "broadcast_loop": ("napari_cuda.server.engine.pixel.broadcaster", "broadcast_loop"),
    "PixelChannelConfig": ("napari_cuda.server.engine.pixel.channel", "PixelChannelConfig"),
    "PixelChannelState": ("napari_cuda.server.engine.pixel.channel", "PixelChannelState"),
    "build_notify_stream_payload": ("napari_cuda.server.engine.pixel.channel", "build_notify_stream_payload"),
    "enqueue_frame": ("napari_cuda.server.engine.pixel.channel", "enqueue_frame"),
    "ensure_keyframe": ("napari_cuda.server.engine.pixel.channel", "ensure_keyframe"),
    "ingest_client": ("napari_cuda.server.engine.pixel.channel", "ingest_client"),
    "mark_stream_config_dirty": ("napari_cuda.server.engine.pixel.channel", "mark_stream_config_dirty"),
    "prepare_client_attach": ("napari_cuda.server.engine.pixel.channel", "prepare_client_attach"),
    "publish_avcc": ("napari_cuda.server.engine.pixel.channel", "publish_avcc"),
    "run_channel_loop": ("napari_cuda.server.engine.pixel.channel", "run_channel_loop"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:  # pragma: no cover - standard attribute error path
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - trivial helper
    return sorted(list(globals().keys()) + list(__all__))


if TYPE_CHECKING:  # pragma: no cover - assist static type checkers
    from napari_cuda.server.engine.capture.capture import CaptureFacade, FrameTimings, encode_frame
    from napari_cuda.server.engine.capture.debug import DebugConfig, DebugDumper
    from napari_cuda.server.engine.capture.egl_context import EglContext
    from napari_cuda.server.engine.encoding.bitstream import (
        ParamCache,
        build_avcc_config,
        configure_bitstream,
        pack_to_avcc,
    )
    from napari_cuda.server.engine.encoding.encoder import Encoder
    from napari_cuda.server.engine.pixel.broadcaster import (
        PixelBroadcastConfig,
        PixelBroadcastState,
        broadcast_loop,
    )
    from napari_cuda.server.engine.pixel.channel import (
        PixelChannelConfig,
        PixelChannelState,
        build_notify_stream_payload,
        enqueue_frame,
        ensure_keyframe,
        ingest_client,
        mark_stream_config_dirty,
        prepare_client_attach,
        publish_avcc,
        run_channel_loop,
    )

