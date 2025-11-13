from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

from napari_cuda.protocol import NOTIFY_LAYERS_TYPE
from napari_cuda.protocol.envelopes import (
    build_notify_layers_delta,
)
from napari_cuda.protocol.messages import NotifyLayersPayload
from napari_cuda.server.control.control_payload_builder import (
    build_notify_layers_payload,
)
from napari_cuda.server.control.protocol.io import send_frame
from napari_cuda.server.control.protocol.runtime import (
    feature_enabled,
    history_store,
    state_sequencer,
    state_session,
)
from napari_cuda.server.control.resumable_history_store import EnvelopeSnapshot
from napari_cuda.server.scene import LayerVisualState
from napari_cuda.server.scene.layer_block_diff import (
    LayerBlockDelta,
    layer_block_delta_sections,
)

LayerState = Union[LayerVisualState, LayerBlockDelta]

_CONTROL_KEYS = {
    "visible",
    "opacity",
    "blending",
    "interpolation",
    "colormap",
    "rendering",
    "gamma",
    "contrast_limits",
    "iso_threshold",
    "attenuation",
}


def _split_visual_state(
    state: LayerVisualState,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, bool]:
    controls: dict[str, Any] = {}
    for key in _CONTROL_KEYS:
        value = state.get(key)
        if value is not None:
            controls[str(key)] = value

    metadata = state.get("metadata")
    metadata_payload = dict(metadata) if isinstance(metadata, dict) and metadata else None

    thumbnail = state.get("thumbnail")
    thumbnail_payload = dict(thumbnail) if isinstance(thumbnail, dict) else None

    data_payload: dict[str, Any] | None = None
    removed = False
    for key, value in state.extra.items():
        skey = str(key)
        if skey == "removed":
            removed = bool(value)
            continue
        if data_payload is None:
            data_payload = {}
        data_payload[skey] = value

    return (
        controls or None,
        metadata_payload,
        data_payload,
        thumbnail_payload,
        removed,
    )


def _split_layer_state(
    state: LayerState,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, bool]:
    if isinstance(state, LayerVisualState):
        return _split_visual_state(state)
    return layer_block_delta_sections(state)


def _split_layer_visual_state(
    state: LayerVisualState,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Compat wrapper for tests expecting the legacy splitter."""

    return _split_visual_state(state)


async def _deliver_layers_delta(
    server: Any,
    *,
    layer_id: Optional[str],
    state: LayerState,
    intent_id: Optional[str],
    timestamp: Optional[float],
    targets: Optional[Sequence[Any]] = None,
) -> None:
    if isinstance(state, LayerVisualState):
        if not state.keys() and not state.extra and not state.metadata and state.thumbnail is None:
            return

    controls, metadata, data, thumbnail, removed = _split_layer_state(state)
    if not any((controls, metadata, data, thumbnail)) and not removed:
        return

    resolved_layer_id = layer_id or state.layer_id or "layer-0"

    payload = build_notify_layers_payload(
        layer_id=resolved_layer_id,
        controls=controls,
        metadata=metadata,
        data=data,
        thumbnail=thumbnail,
        removed=removed or None,
    )

    clients = list(targets) if targets is not None else list(server._state_clients)
    now = timestamp if timestamp is not None else __import__("time").time()

    store = history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None and targets is None:
        snapshot = store.delta_envelope(
            NOTIFY_LAYERS_TYPE,
            payload=payload.to_dict(),
            timestamp=now,
            intent_id=intent_id,
        )

    tasks = []
    for ws in clients:
        if not feature_enabled(ws, "notify.layers"):
            continue
        session_id = state_session(ws)
        if not session_id:
            continue
        kwargs: dict[str, Any] = {
            "session_id": session_id,
            "payload": payload,
            "timestamp": snapshot.timestamp if snapshot is not None else now,
            "intent_id": intent_id,
        }
        if snapshot is not None:
            kwargs["seq"] = snapshot.seq
            kwargs["delta_token"] = snapshot.delta_token
            kwargs["frame_id"] = snapshot.frame_id
        else:
            kwargs["sequencer"] = state_sequencer(ws, NOTIFY_LAYERS_TYPE)
        frame = build_notify_layers_delta(**kwargs)
        tasks.append(send_frame(server, ws, frame))

    if tasks:
        await __import__("asyncio").gather(*tasks, return_exceptions=True)

    if snapshot is not None:
        for ws in clients:
            sequencer = state_sequencer(ws, NOTIFY_LAYERS_TYPE)
            sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def broadcast_layers_delta(
    server: Any,
    *,
    layer_id: Optional[str],
    state: LayerState,
    intent_id: Optional[str],
    timestamp: Optional[float],
) -> None:
    await _deliver_layers_delta(
        server,
        layer_id=layer_id,
        state=state,
        intent_id=intent_id,
        timestamp=timestamp,
        targets=None,
    )


async def send_layers_delta(
    server: Any,
    ws: Any,
    *,
    layer_id: Optional[str],
    state: LayerState,
    intent_id: Optional[str],
    timestamp: Optional[float],
) -> None:
    await _deliver_layers_delta(
        server,
        layer_id=layer_id,
        state=state,
        intent_id=intent_id,
        timestamp=timestamp,
        targets=[ws],
    )


async def send_layer_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = state_session(ws)
    if not session_id:
        return
    payload = NotifyLayersPayload.from_dict(snapshot.payload)
    frame = build_notify_layers_delta(
        session_id=session_id,
        payload=payload,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        intent_id=snapshot.intent_id,
        seq=snapshot.seq,
        delta_token=snapshot.delta_token,
    )
    await send_frame(server, ws, frame)
    sequencer = state_sequencer(ws, NOTIFY_LAYERS_TYPE)
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def send_layer_baseline(
    server: Any,
    ws: Any,
    default_visuals: Sequence[LayerVisualState],
) -> None:
    if not default_visuals:
        return

    store = history_store(server)
    if store is not None:
        now = __import__("time").time()
        for visual in default_visuals:
            controls, metadata, data, thumbnail, removed = _split_layer_visual_state(visual)
            if not any((controls, metadata, data, thumbnail)) and not removed:
                continue
            resolved_layer_id = visual.layer_id or "layer-0"
            payload = build_notify_layers_payload(
                layer_id=resolved_layer_id,
                controls=controls,
                metadata=metadata,
                data=data,
                thumbnail=thumbnail,
                removed=removed or None,
            )
            snapshot = store.delta_envelope(
                NOTIFY_LAYERS_TYPE,
                payload=payload.to_dict(),
                timestamp=now,
                intent_id=None,
            )
            await send_layer_snapshot(server, ws, snapshot)
        return

    for visual in default_visuals:
        await send_layers_delta(
            server,
            ws,
            layer_id=visual.layer_id,
            state=visual,
            intent_id=None,
            timestamp=__import__("time").time(),
        )


__all__ = [
    "send_layers_delta",
    "broadcast_layers_delta",
    "send_layer_baseline",
    "send_layer_snapshot",
]
