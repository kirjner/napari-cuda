from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

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


def classify_layer_changes(
    changes: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, bool]:
    ctrl_keys = {
        "opacity",
        "visible",
        "blending",
        "interpolation",
        "colormap",
        "rendering",
        "gamma",
        "contrast_limits",
        "iso_threshold",
        "attenuation",
    }
    controls: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    thumbnail: dict[str, Any] | None = None
    removed = False

    for key, value in changes.items():
        skey = str(key)
        if skey in ctrl_keys:
            controls[skey] = value
        elif skey == "metadata":
            md = dict(value) if isinstance(value, Mapping) else {"value": value}
            if "thumbnail" in md:
                th_value = md.pop("thumbnail")
                thumbnail = {"array": th_value}
            metadata = md
        elif skey == "thumbnail":
            thumbnail = dict(value) if isinstance(value, Mapping) else {"array": value}
        elif skey == "removed" and bool(value):
            removed = True
        else:
            if data is None:
                data = {}
            data[skey] = value

    return (controls or None), metadata, data, thumbnail, removed


async def broadcast_layers_delta(
    server: Any,
    *,
    layer_id: Optional[str],
    changes: Mapping[str, Any],
    intent_id: Optional[str],
    timestamp: Optional[float],
    targets: Optional[Sequence[Any]] = None,
) -> None:
    if not changes:
        return

    controls, metadata, data, thumbnail, removed = classify_layer_changes(changes)
    payload = build_notify_layers_payload(
        layer_id=layer_id or "layer-0",
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
    default_controls: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    if not default_controls:
        return

    store = history_store(server)
    if store is not None:
        now = __import__("time").time()
        for layer_id, changes in default_controls:
            # Classify the mapping so 'metadata' or 'thumbnail' nested
            # in the provided mapping do not leak into controls.
            controls, metadata, data, thumbnail, removed = classify_layer_changes(changes)
            payload = build_notify_layers_payload(
                layer_id=layer_id or "layer-0",
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

    for layer_id, changes in default_controls:
        await broadcast_layers_delta(
            server,
            layer_id=layer_id,
            changes=changes,
            intent_id=None,
            timestamp=__import__("time").time(),
            targets=[ws],
        )


__all__ = [
    "broadcast_layers_delta",
    "classify_layer_changes",
    "send_layer_baseline",
    "send_layer_snapshot",
]
