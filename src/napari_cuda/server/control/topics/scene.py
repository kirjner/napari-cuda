from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from napari_cuda.protocol import (
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
)
import logging
from napari_cuda.protocol.envelopes import build_notify_scene_snapshot
from napari_cuda.protocol.messages import NotifyScenePayload
from napari_cuda.server.control.protocol.io import send_frame
from napari_cuda.server.control.protocol.runtime import (
    feature_enabled,
    history_store,
    state_sequencer,
    state_session,
)
from napari_cuda.server.control.resumable_history_store import (
    EnvelopeSnapshot,
    ResumeDecision,
    ResumePlan,
)

logger = logging.getLogger(__name__)

async def send_scene_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = state_session(ws)
    if not session_id:
        return
    payload = NotifyScenePayload.from_dict(snapshot.payload)
    sequencer = state_sequencer(ws, NOTIFY_SCENE_TYPE)
    # Resume first, so emitted frame carries the resumed seq/delta state
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    frame = build_notify_scene_snapshot(
        session_id=session_id,
        viewer=payload.viewer,
        layers=payload.layers,
        policies=payload.policies,
        metadata=payload.metadata,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        delta_token=snapshot.delta_token,
        intent_id=snapshot.intent_id,
        sequencer=sequencer,
    )
    await send_frame(server, ws, frame)
    if server._log_dims_info:
        logger.info("notify.scene snapshot sent (resume)")
    else:
        logger.debug("notify.scene snapshot sent (resume)")


async def send_scene_baseline(
    server: Any,
    ws: Any,
    *,
    payload: NotifyScenePayload,
    timestamp: Optional[float] = None,
    plan: Optional[ResumePlan] = None,
) -> None:
    """Emit a scene baseline to a single client, snapshotting when a history store exists.

    - If a history store is available, snapshot `notify.scene`, optionally resetting
      the `layers` and `stream` epochs, and resume the client's scene sequencer.
    - Otherwise, use the per-client sequencer to build and send a frame.
    """
    session_id = state_session(ws)
    if not session_id:
        return

    now = time.time() if timestamp is None else float(timestamp)
    store = history_store(server)
    snapshot: EnvelopeSnapshot | None = None

    if store is not None:
        current = store.current_snapshot(NOTIFY_SCENE_TYPE)
        need_reset = plan is not None and plan.decision == ResumeDecision.RESET
        if current is None or need_reset:
            snapshot = store.snapshot_envelope(NOTIFY_SCENE_TYPE, payload=payload.to_dict(), timestamp=now)
            # Reset downstream epochs when we reset the scene
            store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=now)
            store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=now)
            state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
            state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()
        else:
            snapshot = current

        await send_scene_snapshot(server, ws, snapshot)
    else:
        frame = build_notify_scene_snapshot(
            session_id=session_id,
            viewer=payload.viewer,
            layers=payload.layers,
            policies=payload.policies,
            metadata=payload.metadata,
            timestamp=now,
            sequencer=state_sequencer(ws, NOTIFY_SCENE_TYPE),
        )
        await send_frame(server, ws, frame)
    if server._log_dims_info:
        logger.info("notify.scene baseline sent")
    else:
        logger.debug("notify.scene baseline sent")


async def send_scene_snapshot_direct(
    server: Any,
    ws: Any,
    *,
    payload: NotifyScenePayload,
    timestamp: Optional[float] = None,
) -> None:
    """Send a one-off scene frame to a single client and record it in history if available."""
    session_id = state_session(ws)
    if not session_id:
        return
    now = time.time() if timestamp is None else float(timestamp)
    store = history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None:
        snapshot = store.snapshot_envelope(NOTIFY_SCENE_TYPE, payload=payload.to_dict(), timestamp=now)

    kwargs: dict[str, Any] = {
        "session_id": session_id,
        "viewer": payload.viewer,
        "layers": payload.layers,
        "policies": payload.policies,
        "metadata": payload.metadata,
        "timestamp": snapshot.timestamp if snapshot is not None else now,
    }
    if snapshot is not None:
        kwargs["frame_id"] = snapshot.frame_id
        kwargs["delta_token"] = snapshot.delta_token
        kwargs["seq"] = snapshot.seq

    frame = build_notify_scene_snapshot(**kwargs)
    await send_frame(server, ws, frame)
    if snapshot is not None:
        state_sequencer(ws, NOTIFY_SCENE_TYPE).resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    if server._log_dims_info:
        logger.info("notify.scene direct sent")
    else:
        logger.debug("notify.scene direct sent")


async def broadcast_scene_snapshot(
    server: Any,
    *,
    payload: NotifyScenePayload,
    timestamp: Optional[float] = None,
) -> None:
    """Broadcast a scene snapshot to all eligible clients; snapshot/reset epochs when history store exists."""
    clients = list(getattr(server, "_state_clients", []))
    if not clients:
        # Even without clients, ensure the history reflects the latest scene
        store = history_store(server)
        if store is not None:
            now = time.time() if timestamp is None else float(timestamp)
            if store.current_snapshot(NOTIFY_SCENE_TYPE) is None:
                store.snapshot_envelope(NOTIFY_SCENE_TYPE, payload=payload.to_dict(), timestamp=now)
            else:
                store.delta_envelope(NOTIFY_SCENE_TYPE, payload=payload.to_dict(), timestamp=now)
        return

    now = time.time() if timestamp is None else float(timestamp)
    store = history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None:
        # Snapshot if none exists yet; otherwise record delta
        if store.current_snapshot(NOTIFY_SCENE_TYPE) is None:
            snapshot = store.snapshot_envelope(NOTIFY_SCENE_TYPE, payload=payload.to_dict(), timestamp=now)
        else:
            snapshot = store.delta_envelope(NOTIFY_SCENE_TYPE, payload=payload.to_dict(), timestamp=now)
        # Reset downstream epochs when scene changes
        store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=now)
        store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=now)

    tasks: list[Awaitable[None]] = []
    for ws in clients:
        if not feature_enabled(ws, "notify.scene"):
            continue
        session_id = state_session(ws)
        if not session_id:
            continue
        if snapshot is not None:
            state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
            state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()
        kwargs: dict[str, Any] = {
            "session_id": session_id,
            "viewer": payload.viewer,
            "layers": payload.layers,
            "policies": payload.policies,
            "metadata": payload.metadata,
            "timestamp": snapshot.timestamp if snapshot is not None else now,
        }
        if snapshot is not None:
            kwargs["frame_id"] = snapshot.frame_id
            kwargs["delta_token"] = snapshot.delta_token
            kwargs["seq"] = snapshot.seq
        else:
            kwargs["sequencer"] = state_sequencer(ws, NOTIFY_SCENE_TYPE)
        frame = build_notify_scene_snapshot(**kwargs)
        tasks.append(send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        if snapshot is not None:
            for ws in clients:
                state_sequencer(ws, NOTIFY_SCENE_TYPE).resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    count = len(tasks)
    if server._log_dims_info:
        logger.info("notify.scene broadcast to %d clients", count)
    else:
        logger.debug("notify.scene broadcast to %d clients", count)


__all__ = [
    "broadcast_scene_snapshot",
    "send_scene_baseline",
    "send_scene_snapshot",
    "send_scene_snapshot_direct",
]
