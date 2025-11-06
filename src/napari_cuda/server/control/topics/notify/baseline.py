"""Baseline orchestration for state-channel connections."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from napari_cuda.protocol import (
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
)
from napari_cuda.protocol.snapshots import SceneSnapshot
from napari_cuda.server.control.control_payload_builder import (
    build_notify_scene_payload,
)
from napari_cuda.server.control.protocol.runtime import history_store
from napari_cuda.server.control.resumable_history_store import (
    ResumeDecision,
    ResumePlan,
)
from napari_cuda.server.scene import LayerVisualState, build_ledger_snapshot
from napari_cuda.shared.dims_spec import dims_spec_from_payload
from .dims import broadcast_dims_state, send_dims_state
from .layers import (
    send_layer_baseline,
    send_layer_snapshot,
)
from .scene import (
    send_scene_baseline,
)
from .stream import (
    send_stream_frame,
    send_stream_snapshot,
)

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


async def orchestrate_connect(
    server: Any,
    ws: Any,
    resume_map: Mapping[str, ResumePlan] | None,
) -> None:
    """Emit baseline state for a newly connected websocket client."""

    plan_map = dict(resume_map or {})
    scene_plan = plan_map.get(NOTIFY_SCENE_TYPE)
    layers_plan = plan_map.get(NOTIFY_LAYERS_TYPE)
    stream_plan = plan_map.get(NOTIFY_STREAM_TYPE)

    scene_snapshot = _ensure_scene_snapshot(server)
    ledger_snapshot = server._state_ledger.snapshot()
    scene_payload = build_notify_scene_payload(
        scene_snapshot=scene_snapshot,
        ledger_snapshot=ledger_snapshot,
        viewer_settings=_viewer_settings(server),
    )

    default_visuals = _collect_default_visuals(server, scene_snapshot)
    _record_default_visuals(server, layers_plan, default_visuals)

    await send_scene_baseline(
        server,
        ws,
        payload=scene_payload,
        plan=scene_plan,
    )
    (logger.info if server._log_dims_info else logger.debug)("connect: notify.scene sent")

    await _send_dims_baseline(server, ws)
    await _send_layer_baseline(
        server,
        ws,
        plan=layers_plan,
        default_visuals=default_visuals,
    )
    await _send_stream_baseline(server, ws, plan=stream_plan)

    _schedule_keyframe_and_thumbnail(server)

    ws._napari_cuda_resume_plan = {}


def _ensure_scene_snapshot(server: Any) -> SceneSnapshot:
    # Server must expose a snapshot refresher and snapshot storage
    server._refresh_scene_snapshot()
    snapshot = server._scene_snapshot
    assert isinstance(snapshot, SceneSnapshot), "scene snapshot unavailable"
    return snapshot


def _viewer_settings(server: Any) -> dict[str, Any]:
    width = int(server.width)
    height = int(server.height)
    fps = float(server.cfg.fps)
    spec_entry = server._state_ledger.get("dims", "main", "dims_spec")
    assert spec_entry is not None, "viewer settings require dims_spec entry"
    spec = dims_spec_from_payload(getattr(spec_entry, "value", None))
    assert spec is not None, "dims_spec payload missing"
    use_volume = int(spec.ndisplay) >= 3
    return {
        "fps_target": fps,
        "canvas_size": [width, height],
        "volume_enabled": use_volume,
    }


def _collect_default_visuals(
    server: Any,
    scene_snapshot: SceneSnapshot,
) -> list[LayerVisualState]:
    mirror = getattr(server, "_layer_mirror", None)
    visual_map: dict[str, LayerVisualState] = {}
    if mirror is not None:
        visual_map = mirror.latest_visual_states()
    else:
        logger.debug("layer mirror not initialised; falling back to ledger snapshot controls")
        snapshot = build_ledger_snapshot(server._state_ledger)
        layer_values = snapshot.layer_values or {}
        visual_map = {str(layer_id): state for layer_id, state in layer_values.items()}

    defaults: list[LayerVisualState] = []
    for layer_snapshot in scene_snapshot.layers:
        layer_id = layer_snapshot.layer_id
        if not layer_id:
            continue
        state = visual_map.get(layer_id)
        if state is None:
            updates: dict[str, Any] = {}
            block_controls = layer_snapshot.block.get("controls")
            if isinstance(block_controls, Mapping):
                updates.update({str(key): value for key, value in block_controls.items()})
            metadata_block = layer_snapshot.block.get("metadata")
            if isinstance(metadata_block, Mapping) and metadata_block:
                updates["metadata"] = dict(metadata_block)
            if updates:
                state = LayerVisualState(layer_id=layer_id).with_updates(updates=updates)
        if state is not None:
            defaults.append(state)

    # Include any additional layers not represented in the scene snapshot ordering.
    for layer_id, state in visual_map.items():
        if not any(existing.layer_id == layer_id for existing in defaults):
            defaults.append(state)
    return defaults


_DEFAULT_LAYER_KEYS = {
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
    "metadata",
}


def _record_default_visuals(
    server: Any,
    plan: ResumePlan | None,
    default_visuals: Sequence[LayerVisualState],
) -> None:
    if not default_visuals:
        return
    if plan is not None and plan.decision == ResumeDecision.REPLAY and plan.deltas:
        # Let replay handle the rehydration.
        return

    for visual in default_visuals:
        layer_id = visual.layer_id
        if not layer_id:
            continue
        for key in visual.keys():
            if key not in _DEFAULT_LAYER_KEYS:
                continue
            raw_value = visual.get(key)
            if raw_value is None:
                continue
            entry = server._state_ledger.get("layer", layer_id, key)
            if entry is not None:
                continue
            if key == "contrast_limits" and isinstance(raw_value, (list, tuple)):
                lo, hi = float(raw_value[0]), float(raw_value[1])
                normalized_value: Any = (lo, hi)
            elif key in {"opacity", "gamma", "attenuation", "iso_threshold"}:
                normalized_value = float(raw_value)
            elif key in {"blending", "interpolation", "colormap", "rendering"}:
                normalized_value = str(raw_value)
            elif key == "visible":
                normalized_value = bool(raw_value)
            elif key == "metadata":
                normalized_value = dict(raw_value)
            else:
                normalized_value = raw_value
            server._state_ledger.record_confirmed(
                "layer",
                layer_id,
                key,
                normalized_value,
                origin="bootstrap.layer_defaults",
            )
            if key == "colormap":
                server._state_ledger.record_confirmed(
                    "volume",
                    "main",
                    "colormap",
                    str(normalized_value),
                    origin="bootstrap.layer_defaults",
                )
            elif key == "contrast_limits":
                server._state_ledger.record_confirmed(
                    "volume",
                    "main",
                    "contrast_limits",
                    normalized_value,
                    origin="bootstrap.layer_defaults",
                )
            elif key == "opacity":
                server._state_ledger.record_confirmed(
                    "volume",
                    "main",
                    "opacity",
                    float(normalized_value),
                    origin="bootstrap.layer_defaults",
                )


async def _send_layer_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
    default_visuals: Sequence[LayerVisualState],
) -> None:
    store = history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await send_layer_snapshot(server, ws, snapshot)
            return

    await send_layer_baseline(server, ws, default_visuals)


async def _send_stream_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
) -> None:
    store = history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await send_stream_snapshot(server, ws, snapshot)
        return

    channel = server._pixel_channel
    cfg = server._pixel_config
    if channel is None or cfg is None:
        logger.debug("connect: notify.stream baseline deferred (pixel channel not initialized)")
        return

    avcc = channel.last_avcc
    if avcc is not None:
        stream_payload = server.build_stream_payload(avcc)
        await send_stream_frame(
            server,
            ws,
            payload=stream_payload,
            timestamp=time.time(),
        )
    else:
        server.mark_stream_config_dirty()


async def _send_dims_baseline(server: Any, ws: Any) -> None:
    mirror = server._dims_mirror
    if mirror is None:
        logger.debug("connect: notify.dims baseline deferred (dims mirror not initialized)")
        return
    payload = mirror.latest_payload()
    if payload is None:
        logger.debug("connect: notify.dims baseline skipped (metadata unavailable)")
        return
    await send_dims_state(server, ws, payload=payload, intent_id=None, timestamp=time.time())
    if server._log_dims_info:
        logger.info("connect: notify.dims baseline sent")
    else:
        logger.debug("connect: notify.dims baseline sent")


def _schedule_keyframe_and_thumbnail(server: Any) -> None:
    # Server must expose scheduling and keyframe/thumbnail hooks
    assert hasattr(server, "_schedule_coro"), "server must expose _schedule_coro"
    assert hasattr(server, "_ensure_keyframe"), "server must expose _ensure_keyframe"
    server._schedule_coro(server._ensure_keyframe(), "state-baseline-keyframe")

    # Post-frame thumbnail emission handles updates; no explicit queue here.


__all__ = ["orchestrate_connect"]
