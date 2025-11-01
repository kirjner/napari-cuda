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
from napari_cuda.server.control.protocol_runtime import history_store
from napari_cuda.server.control.resumable_history_store import (
    ResumeDecision,
    ResumePlan,
)
from napari_cuda.server.control.topics.dims import broadcast_dims_state
from napari_cuda.server.control.topics.layers import (
    send_layer_baseline,
    send_layer_snapshot,
)
from napari_cuda.server.control.topics.scene import (
    send_scene_baseline,
)
from napari_cuda.server.control.topics.stream import (
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

    default_controls = _collect_default_controls(server, scene_snapshot)
    _record_default_controls(server, layers_plan, default_controls)

    await send_scene_baseline(
        server,
        ws,
        payload=scene_payload,
        plan=scene_plan,
    )
    (logger.info if server._log_dims_info else logger.debug)("connect: notify.scene sent")

    await _emit_dims_baseline(server, ws)
    await _emit_layer_baseline(
        server,
        ws,
        plan=layers_plan,
        default_controls=default_controls,
    )
    await _emit_stream_baseline(server, ws, plan=stream_plan)

    _schedule_keyframe_and_thumbnail(server)

    if hasattr(ws, "_napari_cuda_resume_plan"):
        delattr(ws, "_napari_cuda_resume_plan")


def _ensure_scene_snapshot(server: Any) -> SceneSnapshot:
    refresher = getattr(server, "_refresh_scene_snapshot", None)
    if callable(refresher):
        refresher()
    snapshot = getattr(server, "_scene_snapshot", None)
    assert isinstance(snapshot, SceneSnapshot), "scene snapshot unavailable"
    return snapshot


def _viewer_settings(server: Any) -> dict[str, Any]:
    settings_provider = getattr(server, "_viewer_settings_for_scene", None)
    if callable(settings_provider):
        settings = settings_provider()
        if isinstance(settings, Mapping):
            return dict(settings)
    return {}


def _collect_default_controls(
    server: Any,
    scene_snapshot: SceneSnapshot,
) -> list[tuple[str, Mapping[str, Any]]]:
    layer_controls_map: dict[str, Mapping[str, Any]] = {}
    mirror = getattr(server, "_layer_mirror", None)
    if mirror is not None:
        layer_controls_map = mirror.latest_controls()
    else:
        logger.debug("layer mirror not initialised; falling back to scene snapshot controls")

    defaults: list[tuple[str, Mapping[str, Any]]] = []
    for layer_snapshot in scene_snapshot.layers:
        layer_id = layer_snapshot.layer_id
        if not layer_id:
            continue

        controls: dict[str, Any] = dict(layer_controls_map.get(layer_id, {}))
        if not controls:
            block_controls = layer_snapshot.block.get("controls")
            if isinstance(block_controls, Mapping):
                controls.update({str(key): value for key, value in block_controls.items()})

        metadata_block = layer_snapshot.block.get("metadata")
        if isinstance(metadata_block, Mapping) and metadata_block:
            controls.setdefault("metadata", dict(metadata_block))

        if controls:
            defaults.append((layer_id, controls))
    return defaults


def _record_default_controls(
    server: Any,
    plan: ResumePlan | None,
    default_controls: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    if not default_controls:
        return
    if plan is not None and plan.decision == ResumeDecision.REPLAY and plan.deltas:
        # Let replay handle the rehydration.
        return

    for layer_id, controls in default_controls:
        if not layer_id:
            continue
        for key, raw_value in controls.items():
            entry = server._state_ledger.get("layer", layer_id, key)
            if entry is not None:
                continue
            if key == "contrast_limits" and isinstance(raw_value, (list, tuple)):
                lo, hi = float(raw_value[0]), float(raw_value[1])
                normalized_value: Any = (lo, hi)
            elif key in {"opacity", "gamma"}:
                normalized_value = float(raw_value)
            elif key in {"blending", "interpolation", "colormap"}:
                normalized_value = str(raw_value)
            elif key == "visible":
                normalized_value = bool(raw_value)
            elif isinstance(raw_value, list):
                normalized_value = tuple(raw_value)
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


async def _emit_layer_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
    default_controls: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    store = history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await send_layer_snapshot(server, ws, snapshot)
            return

    await send_layer_baseline(server, ws, default_controls)


async def _emit_stream_baseline(
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

    channel = getattr(server, "_pixel_channel", None)
    cfg = getattr(server, "_pixel_config", None)
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


async def _emit_dims_baseline(server: Any, ws: Any) -> None:
    mirror = getattr(server, "_dims_mirror", None)
    if mirror is None:
        logger.debug("connect: notify.dims baseline deferred (dims mirror not initialized)")
        return
    payload = mirror.latest_payload()
    if payload is None:
        logger.debug("connect: notify.dims baseline skipped (metadata unavailable)")
        return
    await broadcast_dims_state(server, payload=payload, intent_id=None, timestamp=time.time(), targets=[ws])
    if server._log_dims_info:
        logger.info("connect: notify.dims baseline sent")
    else:
        logger.debug("connect: notify.dims baseline sent")


def _schedule_keyframe_and_thumbnail(server: Any) -> None:
    scheduler = getattr(server, "_schedule_coro", None)
    ensure_keyframe = getattr(server, "_ensure_keyframe", None)
    if callable(scheduler) and callable(ensure_keyframe):
        scheduler(ensure_keyframe(), "state-baseline-keyframe")
    else:
        logger.debug("connect: baseline keyframe scheduling skipped (hooks unavailable)")
        return

    emit_thumbnail = getattr(server, "_emit_layer_thumbnail", None)
    default_layer_id = getattr(server, "_default_layer_id", None)
    if callable(emit_thumbnail) and callable(default_layer_id):
        scheduler(emit_thumbnail(default_layer_id()), "baseline-layer-thumbnail")
    else:
        logger.debug("connect: baseline thumbnail emission skipped (hooks unavailable)")


__all__ = ["orchestrate_connect"]
