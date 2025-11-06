"""State update handlers for view scope."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Mapping, Optional, TYPE_CHECKING

from napari_cuda.server.control.state_reducers import (
    reduce_plane_restore,
    reduce_view_update,
    reduce_volume_restore,
)
from napari_cuda.server.scene import (
    PlaneState,
    RenderMode,
    RenderUpdate,
    VolumeState,
    snapshot_render_state,
)
from napari_cuda.shared.dims_spec import dims_spec_from_payload

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


def _apply_plane_restore_from_ledger(
    server: Any,
    *,
    timestamp: float | None,
    intent_id: str,
) -> bool:
    ledger = server._state_ledger
    with server._state_lock:
        plane_entry = ledger.get("viewport", "plane", "state")

    if plane_entry is None or not isinstance(plane_entry.value, Mapping):
        logger.warning("plane_restore skipped due to missing viewport plane state")
        return False

    plane_state = PlaneState(**dict(plane_entry.value))  # type: ignore[arg-type]

    level_value = plane_state.applied_level or plane_state.target_level
    step_value: Optional[tuple[int, ...]] = None
    if plane_state.applied_step is not None:
        step_value = tuple(int(v) for v in plane_state.applied_step)
    elif plane_state.target_step is not None:
        step_value = tuple(int(v) for v in plane_state.target_step)

    center_value: Optional[tuple[float, float, float]] = None
    if plane_state.pose.center is not None:
        cx, cy = plane_state.pose.center
        center_value = (float(cx), float(cy), 0.0)

    zoom_value = plane_state.pose.zoom
    rect_value = plane_state.pose.rect

    missing: list[str] = []
    if level_value is None:
        missing.append("plane_state.level")
    if step_value is None:
        missing.append("plane_state.step")
    if center_value is None:
        missing.append("viewport.plane.state.pose.center")
    if zoom_value is None:
        missing.append("viewport.plane.state.pose.zoom")
    if rect_value is None:
        missing.append("viewport.plane.state.pose.rect")

    if missing:
        logger.warning("plane_restore skipped due to missing ledger entries: %s", ", ".join(missing))
        return False

    level_idx = int(level_value)

    def _as_step(value: Any) -> tuple[int, ...]:
        if isinstance(value, Mapping):
            # unexpected, fall back to tuple()
            return tuple(int(v) for v in value.values())
        return tuple(int(v) for v in value)

    def _as_center(value: Any) -> tuple[float, float, float]:
        seq = tuple(float(v) for v in value)
        if len(seq) == 3:
            return seq  # type: ignore[return-value]
        if len(seq) >= 2:
            return (seq[0], seq[1], 0.0)
        raise ValueError("plane restore center requires at least two components")

    def _as_zoom(value: Any) -> float:
        zoom_val = float(value)
        if zoom_val <= 0:
            raise ValueError("zoom must be positive")
        return zoom_val

    def _as_rect(value: Any) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in value[:4])  # type: ignore[return-value]

    step_tuple = _as_step(step_value)
    center_tuple = _as_center(center_value)
    zoom_float = _as_zoom(zoom_value)
    rect_tuple = _as_rect(rect_value)

    reduce_plane_restore(
        server._state_ledger,
        level=level_idx,
        step=step_tuple,
        center=center_tuple,
        zoom=zoom_float,
        rect=rect_tuple,
        intent_id=intent_id,
        timestamp=timestamp,
        origin="client.state.view",
    )
    return True


def _apply_volume_restore_from_ledger(
    server: Any,
    *,
    timestamp: float | None,
    intent_id: str,
) -> Optional[VolumeState]:
    ledger = server._state_ledger
    with server._state_lock:
        volume_entry = ledger.get("viewport", "volume", "state")
        center_entry = ledger.get("camera_volume", "main", "center")
        angles_entry = ledger.get("camera_volume", "main", "angles")
        distance_entry = ledger.get("camera_volume", "main", "distance")
        fov_entry = ledger.get("camera_volume", "main", "fov")

    if volume_entry is None or not isinstance(volume_entry.value, Mapping):
        logger.warning("volume_restore skipped due to missing viewport volume state")
        return None

    volume_state = VolumeState(**dict(volume_entry.value))  # type: ignore[arg-type]

    level_value: Optional[int] = volume_state.level

    center_value = volume_state.pose.center
    if (center_value is None or len(center_value) < 3) and center_entry is not None and center_entry.value is not None:
        center_payload = tuple(float(v) for v in center_entry.value)
        if len(center_payload) >= 3:
            center_value = center_payload  # type: ignore[assignment]

    angles_value = volume_state.pose.angles
    if (angles_value is None or len(angles_value) < 3) and angles_entry is not None and angles_entry.value is not None:
        angles_payload = tuple(float(v) for v in angles_entry.value)
        if len(angles_payload) >= 2:
            roll_component = float(angles_payload[2]) if len(angles_payload) >= 3 else 0.0
            angles_value = (float(angles_payload[0]), float(angles_payload[1]), roll_component)

    distance_value = volume_state.pose.distance
    if distance_value is None and distance_entry is not None and distance_entry.value is not None:
        distance_value = float(distance_entry.value)

    fov_value = volume_state.pose.fov
    if fov_value is None and fov_entry is not None and fov_entry.value is not None:
        fov_value = float(fov_entry.value)

    if (
        level_value is None
        or center_value is None
        or len(center_value) < 3
        or angles_value is None
        or len(angles_value) < 2
        or distance_value is None
        or fov_value is None
    ):
        logger.debug("volume_restore skipped (pose missing); awaiting worker emit")
        return None

    level_idx = int(level_value)
    center_tuple = (
        float(center_value[0]),
        float(center_value[1]),
        float(center_value[2]),
    )
    roll_component = float(angles_value[2]) if len(angles_value) >= 3 else 0.0
    angles_tuple = (
        float(angles_value[0]),
        float(angles_value[1]),
        roll_component,
    )
    distance_float = float(distance_value)
    fov_float = float(fov_value)

    volume_state.level = level_idx
    volume_state.update_pose(
        center=center_tuple,
        angles=angles_tuple,
        distance=distance_float,
        fov=fov_float,
    )

    reduce_volume_restore(
        ledger,
        level=level_idx,
        center=center_tuple,
        angles=angles_tuple,
        distance=distance_float,
        fov=fov_float,
        intent_id=intent_id,
        timestamp=timestamp,
        origin="client.state.view",
    )
    return volume_state


async def handle_view_ndisplay(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    try:
        raw_value = int(ctx.value) if ctx.value is not None else 2
    except Exception:
        logger.debug("state.update view ignored (non-integer ndisplay) value=%r", ctx.value)
        await ctx.reject("state.invalid", "ndisplay must be integer")
        return True

    ndisplay = 3 if int(raw_value) >= 3 else 2
    prev_spec_entry = server._state_ledger.get("dims", "main", "dims_spec")
    prev_spec = dims_spec_from_payload(getattr(prev_spec_entry, "value", None)) if prev_spec_entry is not None else None
    prev_ndisplay = (
        int(prev_spec.ndisplay)
        if prev_spec is not None
        else (3 if server._initial_mode is RenderMode.VOLUME else 2)
    )
    was_volume = prev_ndisplay >= 3
    if getattr(server, "_log_dims_info", False):
        logger.info("intent: view.set_ndisplay ndisplay=%d", int(ndisplay))
    else:
        logger.debug("intent: view.set_ndisplay ndisplay=%d", int(ndisplay))

    try:
        result = reduce_view_update(
            server._state_ledger,
            ndisplay=int(ndisplay),
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin="client.state.view",
        )
    except Exception:
        logger.info("state.update view.set_ndisplay failed", exc_info=True)
        await ctx.reject("state.error", "failed to apply ndisplay")
        return True

    assert result.version is not None, "view reducer must supply version"

    new_ndisplay = int(result.value)

    broadcast = server._pixel_channel.broadcast
    broadcast.bypass_until_key = True
    broadcast.waiting_for_keyframe = True
    server.mark_stream_config_dirty()
    server._schedule_coro(server._ensure_keyframe(), "ndisplay-keyframe")

    restored_volume_state: Optional[VolumeState] = None
    if new_ndisplay >= 3:
        restored_volume_state = _apply_volume_restore_from_ledger(
            server,
            timestamp=ctx.timestamp,
            intent_id=ctx.intent_id,
        )
    if was_volume and new_ndisplay == 2:
        _apply_plane_restore_from_ledger(
            server,
            timestamp=ctx.timestamp,
            intent_id=ctx.intent_id,
        )

    await ctx.ack(applied_value=new_ndisplay, version=int(result.version))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "state.update view intent=%s frame=%s ndisplay=%s accepted",
            ctx.intent_id,
            ctx.frame_id,
            new_ndisplay,
        )

    # Post-frame thumbnail emission handles updates; no explicit queue here.

    runtime = getattr(server, "runtime", None)
    if runtime is not None and runtime.is_ready:
        snapshot = runtime.viewport_snapshot()
        if snapshot is None:
            return True
        plane_state = snapshot.plane
        plane_state.target_ndisplay = int(new_ndisplay)
        volume_state = replace(restored_volume_state) if restored_volume_state is not None else snapshot.volume
        desired_mode = RenderMode.VOLUME if new_ndisplay >= 3 else RenderMode.PLANE

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "view.toggle enqueue: mode=%s plane_target_level=%s volume_level=%s",
                desired_mode.name,
                plane_state.target_level,
                volume_state.level,
            )
            logger.debug(
                "view.toggle plane_state=%s volume_state=%s",
                plane_state,
                volume_state,
            )

        snapshot_state = snapshot_render_state(server._state_ledger)
        runtime.enqueue_render_update(
            RenderUpdate(
                scene_state=snapshot_state,
                mode=desired_mode,
                plane_state=plane_state,
                volume_state=volume_state,
            )
        )

    return True
