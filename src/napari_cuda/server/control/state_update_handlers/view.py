"""State update handlers for view scope."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, TYPE_CHECKING

from napari_cuda.server.control.state_reducers import (
    load_plane_restore_cache,
    load_volume_restore_cache,
    reduce_view_update,
)
from napari_cuda.server.scene.blocks import (
    PlaneRestoreCacheBlock,
    VolumeRestoreCacheBlock,
)
from napari_cuda.server.scene import (
    PlaneViewportCache,
    RenderMode,
    VolumeViewportCache,
)
from napari_cuda.shared.dims_spec import dims_spec_from_payload

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


def _plane_cache_from_restore_block(cache: PlaneRestoreCacheBlock) -> PlaneViewportCache:
    pose = cache.pose
    assert pose.center is not None, "plane restore cache missing center"
    assert pose.rect is not None, "plane restore cache missing rect"
    assert pose.zoom is not None, "plane restore cache missing zoom"

    plane_state = PlaneViewportCache()
    plane_state.target_ndisplay = 2
    plane_state.target_level = cache.level
    plane_state.target_step = cache.index
    plane_state.applied_level = cache.level
    plane_state.applied_step = cache.index
    plane_state.update_pose(
        center=pose.center,
        rect=pose.rect,
        zoom=pose.zoom,
    )
    plane_state.camera_pose_dirty = False
    plane_state.awaiting_level_confirm = False
    return plane_state


def _volume_cache_from_restore_block(cache: VolumeRestoreCacheBlock) -> VolumeViewportCache:
    pose = cache.pose
    assert pose.center is not None, "volume restore cache missing center"
    assert pose.angles is not None, "volume restore cache missing angles"
    assert pose.distance is not None, "volume restore cache missing distance"
    assert pose.fov is not None, "volume restore cache missing fov"

    volume_state = VolumeViewportCache()
    volume_state.level = cache.level
    volume_state.update_pose(
        center=pose.center,
        angles=pose.angles,
        distance=pose.distance,
        fov=pose.fov,
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

    ledger = server._state_ledger
    restored_volume_state: Optional[VolumeViewportCache] = None
    plane_state_override: Optional[PlaneViewportCache] = None
    if new_ndisplay >= 3:
        cache = load_volume_restore_cache(ledger)
        restored_volume_state = _volume_cache_from_restore_block(cache)
    if was_volume and new_ndisplay == 2:
        cache = load_plane_restore_cache(ledger)
        plane_state_override = _plane_cache_from_restore_block(cache)

    await ctx.ack(applied_value=new_ndisplay, version=int(result.version))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "state.update view intent=%s frame=%s ndisplay=%s accepted",
            ctx.intent_id,
            ctx.frame_id,
            new_ndisplay,
        )

    # Post-frame thumbnail emission handles updates; no explicit queue here.

    return True
