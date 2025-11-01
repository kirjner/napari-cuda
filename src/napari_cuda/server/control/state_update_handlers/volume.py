"""State update handler for volume scope."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari_cuda.server.control.state_reducers import (
    clamp_opacity,
    clamp_sample_step,
    is_valid_render_mode,
    normalize_clim,
    reduce_volume_colormap,
    reduce_volume_contrast_limits,
    reduce_volume_opacity,
    reduce_volume_render_mode,
    reduce_volume_sample_step,
)

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


async def handle_volume_update(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    key = ctx.key
    value = ctx.value

    if key == 'render_mode':
        mode = str(value or '').lower().strip()
        if not is_valid_render_mode(mode, server._allowed_render_modes):
            logger.debug("state.update volume ignored invalid render_mode=%r", value)
            await ctx.reject(
                "state.invalid",
                "unknown render_mode",
                details={"scope": ctx.scope, "key": key, "value": value},
            )
            return True
        result = reduce_volume_render_mode(
            server._state_ledger,
            mode=mode,
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin='client.state.volume',
        )
    elif key == 'contrast_limits':
        pair = value
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            logger.debug("state.update volume ignored invalid contrast_limits=%r", pair)
            await ctx.reject(
                "state.invalid",
                "contrast_limits requires [lo, hi]",
                details={"scope": ctx.scope, "key": key},
            )
            return True
        try:
            lo, hi = normalize_clim(pair[0], pair[1])
        except (TypeError, ValueError):
            logger.debug("state.update volume ignored invalid clim=%r", pair)
            await ctx.reject(
                "state.invalid",
                "contrast_limits invalid",
                details={"scope": ctx.scope, "key": key},
            )
            return True
        result = reduce_volume_contrast_limits(
            server._state_ledger,
            lo=lo,
            hi=hi,
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin='client.state.volume',
        )
    elif key == 'colormap':
        name = value
        if not isinstance(name, str) or not name.strip():
            logger.debug("state.update volume ignored invalid colormap=%r", name)
            await ctx.reject(
                "state.invalid",
                "colormap must be non-empty",
                details={"scope": ctx.scope, "key": key},
            )
            return True
        result = reduce_volume_colormap(
            server._state_ledger,
            name=str(name),
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin='client.state.volume',
        )
    elif key == 'opacity':
        try:
            norm_alpha = clamp_opacity(value)
        except Exception:
            logger.debug("state.update volume ignored invalid opacity=%r", value)
            await ctx.reject(
                "state.invalid",
                "opacity must be float",
                details={"scope": ctx.scope, "key": key},
            )
            return True
        result = reduce_volume_opacity(
            server._state_ledger,
            alpha=float(norm_alpha),
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin='client.state.volume',
        )
    elif key == 'sample_step':
        try:
            norm_step = clamp_sample_step(value)
        except Exception:
            logger.debug("state.update volume ignored invalid sample_step=%r", value)
            await ctx.reject(
                "state.invalid",
                "sample_step must be float",
                details={"scope": ctx.scope, "key": key},
            )
            return True
        result = reduce_volume_sample_step(
            server._state_ledger,
            sample_step=float(norm_step),
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin='client.state.volume',
        )
    else:
        logger.debug("state.update volume ignored key=%s", key)
        await ctx.reject(
            "state.invalid",
            f"unsupported volume key {key}",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    assert result.version is not None, "volume reducer must supply version"
    await ctx.ack(applied_value=result.value, version=int(result.version))

    logger.debug(
        "state.update volume intent=%s frame=%s key=%s value=%s",
        ctx.intent_id,
        ctx.frame_id,
        key,
        result.value,
    )
    return True
