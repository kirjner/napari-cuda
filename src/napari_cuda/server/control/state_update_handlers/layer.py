"""State update handler for layer scope."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari_cuda.server.control.state_reducers import reduce_layer_property

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


async def handle_layer_property(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    layer_id = ctx.target or "layer-0"
    try:
        result = reduce_layer_property(
            server._state_ledger,
            layer_id=layer_id,
            prop=ctx.key,
            value=ctx.value,
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin='client.state.layer',
        )
    except KeyError:
        logger.debug("state.update unknown layer prop=%s", ctx.key)
        await ctx.reject(
            "state.invalid",
            f"unsupported layer key {ctx.key}",
            details={"scope": ctx.scope, "key": ctx.key, "target": layer_id},
        )
        return True
    except Exception:
        logger.debug(
            "state.update failed for layer=%s key=%s",
            layer_id,
            ctx.key,
            exc_info=True,
        )
        await ctx.reject(
            "state.error",
            "layer update failed",
            details={"scope": ctx.scope, "key": ctx.key, "target": layer_id},
        )
        return True

    assert result.version is not None, "layer reducer must supply version"
    await ctx.ack(applied_value=result.value, version=int(result.version))

    logger.debug(
        "state.update layer intent=%s frame=%s layer_id=%s key=%s value=%s version=%s",
        ctx.intent_id,
        ctx.frame_id,
        layer_id,
        ctx.key,
        result.value,
        result.version,
    )
    return True
