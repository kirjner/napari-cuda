"""State update handler for multiscale scope."""

from __future__ import annotations

import logging
from typing import Mapping, TYPE_CHECKING

from napari_cuda.server.control.state_reducers import (
    clamp_level,
    reduce_level_update,
)
from napari_cuda.server.scene import snapshot_multiscale_state

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


async def handle_multiscale_level(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    key = ctx.key

    if key == 'policy':
        logger.debug("state.update multiscale policy ignored (unsupported)")
        await ctx.reject(
            "state.unsupported",
            "multiscale policy is no longer configurable",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    if key != 'level':
        logger.debug("state.update multiscale ignored key=%s", key)
        await ctx.reject(
            "state.invalid",
            f"unsupported multiscale key {key}",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    multiscale_state = snapshot_multiscale_state(server._state_ledger.snapshot())
    levels = multiscale_state.get('levels') or []
    try:
        level = clamp_level(ctx.value, levels)
    except (TypeError, ValueError):
        logger.debug("state.update multiscale ignored invalid level=%r", ctx.value)
        await ctx.reject(
            "state.invalid",
            "level out of range",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    try:
        result = reduce_level_update(
            server._state_ledger,
            applied={"level": int(level)},
            intent_id=ctx.intent_id,
            timestamp=ctx.timestamp,
            origin='client.state.multiscale',
        )
    except Exception:
        logger.exception("state.update multiscale level failed level=%s", level)
        await ctx.reject(
            "state.error",
            "multiscale level update failed",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    runtime = getattr(server, "runtime", None)
    if runtime is not None and runtime.is_ready:
        log_fn = logger.info if server._log_state_traces else logger.debug
        try:
            levels_meta = multiscale_state.get('levels') or []
            path = None
            if isinstance(levels_meta, list) and 0 <= int(level) < len(levels_meta):
                entry = levels_meta[int(level)]
                if isinstance(entry, Mapping):
                    path = entry.get('path')
            log_fn("state: multiscale level -> runtime.request start level=%s", level)
            runtime.request_level(int(level), path)
            log_fn("state: multiscale level -> runtime.request done")
            log_fn("state: multiscale level -> runtime.force_idr start")
            runtime.force_idr()
            log_fn("state: multiscale level -> runtime.force_idr done")
            server._pixel.bypass_until_key = True
        except Exception:
            logger.exception("multiscale level switch request failed")

    assert result.version is not None, "multiscale reducer must supply version"
    await ctx.ack(applied_value=result.value, version=int(result.version))
    return True
