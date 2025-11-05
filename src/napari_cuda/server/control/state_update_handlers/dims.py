"""State update handler for dims scope."""

from __future__ import annotations

import logging
from numbers import Integral
from typing import Any, Optional, TYPE_CHECKING

from napari_cuda.server.control.state_models import ClientStateUpdateRequest
from napari_cuda.server.control.state_reducers import reduce_dims_update

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


async def handle_dims_update(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    key = ctx.key
    target = ctx.target

    if not target:
        await ctx.reject(
            "state.invalid",
            "dims axis target required",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    step_delta: Optional[int] = None
    value_arg_int: Optional[int] = None
    value_arg_float: Optional[float] = None

    if key == 'index':
        if isinstance(ctx.value, Integral):
            value_arg_int = int(ctx.value)
        else:
            logger.debug("state.update dims ignored (non-integer index) axis=%s value=%r", target, ctx.value)
            await ctx.reject(
                "state.invalid",
                "index must be integer",
                details={"scope": ctx.scope, "key": key, "target": target},
            )
            return True
    elif key == 'step':
        if isinstance(ctx.value, Integral):
            step_delta = int(ctx.value)
        else:
            logger.debug("state.update dims ignored (non-integer step delta) axis=%s value=%r", target, ctx.value)
            await ctx.reject(
                "state.invalid",
                "step delta must be integer",
                details={"scope": ctx.scope, "key": key, "target": target},
            )
            return True
    elif key in ('margin_left', 'margin_right'):
        try:
            value_arg_float = float(ctx.value)
        except Exception:
            logger.debug("state.update dims ignored (non-float margin) axis=%s value=%r", target, ctx.value)
            await ctx.reject(
                "state.invalid",
                "margin must be float",
                details={"scope": ctx.scope, "key": key, "target": target},
            )
            return True
    else:
        logger.debug("state.update dims ignored (unsupported key) axis=%s key=%s", target, key)
        await ctx.reject(
            "state.invalid",
            f"unsupported dims key {key}",
            details={"scope": ctx.scope, "key": key, "target": target},
        )
        return True

    metadata: dict[str, Any] = {}
    if step_delta is not None:
        metadata["step_delta"] = step_delta
    if value_arg_int is not None:
        metadata["value"] = value_arg_int
    if value_arg_float is not None:
        metadata["value"] = value_arg_float

    request = ClientStateUpdateRequest(
        scope="dims",
        target=str(target),
        key=str(key),
        value=value_arg_int if value_arg_int is not None else value_arg_float,
        intent_id=ctx.intent_id,
        timestamp=ctx.timestamp,
        metadata=metadata or None,
    )

    try:
        if key in ('margin_left', 'margin_right'):
            from napari_cuda.server.control.state_reducers import reduce_dims_margins_update
            result = reduce_dims_margins_update(
                server._state_ledger,
                axis=request.target,
                side=key,
                value=float(request.value),
                intent_id=request.intent_id,
                timestamp=request.timestamp,
                origin='client.state.dims',
            )
        else:
            result = reduce_dims_update(
                server._state_ledger,
                axis=request.target,
                prop=request.key,
                value=request.value,
                step_delta=metadata.get("step_delta"),
                intent_id=request.intent_id,
                timestamp=request.timestamp,
                origin='client.state.dims',
            )
    except Exception:
        logger.exception("state.update dims failed axis=%s key=%s", target, key)
        await ctx.reject(
            "state.error",
            "dims update failed",
            details={"scope": ctx.scope, "key": key, "target": target},
        )
        return True

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "state.update dims applied: axis=%s key=%s step_delta=%s value=%s current_step=%s",
            target,
            key,
            step_delta,
            value_arg,
            result.current_step,
        )

    assert result.version is not None, "dims reducer must supply version"
    await ctx.ack(applied_value=result.value, version=int(result.version))
    return True
