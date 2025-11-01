"""State update handlers for camera scope."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, TYPE_CHECKING

from napari_cuda.server.control.state_reducers import reduce_camera_update
from napari_cuda.server.control.topics.camera import broadcast_camera_update
from napari_cuda.server.scene import CameraDeltaCommand

if TYPE_CHECKING:
    from napari_cuda.server.control.control_channel_server import StateUpdateContext

logger = logging.getLogger("napari_cuda.server.control.control_channel_server")


def _camera_target_or_none(ctx: "StateUpdateContext") -> Optional[str]:
    cam_target = (ctx.target or "").lower()
    if cam_target not in {"", "main"}:
        return None
    return cam_target or "main"


def _camera_logger(server: Any) -> Callable[[str, Any], None]:
    info_enabled = bool(getattr(server, "_log_cam_info", False))
    debug_enabled = bool(getattr(server, "_log_cam_debug", False))

    def _log(fmt: str, *args: Any) -> None:
        if info_enabled:
            logger.info(fmt, *args)
        elif debug_enabled:
            logger.debug(fmt, *args)

    return _log


def _camera_metrics(server: Any) -> Any:
    return server.metrics if hasattr(server, "metrics") else None


def _float_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> Optional[float]:
    result = _float_value(value)
    if result is None or result <= 0:
        return None
    return result


def _float_pair(value: Any) -> Optional[tuple[float, float]]:
    if value is None or not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    first = _float_value(value[0])
    second = _float_value(value[1])
    if first is None or second is None:
        return None
    return float(first), float(second)


def _float_triplet(value: Any) -> Optional[tuple[float, float, float]]:
    if value is None or not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    x = _float_value(value[0])
    y = _float_value(value[1])
    z = _float_value(value[2])
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def _float_sequence(value: Any) -> Optional[tuple[float, ...]]:
    if value is None or not isinstance(value, (tuple, list)):
        return None
    components: list[float] = []
    for item in value:
        component = _float_value(item)
        if component is None:
            return None
        components.append(float(component))
    return tuple(components)


def _float_rect(value: Any) -> Optional[tuple[float, float, float, float]]:
    if value is None or not isinstance(value, (tuple, list)) or len(value) < 4:
        return None
    left = _float_value(value[0])
    bottom = _float_value(value[1])
    width = _float_value(value[2])
    height = _float_value(value[3])
    if left is None or bottom is None or width is None or height is None:
        return None
    return float(left), float(bottom), float(width), float(height)


async def handle_camera_zoom(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await ctx.reject(
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await ctx.reject(
            "state.invalid",
            "camera.zoom requires mapping payload",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    factor = _positive_float(payload.get("factor"))
    anchor = _float_pair(payload.get("anchor_px"))
    if factor is None or anchor is None:
        await ctx.reject(
            "state.invalid",
            "zoom payload requires factor and anchor_px",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    ack_value = {"factor": factor, "anchor_px": [anchor[0], anchor[1]]}

    log_camera = _camera_logger(server)
    metrics = _camera_metrics(server)

    log_camera("state: camera.zoom factor=%.4f anchor=(%.1f,%.1f)", factor, anchor[0], anchor[1])
    if metrics is not None:
        metrics.inc('napari_cuda_state_camera_updates')

    seq = server._next_camera_command_seq(target)
    await ctx.ack(applied_value=ack_value, version=seq)
    server._enqueue_camera_delta(
        CameraDeltaCommand(
            kind='zoom',
            target=target,
            command_seq=int(seq),
            factor=factor,
            anchor_px=anchor,
        ),
    )

    server._schedule_coro(
        broadcast_camera_update(
            server,
            mode='zoom',
            delta=ack_value,
            intent_id=ctx.intent_id,
            origin='state.update',
        ),
        'state-camera-zoom',
    )
    return True


async def handle_camera_pan(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await ctx.reject(
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await ctx.reject(
            "state.invalid",
            "camera.pan requires mapping payload",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    dx_value = _float_value(payload.get('dx_px'))
    dy_value = _float_value(payload.get('dy_px'))
    if dx_value is None or dy_value is None:
        await ctx.reject(
            "state.invalid",
            "pan payload requires dx_px and dy_px",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    dx = float(dx_value)
    dy = float(dy_value)
    ack_value = {'dx_px': dx, 'dy_px': dy}
    seq = server._next_camera_command_seq(target)
    await ctx.ack(applied_value=ack_value, version=seq)
    if dx != 0.0 or dy != 0.0:
        log_camera = _camera_logger(server)
        log_camera("state: camera.pan dx=%.1f dy=%.1f", dx, dy)

        metrics = _camera_metrics(server)
        if metrics is not None:
            metrics.inc('napari_cuda_state_camera_updates')

        server._enqueue_camera_delta(
            CameraDeltaCommand(
                kind='pan',
                target=target,
                command_seq=int(seq),
                dx_px=dx,
                dy_px=dy,
            ),
        )

        server._schedule_coro(
            broadcast_camera_update(
                server,
                mode='pan',
                delta=ack_value,
                intent_id=ctx.intent_id,
                origin='state.update',
            ),
            'state-camera-pan',
        )
    return True


async def handle_camera_orbit(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await ctx.reject(
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await ctx.reject(
            "state.invalid",
            "camera.orbit requires mapping payload",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    d_az = _float_value(payload.get('d_az_deg'))
    d_el = _float_value(payload.get('d_el_deg'))
    if d_az is None or d_el is None:
        await ctx.reject(
            "state.invalid",
            "orbit payload requires d_az_deg and d_el_deg",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    metrics = _camera_metrics(server)
    if metrics is not None:
        metrics.inc('napari_cuda_state_camera_updates')

    ack_value = {'d_az_deg': d_az, 'd_el_deg': d_el}
    seq = server._next_camera_command_seq(target)
    await ctx.ack(applied_value=ack_value, version=seq)
    server._enqueue_camera_delta(
        CameraDeltaCommand(
            kind='orbit',
            target=target,
            command_seq=int(seq),
            d_az_deg=float(d_az),
            d_el_deg=float(d_el),
        ),
    )

    log_camera = _camera_logger(server)
    log_camera("state: camera.orbit d_az=%.1f d_el=%.1f", d_az, d_el)

    server._schedule_coro(
        broadcast_camera_update(
            server,
            mode='orbit',
            delta=ack_value,
            intent_id=ctx.intent_id,
            origin='state.update',
        ),
        'state-camera-orbit',
    )
    return True


async def handle_camera_reset(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await ctx.reject(
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    reason_value = None
    if isinstance(ctx.value, Mapping):
        raw = ctx.value.get('reason')
        if isinstance(raw, str) and raw.strip():
            reason_value = str(raw)
    elif isinstance(ctx.value, str) and ctx.value.strip():
        reason_value = str(ctx.value)
    reason = reason_value if reason_value is not None else 'state.update'

    metrics = _camera_metrics(server)
    if metrics is not None:
        metrics.inc('napari_cuda_state_camera_updates')

    seq = server._next_camera_command_seq(target)
    await ctx.ack(applied_value={'reason': reason}, version=seq)
    server._enqueue_camera_delta(
        CameraDeltaCommand(kind='reset', target=target, command_seq=int(seq)),
    )

    runtime = getattr(server, "runtime", None)
    if getattr(server, "_idr_on_reset", False) and runtime is not None and runtime.is_ready:
        server._schedule_coro(server._ensure_keyframe(), 'state-camera-reset-keyframe')

    server._schedule_coro(
        broadcast_camera_update(
            server,
            mode='reset',
            delta={'reason': reason},
            intent_id=ctx.intent_id,
            origin='state.update',
        ),
        'state-camera-reset',
    )
    return True


async def handle_camera_set(ctx: "StateUpdateContext") -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await ctx.reject(
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await ctx.reject(
            "state.invalid",
            "camera.set requires mapping payload",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    center_tuple = _float_sequence(payload.get('center'))
    zoom_float = _float_value(payload.get('zoom'))
    angles_tuple = _float_triplet(payload.get('angles'))
    rect_tuple = _float_rect(payload.get('rect'))

    if (
        center_tuple is None
        and zoom_float is None
        and angles_tuple is None
        and rect_tuple is None
    ):
        await ctx.reject(
            "state.invalid",
            "camera.set payload must include center/zoom/angles/rect",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    ack_components, ack_version = reduce_camera_update(
        server._state_ledger,
        center=center_tuple,
        zoom=zoom_float,
        angles=angles_tuple,
        rect=rect_tuple,
        timestamp=ctx.timestamp,
        origin='client.state.camera',
    )

    await ctx.ack(applied_value=ack_components, version=ack_version)

    log_camera = _camera_logger(server)
    log_camera(
        "state: camera.set center=%s zoom=%s angles=%s",
        ack_components.get('center'),
        ack_components.get('zoom'),
        ack_components.get('angles'),
    )

    metrics = _camera_metrics(server)
    if metrics is not None:
        metrics.inc('napari_cuda_state_camera_updates')

    server._schedule_coro(
        broadcast_camera_update(
            server,
            mode='set',
            delta=ack_components,
            intent_id=ctx.intent_id,
            origin='state.update',
        ),
        'state-camera-set',
    )
    return True
