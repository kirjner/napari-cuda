"""State-channel orchestration helpers for `EGLHeadlessServer`."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid
from contextlib import suppress
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from numbers import Integral

# Protocol builders & dataclasses
from napari_cuda.protocol import (
    EnvelopeParser,
    FeatureToggle,
    PROTO_VERSION,
    NOTIFY_CAMERA_TYPE,
    NOTIFY_DIMS_TYPE,
    NOTIFY_ERROR_TYPE,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    NOTIFY_TELEMETRY_TYPE,
    SESSION_ACK_TYPE,
    SESSION_GOODBYE_TYPE,
    SESSION_HELLO_TYPE,
    SessionHello,
    SessionReject,
    SessionWelcome,
    CallCommand,
    ResumableTopicSequencer,
    build_ack_state,
    build_notify_camera,
    build_notify_dims,
    build_notify_error,
    build_notify_layers_delta,
    build_notify_scene_snapshot,
    build_notify_stream,
    build_notify_telemetry,
    build_error_command,
    build_reply_command,
    build_session_ack,
    build_session_goodbye,
    build_session_heartbeat,
    build_session_reject,
    build_session_welcome,
)
from napari_cuda.protocol.messages import NotifyDimsPayload, NotifyLayersPayload, NotifyScenePayload, NotifyStreamPayload
from napari_cuda.protocol.messages import STATE_UPDATE_TYPE
from napari_cuda.server.control.state_models import ClientStateUpdateRequest
from napari_cuda.server.control.state_reducers import (
    clamp_level,
    clamp_opacity,
    clamp_sample_step,
    is_valid_render_mode,
    normalize_clim,
    reduce_camera_update,
    reduce_dims_update,
    reduce_layer_property,
    reduce_level_update,
    reduce_plane_restore,
    reduce_volume_restore,
    reduce_volume_colormap,
    reduce_volume_contrast_limits,
    reduce_volume_opacity,
    reduce_volume_render_mode,
    reduce_volume_sample_step,
    reduce_view_update,
)
from napari_cuda.server.control.transactions.plane_restore import (
    apply_plane_restore_transaction,
)
from napari_cuda.server.scene import (
    CameraDeltaCommand,
    PlaneState,
    RenderMode,
    RenderUpdate,
    VolumeState,
    snapshot_render_state,
    snapshot_multiscale_state,
)
from napari_cuda.server.control.control_payload_builder import (
    build_notify_layers_delta_payload,
    build_notify_layers_payload,
    build_notify_scene_payload,
)
from napari_cuda.protocol.snapshots import LayerDelta, SceneSnapshot

from napari_cuda.server.control.resumable_history_store import (
    EnvelopeSnapshot,
    ResumableHistoryStore,
    ResumeDecision,
    ResumePlan,
)
from napari_cuda.server.control.command_registry import (
    COMMAND_REGISTRY,
    CommandHandler,
    register_command,
)

logger = logging.getLogger(__name__)

_HANDSHAKE_TIMEOUT_S = 5.0
_REQUIRED_NOTIFY_FEATURES = ("notify.scene", "notify.layers", "notify.stream")
_COMMAND_KEYFRAME = "napari.pixel.request_keyframe"
_COMMAND_LISTDIR = "fs.listdir"
_COMMAND_ZARR_LOAD = "napari.zarr.load"


@dataclass(slots=True)
class CommandResult:
    result: Any | None = None
    idempotency_key: str | None = None


class CommandRejected(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details) if details else None
        self.idempotency_key = idempotency_key


_RESUMABLE_TOPICS = (NOTIFY_SCENE_TYPE, NOTIFY_LAYERS_TYPE, NOTIFY_STREAM_TYPE)
_ENVELOPE_PARSER = EnvelopeParser()

async def _command_request_keyframe(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    if not hasattr(server, "_ensure_keyframe"):
        raise CommandRejected(
            code="command.forbidden",
            message="Keyframe request not supported",
        )
    await server._ensure_keyframe()
    return CommandResult()


register_command(_COMMAND_KEYFRAME, _command_request_keyframe)


async def _command_fs_listdir(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    kwargs = dict(frame.payload.kwargs or {})
    path = kwargs.get("path")
    show_hidden = bool(kwargs.get("show_hidden", False))
    only = kwargs.get("only")
    filters: Optional[Sequence[str]] = None
    if only is not None:
        if isinstance(only, (list, tuple)):
            filters = tuple(str(item) for item in only if item is not None)
        else:
            raise CommandRejected(
                code="command.invalid",
                message="'only' filter must be a list of suffixes",
            )
    if only is None:
        filters = (".zarr",)
    try:
        listing = server._list_directory(
            path,
            only=filters,
            show_hidden=show_hidden,
        )
    except RuntimeError as exc:
        raise CommandRejected(code="fs.forbidden", message=str(exc)) from exc
    except FileNotFoundError:
        raise CommandRejected(
            code="fs.not_found",
            message="Directory not found",
            details={"path": path},
        )
    except NotADirectoryError:
        raise CommandRejected(
            code="fs.not_found",
            message="Path is not a directory",
            details={"path": path},
        )
    return CommandResult(result=listing)


async def _command_zarr_load(server: Any, frame: CallCommand, ws: Any) -> CommandResult:
    kwargs = dict(frame.payload.kwargs or {})
    path = kwargs.get("path")
    if not isinstance(path, str) or not path:
        raise CommandRejected(
            code="command.invalid",
            message="'path' must be a non-empty string",
        )
    try:
        await server._handle_zarr_load(path)
    except RuntimeError as exc:
        raise CommandRejected(code="fs.forbidden", message=str(exc)) from exc
    except FileNotFoundError:
        raise CommandRejected(
            code="fs.not_found",
            message="Dataset not found",
            details={"path": path},
        )
    except NotADirectoryError:
        raise CommandRejected(
            code="fs.not_found",
            message="Dataset path is not a directory",
            details={"path": path},
        )
    except ValueError as exc:
        raise CommandRejected(
            code="zarr.invalid",
            message=str(exc),
            details={"path": path},
        ) from exc
    return CommandResult(result={"ok": True})


register_command(_COMMAND_LISTDIR, _command_fs_listdir)
register_command(_COMMAND_ZARR_LOAD, _command_zarr_load)


_SERVER_FEATURES: dict[str, FeatureToggle] = {
    "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
    "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
    "notify.camera": FeatureToggle(enabled=True, version=1, resume=False),
    "notify.telemetry": FeatureToggle(enabled=False),
    "call.command": FeatureToggle(
        enabled=True,
        version=1,
        resume=False,
    ),
}


@dataclass(slots=True)
class StateUpdateContext:
    """Envelope metadata shared across state.update handlers."""

    server: Any
    ws: Any
    scope: str
    key: str
    target: str
    value: Any
    session_id: str
    frame_id: str
    intent_id: str
    timestamp: float | None


async def _state_update_reject(
    ctx: StateUpdateContext,
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    error_payload: Dict[str, Any] = {"code": code, "message": message}
    if details:
        error_payload["details"] = dict(details)
    await _send_state_ack(
        ctx.server,
        ctx.ws,
        session_id=ctx.session_id,
        intent_id=ctx.intent_id,
        in_reply_to=ctx.frame_id,
        status="rejected",
        error=error_payload,
    )
async def _state_update_ack(
    ctx: StateUpdateContext,
    *,
    applied_value: Any,
    version: int,
) -> None:
    await _send_state_ack(
        ctx.server,
        ctx.ws,
        session_id=ctx.session_id,
        intent_id=ctx.intent_id,
        in_reply_to=ctx.frame_id,
        status="accepted",
        applied_value=applied_value,
        version=int(version),
    )


def _camera_target_or_none(ctx: StateUpdateContext) -> Optional[str]:
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


def _apply_plane_restore_from_ledger(
    server: Any,
    *,
    timestamp: float | None,
    intent_id: str,
) -> bool:
    ledger = server._state_ledger
    with server._state_lock:
        lvl_entry = ledger.get("view_cache", "plane", "level")
        step_entry = ledger.get("view_cache", "plane", "step")
        plane_entry = ledger.get("viewport", "plane", "state")

    if plane_entry is None or not isinstance(plane_entry.value, Mapping):
        logger.warning("plane_restore skipped due to missing viewport plane state")
        return False

    plane_state = PlaneState(**dict(plane_entry.value))  # type: ignore[arg-type]

    level_value: Optional[int] = None
    if lvl_entry is not None and isinstance(lvl_entry.value, int):
        level_value = int(lvl_entry.value)
    if level_value is None:
        level_value = plane_state.applied_level or plane_state.target_level

    step_value: Optional[tuple[int, ...]] = None
    if step_entry is not None and isinstance(step_entry.value, (tuple, list)):
        step_value = tuple(int(v) for v in step_entry.value)
    if step_value is None:
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
        missing.append("view_cache.plane.level")
    if step_value is None:
        missing.append("view_cache.plane.step")
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

    def _as_rect(value: Any) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in value[:4])  # type: ignore[return-value]

    step_tuple = _as_step(step_value)
    center_tuple = _as_center(center_value)
    zoom_float = float(zoom_value)
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
        level_entry = ledger.get("multiscale", "main", "level")
        center_entry = ledger.get("camera_volume", "main", "center")
        angles_entry = ledger.get("camera_volume", "main", "angles")
        distance_entry = ledger.get("camera_volume", "main", "distance")
        fov_entry = ledger.get("camera_volume", "main", "fov")

    if volume_entry is None or not isinstance(volume_entry.value, Mapping):
        logger.warning("volume_restore skipped due to missing viewport volume state")
        return None

    volume_state = VolumeState(**dict(volume_entry.value))  # type: ignore[arg-type]

    level_value: Optional[int] = volume_state.level
    if level_value is None and level_entry is not None and isinstance(level_entry.value, int):
        level_value = int(level_entry.value)

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


async def _handle_view_ndisplay(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    try:
        raw_value = int(ctx.value) if ctx.value is not None else 2
    except Exception:
        logger.debug("state.update view ignored (non-integer ndisplay) value=%r", ctx.value)
        await _state_update_reject(ctx, "state.invalid", "ndisplay must be integer")
        return True

    ndisplay = 3 if int(raw_value) >= 3 else 2
    prev_entry = server._state_ledger.get("view", "main", "ndisplay")
    prev_ndisplay = int(prev_entry.value) if prev_entry is not None and isinstance(prev_entry.value, int) else (
        3 if server._initial_mode is RenderMode.VOLUME else 2
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
        await _state_update_reject(ctx, "state.error", "failed to apply ndisplay")
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

    await _state_update_ack(ctx, applied_value=new_ndisplay, version=int(result.version))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "state.update view intent=%s frame=%s ndisplay=%s accepted",
            ctx.intent_id,
            ctx.frame_id,
            new_ndisplay,
        )

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


async def _handle_camera_zoom(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            "camera.zoom requires mapping payload",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    factor = _positive_float(payload.get("factor"))
    anchor = _float_pair(payload.get("anchor_px"))
    if factor is None or anchor is None:
        await _state_update_reject(
            ctx,
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
    await _state_update_ack(ctx, applied_value=ack_value, version=seq)
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
        _broadcast_camera_update(
            server,
            mode='zoom',
            delta=ack_value,
            intent_id=ctx.intent_id,
            origin='state.update',
        ),
        'state-camera-zoom',
    )
    return True


async def _handle_camera_pan(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            "camera.pan requires mapping payload",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    dx = _float_value(payload.get('dx_px'))
    dy = _float_value(payload.get('dy_px'))
    if dx is None or dy is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            "pan payload must provide numeric dx_px and dy_px",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    ack_value = {"dx_px": float(dx), "dy_px": float(dy)}

    seq = server._next_camera_command_seq(target)
    await _state_update_ack(ctx, applied_value=ack_value, version=seq)
    if dx != 0.0 or dy != 0.0:
        _camera_logger(server)("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
        metrics = _camera_metrics(server)
        if metrics is not None:
            metrics.inc('napari_cuda_state_camera_updates')
        server._enqueue_camera_delta(
            CameraDeltaCommand(
                kind='pan',
                target=target,
                command_seq=int(seq),
                dx_px=float(dx),
                dy_px=float(dy),
            ),
        )

        server._schedule_coro(
            _broadcast_camera_update(
                server,
                mode='pan',
                delta=ack_value,
                intent_id=ctx.intent_id,
                origin='state.update',
            ),
            'state-camera-pan',
        )
    return True


async def _handle_camera_orbit(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            "camera.orbit requires mapping payload",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    d_az = _float_value(payload.get('d_az_deg'))
    d_el = _float_value(payload.get('d_el_deg'))
    if d_az is None or d_el is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            "orbit payload must provide numeric d_az_deg and d_el_deg",
            details={"scope": ctx.scope, "key": ctx.key},
        )
        return True

    ack_value = {"d_az_deg": float(d_az), "d_el_deg": float(d_el)}

    seq = server._next_camera_command_seq(target)
    await _state_update_ack(ctx, applied_value=ack_value, version=seq)
    if d_az != 0.0 or d_el != 0.0:
        _camera_logger(server)("state: camera.orbit daz=%.2f del=%.2f", d_az, d_el)
        metrics = _camera_metrics(server)
        if metrics is not None:
            metrics.inc('napari_cuda_state_camera_updates')
        server._enqueue_camera_delta(
            CameraDeltaCommand(
                kind='orbit',
                target=target,
                command_seq=int(seq),
                d_az_deg=float(d_az),
                d_el_deg=float(d_el),
            ),
        )

        server._schedule_coro(
            _broadcast_camera_update(
                server,
                mode='orbit',
                delta=ack_value,
                intent_id=ctx.intent_id,
                origin='state.update',
            ),
            'state-camera-orbit',
        )
    return True


async def _handle_camera_reset(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await _state_update_reject(
            ctx,
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
    await _state_update_ack(ctx, applied_value={'reason': reason}, version=seq)
    server._enqueue_camera_delta(
        CameraDeltaCommand(kind='reset', target=target, command_seq=int(seq)),
    )

    runtime = getattr(server, "runtime", None)
    if getattr(server, "_idr_on_reset", False) and runtime is not None and runtime.is_ready:
        server._schedule_coro(server._ensure_keyframe(), 'state-camera-reset-keyframe')

    server._schedule_coro(
        _broadcast_camera_update(
            server,
            mode='reset',
            delta={'reason': reason},
            intent_id=ctx.intent_id,
            origin='state.update',
        ),
        'state-camera-reset',
    )
    return True


async def _handle_camera_set(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    target = _camera_target_or_none(ctx)
    if target is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            f"unsupported camera target {ctx.target}",
            details={"scope": ctx.scope, "target": ctx.target},
        )
        return True

    payload = ctx.value if isinstance(ctx.value, Mapping) else None
    if payload is None:
        await _state_update_reject(
            ctx,
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
        await _state_update_reject(
            ctx,
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

    await _state_update_ack(ctx, applied_value=ack_components, version=ack_version)

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
        _broadcast_camera_update(
            server,
            mode='set',
            delta=ack_components,
            intent_id=ctx.intent_id,
            origin='state.update',
        ),
        'state-camera-set',
    )
    return True


async def _handle_layer_property(ctx: StateUpdateContext) -> bool:
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
        await _state_update_reject(
            ctx,
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
        await _state_update_reject(
            ctx,
            "state.error",
            "layer update failed",
            details={"scope": ctx.scope, "key": ctx.key, "target": layer_id},
        )
        return True

    assert result.version is not None, "layer reducer must supply version"
    await _state_update_ack(ctx, applied_value=result.value, version=int(result.version))

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


async def _handle_volume_update(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    key = ctx.key
    value = ctx.value

    if key == 'render_mode':
        mode = str(value or '').lower().strip()
        if not is_valid_render_mode(mode, server._allowed_render_modes):
            logger.debug("state.update volume ignored invalid render_mode=%r", value)
            await _state_update_reject(
                ctx,
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
            await _state_update_reject(
                ctx,
                "state.invalid",
                "contrast_limits requires [lo, hi]",
                details={"scope": ctx.scope, "key": key},
            )
            return True
        try:
            lo, hi = normalize_clim(pair[0], pair[1])
        except (TypeError, ValueError):
            logger.debug("state.update volume ignored invalid clim=%r", pair)
            await _state_update_reject(
                ctx,
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
            await _state_update_reject(
                ctx,
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
            await _state_update_reject(
                ctx,
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
            await _state_update_reject(
                ctx,
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
        await _state_update_reject(
            ctx,
            "state.invalid",
            f"unsupported volume key {key}",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    assert result.version is not None, "volume reducer must supply version"
    await _state_update_ack(ctx, applied_value=result.value, version=int(result.version))

    logger.debug(
        "state.update volume intent=%s frame=%s key=%s value=%s",
        ctx.intent_id,
        ctx.frame_id,
        key,
        result.value,
    )
    return True


async def _handle_multiscale_level(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    key = ctx.key

    if key == 'policy':
        logger.debug("state.update multiscale policy ignored (unsupported)")
        await _state_update_reject(
            ctx,
            "state.unsupported",
            "multiscale policy is no longer configurable",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    if key != 'level':
        logger.debug("state.update multiscale ignored key=%s", key)
        await _state_update_reject(
            ctx,
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
        await _state_update_reject(
            ctx,
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
        await _state_update_reject(
            ctx,
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
    await _state_update_ack(ctx, applied_value=result.value, version=int(result.version))
    return True


async def _handle_dims_update(ctx: StateUpdateContext) -> bool:
    server = ctx.server
    key = ctx.key
    target = ctx.target

    if not target:
        await _state_update_reject(
            ctx,
            "state.invalid",
            "dims axis target required",
            details={"scope": ctx.scope, "key": key},
        )
        return True

    step_delta: Optional[int] = None
    value_arg: Optional[int] = None

    if key == 'index':
        if isinstance(ctx.value, Integral):
            value_arg = int(ctx.value)
        else:
            logger.debug("state.update dims ignored (non-integer index) axis=%s value=%r", target, ctx.value)
            await _state_update_reject(
                ctx,
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
            await _state_update_reject(
                ctx,
                "state.invalid",
                "step delta must be integer",
                details={"scope": ctx.scope, "key": key, "target": target},
            )
            return True
    else:
        logger.debug("state.update dims ignored (unsupported key) axis=%s key=%s", target, key)
        await _state_update_reject(
            ctx,
            "state.invalid",
            f"unsupported dims key {key}",
            details={"scope": ctx.scope, "key": key, "target": target},
        )
        return True

    metadata: Dict[str, Any] = {}
    if step_delta is not None:
        metadata["step_delta"] = step_delta
    if value_arg is not None:
        metadata["value"] = value_arg

    request = ClientStateUpdateRequest(
        scope="dims",
        target=str(target),
        key=str(key),
        value=value_arg,
        intent_id=ctx.intent_id,
        timestamp=ctx.timestamp,
        metadata=metadata or None,
    )

    try:
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
        await _state_update_reject(
            ctx,
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
    await _state_update_ack(ctx, applied_value=result.value, version=int(result.version))
    return True
async def _send_command_error(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    in_reply_to: str,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "in_reply_to": in_reply_to,
        "status": "error",
        "code": str(code),
        "message": str(message),
    }
    if details:
        payload["details"] = dict(details)
    if idempotency_key is not None:
        payload["idempotency_key"] = str(idempotency_key)
    frame = build_error_command(
        session_id=session_id,
        frame_id=None,
        payload=payload,
        timestamp=time.time(),
    )
    await _send_frame(server, ws, frame)


async def _ingest_call_command(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    session_id = _state_session(ws)
    if not session_id:
        logger.debug("call.command ignored: missing session id")
        return True

    try:
        frame = _ENVELOPE_PARSER.parse_call_command(data)
    except (ValueError, TypeError) as exc:
        logger.debug("call.command parse failed", exc_info=True)
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=str(data.get("frame_id") or "unknown"),
            code="command.forbidden",
            message=str(exc) or "invalid call.command payload",
        )
        return True

    envelope = frame.envelope
    payload = frame.payload
    in_reply_to = envelope.frame_id or "unknown"

    if not _feature_enabled(ws, "call.command"):
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code="command.forbidden",
            message="call.command feature disabled",
        )
        return True

    handler = COMMAND_REGISTRY.get_handler(payload.command)
    if handler is None:
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code="command.not_found",
            message=f"unknown command '{payload.command}'",
        )
        return True

    try:
        result = await handler(server, frame, ws)
    except CommandRejected as exc:
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            idempotency_key=exc.idempotency_key,
        )
        return True
    except Exception:
        logger.exception("command handler failed", extra={"command": payload.command})
        await _send_command_error(
            server,
            ws,
            session_id=session_id,
            in_reply_to=in_reply_to,
            code="command.retryable",
            message="command handler raised an exception",
        )
        return True

    response_payload: Dict[str, Any] = {
        "in_reply_to": in_reply_to,
        "status": "ok",
    }
    if result.result is not None:
        response_payload["result"] = result.result
    if result.idempotency_key is not None:
        response_payload["idempotency_key"] = result.idempotency_key

    frame_out = build_reply_command(
        session_id=session_id,
        frame_id=None,
        payload=response_payload,
        timestamp=time.time(),
    )
    await _send_frame(server, ws, frame_out)
    return True

StateMessageHandler = Callable[[Any, Mapping[str, Any], Any], Awaitable[bool]]


def _state_session(ws: Any) -> Optional[str]:
    return getattr(ws, "_napari_cuda_session", None)


def _history_store(server: Any) -> Optional[ResumableHistoryStore]:
    store = getattr(server, "_resumable_store", None)
    if isinstance(store, ResumableHistoryStore):
        return store
    return None


def _ensure_scene_snapshot(server: Any) -> SceneSnapshot:
    refresher = getattr(server, "_refresh_scene_snapshot", None)
    if callable(refresher):
        refresher()
    snapshot = getattr(server, "_scene_snapshot", None)
    assert isinstance(snapshot, SceneSnapshot), "scene snapshot unavailable"
    return snapshot


def _mark_client_activity(ws: Any) -> None:
    now = time.time()
    setattr(ws, "_napari_cuda_last_activity", now)
    setattr(ws, "_napari_cuda_waiting_ack", False)
    setattr(ws, "_napari_cuda_missed_heartbeats", 0)


def _heartbeat_shutting_down(ws: Any) -> bool:
    return bool(getattr(ws, "_napari_cuda_shutdown", False))


def _flag_heartbeat_shutdown(ws: Any) -> None:
    setattr(ws, "_napari_cuda_shutdown", True)


async def _send_session_goodbye(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    code: str,
    message: str,
    reason: Optional[str] = None,
) -> None:
    if getattr(ws, "_napari_cuda_goodbye_sent", False):
        return
    frame = build_session_goodbye(
        session_id=session_id,
        code=code,
        message=message,
        reason=reason,
        timestamp=time.time(),
    )
    await _send_frame(server, ws, frame)
    setattr(ws, "_napari_cuda_goodbye_sent", True)


async def _state_heartbeat_loop(server: Any, ws: Any) -> None:
    try:
        interval = float(getattr(ws, "_napari_cuda_heartbeat_interval", 0.0))
    except Exception:
        interval = 0.0
    if interval <= 0.0:
        return

    log_debug = logger.debug
    session_id = _state_session(ws)
    missed = int(getattr(ws, "_napari_cuda_missed_heartbeats", 0))

    try:
        while True:
            await asyncio.sleep(interval)
            if ws.state is not State.OPEN or _heartbeat_shutting_down(ws):
                break
            session_id = _state_session(ws)
            if not session_id:
                break

            waiting_ack = bool(getattr(ws, "_napari_cuda_waiting_ack", False))
            if waiting_ack:
                missed += 1
            else:
                missed = 0
            setattr(ws, "_napari_cuda_missed_heartbeats", missed)

            if missed >= 2:
                logger.warning(
                    "heartbeat timeout for session %s; closing connection",
                    session_id,
                )
                with suppress(Exception):
                    await _send_session_goodbye(
                        server,
                        ws,
                        session_id=session_id,
                        code="timeout",
                        message="heartbeat timeout",
                        reason="heartbeat_timeout",
                    )
                _flag_heartbeat_shutdown(ws)
                with suppress(Exception):
                    await ws.close(code=1011, reason="heartbeat timeout")
                break

            try:
                frame = build_session_heartbeat(
                    session_id=session_id,
                    timestamp=time.time(),
                )
                await _send_frame(server, ws, frame)
                setattr(ws, "_napari_cuda_waiting_ack", True)
                setattr(ws, "_napari_cuda_last_heartbeat", time.time())
            except Exception:
                log_debug("session.heartbeat send failed", exc_info=True)
                break
    except asyncio.CancelledError:
        pass



def _state_features(ws: Any) -> Mapping[str, FeatureToggle]:
    features = getattr(ws, "_napari_cuda_features", None)
    if isinstance(features, Mapping):
        return features
    return {}


def _feature_enabled(ws: Any, name: str) -> bool:
    toggle = _state_features(ws).get(name)
    return bool(getattr(toggle, "enabled", False))


def _state_sequencer(ws: Any, topic: str) -> ResumableTopicSequencer:
    sequencers = getattr(ws, "_napari_cuda_sequencers", None)
    if not isinstance(sequencers, dict):
        sequencers = {}
        setattr(ws, "_napari_cuda_sequencers", sequencers)
    sequencer = sequencers.get(topic)
    if not isinstance(sequencer, ResumableTopicSequencer):
        sequencer = ResumableTopicSequencer(topic=topic)
        sequencers[topic] = sequencer
    return sequencer


def _viewer_settings(server: Any) -> Dict[str, Any]:
    width = int(server.width)
    height = int(server.height)
    fps = float(server.cfg.fps)
    ndisplay_entry = server._state_ledger.get("view", "main", "ndisplay")
    assert (
        ndisplay_entry is not None and isinstance(ndisplay_entry.value, int)
    ), "viewer settings require ledger ndisplay entry"
    use_volume = int(ndisplay_entry.value) >= 3

    return {
        "fps_target": fps,
        "canvas_size": [width, height],
        "volume_enabled": use_volume,
    }

def _resolve_dims_mode_from_ndisplay(ndisplay: int) -> str:
    return "volume" if int(ndisplay) >= 3 else "plane"


async def _broadcast_layers_delta(
    server: Any,
    *,
    layer_id: Optional[str],
    changes: Mapping[str, Any],
    intent_id: Optional[str],
    timestamp: Optional[float],
    targets: Optional[Sequence[Any]] = None,
) -> None:
    if not changes:
        return

    delta = LayerDelta(layer_id=layer_id or "layer-0", changes=dict(changes))
    payload = delta.to_payload()
    clients = list(targets) if targets is not None else list(server._state_clients)
    now = time.time() if timestamp is None else float(timestamp)

    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None and targets is None:
        snapshot = store.delta_envelope(
            NOTIFY_LAYERS_TYPE,
            payload=payload.to_dict(),
            timestamp=now,
            intent_id=intent_id,
        )

    tasks: list[Awaitable[None]] = []
    for ws in clients:
        if not _feature_enabled(ws, "notify.layers"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        kwargs: Dict[str, Any] = {
            "session_id": session_id,
            "payload": payload,
            "timestamp": snapshot.timestamp if snapshot is not None else now,
            "intent_id": intent_id,
        }
        if snapshot is not None:
            kwargs["seq"] = snapshot.seq
            kwargs["delta_token"] = snapshot.delta_token
            kwargs["frame_id"] = snapshot.frame_id
        else:
            kwargs["sequencer"] = _state_sequencer(ws, NOTIFY_LAYERS_TYPE)
        frame = build_notify_layers_delta(**kwargs)
        tasks.append(_send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    if snapshot is not None:
        for ws in clients:
            sequencer = _state_sequencer(ws, NOTIFY_LAYERS_TYPE)
            sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def _broadcast_dims_state(
    server: Any,
    *,
    payload: NotifyDimsPayload,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    targets: Optional[Sequence[Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    if getattr(server, "_log_dims_info", False):
        logger.info(
            "notify.dims: step=%s order=%s displayed=%s axis_labels=%s level_shapes=%s current_level=%s",
            payload.current_step,
            payload.order,
            payload.displayed,
            payload.axis_labels,
            payload.level_shapes,
            payload.current_level,
        )

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not _feature_enabled(ws, "notify.dims"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        frame = build_notify_dims(
            session_id=session_id,
            payload=payload,
            timestamp=now,
            intent_id=intent_id,
        )
        tasks.append(_send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _normalize_camera_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_camera_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_camera_value(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _float_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _positive_float(value: Any) -> Optional[float]:
    result = _float_value(value)
    if result is None or result <= 0.0:
        return None
    return result


def _float_pair(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    first = _float_value(value[0])
    second = _float_value(value[1])
    if first is None or second is None:
        return None
    return float(first), float(second)


def _float_triplet(value: Any) -> Optional[tuple[float, float, float]]:
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    x = _float_value(value[0])
    y = _float_value(value[1])
    z = _float_value(value[2])
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def _float_rect(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, Sequence) or len(value) < 4:
        return None
    left = _float_value(value[0])
    bottom = _float_value(value[1])
    width = _float_value(value[2])
    height = _float_value(value[3])
    if None in (left, bottom, width, height):
        return None
    return float(left), float(bottom), float(width), float(height)


def _float_sequence(value: Any) -> Optional[tuple[float, ...]]:
    if not isinstance(value, Sequence) or not value:
        return None
    components: list[float] = []
    for item in value:
        component = _float_value(item)
        if component is None:
            return None
        components.append(component)
    return tuple(components)


def _string_value(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return str(value)
    return None


async def _broadcast_camera_update(
    server: Any,
    *,
    mode: str,
    delta: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    intent_id: Optional[str],
    origin: str,
    timestamp: Optional[float] = None,
    targets: Optional[Sequence[Any]] = None,
) -> None:
    clients = list(targets) if targets is not None else list(server._state_clients)
    if not clients:
        return

    payload: Dict[str, Any] = {
        "mode": str(mode),
        "origin": str(origin),
    }
    if delta is not None:
        payload["delta"] = _normalize_camera_value(delta)
    if state is not None:
        payload["state"] = _normalize_camera_value(state)
    if "delta" not in payload and "state" not in payload:
        return

    tasks: list[Awaitable[None]] = []
    now = time.time() if timestamp is None else float(timestamp)

    for ws in clients:
        if not _feature_enabled(ws, "notify.camera"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        frame = build_notify_camera(
            session_id=session_id,
            payload=payload,
            timestamp=now,
            intent_id=intent_id,
        )
        tasks.append(_send_frame(server, ws, frame))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_stream_frame(
    server: Any,
    ws: Any,
    *,
    payload: NotifyStreamPayload | Mapping[str, Any],
    timestamp: Optional[float],
    snapshot: EnvelopeSnapshot | None = None,
    force_snapshot: bool = False,
) -> EnvelopeSnapshot | None:
    session_id = _state_session(ws)
    if not session_id:
        return snapshot
    if not isinstance(payload, NotifyStreamPayload):
        payload = NotifyStreamPayload.from_dict(payload)
    now = time.time() if timestamp is None else float(timestamp)
    store = _history_store(server)
    if snapshot is None and store is not None:
        payload_dict = payload.to_dict()
        if force_snapshot:
            snapshot = store.snapshot_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )
        else:
            snapshot = store.delta_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )

    kwargs: Dict[str, Any] = {
        "session_id": session_id,
        "payload": payload,
        "timestamp": snapshot.timestamp if snapshot is not None else now,
    }
    sequencer = _state_sequencer(ws, NOTIFY_STREAM_TYPE)
    if snapshot is not None:
        kwargs["seq"] = snapshot.seq
        kwargs["delta_token"] = snapshot.delta_token
        kwargs["frame_id"] = snapshot.frame_id
        sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)
    else:
        if force_snapshot or sequencer.seq is None:
            cursor = sequencer.snapshot()
            kwargs["seq"] = cursor.seq
            kwargs["delta_token"] = cursor.delta_token
        else:
            kwargs["sequencer"] = sequencer
    frame = build_notify_stream(**kwargs)
    await _send_frame(server, ws, frame)
    return snapshot


async def _emit_scene_baseline(
    server: Any,
    ws: Any,
    *,
    payload: NotifyScenePayload | None,
    plan: ResumePlan | None,
    reason: str,
) -> None:
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    timestamp = time.time()

    if store is not None:
        snapshot = store.current_snapshot(NOTIFY_SCENE_TYPE)
        need_reset = plan is not None and plan.decision == ResumeDecision.RESET
        if snapshot is None or need_reset:
            if payload is None:
                scene_snapshot = _ensure_scene_snapshot(server)
                payload = build_notify_scene_payload(
                    scene_snapshot=scene_snapshot,
                    ledger_snapshot=server._state_ledger.snapshot(),
                    viewer_settings=_viewer_settings(server),
                )
            snapshot = store.snapshot_envelope(
                NOTIFY_SCENE_TYPE,
                payload=payload.to_dict(),
                timestamp=timestamp,
            )
            store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=timestamp)
            store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=timestamp)
            _state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
            _state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()
        await _send_scene_snapshot_from_cache(server, ws, snapshot)
    else:
        if payload is None:
            scene_snapshot = _ensure_scene_snapshot(server)
            payload = build_notify_scene_payload(
                scene_snapshot=scene_snapshot,
                ledger_snapshot=server._state_ledger.snapshot(),
                viewer_settings=_viewer_settings(server),
            )
        session_id = _state_session(ws)
        if not session_id:
            return
        sequencer = _state_sequencer(ws, NOTIFY_SCENE_TYPE)
        frame = build_notify_scene_snapshot(
            session_id=session_id,
            viewer=payload.viewer,
            layers=payload.layers,
            policies=payload.policies,
            metadata=payload.metadata,
            timestamp=timestamp,
            sequencer=sequencer,
        )
        await _send_frame(server, ws, frame)

    if server._log_dims_info:
        logger.info("%s: notify.scene sent", reason)
    else:
        logger.debug("%s: notify.scene sent", reason)

async def _emit_layer_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
    default_controls: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    store = _history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await _send_layer_snapshot(server, ws, snapshot)
            return
        # No deltas to replay; fall through to emit the current defaults.

    if not default_controls:
        return

    if store is not None:
        now = time.time()
        for layer_id, changes in default_controls:
            payload = build_notify_layers_payload(layer_id=layer_id or "layer-0", changes=changes)
            snapshot = store.delta_envelope(
                NOTIFY_LAYERS_TYPE,
                payload=payload.to_dict(),
                timestamp=now,
                intent_id=None,
            )
            await _send_layer_snapshot(server, ws, snapshot)
        return

    for layer_id, changes in default_controls:
        await _broadcast_layers_delta(
            server,
            layer_id=layer_id,
            changes=changes,
            intent_id=None,
            timestamp=time.time(),
            targets=[ws],
        )


async def _emit_stream_baseline(
    server: Any,
    ws: Any,
    *,
    plan: ResumePlan | None,
) -> None:
    store = _history_store(server)
    if store is not None and plan is not None and plan.decision == ResumeDecision.REPLAY:
        if plan.deltas:
            for snapshot in plan.deltas:
                await _send_stream_snapshot(server, ws, snapshot)
        return

    channel = getattr(server, "_pixel_channel", None)
    cfg = getattr(server, "_pixel_config", None)
    if channel is None or cfg is None:
        raise AssertionError("Pixel channel not initialized")

    avcc = channel.last_avcc
    if avcc is not None:
        stream_payload = server.build_stream_payload(avcc)
        await _send_stream_frame(
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
        raise AssertionError("dims mirror not initialized")
    payload = mirror.latest_payload()
    if payload is None:
        logger.debug("connect: notify.dims baseline skipped (metadata unavailable)")
        return
    await _broadcast_dims_state(server, payload=payload, intent_id=None, timestamp=time.time(), targets=[ws])
    if server._log_dims_info:
        logger.info("connect: notify.dims baseline sent")
    else:
        logger.debug("connect: notify.dims baseline sent")


async def broadcast_stream_config(
    server: Any,
    *,
    payload: NotifyStreamPayload,
    timestamp: Optional[float] = None,
) -> None:
    clients = list(server._state_clients)
    if not clients:
        store = _history_store(server)
        if store is not None:
            now = time.time() if timestamp is None else float(timestamp)
            if store.current_snapshot(NOTIFY_STREAM_TYPE) is None:
                store.snapshot_envelope(
                    NOTIFY_STREAM_TYPE,
                    payload=payload.to_dict(),
                    timestamp=now,
                )
            else:
                store.delta_envelope(
                    NOTIFY_STREAM_TYPE,
                    payload=payload.to_dict(),
                    timestamp=now,
                )
        return

    now = time.time() if timestamp is None else float(timestamp)
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    snapshot_mode = False
    if store is not None:
        payload_dict = payload.to_dict()
        snapshot_mode = store.current_snapshot(NOTIFY_STREAM_TYPE) is None
        if snapshot_mode:
            snapshot = store.snapshot_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )
        else:
            snapshot = store.delta_envelope(
                NOTIFY_STREAM_TYPE,
                payload=payload_dict,
                timestamp=now,
            )

    tasks: list[Awaitable[None]] = []
    for ws in clients:
        if not _feature_enabled(ws, "notify.stream"):
            continue
        session_id = _state_session(ws)
        if not session_id:
            continue
        tasks.append(
            _send_stream_frame(
                server,
                ws,
                payload=payload,
                timestamp=now,
                snapshot=snapshot,
                force_snapshot=snapshot_mode,
            )
        )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def ingest_state(server: Any, ws: Any) -> None:
    """Ingest a state-channel websocket connection."""

    heartbeat_task: asyncio.Task[Any] | None = None
    try:
        _disable_nagle(ws)
        handshake_ok = await _perform_state_handshake(server, ws)
        if not handshake_ok:
            return
        try:
            heartbeat_task = asyncio.create_task(_state_heartbeat_loop(server, ws))

            def _log_heartbeat_completion(task: asyncio.Task[Any]) -> None:
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.debug("heartbeat task completion probe failed", exc_info=True)
                    return
                if exc is not None:
                    logger.exception("Heartbeat task failed", exc_info=exc)

            heartbeat_task.add_done_callback(_log_heartbeat_completion)
            setattr(ws, "_napari_cuda_heartbeat_task", heartbeat_task)
        except Exception:
            logger.debug("failed to start heartbeat loop", exc_info=True)
        server._state_clients.add(ws)
        server.metrics.inc('napari_cuda_state_connects')
        server._update_client_gauges()
        await _send_state_baseline(server, ws)
        remote = getattr(ws, 'remote_address', None)
        log = logger.info if server._log_state_traces else logger.debug
        log("state client loop start remote=%s id=%s", remote, id(ws))
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    logger.debug('state client sent invalid JSON; ignoring')
                    continue
                await process_state_message(server, data, ws)
        except ConnectionClosed as exc:
            logger.info(
                "state client closed remote=%s id=%s code=%s reason=%s",
                remote,
                id(ws),
                getattr(exc, 'code', None),
                getattr(exc, 'reason', None),
            )
        except Exception:
            logger.exception("state client error remote=%s id=%s", remote, id(ws))
    finally:
        _flag_heartbeat_shutdown(ws)
        if heartbeat_task is None:
            heartbeat_task = getattr(ws, "_napari_cuda_heartbeat_task", None)
        if isinstance(heartbeat_task, asyncio.Task):
            heartbeat_task.cancel()
            with suppress(Exception):
                await heartbeat_task
        if hasattr(ws, "_napari_cuda_heartbeat_task"):
            delattr(ws, "_napari_cuda_heartbeat_task")
        try:
            await ws.close()
        except Exception as exc:
            logger.debug("State WS close error: %s", exc)
        server._state_clients.discard(ws)
        server._update_client_gauges()



async def _ingest_session_ack(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    _ENVELOPE_PARSER.parse_ack(data)
    return True


async def _ingest_session_goodbye(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    reason = None
    code = None
    message = "client requested shutdown"
    try:
        goodbye = _ENVELOPE_PARSER.parse_goodbye(data)
        payload = goodbye.payload
        reason = payload.reason
        if payload.message:
            message = payload.message
        if payload.code:
            code = payload.code
    except Exception:
        logger.debug("session.goodbye payload rejected", exc_info=True)

    session_id = _state_session(ws)
    if session_id:
        with suppress(Exception):
            await _send_session_goodbye(
                server,
                ws,
                session_id=session_id,
                code=code or "goodbye",
                message=message,
                reason=reason or "client_goodbye",
            )
    _flag_heartbeat_shutdown(ws)
    with suppress(Exception):
        await ws.close()
    return True


async def _ingest_state_update(server: Any, data: Mapping[str, Any], ws: Any) -> bool:
    try:
        frame = _ENVELOPE_PARSER.parse_state_update(data)
    except Exception:
        logger.debug("state.update payload rejected by parser", exc_info=True)
        frame_id = data.get("frame_id")
        intent_id = data.get("intent_id")
        session_id = data.get("session") or _state_session(ws)
        if frame_id and intent_id and session_id:
            await _send_state_ack(
                server,
                ws,
                session_id=str(session_id),
                intent_id=str(intent_id),
                in_reply_to=str(frame_id),
                status="rejected",
                error={"code": "state.invalid", "message": "state.update payload invalid"},
            )
        return True

    envelope = frame.envelope
    payload = frame.payload

    session_id = envelope.session or _state_session(ws)
    frame_id = envelope.frame_id
    intent_id = envelope.intent_id
    timestamp = envelope.timestamp

    if session_id is None or frame_id is None or intent_id is None:
        raise ValueError("state.update envelope missing session/frame/intent identifiers")

    scope = str(payload.scope or "").strip().lower()
    key = str(payload.key or "").strip().lower()
    target = str(payload.target or "").strip()
    value = payload.value

    ctx = StateUpdateContext(
        server=server,
        ws=ws,
        scope=scope,
        key=key,
        target=target,
        value=value,
        session_id=str(session_id),
        frame_id=str(frame_id),
        intent_id=str(intent_id),
        timestamp=timestamp,
    )

    if not scope or not key:
        await _state_update_reject(
            ctx,
            "state.invalid",
            "scope/key required",
            details={"scope": scope, "key": key, "target": target},
        )
        return True

    handler = _STATE_UPDATE_HANDLERS.get(f"{scope}:{key}")
    if handler is None:
        handler = _STATE_UPDATE_HANDLERS.get(f"{scope}:*")
    if handler is None:
        await _state_update_reject(
            ctx,
            "state.invalid",
            f"unsupported {scope} key {key}",
            details={"scope": scope, "key": key},
        )
        return True

    return await handler(ctx)

    return True

MESSAGE_HANDLERS: dict[str, StateMessageHandler] = {
    STATE_UPDATE_TYPE: _ingest_state_update,
    'call.command': _ingest_call_command,
    SESSION_ACK_TYPE: _ingest_session_ack,
    SESSION_GOODBYE_TYPE: _ingest_session_goodbye,
}


async def process_state_message(server: Any, data: dict, ws: Any) -> None:
    msg_type = data.get('type')
    frame_id = data.get('frame_id')
    _mark_client_activity(ws)
    if server._log_state_traces:
        logger.info("state message start type=%s frame=%s", msg_type, frame_id)
    handled = False
    try:
        handler: StateMessageHandler | None = None
        if isinstance(msg_type, str):
            handler = MESSAGE_HANDLERS.get(msg_type)
        if handler is None:
            if server._log_state_traces:
                logger.info("state message ignored type=%s", msg_type)
            return
        handled = bool(await handler(server, data, ws))
        return
    finally:
        if server._log_state_traces:
            logger.info("state message end type=%s frame=%s handled=%s", msg_type, frame_id, handled)


def _log_volume_event(server: Any, fmt: str, *args: Any) -> None:
    handler = getattr(server, "_log_volume_update", None)
    if callable(handler):
        try:
            handler(fmt, *args)
            return
        except Exception:
            logger.debug("volume update log failed", exc_info=True)
    try:
        logger.debug(fmt, *args)
    except Exception:
        logger.debug("volume fallback log failed", exc_info=True)


def _ndim_from_meta(meta: Mapping[str, Any]) -> int:
    try:
        ndim = int(meta.get("ndim") or 0)
    except Exception:
        ndim = 0

    if ndim <= 0:
        for key in ("order", "axes", "level_shapes"):
            values = meta.get(key)
            if not isinstance(values, Sequence):
                continue
            if key == "level_shapes":
                assert values, "level_shapes must contain at least one entry"
                first = values[0]
                assert isinstance(first, Sequence), "level_shapes entries must be sequences"
                length = len(first)
            else:
                length = len(values)
            ndim = max(ndim, length)

    if ndim <= 0:
        current = meta.get("current_step")
        if isinstance(current, Sequence):
            try:
                ndim = max(ndim, len(tuple(current)))
            except Exception:
                ndim = max(ndim, 0)

    return max(1, int(ndim))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _disable_nagle(ws: Any) -> None:
    try:
        sock = ws.transport.get_extra_info('socket')  # type: ignore[attr-defined]
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        logger.debug('state ws: TCP_NODELAY toggle failed', exc_info=True)


async def _perform_state_handshake(server: Any, ws: Any) -> bool:
    """Negotiate the notify-only protocol handshake with the client."""

    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=_HANDSHAKE_TIMEOUT_S)
    except asyncio.TimeoutError:
        await _send_handshake_reject(
            server,
            ws,
            code="timeout",
            message="session.hello not received",
        )
        return False
    except ConnectionClosed:
        logger.debug("state client closed during handshake")
        return False
    except Exception:
        logger.debug("state client handshake recv failed", exc_info=True)
        return False

    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            await _send_handshake_reject(
                server,
                ws,
                code="invalid_payload",
                message="session.hello must be UTF-8 text",
            )
            return False

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await _send_handshake_reject(
            server,
            ws,
            code="invalid_json",
            message="session.hello must be valid JSON",
        )
        return False

    msg_type = str(data.get("type") or "").lower()
    if msg_type != SESSION_HELLO_TYPE:
        await _send_handshake_reject(
            server,
            ws,
            code="unexpected_message",
            message="expected session.hello",
            details={"received": msg_type},
        )
        return False

    try:
        hello = _ENVELOPE_PARSER.parse_hello(data)
    except Exception as exc:
        await _send_handshake_reject(
            server,
            ws,
            code="invalid_payload",
            message="session.hello payload rejected",
            details={"error": str(exc)},
        )
        return False

    client_protocols = hello.payload.protocols
    if PROTO_VERSION not in client_protocols:
        await _send_handshake_reject(
            server,
            ws,
            code="protocol_mismatch",
            message=f"server requires protocol {PROTO_VERSION}",
            details={"client_protocols": list(client_protocols)},
        )
        return False

    negotiated_features = _resolve_handshake_features(hello)
    missing = [name for name in _REQUIRED_NOTIFY_FEATURES if not negotiated_features[name].enabled]
    if missing:
        await _send_handshake_reject(
            server,
            ws,
            code="unsupported_client",
            message="client missing notify features",
            details={"missing": missing},
        )
        return False

    history_store = _history_store(server)
    client_tokens = hello.payload.resume_tokens
    resume_plan: Dict[str, ResumePlan] = {}

    for topic in _RESUMABLE_TOPICS:
        toggle = negotiated_features.get(topic)
        if toggle is None or not toggle.enabled or not toggle.resume:
            continue
        token = client_tokens.get(topic)
        if history_store is None:
            resume_plan[topic] = ResumePlan(topic=topic, decision=ResumeDecision.RESET, deltas=[])
            continue
        plan = history_store.plan_resume(topic, token)
        resume_plan[topic] = plan
        if plan.decision == ResumeDecision.REJECT:
            await _send_handshake_reject(
                server,
                ws,
                code="invalid_resume_token",
                message=f"resume token for {topic} rejected",
                details={"topic": topic},
            )
            return False
        if plan.decision == ResumeDecision.REPLAY:
            cursor = history_store.latest_resume_state(topic)
            if cursor is not None:
                negotiated_features[topic] = replace(toggle, resume_state=cursor)
            continue
        cursor = history_store.latest_resume_state(topic)
        if cursor is not None:
            negotiated_features[topic] = replace(toggle, resume_state=cursor)

    setattr(ws, "_napari_cuda_resume_plan", resume_plan)

    session_id = str(uuid.uuid4())
    heartbeat_s = getattr(server, "state_heartbeat_s", 15.0)
    ack_timeout_ms = getattr(server, "state_ack_timeout_ms", 250)
    welcome = build_session_welcome(
        session_id=session_id,
        heartbeat_s=heartbeat_s,
        ack_timeout_ms=ack_timeout_ms,
        features=negotiated_features,
        timestamp=time.time(),
    )

    try:
        await _send_handshake_envelope(server, ws, welcome)
    except Exception:
        logger.debug("session.welcome send failed", exc_info=True)
        return False

    setattr(ws, "_napari_cuda_session", session_id)
    setattr(ws, "_napari_cuda_features", negotiated_features)
    setattr(ws, "_napari_cuda_heartbeat_interval", float(heartbeat_s))
    setattr(ws, "_napari_cuda_shutdown", False)
    setattr(ws, "_napari_cuda_goodbye_sent", False)
    setattr(
        ws,
        "_napari_cuda_sequencers",
        {
            NOTIFY_SCENE_TYPE: ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE),
            NOTIFY_LAYERS_TYPE: ResumableTopicSequencer(topic=NOTIFY_LAYERS_TYPE),
            NOTIFY_STREAM_TYPE: ResumableTopicSequencer(topic=NOTIFY_STREAM_TYPE),
        },
    )

    _mark_client_activity(ws)

    client_info = hello.payload.client
    client_name = client_info.name
    client_version = client_info.version
    logger.info(
        "state handshake accepted name=%s version=%s features=%s",
        client_name,
        client_version,
        sorted(name for name, toggle in negotiated_features.items() if toggle.enabled),
    )

    return True


def _resolve_handshake_features(hello: SessionHello) -> Dict[str, FeatureToggle]:
    client_features = hello.payload.features
    negotiated: Dict[str, FeatureToggle] = {}
    for name, server_toggle in _SERVER_FEATURES.items():
        client_enabled = client_features.get(name, False)
        # Required features must be explicitly true; optional ones inherit server toggle.
        enabled = server_toggle.enabled and (client_enabled or name not in _REQUIRED_NOTIFY_FEATURES)
        negotiated[name] = replace(server_toggle, enabled=enabled)

    command_toggle = negotiated.get("call.command")
    if command_toggle is not None:
        commands = COMMAND_REGISTRY.command_names()
        negotiated["call.command"] = replace(
            command_toggle,
            commands=commands if (command_toggle.enabled and commands) else (),
        )
    return negotiated


async def _send_handshake_envelope(server: Any, ws: Any, envelope: SessionWelcome | SessionReject) -> None:
    text = json.dumps(envelope.to_dict(), separators=(",", ":"))
    await _await_state_send(server, ws, text)


async def _send_handshake_reject(
    server: Any,
    ws: Any,
    *,
    code: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    reject = build_session_reject(
        code=str(code),
        message=str(message),
        details=dict(details) if details else None,
        timestamp=time.time(),
    )
    try:
        await _send_handshake_envelope(server, ws, reject)
    except Exception:
        logger.debug("session.reject send failed", exc_info=True)
    try:
        await ws.close()
    except Exception:
        logger.debug("state ws close after reject failed", exc_info=True)


async def _await_state_send(server: Any, ws: Any, text: str) -> None:
    try:
        result = server._state_send(ws, text)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.debug("state send helper failed", exc_info=True)


async def _send_state_ack(
    server: Any,
    ws: Any,
    *,
    session_id: str,
    intent_id: str,
    in_reply_to: str,
    status: str,
    applied_value: Any | None = None,
    error: Mapping[str, Any] | Any | None = None,
    version: int | None = None,
) -> None:
    """Build and emit an ``ack.state`` frame that mirrors the incoming update."""

    if not intent_id or not in_reply_to:
        raise ValueError("ack.state requires intent_id and in_reply_to identifiers")

    normalized_status = str(status).lower()
    if normalized_status not in {"accepted", "rejected"}:
        raise ValueError("ack.state status must be 'accepted' or 'rejected'")

    payload: Dict[str, Any] = {
        "intent_id": str(intent_id),
        "in_reply_to": str(in_reply_to),
        "status": normalized_status,
    }

    if normalized_status == "accepted":
        if error is not None:
            raise ValueError("accepted ack.state payload cannot include error details")
        if version is None:
            raise ValueError("accepted ack.state payload requires version")
        if not isinstance(version, Integral):
            raise ValueError("ack.state version must be integer")
        if applied_value is not None:
            payload["applied_value"] = applied_value
        payload["version"] = int(version)
    else:
        if not isinstance(error, Mapping):
            raise ValueError("rejected ack.state payload requires {code, message}")
        if "code" not in error or "message" not in error:
            raise ValueError("ack.state error payload must include 'code' and 'message'")
        payload["error"] = dict(error)
        if version is not None:
            if not isinstance(version, Integral):
                raise ValueError("ack.state version must be integer")
            payload["version"] = int(version)

    frame = build_ack_state(
        session_id=str(session_id),
        frame_id=None,
        payload=payload,
        timestamp=time.time(),
    )

    await _send_frame(server, ws, frame)


async def _send_frame(server: Any, ws: Any, frame: Any) -> None:
    try:
        payload = frame.to_dict()
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError("frame must provide to_dict()") from exc
    text = json.dumps(payload, separators=(",", ":"))
    await _await_state_send(server, ws, text)


async def _send_layer_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = _state_session(ws)
    if not session_id:
        return
    payload = NotifyLayersPayload.from_dict(snapshot.payload)
    frame = build_notify_layers_delta(
        session_id=session_id,
        payload=payload,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        intent_id=snapshot.intent_id,
        seq=snapshot.seq,
        delta_token=snapshot.delta_token,
    )
    await _send_frame(server, ws, frame)
    sequencer = _state_sequencer(ws, NOTIFY_LAYERS_TYPE)
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def _send_stream_snapshot(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = _state_session(ws)
    if not session_id:
        return
    payload = NotifyStreamPayload.from_dict(snapshot.payload)
    frame = build_notify_stream(
        session_id=session_id,
        payload=payload,
        timestamp=snapshot.timestamp,
        frame_id=snapshot.frame_id,
        seq=snapshot.seq,
        delta_token=snapshot.delta_token,
    )
    await _send_frame(server, ws, frame)
    sequencer = _state_sequencer(ws, NOTIFY_STREAM_TYPE)
    sequencer.resume(seq=snapshot.seq, delta_token=snapshot.delta_token)


async def _send_scene_snapshot_from_cache(server: Any, ws: Any, snapshot: EnvelopeSnapshot) -> None:
    session_id = _state_session(ws)
    if not session_id:
        return
    payload = NotifyScenePayload.from_dict(snapshot.payload)
    sequencer = _state_sequencer(ws, NOTIFY_SCENE_TYPE)
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
    await _send_frame(server, ws, frame)


async def _send_state_baseline(server: Any, ws: Any) -> None:
    resume_map: Dict[str, ResumePlan] = getattr(ws, "_napari_cuda_resume_plan", {}) or {}
    scene_plan = resume_map.get(NOTIFY_SCENE_TYPE)
    layers_plan = resume_map.get(NOTIFY_LAYERS_TYPE)
    stream_plan = resume_map.get(NOTIFY_STREAM_TYPE)

    scene_payload: NotifyScenePayload | None = None
    default_controls: list[tuple[str, Mapping[str, Any]]] = []

    try:
        scene_snapshot = _ensure_scene_snapshot(server)
        ledger_snapshot = server._state_ledger.snapshot()
        scene_payload = build_notify_scene_payload(
            scene_snapshot=scene_snapshot,
            ledger_snapshot=ledger_snapshot,
            viewer_settings=_viewer_settings(server),
        )

        layer_controls_map: Dict[str, Dict[str, Any]] = {}
        mirror = getattr(server, "_layer_mirror", None)
        if mirror is not None:
            layer_controls_map = mirror.latest_controls()
        else:
            logger.debug("layer mirror not initialised; falling back to scene snapshot controls")
        for layer_snapshot in scene_snapshot.layers:
            layer_id = layer_snapshot.layer_id
            controls: Dict[str, Any] = dict(layer_controls_map.get(layer_id, {}))
            if not controls and "controls" in layer_snapshot.block:
                block_controls = layer_snapshot.block["controls"]
                assert isinstance(block_controls, Mapping), "layer snapshot controls missing mapping"
                controls.update({str(key): value for key, value in block_controls.items()})
            if controls:
                default_controls.append((layer_id, controls))

    except Exception:
        logger.debug("Initial scene baseline prep failed", exc_info=True)

    if default_controls and not (
        layers_plan is not None
        and layers_plan.decision == ResumeDecision.REPLAY
        and layers_plan.deltas
    ):
        for layer_id, controls in default_controls:
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

    try:
        await _emit_scene_baseline(
            server,
            ws,
            payload=scene_payload,
            plan=scene_plan,
            reason="connect",
        )
    except Exception:
        logger.exception("Initial notify.scene send failed")

    try:
        await _emit_layer_baseline(
            server,
            ws,
            plan=layers_plan,
            default_controls=default_controls,
        )
    except Exception:
        logger.exception("Initial layer baseline send failed")

    try:
        await _emit_stream_baseline(server, ws, plan=stream_plan)
    except Exception:
        logger.exception("Initial state config send failed")

    await _emit_dims_baseline(server, ws)

    assert hasattr(server, "_ensure_keyframe"), "server must expose _ensure_keyframe"
    assert hasattr(server, "_schedule_coro"), "server must expose _schedule_coro"
    server._schedule_coro(server._ensure_keyframe(), "state-baseline-keyframe")

    if hasattr(ws, "_napari_cuda_resume_plan"):
        delattr(ws, "_napari_cuda_resume_plan")


async def _send_scene_snapshot_direct(server: Any, ws: Any, *, reason: str) -> None:
    session_id = _state_session(ws)
    if not session_id:
        logger.debug("Skipping notify.scene send without session id")
        return

    scene_snapshot = _ensure_scene_snapshot(server)
    payload = build_notify_scene_payload(
        scene_snapshot=scene_snapshot,
        ledger_snapshot=server._state_ledger.snapshot(),
        viewer_settings=_viewer_settings(server),
    )
    timestamp = time.time()
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None:
        snapshot = store.snapshot_envelope(
            NOTIFY_SCENE_TYPE,
            payload=payload.to_dict(),
            timestamp=timestamp,
        )
    frame = build_notify_scene_snapshot(
        session_id=session_id,
        viewer=payload.viewer,
        layers=payload.layers,
        policies=payload.policies,
        metadata=payload.metadata,
        timestamp=snapshot.timestamp if snapshot is not None else timestamp,
        delta_token=snapshot.delta_token if snapshot is not None else None,
        frame_id=snapshot.frame_id if snapshot is not None else None,
    )
    await _send_frame(server, ws, frame)
    if server._log_dims_info:
        logger.info("%s: notify.scene sent", reason)
    else:
        logger.debug("%s: notify.scene sent", reason)


async def broadcast_scene_snapshot(server: Any, *, reason: str) -> None:
    clients = list(server._state_clients)
    if not clients:
        return
    scene_snapshot = _ensure_scene_snapshot(server)
    payload = build_notify_scene_payload(
        scene_snapshot=scene_snapshot,
        ledger_snapshot=server._state_ledger.snapshot(),
        viewer_settings=_viewer_settings(server),
    )
    timestamp = time.time()
    store = _history_store(server)
    snapshot: EnvelopeSnapshot | None = None
    if store is not None:
        snapshot = store.snapshot_envelope(
            NOTIFY_SCENE_TYPE,
            payload=payload.to_dict(),
            timestamp=timestamp,
        )
        store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=timestamp)
        store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=timestamp)
    tasks: list[Awaitable[None]] = []
    for ws in clients:
        session_id = _state_session(ws)
        if not session_id:
            continue
        if snapshot is not None:
            _state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
            _state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()
        frame = build_notify_scene_snapshot(
            session_id=session_id,
            viewer=payload.viewer,
            layers=payload.layers,
            policies=payload.policies,
            metadata=payload.metadata,
            timestamp=snapshot.timestamp if snapshot is not None else timestamp,
            delta_token=snapshot.delta_token if snapshot is not None else None,
            frame_id=snapshot.frame_id if snapshot is not None else None,
        )
        tasks.append(_send_frame(server, ws, frame))
    if not tasks:
        return
    await asyncio.gather(*tasks, return_exceptions=True)
    if server._log_dims_info:
        logger.info("%s: notify.scene broadcast to %d clients", reason, len(tasks))
    else:
        logger.debug("%s: notify.scene broadcast to %d clients", reason, len(tasks))


async def _send_layer_baseline(server: Any, ws: Any) -> None:
    """Send canonical layer controls for all known layers to *ws*."""

    scene_snapshot = _ensure_scene_snapshot(server)
    if not scene_snapshot.layers:
        return

    snapshot_state = snapshot_render_state(server._state_ledger)
    latest_values = snapshot_state.layer_values or {}

    for layer in scene_snapshot.layers:
        layer_id = layer.layer_id
        if not layer_id:
            continue

        values = latest_values.get(layer_id, {})
        if not values:
            continue

        controls = {str(key): value for key, value in values.items()}

        await _broadcast_layers_delta(
            server,
            layer_id=layer_id,
            changes=controls,
            intent_id=None,
            timestamp=time.time(),
            targets=[ws],
        )
StateUpdateHandler = Callable[[StateUpdateContext], Awaitable[bool]]

_STATE_UPDATE_HANDLERS: Dict[str, StateUpdateHandler] = {
    "view:ndisplay": _handle_view_ndisplay,
    "camera:zoom": _handle_camera_zoom,
    "camera:pan": _handle_camera_pan,
    "camera:orbit": _handle_camera_orbit,
    "camera:reset": _handle_camera_reset,
    "camera:set": _handle_camera_set,
    "layer:*": _handle_layer_property,
    "volume:*": _handle_volume_update,
    "multiscale:level": _handle_multiscale_level,
    "multiscale:policy": _handle_multiscale_level,
    "dims:index": _handle_dims_update,
    "dims:step": _handle_dims_update,
}
