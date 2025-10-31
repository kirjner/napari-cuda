"""Control-side helpers for probing initial scene metadata and worker bootstrap."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Optional

from napari_cuda.server.control.state_models import BootstrapSceneMetadata
from napari_cuda.server.data import lod
from napari_cuda.server.data.level_logging import (
    LayerAssignmentLogger,
    LevelSwitchLogger,
)
from napari_cuda.server.data.roi import plane_wh_for_level
from napari_cuda.server.data.zarr_source import (
    LevelDescriptor,
    ZarrSceneSource,
)
from napari_cuda.server.runtime.bootstrap.scene_setup import create_scene_source
from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdateMailbox


def _resolve_level_shapes(descriptors: Sequence[LevelDescriptor]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(size) for size in descriptor.shape) for descriptor in descriptors)


def _resolve_levels(descriptors: Sequence[LevelDescriptor]) -> tuple[dict[str, object], ...]:
    payload: list[dict[str, object]] = []
    for descriptor in descriptors:
        entry: dict[str, object] = {
            "index": int(descriptor.index),
            "shape": [int(size) for size in descriptor.shape],
            "downsample": [float(value) for value in descriptor.downsample],
        }
        if descriptor.path:
            entry["path"] = str(descriptor.path)
        payload.append(entry)
    return tuple(payload)


def _normalize_step(
    *,
    initial: tuple[int, ...],
    axes: Sequence[str],
    use_volume: bool,
) -> tuple[int, ...]:
    values = list(int(value) for value in initial)
    if use_volume and axes:
        axes_lower = [axis.lower() for axis in axes]
        if "z" in axes_lower:
            z_index = axes_lower.index("z")
            while len(values) <= z_index:
                values.append(0)
            values[z_index] = 0
    return tuple(values)


def probe_scene_bootstrap(
    *,
    path: str,
    use_volume: bool,
    preferred_level: Optional[str] = None,
    axes_override: Optional[Sequence[str]] = None,
    z_override: Optional[int] = None,
    canvas_size: tuple[int, int] = (1, 1),
    oversampling_thresholds: Optional[Mapping[int, float]] = None,
    oversampling_hysteresis: float = 0.1,
    threshold_in: float = 1.05,
    threshold_out: float = 1.35,
    fine_threshold: float = 1.05,
    policy_hysteresis: float = 0.0,
    cooldown_ms: float = 0.0,
) -> BootstrapSceneMetadata:
    """Inspect the scene source and return bootstrap metadata for the ledger."""

    source = ZarrSceneSource(
        path,
        preferred_level=preferred_level,
        axis_override=tuple(axes_override) if axes_override is not None else None,
    )

    current_level = source.current_level
    descriptors = source.level_descriptors
    level_shapes = _resolve_level_shapes(descriptors)
    levels = _resolve_levels(descriptors)

    axes = tuple(source.axes)
    viewport_w = max(1, int(canvas_size[0]))
    viewport_h = max(1, int(canvas_size[1]))
    overs_map: dict[int, float] = {}
    for descriptor in descriptors:
        level_index = int(descriptor.index)
        plane_h, plane_w = plane_wh_for_level(source, level_index)
        overs_map[level_index] = max(plane_h / viewport_h, plane_w / viewport_w)

    policy_config = lod.LevelPolicyConfig(
        threshold_in=float(threshold_in),
        threshold_out=float(threshold_out),
        fine_threshold=float(fine_threshold),
        hysteresis=float(policy_hysteresis),
        cooldown_ms=float(cooldown_ms),
    )

    oversampling_map = {int(k): float(v) for k, v in overs_map.items()}
    current_idx = int(current_level)
    remaining = max(1, len(descriptors) + 2)
    while remaining > 0:
        level_inputs = lod.LevelPolicyInputs(
            current_level=current_idx,
            oversampling=oversampling_map,
            zoom_ratio=None,
            lock_level=None,
            last_switch_ts=0.0,
            now_ts=0.0,
        )
        decision = lod.select_level(policy_config, level_inputs)
        selected_idx = int(decision.selected_level)
        if not decision.should_switch or selected_idx == current_idx:
            current_idx = selected_idx
            break
        current_idx = selected_idx
        remaining -= 1

    selected_level = current_idx
    level_count = len(descriptors)
    if level_count > 0:
        selected_level = max(0, min(int(selected_level), level_count - 1))
    else:
        selected_level = 0

    initial_step = source.initial_step(step_or_z=z_override, level=int(selected_level))
    applied_step = source.set_current_slice(initial_step, int(selected_level))
    resolved_step = _normalize_step(initial=applied_step, axes=axes, use_volume=use_volume)

    ndim = len(axes) if axes else len(resolved_step)
    if ndim <= 0:
        ndim = 1
    order = tuple(range(ndim))
    ndisplay = 3 if use_volume and ndim >= 3 else min(2, ndim)

    plane_h, plane_w = plane_wh_for_level(source, int(selected_level))
    rect = (0.0, 0.0, float(plane_w), float(plane_h))
    center = (float(plane_w) * 0.5, float(plane_h) * 0.5)
    zoom = 1.0

    return BootstrapSceneMetadata(
        step=resolved_step,
        axis_labels=axes if axes else tuple(f"axis-{idx}" for idx in range(ndim)),
        order=order,
        level_shapes=level_shapes,
        levels=levels,
        current_level=int(selected_level),
        ndisplay=int(ndisplay),
        plane_rect=rect,
        plane_center=center,
        plane_zoom=zoom,
    )


__all__ = [
    "cleanup_render_worker",
    "init_egl",
    "init_vispy_scene",
    "probe_scene_bootstrap",
    "setup_worker_runtime",
]


def setup_worker_runtime(
    worker: object,
    *,
    use_volume: bool,
    animate: bool,
    animate_dps: float,
    camera_queue,
    camera_pose_cb,
    level_intent_cb,
    policy_name: Optional[str],
) -> None:
    """Initialize render resources, viewport state, and policy config on the worker."""

    from napari_cuda.server.runtime.viewport import (
        RenderMode,
        ViewportRunner,
        ViewportState,
    )
    from napari_cuda.server.runtime.worker.resources import WorkerResources

    logger = logging.getLogger(worker.__class__.__module__)
    viewport_state = ViewportState()
    viewport_state.mode = RenderMode.VOLUME if use_volume else RenderMode.PLANE
    worker._viewport_state = viewport_state  # type: ignore[attr-defined]
    worker._viewport_runner = ViewportRunner(viewport_state.plane)  # type: ignore[attr-defined]

    try:
        worker._animate_dps = float(animate_dps)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive path
        logger.debug("Invalid animate_dps=%r; using default 30.0: %s", animate_dps, exc)
        worker._animate_dps = 30.0  # type: ignore[attr-defined]
    worker._animate = bool(animate)  # type: ignore[attr-defined]
    worker._anim_start = time.perf_counter()  # type: ignore[attr-defined]

    worker._resources = WorkerResources(width=worker.width, height=worker.height)  # type: ignore[attr-defined]
    worker.cuda_ctx = None  # type: ignore[attr-defined]
    worker.canvas = None  # type: ignore[attr-defined]
    worker.view = None  # type: ignore[attr-defined]
    worker._plane_visual_handle = None  # type: ignore[attr-defined]
    worker._volume_visual_handle = None  # type: ignore[attr-defined]

    worker._viewer = None  # type: ignore[attr-defined]
    worker._napari_layer = None  # type: ignore[attr-defined]
    worker._scene_source = None  # type: ignore[attr-defined]
    worker._set_current_level_index(0)  # type: ignore[attr-defined]
    worker._viewport_state.volume.downgraded = False  # type: ignore[attr-defined]
    worker._viewport_state.volume.scale = (1.0, 1.0, 1.0)  # type: ignore[attr-defined]
    worker._data_wh = (int(worker.width), int(worker.height))  # type: ignore[attr-defined]
    worker._data_d = None  # type: ignore[attr-defined]

    layer_logger = LayerAssignmentLogger(logger)
    worker._layer_logger = layer_logger  # type: ignore[attr-defined]
    worker._switch_logger = LevelSwitchLogger(logger)  # type: ignore[attr-defined]

    worker._pose_seq = 1  # type: ignore[attr-defined]
    worker._max_camera_command_seq = 0  # type: ignore[attr-defined]
    worker._level_policy_suppressed = False  # type: ignore[attr-defined]
    worker._last_slice_signature = None  # type: ignore[attr-defined]
    worker._last_snapshot_signature = None  # type: ignore[attr-defined]
    worker._last_dims_signature = None  # type: ignore[attr-defined]
    worker._applied_versions = {}  # type: ignore[attr-defined]
    worker._last_plane_pose = None  # type: ignore[attr-defined]
    worker._last_volume_pose = None  # type: ignore[attr-defined]

    worker._camera_queue = camera_queue  # type: ignore[attr-defined]
    worker._state_lock = threading.RLock()  # type: ignore[attr-defined]
    worker._render_mailbox = RenderUpdateMailbox()  # type: ignore[attr-defined]
    worker._last_ensure_log = None  # type: ignore[attr-defined]
    worker._last_ensure_log_ts = 0.0  # type: ignore[attr-defined]
    worker._render_tick_required = False  # type: ignore[attr-defined]
    worker._render_loop_started = False  # type: ignore[attr-defined]

    if camera_pose_cb is None:
        raise ValueError("EGLRendererWorker requires camera_pose callback")
    worker._camera_pose_callback = camera_pose_cb  # type: ignore[attr-defined]
    worker._level_intent_callback = level_intent_cb  # type: ignore[attr-defined]

    dbg_policy = worker._debug_policy  # type: ignore[attr-defined]
    worker_dbg = dbg_policy.worker
    worker._debug_zoom_drift = bool(worker_dbg.debug_zoom_drift)  # type: ignore[attr-defined]
    worker._debug_pan = bool(worker_dbg.debug_pan)  # type: ignore[attr-defined]
    worker._debug_reset = bool(worker_dbg.debug_reset)  # type: ignore[attr-defined]
    worker._debug_orbit = bool(worker_dbg.debug_orbit)  # type: ignore[attr-defined]
    worker._orbit_el_min = float(worker_dbg.orbit_el_min)  # type: ignore[attr-defined]
    worker._orbit_el_max = float(worker_dbg.orbit_el_max)  # type: ignore[attr-defined]

    policy_cfg = worker._ctx.policy  # type: ignore[attr-defined]
    worker._policy_func = lod.select_level  # type: ignore[attr-defined]
    worker._policy_name = "oversampling"  # type: ignore[attr-defined]
    worker._last_interaction_ts = time.perf_counter()  # type: ignore[attr-defined]
    policy_logging = dbg_policy.logging
    worker._log_layer_debug = bool(policy_logging.log_layer_debug)  # type: ignore[attr-defined]
    worker._log_policy_eval = bool(policy_cfg.log_policy_eval)  # type: ignore[attr-defined]
    worker._lock_level = worker_dbg.lock_level  # type: ignore[attr-defined]
    worker._level_threshold_in = float(policy_cfg.threshold_in)  # type: ignore[attr-defined]
    worker._level_threshold_out = float(policy_cfg.threshold_out)  # type: ignore[attr-defined]
    worker._level_hysteresis = float(policy_cfg.hysteresis)  # type: ignore[attr-defined]
    worker._level_fine_threshold = float(policy_cfg.fine_threshold)  # type: ignore[attr-defined]
    worker._sticky_contrast = bool(policy_cfg.sticky_contrast)  # type: ignore[attr-defined]
    worker._level_switch_cooldown_ms = float(policy_cfg.cooldown_ms)  # type: ignore[attr-defined]
    worker._last_level_switch_ts = 0.0  # type: ignore[attr-defined]
    worker._oversampling_thresholds = (
        {int(k): float(v) for k, v in policy_cfg.oversampling_thresholds.items()}
        if getattr(policy_cfg, "oversampling_thresholds", None)
        else None
    )  # type: ignore[attr-defined]
    worker._oversampling_hysteresis = float(getattr(policy_cfg, "oversampling_hysteresis", 0.1))  # type: ignore[attr-defined]

    worker._roi_edge_threshold = int(worker_dbg.roi_edge_threshold)  # type: ignore[attr-defined]
    worker._roi_align_chunks = bool(worker_dbg.roi_align_chunks)  # type: ignore[attr-defined]
    worker._roi_ensure_contains_viewport = bool(worker_dbg.roi_ensure_contains_viewport)  # type: ignore[attr-defined]
    worker._roi_pad_chunks = 1  # type: ignore[attr-defined]
    worker._idr_on_z = False  # type: ignore[attr-defined]

    cfg = worker._ctx.cfg  # type: ignore[attr-defined]
    worker._slice_max_bytes = int(max(0, getattr(cfg, "max_slice_bytes", 0)))  # type: ignore[attr-defined]
    worker._volume_max_bytes = int(max(0, getattr(cfg, "max_volume_bytes", 0)))  # type: ignore[attr-defined]
    worker._volume_max_voxels = int(max(0, getattr(cfg, "max_volume_voxels", 0)))  # type: ignore[attr-defined]

    if policy_name:
        try:
            worker.set_policy(policy_name)
        except Exception:  # pragma: no cover - defensive path
            logger.exception("policy init set failed; continuing with default")


def init_egl(worker: object) -> None:
    """Initialize EGL + CUDA stack for the worker."""

    worker.cuda_ctx = worker._resources.bootstrap(  # type: ignore[attr-defined]
        server_ctx=worker._ctx,  # type: ignore[attr-defined]
        fps_hint=int(worker.fps),  # type: ignore[attr-defined]
    )


def init_vispy_scene(worker: object) -> None:
    """Create the napari viewer + VisPy scene on the worker thread."""

    from napari_cuda.server.runtime.bootstrap.setup_viewer import (
        _init_viewer_scene,
    )  # Local import to avoid circular dependency

    source = None
    if getattr(worker, "_zarr_path", None):
        source = create_scene_source(worker)
        if source is not None:
            worker._scene_source = source  # type: ignore[attr-defined]
            try:
                worker._zarr_shape = source.level_shape(0)  # type: ignore[attr-defined]
                worker._zarr_dtype = str(source.dtype)  # type: ignore[attr-defined]
                worker._set_current_level_index(source.current_level)  # type: ignore[attr-defined]
                descriptor = source.level_descriptors[source.current_level]
                worker._zarr_level = descriptor.path or None  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive path
                logging.getLogger(worker.__class__.__module__).debug(
                    "scene source metadata bootstrap failed", exc_info=True
                )

    try:
        _init_viewer_scene(worker, source)
    except Exception:
        logging.getLogger(worker.__class__.__module__).exception(
            "Adapter scene initialization failed (legacy path removed)"
        )
        raise


def cleanup_render_worker(worker: object) -> None:
    """Tear down render resources and viewer state."""

    worker._resources.cleanup()  # type: ignore[attr-defined]
    worker.cuda_ctx = None  # type: ignore[attr-defined]

    try:
        if worker.canvas is not None:  # type: ignore[attr-defined]
            worker.canvas.close()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive path
        logging.getLogger(worker.__class__.__module__).debug(
            "Cleanup: canvas close failed", exc_info=True
        )
    worker.canvas = None  # type: ignore[attr-defined]
    worker.view = None  # type: ignore[attr-defined]
    worker._viewer = None  # type: ignore[attr-defined]
    worker._napari_layer = None  # type: ignore[attr-defined]
    if getattr(worker, "_plane_visual_handle", None) is not None:
        worker._plane_visual_handle.detach()  # type: ignore[attr-defined]
    if getattr(worker, "_volume_visual_handle", None) is not None:
        worker._volume_visual_handle.detach()  # type: ignore[attr-defined]
    worker._plane_visual_handle = None  # type: ignore[attr-defined]
    worker._volume_visual_handle = None  # type: ignore[attr-defined]
    worker._is_ready = False  # type: ignore[attr-defined]
