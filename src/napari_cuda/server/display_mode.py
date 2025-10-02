"""Display mode helpers for toggling between 2D and 3D rendering."""

from __future__ import annotations

from typing import Optional
import logging

from vispy import scene  # type: ignore

from napari_cuda.server.worker_runtime import perform_level_switch

logger = logging.getLogger(__name__)


def apply_ndisplay_switch(worker, ndisplay: int) -> None:
    """Apply a 2D/3D toggle using the worker context."""
    view = worker.view
    assert view is not None and view.scene is not None, "VisPy view must be initialized"

    target = 3 if int(ndisplay) >= 3 else 2
    previous_volume = bool(worker.use_volume)
    worker.use_volume = bool(target == 3)

    viewer_model = worker._viewer
    if viewer_model is not None:
        viewer_dims = viewer_model.dims
        viewer_dims.ndisplay = target

    if worker.use_volume:
        worker._last_roi = None
        source = worker._ensure_scene_source()
        level = _coarsest_level_index(source)
        assert level is not None, "Volume mode requires a multiscale level"
        perform_level_switch(
            worker,
            target_level=int(level),
            reason="ndisplay-3d",
            requested_level=int(level),
            selected_level=int(level),
            source=source,
            budget_error=getattr(worker, "_budget_error_cls", RuntimeError),
        )
        _reset_volume_step(worker, source, int(level))
        worker._level_policy_refresh_needed = False
    else:
        if previous_volume:
            worker._level_policy_refresh_needed = True
            worker._mark_render_tick_needed()

    _configure_camera_for_mode(worker)

    if viewer_model is not None:
        _update_viewer_dims(worker, viewer_model, target)

    logger.info("ndisplay switch: %s", "3D" if target == 3 else "2D")
    if not worker.use_volume:
        worker._evaluate_level_policy()
        worker._level_policy_refresh_needed = False
    worker._mark_render_tick_needed()


def _reset_volume_step(worker, source, level: int) -> None:
    descriptors = getattr(source, "level_descriptors", None)
    assert descriptors, "Zarr source missing level descriptors"
    descriptor = descriptors[int(level)]
    axes = [str(a).lower() for a in getattr(source, "axes", []) or []]
    z_axis = axes.index("z") if "z" in axes else 0
    step = list(getattr(source, "current_step", ()) or [])
    shape = list(getattr(descriptor, "shape", ()) or [])
    dims = max(len(step), len(shape), 3)
    if len(step) < dims:
        step.extend([0] * (dims - len(step)))
    if len(shape) < dims:
        shape.extend([1] * (dims - len(shape)))
    if z_axis >= len(step):
        step.extend([0] * (z_axis - len(step) + 1))
    if z_axis >= len(shape):
        shape.extend([1] * (z_axis - len(shape) + 1))
    step[z_axis] = 0
    clamped_step = []
    for idx, val in enumerate(step):
        sh = max(1, int(shape[idx] if idx < len(shape) else 1))
        clamped_step.append(int(max(0, min(int(val), sh - 1))))
    state_lock = getattr(worker, "_state_lock", None)
    assert state_lock is not None, "Worker must expose _state_lock"
    with state_lock:
        source.set_current_level(int(level), step=tuple(clamped_step))
    worker._z_index = 0
    worker._last_step = tuple(clamped_step)


def _configure_camera_for_mode(worker) -> None:
    view = worker.view
    if view is None:
        return

    if worker.use_volume:
        if not isinstance(view.camera, scene.cameras.TurntableCamera):
            view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
        camera = view.camera
        assert isinstance(camera, scene.cameras.TurntableCamera)
        extent = worker._volume_world_extents()
        if extent is None:
            w_px, h_px = worker._data_wh
            d_px = worker._data_d or 1
            extent = (float(w_px), float(h_px), float(d_px))
        world_w, world_h, world_d = extent
        camera.set_range(
            x=(0.0, max(1.0, world_w)),
            y=(0.0, max(1.0, world_h)),
            z=(0.0, max(1.0, world_d)),
        )
        worker._frame_volume_camera(world_w, world_h, world_d)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "configure_camera_for_mode: use_volume extent=(%.3f, %.3f, %.3f) center=%s distance=%.3f",
                world_w,
                world_h,
                world_d,
                getattr(camera, 'center', None),
                float(getattr(camera, 'distance', 0.0)),
            )
    else:
        if not isinstance(view.camera, scene.cameras.PanZoomCamera):
            view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
        camera = view.camera
        assert camera is not None
        worker._apply_camera_reset(camera)


def _update_viewer_dims(worker, viewer, target: int) -> None:
    viewer_dims = viewer.dims
    viewer_dims.ndisplay = target
    if target != 3:
        return

    layer = worker._napari_layer
    layer_ndim = int(getattr(layer, "ndim", 0)) if layer is not None else 0
    data = getattr(layer, "data", None) if layer is not None else None
    data_ndim = int(getattr(data, "ndim", 0)) if data is not None else 0
    dims_attr = int(getattr(viewer_dims, "ndim", 0))
    ndim = max(layer_ndim, data_ndim, dims_attr, 3)
    viewer_dims.ndim = ndim
    displayed = tuple(range(max(0, ndim - 3), ndim))

    # Keep the displayed axes as the tail of the order tuple so napari marks
    # them as active. This mirrors the old `displayed` setter without relying on it.
    order = list(getattr(viewer_dims, "order", ()) or ())
    if len(order) != ndim or any(ax >= ndim for ax in order):
        order = list(range(ndim))
    else:
        order = [ax for ax in order if ax not in displayed]
        order.extend(displayed)
    viewer_dims.order = tuple(order)
    steps = list(getattr(viewer_dims, "current_step", ()) or ())
    if len(steps) < ndim:
        steps.extend([0] * (ndim - len(steps)))
    elif len(steps) > ndim:
        steps = steps[:ndim]
    if steps and worker._z_index is not None:
        steps[0] = int(worker._z_index)
    shape = worker._volume_shape_for_view()
    if shape is not None:
        axes = tuple(getattr(viewer_dims, "axis_labels", []) or [])
        for axis_idx in range(min(len(steps), len(shape))):
            size = int(max(1, shape[axis_idx]))
            steps[axis_idx] = int(max(0, min(int(steps[axis_idx]), size - 1)))
        if axes and shape:
            axis_map = {label.lower(): idx for idx, label in enumerate(axes)}
            for label_key, dim_size in zip(("z", "y", "x"), shape):
                idx = axis_map.get(label_key)
                if idx is not None and idx < len(steps):
                    steps[idx] = int(max(0, min(int(steps[idx]), int(dim_size) - 1)))
    viewer_dims.current_step = tuple(int(s) for s in steps)


def _coarsest_level_index(source) -> Optional[int]:
    descriptors = getattr(source, "level_descriptors", None)
    if not descriptors:
        return None
    last = descriptors[-1]
    index_attr = getattr(last, "index", None)
    if index_attr is not None:
        return int(index_attr)
    return int(len(descriptors) - 1)


__all__ = ["apply_ndisplay_switch"]
