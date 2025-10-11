"""Display mode helpers for toggling between 2D and 3D rendering."""

from __future__ import annotations

from typing import Optional
import logging

from vispy import scene  # type: ignore

from napari_cuda.server.runtime.worker_runtime import perform_level_switch, notify_scene_refresh

logger = logging.getLogger(__name__)


def apply_ndisplay_switch(worker, ndisplay: int) -> None:
    """Apply a 2D/3D toggle using the worker context."""

    assert worker.view is not None and worker.view.scene is not None, "VisPy view must be initialised"

    target = 3 if int(ndisplay) >= 3 else 2
    previous_volume = worker.use_volume

    worker.use_volume = target == 3

    viewer_model = worker._viewer
    viewer_dims = viewer_model.dims if viewer_model is not None else None

    if worker.use_volume:
        worker._last_roi = None
        source = worker._ensure_scene_source()
        level = _coarsest_level_index(source)
        assert level is not None, "Volume mode requires a multiscale level"
        perform_level_switch(
            worker,
            target_level=level,
            reason="ndisplay-3d",
            requested_level=level,
            selected_level=level,
            source=source,
            budget_error=worker._budget_error_cls,
            restoring_plane_state=False,
        )
        worker._level_policy_refresh_needed = False
    else:
        if previous_volume:
            worker._level_policy_refresh_needed = True
            worker._mark_render_tick_needed()

    _configure_camera_for_mode(worker)

    if viewer_dims is not None:
        _update_viewer_dims(worker, viewer_dims, target)

    mode_label = "3D" if target == 3 else "2D"
    logger.info("ndisplay switch: %s", mode_label)
    if not worker.use_volume:
        worker._evaluate_level_policy()
        worker._level_policy_refresh_needed = False
    worker._mark_render_tick_needed()
    notify_scene_refresh(worker)


def _configure_camera_for_mode(worker) -> None:
    view = worker.view
    if view is None:
        return

    if worker.use_volume:
        view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
        extent = worker._volume_world_extents()
        if extent is None:
            width_px, height_px = worker._data_wh
            depth_px = worker._data_d or 1
            extent = (float(width_px), float(height_px), float(depth_px))
        view.camera.set_range(
            x=(0.0, max(1.0, extent[0])),
            y=(0.0, max(1.0, extent[1])),
            z=(0.0, max(1.0, extent[2])),
        )
        worker._frame_volume_camera(extent[0], extent[1], extent[2])
    else:
        view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
        worker._apply_camera_reset(view.camera)


def _update_viewer_dims(worker, viewer_dims, target: int) -> None:
    layer = worker._napari_layer
    layer_ndim = layer.ndim if layer is not None else 0
    data = layer.data if layer is not None else None
    data_ndim = data.ndim if data is not None else 0
    current_ndim = viewer_dims.ndim
    ndim = max(layer_ndim, data_ndim, current_ndim, target)
    if ndim <= 0:
        ndim = target
    if current_ndim != ndim:
        viewer_dims.ndim = ndim

    displayed_count = min(target, ndim)
    displayed = tuple(range(ndim - displayed_count, ndim))

    order = list(viewer_dims.order)
    valid_order = len(order) == ndim and all(0 <= axis < ndim for axis in order)
    if not valid_order:
        order = list(range(ndim))
    else:
        order = [axis for axis in order if axis not in displayed]
        order.extend(displayed)
    viewer_dims.order = tuple(order)

    ledger_step = worker._ledger_step()
    if ledger_step is not None:
        steps = list(ledger_step)
    else:
        steps = list(viewer_dims.current_step)
    if len(steps) < ndim:
        steps.extend([0] * (ndim - len(steps)))
    elif len(steps) > ndim:
        steps = steps[:ndim]

    if target == 3 and worker._z_index is not None:
        anchor_axis = displayed[0] if displayed else 0
        steps[anchor_axis] = int(worker._z_index)

    if target == 2:
        nsteps = list(viewer_dims.nsteps)
        for idx in range(min(len(steps), len(nsteps))):
            max_index = max(0, int(nsteps[idx]) - 1)
            steps[idx] = int(max(0, min(steps[idx], max_index)))
        if ndim > displayed_count:
            anchor_axis = ndim - displayed_count - 1
            worker._z_index = int(steps[anchor_axis])

    updated_step = tuple(int(value) for value in steps)
    viewer_dims.current_step = updated_step


def _coarsest_level_index(source) -> Optional[int]:
    descriptors = source.level_descriptors
    if not descriptors:
        return None
    return len(descriptors) - 1


__all__ = ["apply_ndisplay_switch"]
