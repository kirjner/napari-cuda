"""Construct the napari + VisPy viewer stack for the render worker.

The worker invokes this helper once during startup to create the EGL-backed
`SceneCanvas`, attach a `ViewBox`, and bootstrap a headless napari `ViewerModel`
with either an image slice or volume layer. All worker-specific policy hooks
are delegated back via the bridge object so the builder stays focused on the
initial scene wiring.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from vispy import scene

from napari.components.viewer_model import ViewerModel
from napari_cuda.server.data.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
)
from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.runtime.viewport import RenderMode

from napari_cuda.server.runtime.bootstrap.interface import ViewerBootstrapInterface
from napari_cuda.server.runtime.bootstrap.setup_camera import (
    _bootstrap_camera_pose,
    _configure_camera_for_mode,
)
from napari_cuda.server.runtime.bootstrap.setup_visuals import (
    _ensure_plane_visual,
    _ensure_volume_visual,
    _register_plane_visual,
    _register_volume_visual,
    _VisualHandle,
)

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalAxes:
    ndim: int
    axis_labels: tuple[str, ...]
    order: tuple[int, ...]
    ndisplay: int
    current_step: tuple[int, ...]
    ranges: tuple[tuple[float, float, float], ...]
    sizes: tuple[int, ...]


def canonical_axes_from_source(
    *,
    axes: Sequence[str],
    shape: Sequence[int],
    step: Sequence[int],
    use_volume: bool,
) -> CanonicalAxes:
    ndim = len(shape)
    assert ndim > 0, "canonical axes requires at least one dimension"

    labels = tuple(str(axis) for axis in axes)
    assert len(labels) == ndim, f"source axes {labels} must match shape {shape}"

    step_tuple = tuple(int(value) for value in step)
    assert len(step_tuple) == ndim, f"step {step_tuple} must match ndim={ndim}"

    sizes = tuple(int(max(1, int(size))) for size in shape)
    ranges = tuple((0.0, float(size - 1), 1.0) for size in sizes)

    ndisplay = 3 if use_volume and ndim >= 3 else min(2, ndim)
    order = tuple(range(ndim))

    return CanonicalAxes(
        ndim=ndim,
        axis_labels=labels,
        order=order,
        ndisplay=ndisplay,
        current_step=step_tuple,
        ranges=ranges,
        sizes=sizes,
    )


def apply_canonical_axes(viewer: ViewerModel, meta: CanonicalAxes) -> None:
    dims = viewer.dims
    dims.ndim = meta.ndim
    dims.axis_labels = meta.axis_labels
    dims.order = meta.order
    dims.ndisplay = meta.ndisplay
    dims.range = meta.ranges
    dims.current_step = meta.current_step


class ViewerBuilder:
    """Bootstrap the viewer and VisPy canvas for the render worker."""

    def __init__(self, facade: ViewerBootstrapInterface) -> None:
        self._facade = facade

    def build(
        self,
        source: Optional[ZarrSceneSource],
        *,
        level: Optional[int] = None,
        step: Optional[Sequence[int]] = None,
        axis_labels: Optional[Sequence[str]] = None,
        order: Optional[Sequence[int]] = None,
        ndisplay: Optional[int] = None,
    ) -> tuple[scene.SceneCanvas, scene.widgets.ViewBox, ViewerModel]:
        worker_policy = self._facade.debug_policy
        worker_debug = worker_policy.worker
        debug_bg = bool(worker_debug.debug_bg_overlay)
        layer_interpolation = worker_debug.layer_interpolation
        assert layer_interpolation, "worker debug policy missing layer_interpolation"
        bgcolor = (0.08, 0.08, 0.08, 1.0) if debug_bg else "black"
        # Create a native EGL-backed SceneCanvas. The worker will adopt this
        # context for capture to ensure a single GL context is used.
        canvas = scene.SceneCanvas(
            size=(self._facade.width, self._facade.height),
            bgcolor=bgcolor,
            show=False,
            app="egl",
            create_native=True,
        )
        view = canvas.central_widget.add_view()
        # Expose to bridge for downstream code that expects these attributes
        self._facade.set_canvas(canvas)
        self._facade.set_view(view)

        rng = np.random.default_rng(41)
        viewer = ViewerModel()

        layer = None
        adapter = None
        scene_src = "synthetic"
        scene_meta = ""
        applied_step: Optional[tuple[int, ...]] = None
        viewport_state = self._facade.viewport_state
        is_volume_mode = viewport_state.mode is RenderMode.VOLUME

        if source is not None:
            levels = source.level_descriptors
            level_count = len(levels)
            selected_level = int(level) if level is not None else int(source.current_level)
            if level_count > 0:
                max_index = level_count - 1
                selected_level = max(0, min(selected_level, max_index))
                descriptor = levels[selected_level]
            else:
                selected_level = 0
                descriptor = None

            step_hint = tuple(int(v) for v in step) if step is not None else source.initial_step(
                step_or_z=self._facade.zarr_init_z,
                level=selected_level,
            )
            applied_step = source.set_current_slice(step_hint, selected_level)

            self._facade.set_scene_source(source)
            self._facade.set_current_level_index(selected_level)
            level_path = descriptor.path if descriptor is not None and descriptor.path else None  # type: ignore[union-attr]
            self._facade.set_zarr_level(level_path)
            self._facade.set_zarr_axes(''.join(source.axes))
            shape_payload = tuple(int(dim) for dim in descriptor.shape) if descriptor is not None else None  # type: ignore[union-attr]
            self._facade.set_zarr_shape(shape_payload)
            self._facade.set_zarr_dtype(str(source.dtype))
            self._facade.set_zarr_clim(source.ensure_contrast(level=selected_level))

            axes_tuple = tuple(str(ax) for ax in source.axes)
            axes_lower = [axis.lower() for axis in axes_tuple]
            z_index = 0
            if applied_step is not None and 'z' in axes_lower:
                z_pos = axes_lower.index('z')
                if 0 <= z_pos < len(applied_step):
                    z_index = int(applied_step[z_pos])
            self._facade.set_z_index(int(z_index))

            if self._facade.viewport_state.mode is RenderMode.VOLUME:
                data = self._facade.load_volume(
                    source,
                    selected_level,
                )
                level_scale = source.level_scale(selected_level)
                scale_vals = [float(s) for s in level_scale]
                while len(scale_vals) < 3:
                    scale_vals.insert(0, 1.0)
                sz, sy, sx = scale_vals[-3], scale_vals[-2], scale_vals[-1]
                self._facade.set_volume_scale((float(sz), float(sy), float(sx)))
                layer = viewer.add_image(data, name="zarr-volume", scale=level_scale)
                if axis_labels:
                    viewer.dims.axis_labels = tuple(str(a) for a in axis_labels)
                else:
                    viewer.dims.axis_labels = tuple(source.axes)
                viewer.dims.ndisplay = 3
                viewer.dims.current_step = tuple(int(v) for v in applied_step) if applied_step is not None else tuple()
                from napari._vispy.layers.image import (
                    VispyImageLayer,  # type: ignore
                )
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                d, h, w = int(data.shape[0]), int(data.shape[1]), int(data.shape[2])
                self._facade.set_data_wh((w, h))
                self._facade.set_data_depth(d)
                world_w = float(w) * float(sx)
                world_h = float(h) * float(sy)
                world_d = float(d) * float(sz)
                view.camera.set_range(
                    x=(0.0, max(1.0, world_w)),
                    y=(0.0, max(1.0, world_h)),
                    z=(0.0, max(1.0, world_d)),
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "viewer builder volume extent=(%.3f, %.3f, %.3f) translate=%s scale=%s",
                        world_w,
                        world_h,
                        world_d,
                        getattr(layer, 'translate', None),
                    getattr(layer, 'scale', None),
                )
                self._facade.frame_volume_camera(world_w, world_h, world_d)
                layer.rendering = "mip"
                scene_src = "napari-zarr-volume"
                scene_meta = f"level={self._facade.zarr_level or selected_level} shape={d}x{h}x{w}"
            else:
                # Bootstrap: request a full-slab ROI for the first slice load
                self._facade.set_bootstrap_full_roi(True)
                slice_array = self._facade.load_initial_slice(
                    source,
                    selected_level,
                    int(z_index),
                )
                if self._facade.log_layer_debug:
                    smin = float(np.nanmin(slice_array))
                    smax = float(np.nanmax(slice_array))
                    smea = float(np.mean(slice_array))
                    logger.info(
                        "init slab: level=%d z=%d shape=%sx%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                        int(selected_level),
                        int(z_index),
                        int(slice_array.shape[0]),
                        int(slice_array.shape[1]),
                        str(getattr(slice_array, 'dtype', 'na')),
                        smin,
                        smax,
                        smea,
                    )
                sy, sx = plane_scale_for_level(source, selected_level)
                layer = viewer.add_image(
                    slice_array,
                    name="zarr-image",
                    multiscale=False,
                    contrast_limits=(0.0, 1.0),
                    scale=(sy, sx),
                )
                # Make sure layer draws fully opaque and on top for debugging
                layer.opacity = 1.0
                layer.blending = 'opaque'
                layer.gamma = 1.0
                layer.colormap = 'gray'
                layer.interpolation = layer_interpolation
                from napari._vispy.layers.image import (
                    VispyImageLayer,  # type: ignore
                )
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                h, w = plane_wh_for_level(source, selected_level)
                self._facade.set_data_wh((w, h))
                world_w = float(w) * float(sx)
                world_h = float(h) * float(sy)
                view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
                scene_src = "napari-zarr-adapter"
                scene_meta = f"level={self._facade.zarr_level or selected_level} shape={h}x{w}"
                if axis_labels:
                    viewer.dims.axis_labels = tuple(str(a) for a in axis_labels)
                viewer.dims.current_step = tuple(int(v) for v in applied_step) if applied_step is not None else tuple()

        if layer is None:
            if is_volume_mode:
                volume = rng.random((self._facade.volume_depth, self._facade.height, self._facade.width), dtype=np.float32)
                layer = viewer.add_image(volume, name="adapter-volume")
                viewer.dims.ndisplay = 3
                from napari._vispy.layers.image import (
                    VispyImageLayer,  # type: ignore
                )
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                d = int(volume.shape[0]); h = int(volume.shape[1]); w = int(volume.shape[2])
                self._facade.set_data_wh((w, h))
                self._facade.set_data_depth(d)
                view.camera.set_range(x=(0, w), y=(0, h), z=(0, d))
                self._facade.frame_volume_camera(w, h, d)
                layer.rendering = "mip"
                scene_src = "napari-adapter-volume"
                scene_meta = f"synthetic shape={d}x{h}x{w}"
                canonical_meta = canonical_axes_from_source(
                    axes=("z", "y", "x"),
                    shape=volume.shape,
                    step=(0,) * volume.ndim,
                    use_volume=True,
                )
            else:
                image = rng.random((self._facade.height, self._facade.width), dtype=np.float32)
                layer = viewer.add_image(image, name="adapter-image")
                from napari._vispy.layers.image import (
                    VispyImageLayer,  # type: ignore
                )
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                h = int(image.shape[0]); w = int(image.shape[1])
                self._facade.set_data_wh((w, h))
                view.camera.set_range(x=(0, w), y=(0, h))
                scene_src = "napari-adapter-image"
                scene_meta = f"synthetic shape={h}x{w}"
                canonical_meta = canonical_axes_from_source(
                    axes=("y", "x"),
                    shape=image.shape,
                    step=(0,) * image.ndim,
                    use_volume=False,
                )

        if axis_labels:
            viewer.dims.axis_labels = tuple(str(a) for a in axis_labels)
        if order:
            viewer.dims.order = tuple(int(v) for v in order)
        if ndisplay is not None:
            viewer.dims.ndisplay = int(ndisplay)

        self._facade.set_viewer(viewer)
        self._facade.set_napari_layer(layer)
        def _sync_visual_handles() -> None:
            plane_node = adapter._layer_node.get_node(2)
            volume_node = adapter._layer_node.get_node(3)
            self._facade.register_plane_visual(plane_node)
            self._facade.register_volume_visual(volume_node)
            if self._facade.viewport_state.mode is RenderMode.VOLUME:
                self._facade.ensure_volume_visual()
            else:
                self._facade.ensure_plane_visual()

        _sync_visual_handles()

        layer.events.set_data.connect(lambda e=None: _sync_visual_handles())  # type: ignore[attr-defined]
        layer.events.visible.connect(lambda e=None: _sync_visual_handles())  # type: ignore[attr-defined]

        if self._facade.log_layer_debug:
            n = adapter.node
            # Inspect napari slice state to detect empty visibility
            slice_obj = getattr(layer, '_slice', None)
            slice_empty = None
            try:
                if slice_obj is not None and hasattr(slice_obj, 'empty'):
                    slice_empty = bool(slice_obj.empty)
            except Exception:
                slice_empty = 'error'
            # Scene graph snapshot (top level)
            try:
                kids = getattr(view.scene, 'children', []) or []
                kid_types = [type(k).__name__ for k in kids]
            except Exception:
                kid_types = ['error']
            logger.info(
                "adapter node: type=%s visible=%s parent_is_scene=%s layer_visible=%s slice_empty=%s scene_children=%s",
                type(n).__name__, getattr(n, 'visible', None), bool(getattr(n, 'parent', None) is view.scene), getattr(layer, 'visible', None), slice_empty, kid_types,
            )

        if bool(getattr(worker_debug, "debug_overlay", False)):
            # Use scene.visuals API for compatibility with recent VisPy
            try:
                from vispy.scene.visuals import Rectangle  # type: ignore
                overlay = Rectangle(
                    center=(20, 20),
                    width=30,
                    height=30,
                    color=(1, 0, 0, 1),
                    border_color=(1, 1, 1, 1),
                    parent=view.scene,
                )
                # Ensure overlay draws on top of the layer visual
                try:
                    overlay.order = 20000
                except Exception:
                    pass
                # Disable depth test and enable blending for overlays
                try:
                    overlay.set_gl_state(depth_test=False, blend=True)
                except Exception:
                    logger.debug("overlay.set_gl_state failed", exc_info=True)
                self._facade.set_debug_overlay(overlay)
                logger.info("debug overlay created (red square) order=%s", getattr(overlay, 'order', 'n/a'))
            except Exception:
                logger.debug("debug overlay creation failed (Rectangle unavailable)", exc_info=True)

        logger.info("Scene init: source=%s %s", scene_src, scene_meta)
        logger.info("Camera class: %s", type(view.camera).__name__)
        canvas.render()
        return canvas, view, viewer


def _init_viewer_scene(worker: EGLRendererWorker, source: Optional[ZarrSceneSource]) -> None:
    facade = ViewerBootstrapInterface(worker)
    builder = ViewerBuilder(facade)
    ledger = worker._ledger
    step_hint = ledger_step(ledger)
    level_hint = ledger_level(ledger)
    axis_labels = ledger_axis_labels(ledger)
    order = ledger_order(ledger)
    ndisplay = ledger_ndisplay(ledger)
    canvas, view, viewer = builder.build(
        source,
        level=level_hint,
        step=step_hint,
        axis_labels=axis_labels,
        order=order,
        ndisplay=ndisplay,
    )
    worker.canvas = canvas
    worker.view = view
    worker._viewer = viewer
    assert worker.view is not None, "adapter must supply a VisPy view"
    active_mode = worker._viewport_state.mode
    _bootstrap_camera_pose(worker, RenderMode.PLANE, source, reason="bootstrap-plane")
    _bootstrap_camera_pose(worker, RenderMode.VOLUME, source, reason="bootstrap-volume")
    worker._viewport_state.mode = active_mode
    _configure_camera_for_mode(worker)


from napari_cuda.server.runtime.render_loop.plan.ledger_access import (  # noqa: E402  (avoid circular import during module load)
    axis_labels as ledger_axis_labels,
    level as ledger_level,
    ndisplay as ledger_ndisplay,
    order as ledger_order,
    step as ledger_step,
)

__all__ = [
    "CanonicalAxes",
    "ViewerBuilder",
    "_VisualHandle",
    "_ensure_plane_visual",
    "_ensure_volume_visual",
    "_init_viewer_scene",
    "_register_plane_visual",
    "_register_volume_visual",
    "apply_canonical_axes",
    "canonical_axes_from_source",
]
