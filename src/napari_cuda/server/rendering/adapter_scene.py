from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from vispy import scene
from napari.components.viewer_model import ViewerModel

from napari_cuda.server.roi import plane_scale_for_level, plane_wh_for_level
from napari_cuda.server.zarr_source import ZarrSceneSource

logger = logging.getLogger(__name__)


class AdapterScene:
    """Thin wrapper around the napari/vispy adapter scene setup.

    This class still delegates lifecycle hooks back to the worker via ``bridge``
    but relies on shared helpers for plane geometry so scale/ROI math remains in
    a single module.
    """

    def __init__(self, bridge) -> None:
        self._bridge = bridge

    def init(self, source: Optional[ZarrSceneSource]) -> Tuple[scene.SceneCanvas, scene.widgets.ViewBox, ViewerModel]:
        worker_policy = getattr(self._bridge, "_debug_policy", None)
        worker_debug = getattr(worker_policy, "worker", None)
        debug_bg = bool(getattr(worker_debug, "debug_bg_overlay", False))
        layer_interpolation = (getattr(worker_debug, "layer_interpolation", "bilinear") or "bilinear").strip().lower()
        bgcolor = (0.08, 0.08, 0.08, 1.0) if debug_bg else "black"
        # Create a native EGL-backed SceneCanvas. The worker will adopt this
        # context for capture to ensure a single GL context is used.
        canvas = scene.SceneCanvas(
            size=(self._bridge.width, self._bridge.height),
            bgcolor=bgcolor,
            show=False,
            app="egl",
            create_native=True,
        )
        view = canvas.central_widget.add_view()
        # Expose to bridge for downstream code that expects these attributes
        self._bridge.canvas = canvas
        self._bridge.view = view

        rng = np.random.default_rng(41)
        viewer = ViewerModel()

        layer = None
        adapter = None

        scene_src = "synthetic"
        scene_meta = ""

        if source is not None:
            levels = source.level_descriptors
            chosen_level = (len(levels) - 1) if levels else 0
            if levels:
                found = None
                if self._bridge.use_volume:
                    for li in range(len(levels) - 1, -1, -1):
                        try:
                            self._bridge._volume_budget_allows(source, li)
                            found = li
                            break
                        except Exception:
                            if self._bridge._log_layer_debug:
                                logger.info("init budget reject (volume): level=%d", li)
                else:
                    for li in range(len(levels) - 1, -1, -1):
                        try:
                            self._bridge._slice_budget_allows(source, li)
                            found = li
                            break
                        except Exception:
                            if self._bridge._log_layer_debug:
                                logger.info("init budget reject (slice): level=%d", li)
                chosen_level = found if found is not None else (len(levels) - 1)
            current_level = int(chosen_level)
            if self._bridge._log_layer_debug:
                logger.info("adapter init: nlevels=%d chosen=%d use_volume=%s", len(levels), int(current_level), bool(self._bridge.use_volume))
                for li, desc in enumerate(levels):
                    h, w = plane_wh_for_level(source, li)
                    dtype_size = int(np.dtype(source.dtype).itemsize)
                    bytes_est = int(h) * int(w) * dtype_size
                    logger.info("adapter levels: idx=%d shape=%s plane=%dx%d dtype=%s slice_bytes=%d", int(li), 'x'.join(str(int(x)) for x in desc.shape), int(h), int(w), str(source.dtype), bytes_est)

            init_step = source.initial_step(step_or_z=self._bridge._zarr_init_z, level=current_level)
            step = source.set_current_level(current_level, step=init_step)
            descriptor = source.level_descriptors[current_level]
            self._bridge._scene_source = source
            self._bridge._active_ms_level = current_level
            self._bridge._zarr_level = descriptor.path or None
            self._bridge._zarr_axes = ''.join(source.axes)
            self._bridge._zarr_shape = descriptor.shape
            self._bridge._zarr_dtype = str(source.dtype)
            self._bridge._zarr_clim = source.ensure_contrast(level=current_level)

            axes_lower = [str(ax).lower() for ax in source.axes]
            if 'z' in axes_lower:
                try:
                    self._bridge._z_index = int(step[axes_lower.index('z')])
                except Exception:
                    self._bridge._z_index = None

            if self._bridge.use_volume:
                data = self._bridge._get_level_volume(source, current_level)
                level_scale = source.level_scale(current_level)
                try:
                    scale_vals = [float(s) for s in level_scale]
                except Exception:
                    scale_vals = []
                while len(scale_vals) < 3:
                    scale_vals.insert(0, 1.0)
                sz, sy, sx = scale_vals[-3], scale_vals[-2], scale_vals[-1]
                self._bridge._volume_scale = (float(sz), float(sy), float(sx))
                layer = viewer.add_image(data, name="zarr-volume", scale=level_scale)
                viewer.dims.axis_labels = tuple(source.axes)
                viewer.dims.ndisplay = 3
                step_seq = list(step)
                axes_lower = [str(ax).lower() for ax in source.axes]
                if 'z' in axes_lower:
                    z_pos = axes_lower.index('z')
                    if z_pos < len(step_seq) and step_seq[z_pos] != 0:
                        step_seq[z_pos] = 0
                        try:
                            step = source.set_current_level(current_level, step=tuple(step_seq))
                        except Exception:
                            logger.debug("adapter volume: resetting source step to 0 failed", exc_info=True)
                viewer.dims.current_step = tuple(int(s) for s in step)
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                d, h, w = int(data.shape[0]), int(data.shape[1]), int(data.shape[2])
                self._bridge._data_wh = (w, h)
                self._bridge._data_d = d
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
                        "adapter volume init extent=(%.3f, %.3f, %.3f) translate=%s scale=%s",
                        world_w,
                        world_h,
                        world_d,
                        getattr(layer, 'translate', None),
                        getattr(layer, 'scale', None),
                    )
                print(
                    "adapter_scene volume",
                    {
                        'extent': (float(world_w), float(world_h), float(world_d)),
                        'layer_translate': getattr(layer, 'translate', None),
                        'layer_scale': getattr(layer, 'scale', None),
                    },
                    flush=True,
                )
                self._bridge._frame_volume_camera(world_w, world_h, world_d)
                self._bridge._z_index = 0
                layer.rendering = "mip"
                scene_src = "napari-zarr-volume"
                scene_meta = f"level={self._bridge._zarr_level or current_level} shape={d}x{h}x{w}"
            else:
                z_idx = self._bridge._z_index or 0
                # Load initial 2D slab using the worker's slice helper.
                # Metrics plumbing was removed; we no longer return a metrics tuple here.
                slice_array = self._bridge._load_slice(source, current_level, int(z_idx))
                if self._bridge._log_layer_debug:
                    try:
                        smin = float(np.nanmin(slice_array)) if hasattr(np, 'nanmin') else float(np.min(slice_array))
                        smax = float(np.nanmax(slice_array)) if hasattr(np, 'nanmax') else float(np.max(slice_array))
                        smea = float(np.mean(slice_array))
                        logger.info(
                            "init slab: level=%d z=%d shape=%sx%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                            int(current_level),
                            int(z_idx),
                            int(slice_array.shape[0]),
                            int(slice_array.shape[1]),
                            str(getattr(slice_array, 'dtype', 'na')),
                            smin,
                            smax,
                            smea,
                        )
                    except Exception:
                        logger.debug("init slab stats failed", exc_info=True)
                # slice_array is normalized to [0,1] by ZarrSceneSource.slice(compute=True),
                # so use fixed contrast_limits=(0,1) to avoid black output from raw-domain clims.
                sy, sx = plane_scale_for_level(source, current_level)
                layer = viewer.add_image(
                    slice_array,
                    name="zarr-image",
                    multiscale=False,
                    contrast_limits=(0.0, 1.0),
                    scale=(sy, sx),
                )
                viewer.dims.axis_labels = tuple(source.axes)
                viewer.dims.ndisplay = 2
                viewer.dims.current_step = tuple(step)
                # Make sure layer draws fully opaque and on top for debugging
                try:
                    layer.opacity = 1.0
                    layer.blending = 'opaque'
                    layer.gamma = 1.0
                    layer.colormap = 'gray'
                except Exception:
                    logger.debug("adapter: layer visual props set failed", exc_info=True)
                # Configurable interpolation; default to policy-specified value (bilinear by default)
                interp = layer_interpolation or 'bilinear'
                if hasattr(layer, 'interpolation'):
                    try:
                        layer.interpolation = interp
                    except Exception:
                        logger.debug("adapter: setting interpolation failed (interp=%s)", interp, exc_info=True)
                try:
                    self._bridge._set_dims_range_for_level(source, current_level)
                except Exception:
                    logger.debug("adapter: set dims range failed", exc_info=True)
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                h, w = plane_wh_for_level(source, current_level)
                sy, sx = plane_scale_for_level(source, current_level)
                self._bridge._data_wh = (w, h)
                # Use world extents (shape * scale) so the image falls within the view frustum
                world_w = float(w) * float(sx)
                world_h = float(h) * float(sy)
                view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
                scene_src = "napari-zarr-adapter"
                scene_meta = f"level={self._bridge._zarr_level or current_level} shape={h}x{w}"

        if layer is None:
            if self._bridge.use_volume:
                volume = rng.random((self._bridge.volume_depth, self._bridge.height, self._bridge.width), dtype=np.float32)
                layer = viewer.add_image(volume, name="adapter-volume")
                viewer.dims.ndisplay = 3
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                d = int(volume.shape[0]); h = int(volume.shape[1]); w = int(volume.shape[2])
                self._bridge._data_wh = (w, h)
                self._bridge._data_d = d
                view.camera.set_range(x=(0, w), y=(0, h), z=(0, d))
                self._bridge._frame_volume_camera(w, h, d)
                layer.rendering = "mip"
                scene_src = "napari-adapter-volume"
                scene_meta = f"synthetic shape={d}x{h}x{w}"
            else:
                image = rng.random((self._bridge.height, self._bridge.width), dtype=np.float32)
                layer = viewer.add_image(image, name="adapter-image")
                viewer.dims.ndisplay = 2
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                h = int(image.shape[0]); w = int(image.shape[1])
                self._bridge._data_wh = (w, h)
                view.camera.set_range(x=(0, w), y=(0, h))
                scene_src = "napari-adapter-image"
                scene_meta = f"synthetic shape={h}x{w}"

        def _ensure_node_registered() -> None:
            try:
                n = adapter.node
                # Register node with the viewbox scene explicitly
                try:
                    if getattr(n, 'parent', None) is not view.scene:
                        view.add(n)  # ensures proper ViewBox registration
                except Exception:
                    # fallback to direct parent assignment
                    n.parent = view.scene
                # Force visibility/opacity in case layer state hasn’t propagated yet
                try:
                    n.visible = True
                    n.opacity = 1.0
                except Exception:
                    pass
                try:
                    # Prefer adapter property to trigger proper blending updates
                    setattr(adapter, 'order', 10_000)
                except Exception:
                    setattr(n, 'order', 10_000)
                # Update bridge visual pointer so downstream volume params hit the active node
                self._bridge._visual = n
            except Exception:
                logger.debug("adapter: ensure_node_registered failed", exc_info=True)

        _ensure_node_registered()

        # Re-assert parent/order whenever napari swaps node on display/data change
        try:
            layer.events.set_data.connect(lambda e=None: _ensure_node_registered())  # type: ignore[attr-defined]
            layer.events.display.connect(lambda e=None: _ensure_node_registered())  # type: ignore[attr-defined]
            layer.events.visible.connect(lambda e=None: _ensure_node_registered())  # type: ignore[attr-defined]
        except Exception:
            logger.debug("adapter: connecting layer events failed", exc_info=True)

        self._bridge._visual = adapter.node
        self._bridge._viewer = viewer
        self._bridge._napari_layer = layer
        

        if self._bridge._log_layer_debug:
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
                    setattr(overlay, 'order', 20_000)
                except Exception:
                    pass
                # Disable depth test and enable blending for overlays
                try:
                    overlay.set_gl_state(depth_test=False, blend=True)
                except Exception:
                    logger.debug("overlay.set_gl_state failed", exc_info=True)
                setattr(self._bridge, '_debug_overlay', overlay)
                logger.info("debug overlay created (red square) order=%s", getattr(overlay, 'order', 'n/a'))
            except Exception:
                logger.debug("debug overlay creation failed (Rectangle unavailable)", exc_info=True)

        self._bridge._notify_scene_refresh()

        logger.info("Scene init: source=%s %s", scene_src, scene_meta)
        logger.info("Camera class: %s", type(view.camera).__name__)
        canvas.render()
        return canvas, view, viewer
