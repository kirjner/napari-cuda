from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
from vispy import scene
from napari.components.viewer_model import ViewerModel

from napari_cuda.server.zarr_source import ZarrSceneSource

logger = logging.getLogger(__name__)


class AdapterScene:
    """Thin wrapper around the napari/vispy adapter scene setup.

    This class delegates helper computations back to the worker via the provided
    bridge (an EGLRendererWorker-like object) to avoid logic duplication.
    """

    def __init__(self, bridge) -> None:
        self._bridge = bridge

    def init(self, source: Optional[ZarrSceneSource]) -> Tuple[scene.SceneCanvas, scene.widgets.ViewBox, ViewerModel]:
        try:
            _bg_dbg = int(os.getenv('NAPARI_CUDA_DEBUG_BG', '0') or '0')
        except Exception:
            _bg_dbg = 0
        bgcolor = (0.08, 0.08, 0.08, 1.0) if _bg_dbg else "black"
        canvas = scene.SceneCanvas(size=(self._bridge.width, self._bridge.height), bgcolor=bgcolor, show=False, app="egl")
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
                    h, w = self._bridge._plane_wh_for_level(source, li)
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
                layer = viewer.add_image(data, name="zarr-volume", scale=source.level_scale(current_level))
                viewer.dims.axis_labels = tuple(source.axes)
                viewer.dims.ndisplay = 3
                viewer.dims.current_step = tuple(step)
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                d, h, w = int(data.shape[0]), int(data.shape[1]), int(data.shape[2])
                self._bridge._data_wh = (w, h)
                self._bridge._data_d = d
                view.camera.set_range(x=(0, w), y=(0, h), z=(0, d))
                self._bridge._frame_volume_camera(w, h, d)
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
                sy, sx = self._bridge._plane_scale_for_level(source, current_level)
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
                try:
                    # Make sure layer draws fully opaque and on top for debugging
                    layer.opacity = 1.0
                    layer.blending = 'opaque'
                    layer.gamma = 1.0
                    layer.colormap = 'gray'
                except Exception:
                    logger.debug("adapter: layer visual props set failed", exc_info=True)
                try:
                    self._bridge._set_dims_range_for_level(source, current_level)
                except Exception:
                    logger.debug("adapter: set dims range failed", exc_info=True)
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                h, w = self._bridge._plane_wh_for_level(source, current_level)
                sy, sx = self._bridge._plane_scale_for_level(source, current_level)
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

        node = adapter.node
        node.parent = view.scene
        # Force draw order to front to avoid accidental occlusion
        setattr(node, 'order', 10_000)
        self._bridge._visual = node
        self._bridge._viewer = viewer
        self._bridge._napari_layer = layer
        

        if self._bridge._log_layer_debug:
            logger.info("adapter node: type=%s visible=%s parent_is_scene=%s layer_visible=%s", type(node).__name__, getattr(node, 'visible', None), bool(getattr(node, 'parent', None) is view.scene), getattr(layer, 'visible', None))

        if int(os.getenv('NAPARI_CUDA_DEBUG_OVERLAY', '0') or '0'):
            from vispy.visuals import Rectangle
            overlay = Rectangle(center=(20, 20), width=30, height=30, color=(1, 0, 0, 1), parent=view.scene)
            setattr(self._bridge, '_debug_overlay', overlay)
            logger.info("debug overlay created (red square)")

        self._bridge._notify_scene_refresh()

        logger.info("Scene init: source=%s %s", scene_src, scene_meta)
        logger.info("Camera class: %s", type(view.camera).__name__)
        canvas.render()
        return canvas, view, viewer
