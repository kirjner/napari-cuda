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
        self._b = bridge

    def init(self, source: Optional[ZarrSceneSource]) -> Tuple[scene.SceneCanvas, scene.widgets.ViewBox, ViewerModel]:
        try:
            _bg_dbg = int(os.getenv('NAPARI_CUDA_DEBUG_BG', '0') or '0')
        except Exception:
            _bg_dbg = 0
        bgcolor = (0.08, 0.08, 0.08, 1.0) if _bg_dbg else "black"
        canvas = scene.SceneCanvas(size=(self._b.width, self._b.height), bgcolor=bgcolor, show=False, app="egl")
        view = canvas.central_widget.add_view()
        # Expose to bridge for downstream code that expects these attributes
        self._b.canvas = canvas
        self._b.view = view

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
                if self._b.use_volume:
                    for li in range(len(levels) - 1, -1, -1):
                        try:
                            self._b._volume_budget_allows(source, li)
                            found = li
                            break
                        except Exception:
                            if self._b._log_layer_debug:
                                logger.info("init budget reject (volume): level=%d", li)
                else:
                    for li in range(len(levels) - 1, -1, -1):
                        try:
                            self._b._slice_budget_allows(source, li)
                            found = li
                            break
                        except Exception:
                            if self._b._log_layer_debug:
                                logger.info("init budget reject (slice): level=%d", li)
                chosen_level = found if found is not None else (len(levels) - 1)
            current_level = int(chosen_level)
            if self._b._log_layer_debug:
                try:
                    logger.info("adapter init: nlevels=%d chosen=%d use_volume=%s", len(levels), int(current_level), bool(self._b.use_volume))
                except Exception:
                    logger.debug("adapter init debug log failed", exc_info=True)
                try:
                    for li, desc in enumerate(levels):
                        h, w = self._b._plane_wh_for_level(source, li)
                        dtype_size = int(np.dtype(source.dtype).itemsize)
                        bytes_est = int(h) * int(w) * dtype_size
                        logger.info("adapter levels: idx=%d shape=%s plane=%dx%d dtype=%s slice_bytes=%d", int(li), 'x'.join(str(int(x)) for x in desc.shape), int(h), int(w), str(source.dtype), bytes_est)
                except Exception:
                    logger.debug("adapter init level listing failed", exc_info=True)

            init_step = source.initial_step(step_or_z=self._b._zarr_init_z, level=current_level)
            step = source.set_current_level(current_level, step=init_step)
            descriptor = source.level_descriptors[current_level]
            self._b._scene_source = source
            self._b._active_ms_level = current_level
            self._b._zarr_level = descriptor.path or None
            self._b._zarr_axes = ''.join(source.axes)
            self._b._zarr_shape = descriptor.shape
            self._b._zarr_dtype = str(source.dtype)
            self._b._zarr_clim = source.ensure_contrast(level=current_level)

            axes_lower = [str(ax).lower() for ax in source.axes]
            if 'z' in axes_lower:
                try:
                    self._b._z_index = int(step[axes_lower.index('z')])
                except Exception:
                    self._b._z_index = None

            if self._b.use_volume:
                data = self._b._get_level_volume(source, current_level)
                layer = viewer.add_image(data, name="zarr-volume", scale=source.level_scale(current_level))
                viewer.dims.axis_labels = tuple(source.axes)
                viewer.dims.ndisplay = 3
                viewer.dims.current_step = tuple(step)
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                d, h, w = int(data.shape[0]), int(data.shape[1]), int(data.shape[2])
                self._b._data_wh = (w, h)
                self._b._data_d = d
                view.camera.set_range(x=(0, w), y=(0, h), z=(0, d))
                self._b._frame_volume_camera(w, h, d)
                layer.rendering = "mip"
                scene_src = "napari-zarr-volume"
                scene_meta = f"level={self._b._zarr_level or current_level} shape={d}x{h}x{w}"
            else:
                z_idx = self._b._z_index or 0
                slab2d = source.slice(current_level, int(z_idx), compute=True)
                if self._b._log_layer_debug:
                    try:
                        smin = float(np.nanmin(slab2d)) if hasattr(np, 'nanmin') else float(np.min(slab2d))
                        smax = float(np.nanmax(slab2d)) if hasattr(np, 'nanmax') else float(np.max(slab2d))
                        smea = float(np.mean(slab2d))
                        logger.info("init slab: level=%d z=%d shape=%sx%s dtype=%s min=%.6f max=%.6f mean=%.6f", int(current_level), int(z_idx), int(slab2d.shape[0]), int(slab2d.shape[1]), str(getattr(slab2d, 'dtype', 'na')), smin, smax, smea)
                    except Exception:
                        logger.debug("init slab stats failed", exc_info=True)
                # slab2d is normalized to [0,1] by ZarrSceneSource.slice(compute=True),
                # so use fixed contrast_limits=(0,1) to avoid black output from raw-domain clims.
                sy, sx = self._b._plane_scale_for_level(source, current_level)
                layer = viewer.add_image(
                    slab2d,
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
                    self._b._set_dims_range_for_level(source, current_level)
                except Exception:
                    logger.debug("adapter: set dims range failed", exc_info=True)
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                h, w = self._b._plane_wh_for_level(source, current_level)
                sy, sx = self._b._plane_scale_for_level(source, current_level)
                self._b._data_wh = (w, h)
                # Use world extents (shape * scale) so the image falls within the view frustum
                world_w = float(w) * float(sx)
                world_h = float(h) * float(sy)
                view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
                scene_src = "napari-zarr-adapter"
                scene_meta = f"level={self._b._zarr_level or current_level} shape={h}x{w}"

        if layer is None:
            if self._b.use_volume:
                volume = rng.random((self._b.volume_depth, self._b.height, self._b.width), dtype=np.float32)
                layer = viewer.add_image(volume, name="adapter-volume")
                viewer.dims.ndisplay = 3
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30, fov=60)
                d = int(volume.shape[0]); h = int(volume.shape[1]); w = int(volume.shape[2])
                self._b._data_wh = (w, h)
                self._b._data_d = d
                view.camera.set_range(x=(0, w), y=(0, h), z=(0, d))
                self._b._frame_volume_camera(w, h, d)
                layer.rendering = "mip"
                scene_src = "napari-adapter-volume"
                scene_meta = f"synthetic shape={d}x{h}x{w}"
            else:
                image = rng.random((self._b.height, self._b.width), dtype=np.float32)
                layer = viewer.add_image(image, name="adapter-image")
                viewer.dims.ndisplay = 2
                from napari._vispy.layers.image import VispyImageLayer  # type: ignore
                adapter = VispyImageLayer(layer)
                view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
                h = int(image.shape[0]); w = int(image.shape[1])
                self._b._data_wh = (w, h)
                view.camera.set_range(x=(0, w), y=(0, h))
                scene_src = "napari-adapter-image"
                scene_meta = f"synthetic shape={h}x{w}"

        node = adapter.node
        node.parent = view.scene
        try:
            # Force draw order to front to avoid accidental occlusion
            setattr(node, 'order', 10_000)
        except Exception:
            logger.debug("adapter: set node.order failed", exc_info=True)
        self._b._visual = node
        self._b._viewer = viewer
        self._b._napari_layer = layer
        self._b._napari_adapter = adapter

        if self._b._log_layer_debug:
            try:
                logger.info("adapter node: type=%s visible=%s parent_is_scene=%s layer_visible=%s", type(node).__name__, getattr(node, 'visible', None), bool(getattr(node, 'parent', None) is view.scene), getattr(layer, 'visible', None))
            except Exception:
                logger.debug("adapter node visibility log failed", exc_info=True)

        try:
            if int(os.getenv('NAPARI_CUDA_DEBUG_OVERLAY', '0') or '0'):
                from vispy.visuals import Rectangle
                overlay = Rectangle(center=(20, 20), width=30, height=30, color=(1, 0, 0, 1), parent=view.scene)
                setattr(self._b, '_debug_overlay', overlay)
                logger.info("debug overlay created (red square)")
        except Exception:
            logger.debug("debug overlay creation failed", exc_info=True)

        self._b._notify_scene_refresh()

        try:
            logger.info("Scene init: source=%s %s", scene_src, scene_meta)
        except Exception:
            logger.debug("Scene init log failed", exc_info=True)
        try:
            logger.info("Camera class: %s", type(view.camera).__name__)
        except Exception:
            pass
        canvas.render()
        return canvas, view, viewer
