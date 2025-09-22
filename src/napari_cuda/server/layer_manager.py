"""Scene management helpers for the server-side napari viewer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from napari.components.viewer_model import ViewerModel

from napari_cuda.protocol.messages import (
    CameraSpec,
    DimsSpec,
    LayerRenderHints,
    LayerSpec,
    MultiscaleLevelSpec,
    MultiscaleSpec,
    SceneSpec,
    SceneSpecMessage,
)
from napari_cuda.server import scene_spec as _scene


logger = logging.getLogger(__name__)

@dataclass
class _WorkerSnapshot:
    """Lightweight view of worker state needed for scene metadata."""

    ndim: int
    shape: List[int]
    axis_labels: List[str]
    dtype: Optional[str]
    is_volume: bool
    zarr_axes: Optional[str]
    zarr_dtype: Optional[str]


class ViewerSceneManager:
    """Maintains a headless ``ViewerModel`` and derived scene specifications."""

    def __init__(
        self,
        canvas_size: tuple[int, int],
        default_layer_id: str = "layer-0",
        default_layer_name: str = "napari-cuda",
        viewer: Optional[ViewerModel] = None,
    ) -> None:
        self._viewer = viewer or ViewerModel()
        self._owns_viewer = viewer is None
        self._canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self._default_layer_id = default_layer_id
        self._default_layer_name = default_layer_name
        self._capabilities: List[str] = ["layer.update", "layer.remove"]
        self._scene: Optional[SceneSpec] = None

    # ------------------------------------------------------------------
    # Public API
    def update_from_sources(
        self,
        *,
        worker: Optional[object],
        scene_state: Optional[object],
        multiscale_state: Optional[Dict[str, Any]],
        volume_state: Optional[Dict[str, Any]],
        current_step: Optional[Iterable[int]],
        ndisplay: Optional[int],
        zarr_path: Optional[str],
        viewer_model: Optional[ViewerModel] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> SceneSpec:
        """Refresh the cached scene metadata from the latest server state."""

        if viewer_model is not None and viewer_model is not self._viewer:
            self._viewer = viewer_model
            self._owns_viewer = False

        worker_info = self._snapshot_worker(worker, ndisplay, viewer_model=viewer_model)

        adapter_layer = None
        if viewer_model is not None:
            try:
                adapter_layer = viewer_model.layers[0]
            except Exception:
                adapter_layer = None

        extras_map: Dict[str, Any] = {}
        if extras:
            extras_map.update(extras)
        if adapter_layer is not None:
            extras_map.update(self._adapter_layer_extras(adapter_layer))

        layer_metadata: Dict[str, Any] = {}
        if adapter_layer is not None:
            thumbnail = self._adapter_layer_thumbnail(adapter_layer)
            if thumbnail is not None:
                layer_metadata["thumbnail"] = thumbnail

        layer_spec = self._build_layer_spec(
            worker_info,
            multiscale_state=multiscale_state,
            volume_state=volume_state,
            zarr_path=zarr_path,
            extras=extras_map or None,
            metadata=layer_metadata or None,
            adapter_layer=adapter_layer,
        )
        dims_spec = self._build_dims_spec(
            layer_spec,
            current_step=current_step,
            ndisplay=ndisplay,
            viewer_model=viewer_model,
        )
        camera_spec = self._build_camera_spec(scene_state, ndisplay=ndisplay)

        metadata: Dict[str, Any] = {}
        if zarr_path:
            metadata["zarr_path"] = zarr_path
        if extras_map:
            for key, value in extras_map.items():
                if value is not None:
                    metadata[key] = value

        self._scene = SceneSpec(
            layers=[layer_spec],
            dims=dims_spec,
            camera=camera_spec,
            capabilities=list(self._capabilities),
            metadata=metadata or None,
        )

        self._apply_to_viewer(layer_spec, dims_spec, camera_spec)
        return self._scene

    def scene_spec(self) -> Optional[SceneSpec]:
        return self._scene

    def scene_message(self, timestamp: Optional[float] = None) -> SceneSpecMessage:
        scene = self._scene or self.update_from_sources(
            worker=None,
            scene_state=None,
            multiscale_state=None,
            volume_state=None,
            current_step=None,
            ndisplay=2,
            zarr_path=None,
        )
        return SceneSpecMessage(timestamp=timestamp, scene=scene)

    def dims_metadata(self) -> Dict[str, Any]:
        scene = self._scene
        if scene is None or scene.dims is None:
            return {}
        dims_dict = scene.dims.to_dict()
        meta: Dict[str, Any] = {
            "ndim": dims_dict.get("ndim"),
            "order": dims_dict.get("order"),
            "sizes": dims_dict.get("sizes"),
            "range": dims_dict.get("range"),
            "axis_labels": dims_dict.get("axis_labels"),
        }
        layer = scene.layers[0] if scene.layers else None
        if layer is not None:
            extras = layer.extras or {}
            if "is_volume" in extras:
                meta["volume"] = bool(extras["is_volume"])
            if layer.render is not None:
                meta["render"] = layer.render.to_dict()
            if layer.multiscale is not None:
                ms_dict = layer.multiscale.to_dict()
                policy = None
                index_space = None
                md = ms_dict.pop("metadata", None)
                if isinstance(md, dict):
                    policy = md.get("policy")
                    index_space = md.get("index_space")
                if policy is not None:
                    ms_dict["policy"] = policy
                if index_space is not None:
                    ms_dict["index_space"] = index_space
                meta["multiscale"] = ms_dict
            # Robustly derive sizes/range from active level when available.
            try:
                eff_shape = None
                if layer.multiscale is not None and layer.multiscale.levels:
                    cur = int(layer.multiscale.current_level or 0)
                    levels = layer.multiscale.levels
                    if 0 <= cur < len(levels) and levels[cur].shape:
                        eff_shape = [int(s) for s in levels[cur].shape or []]
                if not eff_shape and layer.shape:
                    eff_shape = [int(s) for s in layer.shape]
                if eff_shape:
                    meta["sizes"] = list(eff_shape)
                    meta["range"] = [[0, max(0, int(s) - 1)] for s in eff_shape]
            except Exception:
                logger.debug("dims_metadata: shape/range reconciliation failed", exc_info=True)
        return {key: value for key, value in meta.items() if value is not None}

    # ------------------------------------------------------------------
    # Internal helpers
    def _snapshot_worker(
        self,
        worker: Optional[object],
        ndisplay: Optional[int],
        *,
        viewer_model: Optional[ViewerModel] = None,
    ) -> _WorkerSnapshot:
        if viewer_model is not None:
            adapter_snapshot = self._snapshot_from_viewer(viewer_model, ndisplay)
            if adapter_snapshot is not None:
                try:
                    if worker is not None and not bool(getattr(worker, "use_volume", False)):
                        zshape = getattr(worker, "_zarr_shape", None)
                        if isinstance(zshape, tuple) and len(zshape) >= 3:
                            return _WorkerSnapshot(
                                ndim=3,
                                shape=[int(zshape[0]), int(zshape[1]), int(zshape[2])],
                                axis_labels=["z", "y", "x"],
                                dtype=adapter_snapshot.dtype,
                                is_volume=False,
                                zarr_axes=getattr(worker, "_zarr_axes", None),
                                zarr_dtype=getattr(worker, "_zarr_dtype", None),
                            )
                except Exception:
                    logger.debug("_snapshot_worker: adapter 2D->3D dims adjust failed", exc_info=True)
                return adapter_snapshot

        width, height = self._canvas_size
        if worker is None:
            ndim = 2
            shape = [height, width]
            axis_labels = ["y", "x"]
            return _WorkerSnapshot(
                ndim=ndim,
                shape=shape,
                axis_labels=axis_labels,
                dtype=None,
                is_volume=False,
                zarr_axes=None,
                zarr_dtype=None,
            )

        axes = getattr(worker, "_zarr_axes", None)
        if axes is not None:
            axis_labels = [str(a) for a in axes]
        else:
            axis_labels = ["y", "x"]

        is_volume = bool(getattr(worker, "use_volume", False))
        zarr_shape = getattr(worker, "_zarr_shape", None)

        if is_volume:
            if isinstance(zarr_shape, tuple) and len(zarr_shape) >= 3:
                shape = [int(x) for x in zarr_shape[:3]]
            else:
                depth = int(getattr(worker, "_data_d", 0) or getattr(worker, "volume_depth", 0) or 1)
                w, h = getattr(worker, "_data_wh", (self._canvas_size[0], self._canvas_size[1]))
                shape = [depth, int(h), int(w)]
            if len(axis_labels) != len(shape):
                axis_labels = ["z", "y", "x"]
        else:
            # Prefer exposing a Z axis when slicing a 3D Zarr volume so the client
            # can show a proper Z slider with the active level's depth range.
            if isinstance(zarr_shape, tuple) and len(zarr_shape) >= 3:
                # Keep conventional order Z, Y, X for 3D metadata even in 2D render
                shape = [int(zarr_shape[0]), int(zarr_shape[1]), int(zarr_shape[2])]
                if len(axis_labels) != 3:
                    axis_labels = ["z", "y", "x"]
            elif isinstance(zarr_shape, tuple) and len(zarr_shape) >= 2:
                shape = [int(zarr_shape[-2]), int(zarr_shape[-1])]
                if len(axis_labels) != 2:
                    axis_labels = ["y", "x"]
            else:
                w, h = getattr(worker, "_data_wh", (self._canvas_size[0], self._canvas_size[1]))
                shape = [int(h), int(w)]
                if len(axis_labels) != 2:
                    axis_labels = ["y", "x"]

        dtype = getattr(worker, "_zarr_dtype", None) or getattr(worker, "volume_dtype", None)
        if dtype is not None:
            dtype = str(dtype)

        ndim = len(shape)
        if ndisplay is not None and ndisplay > ndim:
            ndisplay = ndim

        return _WorkerSnapshot(
            ndim=ndim,
            shape=shape,
            axis_labels=axis_labels,
            dtype=dtype,
            is_volume=is_volume or (ndisplay == 3),
            zarr_axes=getattr(worker, "_zarr_axes", None),
            zarr_dtype=getattr(worker, "_zarr_dtype", None),
        )

    def _snapshot_from_viewer(
        self, viewer: ViewerModel, ndisplay: Optional[int]
    ) -> Optional[_WorkerSnapshot]:
        layers = getattr(viewer, "layers", None)
        if not layers:
            return None
        try:
            layer = layers[0]
        except Exception:
            return None

        shape = self._shape_from_data(layer.data)
        if not shape:
            shape = [self._canvas_size[1], self._canvas_size[0]]
        ndim = len(shape)
        axis_labels = self._normalize_axis_labels(viewer, ndim)
        dtype = self._dtype_from_layer(layer)
        zarr_axes = self._zarr_axes_from_layer(layer)
        zarr_dtype = dtype
        is_volume = bool((ndisplay if ndisplay is not None else viewer.dims.ndisplay) == 3)

        return _WorkerSnapshot(
            ndim=ndim,
            shape=shape,
            axis_labels=axis_labels,
            dtype=dtype,
            is_volume=is_volume,
            zarr_axes=zarr_axes,
            zarr_dtype=zarr_dtype,
        )

    @staticmethod
    def _shape_from_data(data: Any) -> List[int]:
        if hasattr(data, "shape"):
            try:
                return [int(max(1, s)) for s in tuple(data.shape)]
            except Exception:
                return []
        if isinstance(data, (list, tuple)) and data:
            return ViewerSceneManager._shape_from_data(data[0])
        return []

    @staticmethod
    def _dtype_from_layer(layer: Any) -> Optional[str]:
        dtype = getattr(layer, "dtype", None)
        if dtype is None:
            data = getattr(layer, "data", None)
            dtype = getattr(data, "dtype", None)
            if dtype is None and isinstance(data, (list, tuple)) and data:
                dtype = getattr(data[0], "dtype", None)
        if dtype is None:
            return None
        try:
            return str(np.dtype(dtype))
        except Exception:
            return str(dtype)

    def _normalize_axis_labels(self, viewer: ViewerModel, ndim: int) -> List[str]:
        defaults: Dict[int, List[str]] = {
            1: ["x"],
            2: ["y", "x"],
            3: ["z", "y", "x"],
            4: ["t", "z", "y", "x"],
        }
        try:
            labels_source = list(viewer.dims.axis_labels or [])
        except Exception:
            labels_source = []
        normalized: List[str] = []
        default_labels = defaults.get(ndim, [f"axis-{i}" for i in range(ndim)])
        for idx in range(ndim):
            label = None
            if idx < len(labels_source):
                label = labels_source[idx]
            if label is None or str(label).strip() == "":
                normalized.append(default_labels[idx] if idx < len(default_labels) else f"axis-{idx}")
            else:
                normalized.append(str(label))
        return normalized

    @staticmethod
    def _zarr_axes_from_layer(layer: Any) -> Optional[str]:
        metadata = getattr(layer, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        axes = metadata.get("axes")
        if isinstance(axes, str):
            return axes
        if isinstance(axes, (list, tuple)):
            try:
                return "".join(str(a) for a in axes)
            except Exception:
                return None
        return None

    @staticmethod
    def _adapter_layer_extras(layer: Any) -> Dict[str, Any]:
        extras: Dict[str, Any] = {"adapter_engine": "napari-vispy"}
        name = getattr(layer, "name", None)
        if name:
            extras["layer_name"] = str(name)
        visible = getattr(layer, "visible", None)
        if visible is not None:
            extras["visible"] = bool(visible)
        opacity_val = getattr(layer, "opacity", None)
        if opacity_val is not None:
            try:
                extras["opacity"] = float(opacity_val)
            except Exception:
                logger.debug("layer_manager: opacity coercion failed", exc_info=True)
        blending = getattr(layer, "blending", None)
        if blending is not None:
            extras["blending"] = str(blending)
        interpolation = getattr(layer, "interpolation", None)
        if interpolation is not None:
            extras["interpolation"] = str(interpolation)
        colormap = getattr(layer, "colormap", None)
        if colormap is not None:
            cmap_name = getattr(colormap, "name", None)
            if cmap_name is None and isinstance(colormap, dict):
                cmap_name = colormap.get("name")
            if cmap_name is not None:
                extras["colormap"] = str(cmap_name)
        rendering = getattr(layer, "rendering", None)
        if rendering is not None:
            extras["rendering"] = str(rendering)
        scale = getattr(layer, "scale", None)
        if scale is not None:
            try:
                extras["scale"] = [float(s) for s in scale]
            except Exception:
                logger.debug("layer_manager: scale coercion failed", exc_info=True)
        clim = getattr(layer, "contrast_limits", None)
        if clim is not None:
            try:
                extras["contrast_limits"] = [float(clim[0]), float(clim[1])]
            except Exception:
                logger.debug("layer_manager: contrast_limits coercion failed", exc_info=True)
        return {k: v for k, v in extras.items() if v is not None}

    @staticmethod
    def _adapter_layer_thumbnail(layer: Any) -> Optional[List[List[List[float]]]]:
        try:
            updater = getattr(layer, '_update_thumbnail', None)
            if callable(updater):
                updater()
        except Exception:
            logger.debug('layer_manager: thumbnail refresh failed', exc_info=True)
        try:
            thumbnail = getattr(layer, 'thumbnail', None)
        except Exception:
            return None
        if thumbnail is None:
            return None
        try:
            arr = np.asarray(thumbnail)
        except Exception:
            return None
        if arr.size == 0:
            return None
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.ndim != 3:
            return None
        try:
            arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            arr = arr.astype(np.float32, copy=False)
        return arr.tolist()

    def _build_layer_spec(
        self,
        worker_snapshot: _WorkerSnapshot,
        *,
        multiscale_state: Optional[Dict[str, Any]],
        volume_state: Optional[Dict[str, Any]],
        zarr_path: Optional[str],
        extras: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
        adapter_layer: Optional[Any] = None,
    ) -> LayerSpec:
        # Collect multiscale descriptors
        ms_adapt = self._adapter_multiscale_spec(adapter_layer, extras, worker_snapshot.shape)
        ms_state = self._build_multiscale_spec(multiscale_state, worker_snapshot.shape)
        levels: Optional[List[Dict[str, Any]]] = None
        cur_level: Optional[int] = None
        ms_metadata: Optional[Dict[str, Any]] = None
        if ms_adapt is not None:
            try:
                levels = [lvl.to_dict() for lvl in ms_adapt.levels]
                cur_level = int(ms_adapt.current_level)
            except Exception:
                logger.debug("layer_manager: adapt ms to_dict failed", exc_info=True)
            ms_metadata = getattr(ms_adapt, "metadata", None)
        elif ms_state is not None:
            try:
                levels = [lvl.to_dict() for lvl in ms_state.levels]
                cur_level = int(ms_state.current_level)
            except Exception:
                logger.debug("layer_manager: state ms to_dict failed", exc_info=True)
            ms_metadata = getattr(ms_state, "metadata", None)

        render_hints = self._build_render_hints(volume_state)
        # Compose extras map
        layer_extras: Dict[str, Any] = {"zarr_axes": worker_snapshot.zarr_axes}
        if extras:
            layer_extras.update({k: v for k, v in extras.items() if v is not None})

        spec = _scene.build_layer_spec(
            shape=worker_snapshot.shape,
            is_volume=worker_snapshot.is_volume,
            zarr_path=zarr_path,
            multiscale_levels=levels,
            multiscale_current_level=cur_level,
            render_hints=render_hints.to_dict() if render_hints else None,  # type: ignore[arg-type]
            extras=layer_extras or None,
            metadata=metadata or None,
        )
        if spec.multiscale is not None and ms_metadata:
            spec.multiscale.metadata = dict(ms_metadata)
        # Preserve dtype and axis labels from snapshot logic
        spec.dtype = worker_snapshot.dtype
        spec.axis_labels = list(worker_snapshot.axis_labels)
        # Preserve any contrast limits derived earlier
        if adapter_layer is not None and hasattr(adapter_layer, "contrast_limits"):
            clim = adapter_layer.contrast_limits  # type: ignore[attr-defined]
            if isinstance(clim, (list, tuple)) and len(clim) >= 2:
                spec.contrast_limits = [float(clim[0]), float(clim[1])]
        elif isinstance(volume_state, dict):
            clim_vs = volume_state.get("clim")
            if isinstance(clim_vs, (list, tuple)) and len(clim_vs) >= 2:
                try:
                    spec.contrast_limits = [float(clim_vs[0]), float(clim_vs[1])]
                except Exception:
                    logger.debug("layer_manager: volume-state contrast coercion failed", exc_info=True)
        return spec

    def _build_dims_spec(
        self,
        layer_spec: LayerSpec,
        *,
        current_step: Optional[Iterable[int]],
        ndisplay: Optional[int],
        viewer_model: Optional[ViewerModel],
    ) -> DimsSpec:
        # Preserve viewer-provided current_step and ndisplay when available
        viewer_dims = viewer_model.dims if viewer_model is not None else None
        if current_step is None and viewer_dims is not None:
            current_step = tuple(viewer_dims.current_step)
        if ndisplay is None and viewer_dims is not None:
            ndisplay = int(viewer_dims.ndisplay)

        return _scene.build_dims_spec(
            layer_spec=layer_spec.to_dict(),
            current_step=current_step,
            ndisplay=ndisplay,
            axis_labels=layer_spec.axis_labels,
        )

    def _build_camera_spec(
        self, scene_state: Optional[object], *, ndisplay: Optional[int]
    ) -> Optional[CameraSpec]:
        if scene_state is None:
            return None
        center = scene_state.center  # type: ignore[attr-defined]
        zoom = scene_state.zoom  # type: ignore[attr-defined]
        angles = scene_state.angles  # type: ignore[attr-defined]
        return _scene.build_camera_spec(
            center=center,
            zoom=zoom,
            angles=angles,
            ndisplay=ndisplay,
        )

    def _build_multiscale_spec(
        self,
        multiscale_state: Optional[Dict[str, Any]],
        base_shape: List[int],
    ) -> Optional[MultiscaleSpec]:
        if not multiscale_state:
            return None

        levels_data = multiscale_state.get("levels")
        if not levels_data:
            return None

        levels: List[MultiscaleLevelSpec] = []
        for entry in levels_data:
            if not isinstance(entry, dict):
                continue
            shape = entry.get("shape")
            down = entry.get("downsample")
            if not shape and down:
                try:
                    shape = [
                        int(max(1, round(dim / float(ds))))
                        for dim, ds in zip(base_shape, down)
                    ]
                except (TypeError, ValueError) as exc:
                    logger.debug("layer_manager: multiscale shape inference failed", exc_info=exc)
                    shape = None
            levels.append(
                MultiscaleLevelSpec(
                    shape=[int(x) for x in (shape or base_shape)],
                    downsample=[float(x) for x in (down or [1.0] * len(base_shape))],
                    path=entry.get("path"),
                )
            )

        if not levels:
            return None

        metadata = {
            "policy": multiscale_state.get("policy", "latency"),
            "index_space": multiscale_state.get("index_space", "base"),
        }

        return MultiscaleSpec(
            levels=levels,
            current_level=int(multiscale_state.get("current_level", 0)),
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    def _adapter_multiscale_spec(
        self,
        layer: Optional[Any],
        extras: Optional[Dict[str, Any]],
        base_shape: List[int],
    ) -> Optional[MultiscaleSpec]:
        if layer is None or not bool(getattr(layer, "multiscale", False)):
            return None

        current_level = None
        if extras and extras.get("multiscale_current_level") is not None:
            current_level = int(extras["multiscale_current_level"])
        elif hasattr(layer, "data_level"):
            current_level = int(getattr(layer, "data_level"))
        else:
            current_level = 0

        levels: List[MultiscaleLevelSpec] = []
        level_entries = extras.get("multiscale_levels") if extras else None
        if isinstance(level_entries, list) and level_entries:
            for entry in level_entries:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                shape = entry.get("shape") or base_shape
                downsample = entry.get("downsample") or [1.0] * len(base_shape)
                levels.append(
                    MultiscaleLevelSpec(
                        shape=[int(x) for x in shape],
                        downsample=[float(x) for x in downsample],
                        path=str(path) if path else None,
                    )
                )

        if not levels:
            data = getattr(layer, "data", None)
            data_seq: Optional[Sequence[Any]] = None
            if isinstance(data, Sequence):
                try:
                    if len(data) > 0:
                        data_seq = data
                except TypeError:
                    data_seq = None
            if data_seq:
                top_shape = base_shape or list(getattr(data_seq[0], "shape", []))
                for idx in range(len(data_seq)):
                    arr = data_seq[idx]
                    arr_shape = list(getattr(arr, "shape", top_shape))
                    downsample: List[float] = []
                    for base, level_dim in zip(top_shape, arr_shape):
                        if level_dim in (0, None):
                            downsample.append(1.0)
                        else:
                            downsample.append(float(base) / float(level_dim))
                    levels.append(
                        MultiscaleLevelSpec(
                            shape=[int(x) for x in arr_shape],
                            downsample=downsample or [1.0] * len(arr_shape),
                            path=None,
                        )
                    )

        if not levels:
            return None

        metadata = {"policy": "latency", "index_space": "base"}
        return MultiscaleSpec(
            levels=levels,
            current_level=int(current_level or 0),
            metadata=metadata,
        )

    def _build_render_hints(
        self, volume_state: Optional[Dict[str, Any]]
    ) -> Optional[LayerRenderHints]:
        if not volume_state:
            return None

        return LayerRenderHints(
            mode=str(volume_state.get("mode")) if volume_state.get("mode") else None,
            colormap=str(volume_state.get("colormap")) if volume_state.get("colormap") else None,
            opacity=float(volume_state["opacity"])
            if volume_state.get("opacity") is not None
            else None,
            iso_threshold=float(volume_state.get("iso_threshold"))
            if volume_state.get("iso_threshold") is not None
            else None,
            attenuation=float(volume_state.get("attenuation"))
            if volume_state.get("attenuation") is not None
            else None,
            gamma=float(volume_state.get("gamma"))
            if volume_state.get("gamma") is not None
            else None,
            shading=str(volume_state.get("shade"))
            if volume_state.get("shade")
            else None,
        )

    def _apply_to_viewer(
        self,
        layer_spec: LayerSpec,
        dims_spec: DimsSpec,
        camera_spec: Optional[CameraSpec],
    ) -> None:
        """Keep the internal ``ViewerModel`` roughly in sync."""

        dims = self._viewer.dims
        dims.ndim = dims_spec.ndim
        if dims_spec.axis_labels:
            dims.axis_labels = tuple(dims_spec.axis_labels)
        if dims_spec.current_step:
            dims.current_step = tuple(dims_spec.current_step)
        if dims_spec.ndisplay is not None:
            dims.ndisplay = int(dims_spec.ndisplay)
        if camera_spec is not None:
            cam = self._viewer.camera
            if camera_spec.center is not None:
                cam.center = tuple(camera_spec.center)
            if camera_spec.zoom is not None:
                cam.zoom = float(camera_spec.zoom)
