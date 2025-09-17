"""Scene management helpers for the server-side napari viewer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

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
    ) -> None:
        self._viewer = ViewerModel()
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
        extras: Optional[Dict[str, Any]] = None,
    ) -> SceneSpec:
        """Refresh the cached scene metadata from the latest server state."""

        worker_info = self._snapshot_worker(worker, ndisplay)
        layer_spec = self._build_layer_spec(
            worker_info,
            multiscale_state=multiscale_state,
            volume_state=volume_state,
            zarr_path=zarr_path,
            extras=extras,
        )
        dims_spec = self._build_dims_spec(
            layer_spec, current_step=current_step, ndisplay=ndisplay
        )
        camera_spec = self._build_camera_spec(scene_state, ndisplay=ndisplay)

        metadata: Dict[str, Any] = {}
        if zarr_path:
            metadata["zarr_path"] = zarr_path
        if extras:
            for key, value in extras.items():
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
        return {key: value for key, value in meta.items() if value is not None}

    # ------------------------------------------------------------------
    # Internal helpers
    def _snapshot_worker(
        self, worker: Optional[object], ndisplay: Optional[int]
    ) -> _WorkerSnapshot:
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
            if isinstance(zarr_shape, tuple) and len(zarr_shape) >= 2:
                shape = [int(zarr_shape[-2]), int(zarr_shape[-1])]
            else:
                w, h = getattr(worker, "_data_wh", (self._canvas_size[0], self._canvas_size[1]))
                shape = [int(h), int(w)]
            if len(axis_labels) != len(shape):
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

    def _build_layer_spec(
        self,
        worker_snapshot: _WorkerSnapshot,
        *,
        multiscale_state: Optional[Dict[str, Any]],
        volume_state: Optional[Dict[str, Any]],
        zarr_path: Optional[str],
        extras: Optional[Dict[str, Any]],
    ) -> LayerSpec:
        multiscale = self._build_multiscale_spec(multiscale_state, worker_snapshot.shape)
        render_hints = self._build_render_hints(volume_state)

        contrast_limits = None
        if isinstance(volume_state, dict):
            clim = volume_state.get("clim")
            if isinstance(clim, (list, tuple)) and len(clim) >= 2:
                try:
                    contrast_limits = [float(clim[0]), float(clim[1])]
                except (TypeError, ValueError) as exc:
                    logger.debug("layer_manager: invalid contrast limits %s", clim, exc_info=exc)
                    contrast_limits = None

        layer_extras: Dict[str, Any] = {
            "is_volume": worker_snapshot.is_volume,
            "zarr_path": zarr_path,
            "zarr_axes": worker_snapshot.zarr_axes,
        }
        if extras:
            layer_extras.update({k: v for k, v in extras.items() if v is not None})

        return LayerSpec(
            layer_id=self._default_layer_id,
            layer_type="image",
            name=self._default_layer_name,
            ndim=worker_snapshot.ndim,
            shape=worker_snapshot.shape,
            dtype=worker_snapshot.dtype,
            axis_labels=worker_snapshot.axis_labels,
            contrast_limits=contrast_limits,
            render=render_hints,
            multiscale=multiscale,
            extras=layer_extras,
        )

    def _build_dims_spec(
        self,
        layer_spec: LayerSpec,
        *,
        current_step: Optional[Iterable[int]],
        ndisplay: Optional[int],
    ) -> DimsSpec:
        ndim = int(layer_spec.ndim)
        axis_labels = layer_spec.axis_labels or [f"axis-{i}" for i in range(ndim)]
        order = axis_labels
        sizes = [int(size) for size in layer_spec.shape]
        ranges = [[0, max(0, int(size) - 1)] for size in sizes]

        current = list(current_step) if current_step is not None else [0 for _ in range(ndim)]
        if len(current) < ndim:
            current.extend([0] * (ndim - len(current)))

        if ndisplay is None:
            ndisplay = 3 if layer_spec.extras and layer_spec.extras.get("is_volume") else 2
        ndisplay = max(1, min(int(ndisplay), ndim))
        displayed_indices = list(range(max(0, ndim - ndisplay), ndim))

        return DimsSpec(
            ndim=ndim,
            axis_labels=axis_labels,
            order=order,
            sizes=sizes,
            range=ranges,
            current_step=current,
            displayed=displayed_indices,
            ndisplay=ndisplay,
        )

    def _build_camera_spec(
        self, scene_state: Optional[object], *, ndisplay: Optional[int]
    ) -> Optional[CameraSpec]:
        if scene_state is None:
            return None

        center = getattr(scene_state, "center", None)
        zoom = getattr(scene_state, "zoom", None)
        angles = getattr(scene_state, "angles", None)
        try:
            center_list = list(center) if center is not None else None
        except (TypeError, ValueError) as exc:
            logger.debug("layer_manager: invalid camera center %r", center, exc_info=exc)
            center_list = None
        try:
            angles_list = list(angles) if angles is not None else None
        except (TypeError, ValueError) as exc:
            logger.debug("layer_manager: invalid camera angles %r", angles, exc_info=exc)
            angles_list = None

        return CameraSpec(
            center=center_list,
            zoom=float(zoom) if zoom is not None else None,
            angles=angles_list,
            ndisplay=int(ndisplay) if ndisplay is not None else None,
        )

    def _build_multiscale_spec(
        self,
        multiscale_state: Optional[Dict[str, Any]],
        base_shape: List[int],
    ) -> Optional[MultiscaleSpec]:
        if not multiscale_state:
            return None

        levels_data = multiscale_state.get("levels") or []
        levels: List[MultiscaleLevelSpec] = []
        for entry in levels_data:
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

        metadata = {
            "policy": multiscale_state.get("policy", "fixed"),
            "index_space": multiscale_state.get("index_space", "base"),
        }

        return MultiscaleSpec(
            levels=levels,
            current_level=int(multiscale_state.get("current_level", 0)),
            metadata={k: v for k, v in metadata.items() if v is not None},
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
