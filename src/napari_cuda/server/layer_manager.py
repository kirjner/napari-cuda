"""Scene management helpers for the server-side napari viewer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from contextlib import ExitStack, suppress

from napari.components.viewer_model import ViewerModel

from napari_cuda.protocol.snapshots import (
    LayerSnapshot,
    SceneSnapshot,
    ViewerSnapshot,
    scene_snapshot,
    viewer_snapshot_from_blocks,
)
from napari_cuda.server.server_scene import (
    CONTROL_KEYS,
    LayerControlState,
    default_layer_controls,
    layer_controls_to_dict,
)


logger = logging.getLogger(__name__)


@dataclass
class _WorkerSnapshot:
    ndim: int
    shape: List[int]
    axis_labels: List[str]
    dtype: Optional[str]
    is_volume: bool
    zarr_axes: Optional[str]
    zarr_dtype: Optional[str]


class ViewerSceneManager:
    """Maintains a headless ``ViewerModel`` and derived scene snapshots."""

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
        self._capabilities: Tuple[str, ...] = ("layer.remove", "state.update")
        self._snapshot: Optional[SceneSnapshot] = None

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
        layer_controls: Optional[Dict[str, LayerControlState]] = None,
    ) -> SceneSnapshot:
        if viewer_model is not None and viewer_model is not self._viewer:
            self._viewer = viewer_model
            self._owns_viewer = False

        worker_snapshot = self._snapshot_worker(worker, ndisplay, viewer_model=viewer_model)

        adapter_layer = self._adapter_layer(viewer_model)
        extras_map = self._compose_extras(extras, adapter_layer)

        controls_state = self._resolve_controls(layer_controls)
        control_map = layer_controls_to_dict(controls_state)

        layer_metadata = self._layer_metadata(adapter_layer)
        layer_block = self._build_layer_block(
            worker_snapshot,
            multiscale_state=multiscale_state,
            volume_state=volume_state,
            zarr_path=zarr_path,
            extras=extras_map,
            controls=control_map,
            metadata=layer_metadata,
            adapter_layer=adapter_layer,
        )

        dims_block = self._build_dims_block(
            layer_block,
            current_step=current_step,
            ndisplay=ndisplay,
            viewer_model=viewer_model,
        )

        camera_block = self._build_camera_block(scene_state, ndisplay=ndisplay)

        viewer_snapshot = viewer_snapshot_from_blocks(
            settings=self._viewer_settings(worker_snapshot.is_volume),
            dims=dims_block,
            camera=camera_block,
        )

        layer_snapshot = LayerSnapshot(layer_id=layer_block["layer_id"], block=layer_block)

        ancillary = self._build_ancillary(extras_map, zarr_path)

        snapshot = scene_snapshot(
            viewer=viewer_snapshot,
            layers=[layer_snapshot],
            policies={},
            ancillary=ancillary,
        )

        self._snapshot = snapshot
        self._apply_to_viewer(dims_block, camera_block)
        return snapshot

    def scene_snapshot(self) -> Optional[SceneSnapshot]:
        return self._snapshot

    def dims_metadata(self) -> Dict[str, Any]:
        snapshot = self._snapshot
        if snapshot is None:
            return {}

        dims_block = dict(snapshot.viewer.dims)
        meta: Dict[str, Any] = dict(dims_block)

        if not snapshot.layers:
            return meta

        layer_block = snapshot.layers[0].block
        extras = layer_block.get("extras", {}) or {}
        if "is_volume" in extras:
            meta["volume"] = bool(extras["is_volume"])
        else:
            meta["volume"] = False

        controls = layer_block.get("controls", {})
        if controls:
            meta["controls"] = dict(controls)

        multiscale_block = layer_block.get("multiscale")
        if isinstance(multiscale_block, Mapping):
            ms_dict = dict(multiscale_block)
            metadata_block = ms_dict.pop("metadata", None)
            if isinstance(metadata_block, Mapping):
                for key, value in metadata_block.items():
                    ms_dict[str(key)] = _normalize_scalar(value)
            meta["multiscale"] = ms_dict
            levels = ms_dict.get("levels")
            if isinstance(levels, Sequence) and levels:
                index = int(ms_dict.get("current_level", 0))
                if 0 <= index < len(levels):
                    level_shape = levels[index].get("shape")
                    if isinstance(level_shape, Sequence):
                        shape_list = [int(x) for x in level_shape]
                        meta["sizes"] = shape_list
                        meta["range"] = [[0, max(0, s - 1)] for s in shape_list]

        return meta

    # ------------------------------------------------------------------
    def _viewer_settings(self, use_volume: bool) -> Dict[str, Any]:
        width, height = self._canvas_size
        return {
            "fps_target": float(getattr(self._viewer, "fps", 60.0)),
            "canvas_size": [width, height],
            "volume_enabled": bool(use_volume),
        }

    def _adapter_layer(self, viewer_model: Optional[ViewerModel]) -> Optional[Any]:
        if viewer_model is None:
            return None
        return viewer_model.layers[0] if viewer_model.layers else None

    def _compose_extras(self, extras: Optional[Dict[str, Any]], adapter_layer: Optional[Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if extras:
            payload.update({k: v for k, v in extras.items() if v is not None})
        if adapter_layer is not None:
            payload.update(self._adapter_layer_extras(adapter_layer))
        for key in CONTROL_KEYS:
            payload.pop(key, None)
        return payload

    def _resolve_controls(self, layer_controls: Optional[Dict[str, LayerControlState]]) -> LayerControlState:
        if not layer_controls:
            return default_layer_controls()
        return layer_controls.get(self._default_layer_id, default_layer_controls())

    def _layer_metadata(self, adapter_layer: Optional[Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if adapter_layer is None:
            return metadata
        thumbnail = self._adapter_layer_thumbnail(adapter_layer)
        if thumbnail is not None:
            metadata["thumbnail"] = thumbnail.tolist()
        return metadata

    def _build_layer_block(
        self,
        worker_snapshot: _WorkerSnapshot,
        *,
        multiscale_state: Optional[Dict[str, Any]],
        volume_state: Optional[Dict[str, Any]],
        zarr_path: Optional[str],
        extras: Dict[str, Any],
        controls: Dict[str, Any],
        metadata: Dict[str, Any],
        adapter_layer: Optional[Any],
    ) -> Dict[str, Any]:
        multiscale = self._build_multiscale_block(multiscale_state, worker_snapshot.shape)
        if multiscale is None:
            multiscale = self._adapter_multiscale_block(adapter_layer, extras, worker_snapshot.shape)
        render_hints = self._build_render_hints(volume_state)

        layer_extras = {k: v for k, v in extras.items() if v is not None}
        layer_extras["zarr_axes"] = worker_snapshot.zarr_axes

        shape = [int(x) for x in worker_snapshot.shape]
        ndim = len(shape)
        axis_labels = self._normalize_axis_labels(worker_snapshot.axis_labels, ndim)

        block: Dict[str, Any] = {
            "layer_id": self._default_layer_id,
            "layer_type": "image",
            "name": self._default_layer_name,
            "ndim": ndim,
            "shape": shape,
            "dtype": worker_snapshot.dtype,
            "axis_labels": axis_labels,
            "metadata": metadata or None,
            "extras": dict(layer_extras) if layer_extras else None,
            "controls": controls or None,
        }

        if render_hints:
            block["render"] = render_hints
        if multiscale:
            block["multiscale"] = multiscale
        extras_block = block.get("extras") if isinstance(block.get("extras"), dict) else {}
        if worker_snapshot.is_volume:
            extras_block.setdefault("is_volume", True)
        if zarr_path:
            extras_block.setdefault("zarr_path", zarr_path)
        if extras_block:
            block["extras"] = extras_block
        elif "extras" in block:
            block["extras"] = None

        self._apply_contrast(block, adapter_layer, volume_state)
        return block

    def _build_dims_block(
        self,
        layer_block: Dict[str, Any],
        *,
        current_step: Optional[Iterable[int]],
        ndisplay: Optional[int],
        viewer_model: Optional[ViewerModel],
    ) -> Dict[str, Any]:
        if viewer_model is not None:
            dims = viewer_model.dims
            axis_labels = list(dims.axis_labels)
            order = list(dims.order)
            displayed = list(dims.displayed)
            current = list(dims.current_step)
            ndisplay_val = int(dims.ndisplay)
        else:
            shape = [int(x) for x in (layer_block.get("shape") or [])]
            axis_labels = self._normalize_axis_labels(layer_block.get("axis_labels"), len(shape))
            order = list(axis_labels)
            current = self._normalize_step(current_step, len(axis_labels))
            ndisplay_val = int(ndisplay or min(2, len(axis_labels)))
            displayed = list(range(max(0, len(axis_labels) - ndisplay_val), len(axis_labels)))

        sizes = [int(x) for x in (layer_block.get("shape") or [])]
        ranges = [[0, max(0, s - 1)] for s in sizes]

        return {
            "ndim": len(axis_labels),
            "axis_labels": axis_labels,
            "order": order,
            "sizes": sizes,
            "range": ranges,
            "current_step": current,
            "displayed": displayed,
            "ndisplay": ndisplay_val,
        }

    def _build_camera_block(
        self,
        scene_state: Optional[object],
        *,
        ndisplay: Optional[int],
    ) -> Dict[str, Any]:
        if scene_state is None:
            return {}
        block: Dict[str, Any] = {}
        center = getattr(scene_state, "center", None)
        if center is not None:
            block["center"] = [float(v) for v in center]
        zoom = getattr(scene_state, "zoom", None)
        if zoom is not None:
            block["zoom"] = float(zoom)
        angles = getattr(scene_state, "angles", None)
        if angles is not None:
            block["angles"] = [float(v) for v in angles]
        if ndisplay is not None:
            block["ndisplay"] = int(ndisplay)
        return block

    def _build_multiscale_block(
        self,
        multiscale_state: Optional[Dict[str, Any]],
        base_shape: List[int],
    ) -> Optional[Dict[str, Any]]:
        if not multiscale_state:
            return None
        levels_data = multiscale_state.get("levels")
        if not levels_data:
            return None
        levels: List[Dict[str, Any]] = []
        for entry in levels_data:
            shape = entry.get("shape")
            downsample = entry.get("downsample")
            if not shape and downsample:
                shape = [int(max(1, round(dim / float(ds)))) for dim, ds in zip(base_shape, downsample)]
            levels.append(
                {
                    "shape": [int(x) for x in (shape or base_shape)],
                    "downsample": [float(x) for x in (downsample or [1.0] * len(base_shape))],
                    "path": entry.get("path"),
                }
            )
        metadata: Dict[str, Any] = {}
        if "policy" in multiscale_state:
            metadata["policy"] = multiscale_state["policy"]
        if "index_space" in multiscale_state:
            metadata["index_space"] = multiscale_state["index_space"]
        current_level = multiscale_state.get("current_level", 0)
        payload: Dict[str, Any] = {
            "levels": levels,
            "current_level": int(current_level),
        }
        if metadata:
            payload["metadata"] = metadata
        return payload

    def _adapter_multiscale_block(
        self,
        layer: Optional[Any],
        extras: Dict[str, Any],
        base_shape: List[int],
    ) -> Optional[Dict[str, Any]]:
        if layer is None or not bool(getattr(layer, "multiscale", False)):
            return None

        if extras.get("multiscale_current_level") is not None:
            current_level = int(extras["multiscale_current_level"])
        elif hasattr(layer, "data_level"):
            current_level = int(getattr(layer, "data_level"))
        else:
            current_level = 0

        levels: List[Dict[str, Any]] = []
        entries = extras.get("multiscale_levels")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                shape = entry.get("shape") or base_shape
                downsample = entry.get("downsample") or [1.0] * len(base_shape)
                levels.append(
                    {
                        "shape": [int(x) for x in shape],
                        "downsample": [float(x) for x in downsample],
                        "path": entry.get("path"),
                    }
                )

        if not levels:
            data = getattr(layer, "data", None)
            if isinstance(data, Sequence) and data:
                base = base_shape or list(getattr(data[0], "shape", []))
                for arr in data:
                    arr_shape = list(getattr(arr, "shape", base))
                    downsample = [float(b) / float(d) if d else 1.0 for b, d in zip(base, arr_shape)]
                    levels.append(
                        {
                            "shape": [int(x) for x in arr_shape],
                            "downsample": [float(x) for x in downsample],
                            "path": None,
                        }
                    )

        if not levels:
            return None

        return {
            "levels": levels,
            "current_level": int(current_level),
            "metadata": {"policy": "latency", "index_space": "base"},
        }

    def _build_render_hints(
        self, volume_state: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not volume_state:
            return None
        shading = volume_state.get("shade")
        if not shading:
            return None
        return {"shading": str(shading)}

    def _apply_contrast(
        self,
        block: Dict[str, Any],
        adapter_layer: Optional[Any],
        volume_state: Optional[Dict[str, Any]],
    ) -> None:
        if adapter_layer is not None and hasattr(adapter_layer, "contrast_limits"):
            clim = adapter_layer.contrast_limits  # type: ignore[attr-defined]
            if isinstance(clim, (list, tuple)) and len(clim) >= 2:
                block["contrast_limits"] = [float(clim[0]), float(clim[1])]
            return
        if volume_state is None:
            return
        clim_vs = volume_state.get("clim")
        if isinstance(clim_vs, (list, tuple)) and len(clim_vs) >= 2:
            block["contrast_limits"] = [float(clim_vs[0]), float(clim_vs[1])]

    @staticmethod
    def _normalize_axis_labels(labels: Optional[Iterable[Any]], ndim: int) -> List[str]:
        if labels is not None:
            normalized = [str(label) for label in labels]
            if len(normalized) == ndim and all(label.strip() for label in normalized):
                return normalized
        if ndim == 3:
            return ["z", "y", "x"]
        if ndim == 2:
            return ["y", "x"]
        return [f"d{i}" for i in range(ndim)]

    @staticmethod
    def _normalize_step(step: Optional[Iterable[int]], ndim: int) -> List[int]:
        values = list(step) if step is not None else []
        result = [int(val) for val in values[:ndim]]
        if len(result) < ndim:
            result.extend([0] * (ndim - len(result)))
        return result

    def _adapter_layer_extras(self, layer: Any) -> Dict[str, Any]:
        extras: Dict[str, Any] = {"adapter_engine": "napari-vispy"}
        name = getattr(layer, "name", None)
        if name:
            extras["layer_name"] = str(name)
        visible = getattr(layer, "visible", None)
        if visible is not None:
            extras["visible"] = bool(visible)
        opacity_val = getattr(layer, "opacity", None)
        if opacity_val is not None:
            extras["opacity"] = float(opacity_val)
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
            extras["scale"] = [float(s) for s in scale]
        clim = getattr(layer, "contrast_limits", None)
        if clim is not None:
            extras["contrast_limits"] = [float(clim[0]), float(clim[1])]
        if hasattr(layer, "source_path"):
            extras["source_path"] = layer.source_path  # type: ignore[attr-defined]
        if hasattr(layer, "data_id"):
            extras["data_id"] = layer.data_id  # type: ignore[attr-defined]
        return {k: v for k, v in extras.items() if v is not None}

    @staticmethod
    def _adapter_layer_thumbnail(layer: Any) -> Optional[np.ndarray]:
        updater = getattr(layer, "_update_thumbnail", None)
        if callable(updater):
            with suppress(Exception):
                updater()
        thumbnail = getattr(layer, "thumbnail", None)
        if thumbnail is None:
            return None
        with suppress(Exception):
            arr = np.asarray(thumbnail)
        if 'arr' not in locals() or arr.size == 0:
            return None
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    def _build_ancillary(self, extras: Dict[str, Any], zarr_path: Optional[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if zarr_path:
            payload.setdefault("metadata", {})["zarr_path"] = zarr_path
        if extras:
            payload.setdefault("metadata", {}).update(extras)
        payload["capabilities"] = list(self._capabilities)
        return payload

    def _snapshot_worker(
        self,
        worker: Optional[object],
        ndisplay: Optional[int],
        *,
        viewer_model: Optional[ViewerModel],
    ) -> _WorkerSnapshot:
        if viewer_model is not None and viewer_model.layers:
            adapter_snapshot = self._snapshot_from_viewer(viewer_model, ndisplay)
            if adapter_snapshot is not None and (worker is None or not getattr(worker, "use_volume", False)):
                return adapter_snapshot

        if worker is None:
            width, height = self._canvas_size
            return _WorkerSnapshot(
                ndim=2,
                shape=[height, width],
                axis_labels=["y", "x"],
                dtype=None,
                is_volume=False,
                zarr_axes=None,
                zarr_dtype=None,
            )

        axes = getattr(worker, "_zarr_axes", None)
        axis_labels = [str(a) for a in axes] if axes is not None else ["y", "x"]

        is_volume = bool(getattr(worker, "use_volume", False))
        zarr_shape = getattr(worker, "_zarr_shape", None)

        if is_volume and isinstance(zarr_shape, tuple) and len(zarr_shape) >= 3:
            shape = [int(x) for x in zarr_shape[:3]]
            axis_labels = ["z", "y", "x"]
        elif isinstance(zarr_shape, tuple) and len(zarr_shape) >= 3:
            shape = [int(zarr_shape[0]), int(zarr_shape[1]), int(zarr_shape[2])]
            axis_labels = ["z", "y", "x"]
        elif isinstance(zarr_shape, tuple) and len(zarr_shape) >= 2:
            shape = [int(zarr_shape[-2]), int(zarr_shape[-1])]
            axis_labels = ["y", "x"]
        else:
            w, h = getattr(worker, "_data_wh", (self._canvas_size[0], self._canvas_size[1]))
            shape = [int(h), int(w)]
            axis_labels = ["y", "x"]

        dtype = getattr(worker, "_zarr_dtype", None) or getattr(worker, "volume_dtype", None)
        dtype_str = str(dtype) if dtype is not None else None

        ndim = len(shape)
        is_volume = is_volume or (ndisplay == 3)

        return _WorkerSnapshot(
            ndim=ndim,
            shape=shape,
            axis_labels=axis_labels,
            dtype=dtype_str,
            is_volume=is_volume,
            zarr_axes="".join(axis_labels),
            zarr_dtype=dtype_str,
        )

    def _snapshot_from_viewer(self, viewer: ViewerModel, ndisplay: Optional[int]) -> Optional[_WorkerSnapshot]:
        if not viewer.layers:
            return None
        layer = viewer.layers[0]
        raw_data = layer.data
        if isinstance(raw_data, Sequence) and raw_data:
            first = raw_data[0]
            arr = np.asarray(first)
        else:
            arr = np.asarray(raw_data)
        shape = list(arr.shape)
        axis_labels = list(layer.axis_labels) if layer.axis_labels else ["y", "x"]
        dtype = str(arr.dtype)
        ndim = len(shape)
        is_volume = bool(ndisplay == 3 or ndim == 3)
        return _WorkerSnapshot(
            ndim=ndim,
            shape=shape,
            axis_labels=axis_labels,
            dtype=dtype,
            is_volume=is_volume,
            zarr_axes="".join(axis_labels),
            zarr_dtype=dtype,
        )

    def _apply_to_viewer(
        self,
        dims_block: Dict[str, Any],
        camera_block: Dict[str, Any],
    ) -> None:
        dims = self._viewer.dims
        with ExitStack() as stack:
            for attr in ("ndisplay", "current_step", "displayed"):
                emitter = getattr(dims.events, attr, None)
                if emitter is not None and hasattr(emitter, "blocker"):
                    stack.enter_context(emitter.blocker())
            ndim = dims_block.get("ndim")
            if ndim is not None:
                dims.ndim = int(ndim)
            axis_labels = dims_block.get("axis_labels")
            if axis_labels:
                dims.axis_labels = tuple(str(label) for label in axis_labels)
            displayed = dims_block.get("displayed")
            if displayed:
                with suppress(AttributeError):
                    dims.displayed = tuple(int(idx) for idx in displayed)  # type: ignore[attr-defined]
            current_step = dims_block.get("current_step")
            if current_step:
                dims.current_step = tuple(int(val) for val in current_step)
            ndisplay = dims_block.get("ndisplay")
            if ndisplay is not None:
                dims.ndisplay = int(ndisplay)

        if not camera_block:
            return
        camera = self._viewer.camera
        center = camera_block.get("center")
        if center is not None:
            camera.center = tuple(float(value) for value in center)
        zoom = camera_block.get("zoom")
        if zoom is not None:
            camera.zoom = float(zoom)
        angles = camera_block.get("angles")
        if angles is not None:
            camera.angles = tuple(float(value) for value in angles)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    return value
