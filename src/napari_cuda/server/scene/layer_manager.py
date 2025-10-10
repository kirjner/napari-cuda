"""Scene management helpers for the server-side napari viewer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from contextlib import ExitStack, suppress

from napari.components.viewer_model import ViewerModel

from napari_cuda.protocol.axis_labels import normalize_axis_labels
from napari_cuda.protocol.snapshots import (
    LayerSnapshot,
    SceneSnapshot,
    ViewerSnapshot,
    scene_snapshot,
    viewer_snapshot_from_blocks,
)
from napari_cuda.server.scene import (
    LayerControlState,
    default_layer_controls,
    layer_controls_to_dict,
)

if TYPE_CHECKING:
    from napari_cuda.server.runtime.egl_worker import EGLRendererWorker


logger = logging.getLogger(__name__)


@dataclass
class _WorkerSnapshot:
    shape: List[int]
    axis_labels: List[str]
    order: List[int]
    current_step: List[int]
    displayed: List[int]
    ndisplay: int
    dtype: Optional[str]
    is_volume: bool


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
        scene_source: Optional[object],
        viewer_model: Optional[ViewerModel] = None,
        layer_controls: Optional[Dict[str, LayerControlState]] = None,
    ) -> SceneSnapshot:
        if viewer_model is not None and viewer_model is not self._viewer:
            self._viewer = viewer_model
            self._owns_viewer = False

        worker_snapshot = self._snapshot_worker(worker, ndisplay, viewer_model=viewer_model)
        adapter_layer = self._adapter_layer(viewer_model)

        controls_state = self._resolve_controls(layer_controls)
        control_map = layer_controls_to_dict(controls_state)

        layer_metadata = self._layer_metadata(adapter_layer)
        layer_block = self._build_layer_block(
            worker_snapshot,
            multiscale_state=multiscale_state,
            volume_state=volume_state,
            controls=control_map,
            metadata=layer_metadata,
            adapter_layer=adapter_layer,
            scene_source=scene_source,
        )

        dims_block = self._build_dims_block(
            layer_block,
            worker_snapshot=worker_snapshot,
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

        snapshot = scene_snapshot(
            viewer=viewer_snapshot,
            layers=[layer_snapshot],
            policies={},
            metadata=self._scene_metadata(zarr_path=zarr_path, scene_source=scene_source),
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
        meta["volume"] = bool(layer_block.get("volume"))

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

        return meta

    # ------------------------------------------------------------------
    def _viewer_settings(self, use_volume: bool) -> Dict[str, Any]:
        width, height = self._canvas_size
        return {
            "fps_target": float(getattr(self._viewer, "fps", 60.0)),
            "canvas_size": [width, height],
            "volume_enabled": bool(use_volume),
        }

    def _scene_metadata(
        self,
        *,
        zarr_path: Optional[str],
        scene_source: Optional[object],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if zarr_path:
            metadata["source_path"] = str(zarr_path)
        if scene_source is not None:
            metadata.setdefault("source_kind", "ome-zarr")
        return metadata

    def _adapter_layer(self, viewer_model: Optional[ViewerModel]) -> Optional[Any]:
        if viewer_model is None:
            return None
        return viewer_model.layers[0] if viewer_model.layers else None

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
        controls: Dict[str, Any],
        metadata: Dict[str, Any],
        adapter_layer: Optional[Any],
        scene_source: Optional[object],
    ) -> Dict[str, Any]:
        shape = [int(x) for x in worker_snapshot.shape]
        axis_labels = normalize_axis_labels(worker_snapshot.axis_labels, len(shape))

        level_shapes_list: List[List[int]] = [list(shape)]

        block: Dict[str, Any] = {
            "layer_id": self._default_layer_id,
            "layer_type": "image",
            "name": self._default_layer_name,
            "ndim": len(shape),
            "shape": shape,
            "dtype": worker_snapshot.dtype,
            "axis_labels": axis_labels,
            "volume": bool(worker_snapshot.is_volume),
        }

        if metadata:
            block["metadata"] = dict(metadata)
        if controls:
            block["controls"] = dict(controls)

        multiscale_block = self._build_multiscale_block(multiscale_state, base_shape=shape)
        if multiscale_block is not None:
            block["multiscale"] = multiscale_block
            levels = multiscale_block.get("levels")
            if isinstance(levels, list) and levels:
                level_shapes_list = []
                first_level = levels[0]
                for entry in levels:
                    if isinstance(entry, Mapping):
                        entry_shape = entry.get("shape")
                        if isinstance(entry_shape, Sequence) and entry_shape:
                            level_shapes_list.append([int(x) for x in entry_shape])
                active_index = int(multiscale_block.get("current_level", 0))
                if level_shapes_list:
                    active_index = max(0, min(active_index, len(level_shapes_list) - 1))
                    active_shape = level_shapes_list[active_index]
                    block["shape"] = list(active_shape)
                    block["ndim"] = len(active_shape)
                    block["axis_labels"] = normalize_axis_labels(
                        worker_snapshot.axis_labels,
                        len(active_shape),
                    )

        if not level_shapes_list:
            level_shapes_list = [list(shape)]
        block["level_shapes"] = [list(entry) for entry in level_shapes_list]

        render_hints = self._build_render_hints(volume_state)
        if render_hints:
            block["render"] = render_hints

        scale = self._resolve_scale(scene_source, adapter_layer)
        if scale is not None:
            block["scale"] = scale

        translate = self._resolve_translate(adapter_layer)
        if translate is not None:
            block["translate"] = translate

        source_block = self._build_source_block(scene_source, adapter_layer)
        if source_block is not None:
            block["source"] = source_block

        self._apply_contrast(block, adapter_layer, volume_state)
        return block

    def _resolve_scale(
        self,
        scene_source: Optional[object],
        adapter_layer: Optional[Any],
    ) -> Optional[List[float]]:
        if scene_source is not None:
            if not hasattr(scene_source, "current_level") or not hasattr(scene_source, "level_scale"):
                raise AttributeError("scene_source must expose current_level and level_scale")
            level_index = int(getattr(scene_source, "current_level"))
            scale_fn = getattr(scene_source, "level_scale")
            scale_tuple = scale_fn(level_index)
            if not isinstance(scale_tuple, Sequence):
                raise TypeError("scene_source.level_scale must return a sequence")
            return [float(value) for value in scale_tuple]
        if adapter_layer is None:
            return None
        scale_value = getattr(adapter_layer, "scale", None)
        if not isinstance(scale_value, Sequence):
            return None
        return [float(value) for value in scale_value]

    def _resolve_translate(self, adapter_layer: Optional[Any]) -> Optional[List[float]]:
        if adapter_layer is None:
            return None
        translate = getattr(adapter_layer, "translate", None)
        if not isinstance(translate, Sequence):
            return None
        return [float(value) for value in translate]

    def _build_source_block(
        self,
        scene_source: Optional[object],
        adapter_layer: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        block: Dict[str, Any] = {}
        if scene_source is not None:
            block["kind"] = "ome-zarr"
        if adapter_layer is not None:
            if hasattr(adapter_layer, "data_id"):
                data_id = getattr(adapter_layer, "data_id")
                if data_id is not None:
                    block["data_id"] = str(data_id)
            if hasattr(adapter_layer, "cache_version"):
                cache_version = getattr(adapter_layer, "cache_version")
                if cache_version is not None:
                    block["cache_version"] = int(cache_version)
        return block or None

    def _build_dims_block(
        self,
        layer_block: Dict[str, Any],
        *,
        worker_snapshot: _WorkerSnapshot,
        current_step: Optional[Iterable[int]],
        ndisplay: Optional[int],
        viewer_model: Optional[ViewerModel],
    ) -> Dict[str, Any]:
        multiscale = layer_block.get("multiscale")
        current_level = 0
        if isinstance(multiscale, Mapping) and "current_level" in multiscale:
            current_level = int(multiscale.get("current_level", 0))
        level_shapes_raw = layer_block.get("level_shapes") or [layer_block.get("shape", [])]
        level_shapes = [[int(x) for x in shape] for shape in level_shapes_raw if isinstance(shape, Sequence)]
        if not level_shapes:
            level_shapes = [[int(x) for x in (layer_block.get("shape") or [])]]
        current_level = max(0, min(current_level, len(level_shapes) - 1))
        active_shape = level_shapes[current_level]

        axis_labels_source = list(layer_block.get("axis_labels") or worker_snapshot.axis_labels)
        axis_count = len(active_shape)
        axis_labels = axis_labels_source[:axis_count]
        if len(axis_labels) < axis_count:
            axis_labels.extend(
                [f"axis-{idx}" for idx in range(len(axis_labels), axis_count)]
            )
        order_source = worker_snapshot.order or list(range(axis_count))
        order = list(order_source)
        order = [idx for idx in order if idx < axis_count]
        missing = [idx for idx in range(axis_count) if idx not in order]
        order.extend(missing)

        if current_step is not None:
            desired_step = self._normalize_step(current_step, axis_count)
        else:
            desired_step = worker_snapshot.current_step[:axis_count]
            if len(desired_step) < axis_count:
                desired_step.extend(0 for _ in range(axis_count - len(desired_step)))

        ndisplay_val = int(ndisplay) if ndisplay is not None else int(worker_snapshot.ndisplay)
        ndisplay_val = max(1, min(ndisplay_val, axis_count))
        displayed_source = worker_snapshot.displayed or order[-ndisplay_val:]
        displayed = [idx for idx in displayed_source if 0 <= idx < axis_count]
        if len(displayed) < ndisplay_val:
            extras = [idx for idx in order if idx not in displayed]
            displayed.extend(extras[: ndisplay_val - len(displayed)])
        displayed = displayed[:ndisplay_val]

        return {
            "ndim": axis_count,
            "axis_labels": axis_labels,
            "order": order,
            "level_shapes": level_shapes,
            "current_level": current_level,
            "current_step": desired_step,
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
            if not isinstance(entry, Mapping):
                raise ValueError("multiscale level entry must be a mapping")
            shape_value = entry.get("shape")
            if not isinstance(shape_value, Sequence) or not shape_value:
                raise ValueError("multiscale level missing shape metadata")
            downsample_value = entry.get("downsample")
            if not isinstance(downsample_value, Sequence) or not downsample_value:
                raise ValueError("multiscale level missing downsample metadata")
            levels.append(
                {
                    "shape": [int(x) for x in shape_value],
                    "downsample": [float(x) for x in downsample_value],
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
    def _normalize_step(step: Optional[Iterable[int]], ndim: int) -> List[int]:
        values = list(step) if step is not None else []
        result = [int(val) for val in values[:ndim]]
        if len(result) < ndim:
            result.extend([0] * (ndim - len(result)))
        return result

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

    def _snapshot_worker(
        self,
        worker: "EGLRendererWorker",
        ndisplay: Optional[int],
        *,
        viewer_model: Optional[ViewerModel],
    ) -> _WorkerSnapshot:
        assert worker.is_ready, "worker snapshot requested before worker ready"

        ledger_labels = worker._ledger_axis_labels()
        assert ledger_labels is not None and len(ledger_labels) > 0, "ledger missing axis labels"
        axis_labels = [str(label) for label in ledger_labels]

        ledger_order = worker._ledger_order()
        assert ledger_order is not None and len(ledger_order) > 0, "ledger missing axis order"
        order = [int(idx) for idx in ledger_order]

        ledger_step = worker._ledger_step()
        assert ledger_step is not None and len(ledger_step) > 0, "ledger missing current step"
        current_step = [int(val) for val in ledger_step]

        ledger_displayed = worker._ledger_displayed()
        assert ledger_displayed is not None and len(ledger_displayed) > 0, "ledger missing displayed axes"
        displayed = [int(idx) for idx in ledger_displayed]

        ledger_ndisplay = worker._ledger_ndisplay()
        assert ledger_ndisplay is not None, "ledger missing ndisplay"
        ndisplay_val = int(ledger_ndisplay)

        ledger_shapes = worker._ledger_level_shapes()
        assert ledger_shapes is not None and len(ledger_shapes) > 0, "ledger missing multiscale level shapes"
        active_index = worker._active_ms_level
        assert active_index is not None, "worker missing active multiscale level"
        assert 0 <= int(active_index) < len(ledger_shapes), "active multiscale level out of range"
        shape = [int(s) for s in ledger_shapes[int(active_index)]]

        assert len(shape) == len(axis_labels), "ledger axis labels mismatch shape"
        assert len(order) == len(axis_labels), "ledger axis order mismatch axis labels"
        assert len(current_step) == len(axis_labels), "ledger current step mismatch axis labels"
        assert len(displayed) == ndisplay_val, "ledger displayed axes mismatch ndisplay"

        dtype_value = worker._zarr_dtype or worker.volume_dtype
        dtype_str = str(dtype_value) if dtype_value is not None else None

        is_volume = bool(worker.use_volume or (ndisplay == 3))

        return _WorkerSnapshot(
            shape=shape,
            axis_labels=axis_labels,
            order=order,
            current_step=current_step,
            displayed=displayed,
            ndisplay=ndisplay_val,
            dtype=dtype_str,
            is_volume=is_volume,
        )

    def _apply_to_viewer(
        self,
        dims_block: Dict[str, Any],
        camera_block: Dict[str, Any],
    ) -> None:
        dims = self._viewer.dims
        with ExitStack() as stack:
            for attr in ("ndisplay", "current_step", "displayed", "order", "axis_labels", "ndim"):
                emitter = getattr(dims.events, attr, None)
                if emitter is not None and hasattr(emitter, "blocker"):
                    stack.enter_context(emitter.blocker())

            required_keys = ("ndim", "axis_labels", "order", "current_step", "displayed", "ndisplay")
            for key in required_keys:
                assert key in dims_block, f"dims block missing {key}"
                assert dims_block[key] is not None, f"dims block null {key}"

            ndim_value = int(dims_block["ndim"])
            axis_labels_seq = tuple(str(val) for val in dims_block["axis_labels"])
            order_seq = tuple(int(val) for val in dims_block["order"])
            current_step_seq = tuple(int(val) for val in dims_block["current_step"])
            displayed_seq = tuple(int(idx) for idx in dims_block["displayed"])
            ndisplay_value = int(dims_block["ndisplay"])

            assert len(axis_labels_seq) == ndim_value, "axis labels length mismatch ndim"
            assert len(order_seq) == ndim_value, "order length mismatch ndim"
            assert len(current_step_seq) == ndim_value, "current step length mismatch ndim"
            assert len(displayed_seq) == ndisplay_value, "displayed length mismatch ndisplay"

            dims.ndim = ndim_value
            dims.axis_labels = axis_labels_seq  # type: ignore[assignment]
            dims.order = order_seq  # type: ignore[assignment]
            dims.ndisplay = ndisplay_value
            expected_displayed = tuple(order_seq[-ndisplay_value:]) if order_seq else tuple(range(ndisplay_value))
            assert expected_displayed == displayed_seq, "displayed axes must align with order and ndisplay"
            dims.current_step = current_step_seq

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
