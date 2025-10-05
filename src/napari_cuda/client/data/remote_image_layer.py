"""napari Image layer subclass backed by remote metadata only."""

from __future__ import annotations

import logging
from dataclasses import replace
import math
from contextlib import nullcontext
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from napari.layers.image.image import Image
from napari.layers._scalar_field._slice import _ScalarFieldSliceResponse, _ScalarFieldView
try:
    from qtpy import QtCore  # type: ignore
except Exception:  # pragma: no cover - best effort import
    QtCore = None

from .remote_data import RemoteArray, RemoteMultiscale, RemotePreview, build_remote_data

logger = logging.getLogger(__name__)

_CONTROL_ONLY_BLOCK_KEYS = {
    "controls",
    "contrast_limits",
    "render",
    "metadata",
}


class RemoteImageLayer(Image):
    """Read-only image layer describing remote content."""

    _REMOTE_ID_META_KEY = "napari_cuda.remote_layer_id"

    def __init__(self, *, layer_id: str, block: Mapping[str, Any]) -> None:
        self._allow_data_update = True
        self._remote_id = str(layer_id)
        self._remote_block = dict(block)

        data_obj, multiscale = build_remote_data(self._remote_block)
        if multiscale is not None:
            arrays = multiscale.arrays
            init_data = multiscale.as_multiscale()
            multiscale_flag = True
        elif isinstance(data_obj, RemoteArray):
            arrays = (data_obj,)
            init_data = data_obj
            multiscale_flag = False
        else:
            arrays = ()
            init_data = data_obj
            multiscale_flag = False

        self._remote_arrays = arrays
        self._remote_preview = RemotePreview()

        metadata = dict(self._remote_block.get("metadata") or {})
        preview = self._preview_from_metadata(metadata)
        metadata.setdefault(self._REMOTE_ID_META_KEY, self._remote_id)

        init_kwargs = self._build_init_kwargs(self._remote_block, metadata, multiscale_flag)
        super().__init__(init_data, **init_kwargs)
        self._allow_data_update = False
        self.editable = False
        self._keep_auto_contrast = False

        self._apply_render(self._remote_block.get("render"))
        self._install_empty_slice()
        fallback_preview = self._extract_preview() if preview is None else preview
        self.update_preview(fallback_preview)
        self._apply_controls(self._remote_block.get("controls") or {})

    # ------------------------------------------------------------------
    def update_from_block(self, block: Mapping[str, Any]) -> None:
        new_block = dict(block)
        control_only = self._is_control_only_update(new_block)
        self._remote_block = new_block

        metadata = dict(self._remote_block.get("metadata") or {})
        preview = self._preview_from_metadata(metadata)
        metadata.setdefault(self._REMOTE_ID_META_KEY, self._remote_id)

        if not control_only:
            data_obj, multiscale = build_remote_data(self._remote_block)
            if multiscale is not None:
                arrays = multiscale.arrays
                new_data = multiscale.as_multiscale()
                multiscale_flag = True
            elif isinstance(data_obj, RemoteArray):
                arrays = (data_obj,)
                new_data = data_obj
                multiscale_flag = False
            else:
                arrays = ()
                new_data = data_obj
                multiscale_flag = False

            self._remote_arrays = arrays
            self._allow_data_update = True
            try:
                Image.data.fset(self, new_data)
            finally:
                self._allow_data_update = False
            self.multiscale = multiscale_flag

            self.name = str(self._remote_block.get("name", self._remote_id))
            axis_labels = self._remote_block.get("axis_labels")
            if isinstance(axis_labels, Sequence) and axis_labels:
                self.axis_labels = tuple(str(label) for label in axis_labels)
            scale = self._remote_block.get("scale")
            if isinstance(scale, Sequence) and scale:
                self.scale = tuple(float(value) for value in scale)
            translate = self._remote_block.get("translate")
            if isinstance(translate, Sequence) and translate:
                self.translate = tuple(float(value) for value in translate)
            contrast_limits = self._remote_block.get("contrast_limits")
            if isinstance(contrast_limits, Sequence) and len(contrast_limits) >= 2:
                self.contrast_limits = (
                    float(contrast_limits[0]),
                    float(contrast_limits[1]),
                )
            self.metadata = metadata

            self._install_empty_slice()
            fallback_preview = self._extract_preview() if preview is None else preview
            self.update_preview(fallback_preview)
        else:
            if metadata:
                try:
                    self.metadata = metadata
                except Exception:
                    self.metadata.update(metadata)

        self._apply_render(self._remote_block.get("render"))
        self._apply_controls(self._remote_block.get("controls") or {})
        if control_only:
            fallback_preview = self._extract_preview() if preview is None else preview
            if fallback_preview is not None:
                self.update_preview(fallback_preview)

    # ------------------------------------------------------------------
    def _build_init_kwargs(self, block: Mapping[str, Any], metadata: dict[str, Any], multiscale_flag: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "name": str(block.get("name", self._remote_id)),
            "metadata": metadata,
            "multiscale": multiscale_flag,
        }
        axis_labels = block.get("axis_labels")
        if isinstance(axis_labels, Sequence) and axis_labels:
            kwargs["axis_labels"] = tuple(str(label) for label in axis_labels)
        scale = block.get("scale")
        if isinstance(scale, Sequence) and scale:
            kwargs["scale"] = tuple(float(value) for value in scale)
        translate = block.get("translate")
        if isinstance(translate, Sequence) and translate:
            kwargs["translate"] = tuple(float(value) for value in translate)
        contrast_limits = block.get("contrast_limits")
        if isinstance(contrast_limits, Sequence) and len(contrast_limits) >= 2:
            kwargs["contrast_limits"] = (
                float(contrast_limits[0]),
                float(contrast_limits[1]),
            )
        return {key: value for key, value in kwargs.items() if value is not None}


    def _install_empty_slice(self) -> None:
        try:
            empty = _ScalarFieldSliceResponse.make_empty(
                slice_input=self._slice_input,
                rgb=bool(getattr(self, "rgb", False)),
                dtype=self.dtype,
            )
            self._update_slice_response(empty)
            self._set_loaded(False)
        except Exception:
            logger.debug("RemoteImageLayer: failed to install empty slice", exc_info=True)

    def _set_view_slice(self) -> None:
        self._install_empty_slice()

    # ------------------------------------------------------------------
    def _apply_render(self, hints: Optional[Mapping[str, Any]]) -> None:
        if not hints:
            return
        if hints.get("mode"):
            try:
                self.rendering = str(hints["mode"])
            except Exception:
                logger.debug("RemoteImageLayer: render mode %s unsupported", hints["mode"], exc_info=True)
        if hints.get("colormap"):
            try:
                self.colormap = str(hints["colormap"])
            except Exception:
                logger.debug("RemoteImageLayer: colormap %s unsupported", hints["colormap"], exc_info=True)
        if hints.get("shading"):
            try:
                setattr(self, "shading", str(hints["shading"]))
            except Exception:
                logger.debug("RemoteImageLayer: shading update failed", exc_info=True)
        if hints.get("opacity") is not None:
            self.opacity = float(hints["opacity"])
        if hints.get("visibility") is not None:
            self.visible = bool(hints["visibility"])
        if hints.get("gamma") is not None:
            self.gamma = float(hints["gamma"])
        if hints.get("iso_threshold") is not None:
            self.iso_threshold = float(hints["iso_threshold"])
        if hints.get("attenuation") is not None:
            self.attenuation = float(hints["attenuation"])

    def _apply_controls(self, controls: Mapping[str, Any]) -> None:
        if not controls:
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RemoteImageLayer[%s]: apply_controls keys=%s",
                self._remote_id,
                list(controls.keys()),
            )

        def _set_attr(attr: str, value: Any, *, equals: Optional[Callable[[Any, Any], bool]] = None) -> None:
            current = getattr(self, attr, None)
            if equals is not None:
                if current is not None and equals(current, value):
                    return
            else:
                if current == value:
                    return
            events_obj = getattr(self, "events", None)
            emitter = getattr(events_obj, attr, None)
            blocker = emitter.blocker() if emitter is not None and hasattr(emitter, "blocker") else nullcontext()
            with blocker:
                setattr(self, attr, value)

        try:
            value = controls.get("visible")
            if value is not None:
                new_val = bool(value)
                if bool(getattr(self, "visible", new_val)) is not new_val:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set visible -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr("visible", new_val)
        except Exception:
            logger.debug("RemoteImageLayer: visible control failed", exc_info=True)

        try:
            value = controls.get("opacity")
            if value is not None:
                new_val = float(value)
                if not math.isclose(float(getattr(self, "opacity", new_val)), new_val, rel_tol=1e-6, abs_tol=1e-6):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set opacity -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr(
                        "opacity",
                        new_val,
                        equals=lambda cur, val: math.isclose(float(cur), float(val), rel_tol=1e-6, abs_tol=1e-6),
                    )
        except Exception:
            logger.debug("RemoteImageLayer: opacity control failed", exc_info=True)

        try:
            value = controls.get("blending")
            if value:
                new_val = str(value)
                if str(getattr(self, "blending", "")) != new_val:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set blending -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr("blending", new_val)
        except Exception:
            logger.debug("RemoteImageLayer: blending control failed", exc_info=True)

        try:
            value = controls.get("interpolation")
            if value:
                new_val = str(value)
                if str(getattr(self, "interpolation", "")) != new_val:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set interpolation -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr("interpolation", new_val)
        except Exception:
            logger.debug("RemoteImageLayer: interpolation control failed", exc_info=True)

        try:
            value = controls.get("colormap")
            if value:
                new_val = str(value)
                if str(getattr(self, "colormap", "")) != new_val:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set colormap -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr("colormap", new_val)
        except Exception:
            logger.debug("RemoteImageLayer: colormap control failed", exc_info=True)

        try:
            value = controls.get("rendering")
            if value:
                new_val = str(value)
                if str(getattr(self, "rendering", "")) != new_val:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set rendering -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr("rendering", new_val)
        except Exception:
            logger.debug("RemoteImageLayer: rendering control failed", exc_info=True)

        try:
            value = controls.get("gamma")
            if value is not None:
                new_val = float(value)
                if not math.isclose(float(getattr(self, "gamma", new_val)), new_val, rel_tol=1e-6, abs_tol=1e-6):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set gamma -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr(
                        "gamma",
                        new_val,
                        equals=lambda cur, val: math.isclose(float(cur), float(val), rel_tol=1e-6, abs_tol=1e-6),
                    )
        except Exception:
            logger.debug("RemoteImageLayer: gamma control failed", exc_info=True)

        try:
            value = controls.get("contrast_limits")
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                new_limits = (float(value[0]), float(value[1]))
                current_limits = getattr(self, "contrast_limits", new_limits)
                if (
                    len(current_limits) < 2
                    or not math.isclose(float(current_limits[0]), new_limits[0], rel_tol=1e-6, abs_tol=1e-6)
                    or not math.isclose(float(current_limits[1]), new_limits[1], rel_tol=1e-6, abs_tol=1e-6)
                ):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set contrast_limits -> %s",
                            self._remote_id,
                            new_limits,
                        )
                    _set_attr(
                        "contrast_limits",
                        new_limits,
                        equals=lambda cur, val: (
                            len(cur) >= 2
                            and math.isclose(float(cur[0]), float(val[0]), rel_tol=1e-6, abs_tol=1e-6)
                            and math.isclose(float(cur[1]), float(val[1]), rel_tol=1e-6, abs_tol=1e-6)
                        ),
                    )
        except Exception:
            logger.debug("RemoteImageLayer: contrast_limits control failed", exc_info=True)

        try:
            value = controls.get("iso_threshold")
            if value is not None:
                new_val = float(value)
                if not math.isclose(float(getattr(self, "iso_threshold", new_val)), new_val, rel_tol=1e-6, abs_tol=1e-6):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set iso_threshold -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr(
                        "iso_threshold",
                        new_val,
                        equals=lambda cur, val: math.isclose(float(cur), float(val), rel_tol=1e-6, abs_tol=1e-6),
                    )
        except Exception:
            logger.debug("RemoteImageLayer: iso_threshold control failed", exc_info=True)

        try:
            value = controls.get("attenuation")
            if value is not None:
                new_val = float(value)
                if not math.isclose(float(getattr(self, "attenuation", new_val)), new_val, rel_tol=1e-6, abs_tol=1e-6):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RemoteImageLayer[%s]: set attenuation -> %s",
                            self._remote_id,
                            new_val,
                        )
                    _set_attr(
                        "attenuation",
                        new_val,
                        equals=lambda cur, val: math.isclose(float(cur), float(val), rel_tol=1e-6, abs_tol=1e-6),
                    )
        except Exception:
            logger.debug("RemoteImageLayer: attenuation control failed", exc_info=True)

    def apply_control_changes(self, block: Mapping[str, Any], changes: Mapping[str, Any]) -> None:
        self._remote_block = dict(block)
        controls = self._remote_block.get("controls")
        if not isinstance(controls, Mapping):
            return
        subset: dict[str, Any] = {}
        for key in changes:
            if key == "removed":
                continue
            if key == "contrast_limits" and key not in controls and key in self._remote_block:
                subset[key] = self._remote_block.get(key)
            else:
                subset[key] = controls.get(key)
        self._apply_controls({k: v for k, v in subset.items() if v is not None})

    def _is_control_only_update(self, new_block: Mapping[str, Any]) -> bool:
        if not self._remote_block:
            return False

        old_block = self._remote_block
        old_keys = {key for key in old_block.keys() if key not in _CONTROL_ONLY_BLOCK_KEYS}
        new_keys = {key for key in new_block.keys() if key not in _CONTROL_ONLY_BLOCK_KEYS}
        if old_keys != new_keys:
            return False

        for key in old_keys:
            if old_block.get(key) != new_block.get(key):
                return False

        return True

    # ------------------------------------------------------------------
    @property
    def remote_id(self) -> str:
        return self._remote_id

    @Image.data.setter  # type: ignore[misc]
    def data(self, data):  # type: ignore[override]
        if getattr(self, "_allow_data_update", False):
            Image.data.fset(self, data)
            return
        logger.debug("RemoteImageLayer: ignoring data mutation attempt")

    # ------------------------------------------------------------------
    def update_preview(self, preview: np.ndarray | None) -> None:
        self._remote_preview.update(preview)
        self._update_thumbnail()
        if preview is not None or self._remote_preview.data is not None:
            try:
                self._set_loaded(True)
            except Exception:
                logger.debug("RemoteImageLayer: failed to mark preview as loaded", exc_info=True)

    # ------------------------------------------------------------------
    def _extract_preview(self) -> Optional[np.ndarray]:
        if not self._remote_arrays:
            return None
        try:
            return self._remote_arrays[0].preview
        except Exception:
            return None

    def _ensure_main_thread(self) -> bool:
        if QtCore is None:
            return True
        app = QtCore.QCoreApplication.instance()
        if app is None:
            return True
        return QtCore.QThread.currentThread() is app.thread()

    def _schedule_thumbnail_refresh(self) -> bool:
        if QtCore is None:
            return False
        app = QtCore.QCoreApplication.instance()
        if app is None:
            return False
        QtCore.QTimer.singleShot(0, self._update_thumbnail)  # type: ignore[arg-type]
        return True

    def _apply_preview_to_slice(self) -> None:
        try:
            if not hasattr(self, "_slice"):
                return
            thumb_shape = getattr(self, "_thumbnail_shape", (16, 16))
            preview = self._remote_preview.as_thumbnail(
                self.rgb, (thumb_shape[0], thumb_shape[1])
            )
            if (
                self._slice_input.ndisplay == 3
                and self.ndim > 2
                and preview.ndim >= 2
            ):
                preview = preview[np.newaxis, ...]
            thumbnail_view = _ScalarFieldView.from_view(preview)
            self._slice = replace(self._slice, thumbnail=thumbnail_view, empty=False)
            try:
                self._set_loaded(True)
            except Exception:
                logger.debug("RemoteImageLayer: failed to mark slice loaded", exc_info=True)
        except Exception:
            logger.debug("RemoteImageLayer: failed to apply preview thumbnail", exc_info=True)

    def _update_thumbnail(self) -> None:  # type: ignore[override]
        if not self._ensure_main_thread():
            if self._schedule_thumbnail_refresh():
                return
        self._apply_preview_to_slice()
        super()._update_thumbnail()

    # ------------------------------------------------------------------
    def _preview_from_metadata(self, metadata: dict[str, Any]) -> Optional[np.ndarray]:
        if not metadata:
            return None
        raw = metadata.pop("thumbnail", None)
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except Exception:
            logger.debug("RemoteImageLayer: failed to coerce thumbnail metadata", exc_info=True)
            return None
        if arr.size == 0:
            return None
        np.clip(arr, 0.0, 1.0, out=arr)
        return arr
