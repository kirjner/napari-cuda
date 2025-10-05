"""napari Image layer subclass backed by remote metadata only."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from napari.layers.image.image import Image
from napari.layers._scalar_field._slice import _ScalarFieldSliceResponse, _ScalarFieldView
try:
    from qtpy import QtCore  # type: ignore
except Exception:  # pragma: no cover - best effort import
    QtCore = None

from .remote_data import RemoteArray, RemoteMultiscale, RemotePreview, build_remote_data

logger = logging.getLogger(__name__)


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
        self._remote_block = dict(block)

        metadata = dict(self._remote_block.get("metadata") or {})
        preview = self._preview_from_metadata(metadata)
        metadata.setdefault(self._REMOTE_ID_META_KEY, self._remote_id)

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
            self.contrast_limits = (float(contrast_limits[0]), float(contrast_limits[1]))  # type: ignore[arg-type]
        self.metadata = metadata

        self._apply_render(self._remote_block.get("render"))
        self._apply_controls(self._remote_block.get("controls") or {})
        self._install_empty_slice()
        fallback_preview = self._extract_preview() if preview is None else preview
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

        try:
            value = controls.get("visible")
            if value is not None:
                self.visible = bool(value)
        except Exception:
            logger.debug("RemoteImageLayer: visible control failed", exc_info=True)

        try:
            value = controls.get("opacity")
            if value is not None:
                self.opacity = float(value)
        except Exception:
            logger.debug("RemoteImageLayer: opacity control failed", exc_info=True)

        try:
            value = controls.get("blending")
            if value:
                self.blending = str(value)
        except Exception:
            logger.debug("RemoteImageLayer: blending control failed", exc_info=True)

        try:
            value = controls.get("interpolation")
            if value:
                self.interpolation = str(value)
        except Exception:
            logger.debug("RemoteImageLayer: interpolation control failed", exc_info=True)

        try:
            value = controls.get("colormap")
            if value:
                self.colormap = str(value)
        except Exception:
            logger.debug("RemoteImageLayer: colormap control failed", exc_info=True)

        try:
            value = controls.get("rendering")
            if value:
                self.rendering = str(value)
        except Exception:
            logger.debug("RemoteImageLayer: rendering control failed", exc_info=True)

        try:
            value = controls.get("gamma")
            if value is not None:
                self.gamma = float(value)
        except Exception:
            logger.debug("RemoteImageLayer: gamma control failed", exc_info=True)

        try:
            value = controls.get("contrast_limits")
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                self.contrast_limits = (float(value[0]), float(value[1]))
        except Exception:
            logger.debug("RemoteImageLayer: contrast_limits control failed", exc_info=True)

        try:
            value = controls.get("iso_threshold")
            if value is not None:
                self.iso_threshold = float(value)
        except Exception:
            logger.debug("RemoteImageLayer: iso_threshold control failed", exc_info=True)

        try:
            value = controls.get("attenuation")
            if value is not None:
                self.attenuation = float(value)
        except Exception:
            logger.debug("RemoteImageLayer: attenuation control failed", exc_info=True)

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
