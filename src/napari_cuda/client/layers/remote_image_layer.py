"""napari Image layer subclass backed by remote metadata only."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Mapping, Optional

import numpy as np

from napari.layers.image.image import Image
from napari.layers._scalar_field._slice import _ScalarFieldSliceResponse, _ScalarFieldView
try:  # Qt may be unavailable in headless test environments
    from qtpy import QtCore  # type: ignore
except Exception:  # pragma: no cover - best effort import
    QtCore = None

from napari_cuda.protocol.messages import LayerRenderHints, LayerSpec, MultiscaleSpec

from .remote_data import RemoteArray, RemoteMultiscale, RemotePreview, build_remote_data

logger = logging.getLogger(__name__)


class RemoteImageLayer(Image):
    """Read-only image layer describing remote content."""

    _REMOTE_ID_META_KEY = "napari_cuda.remote_layer_id"

    def __init__(self, spec: LayerSpec) -> None:
        self._allow_data_update = True
        data, multiscale = build_remote_data(spec)
        arrays: tuple[RemoteArray, ...]
        if isinstance(multiscale, RemoteMultiscale):
            arrays = multiscale.arrays
        elif isinstance(data, RemoteArray):
            arrays = (data,)
        else:
            arrays = ()
        self._remote_arrays = arrays
        self._remote_preview = RemotePreview()
        metadata = dict(spec.metadata or {})
        preview = self._preview_from_metadata(metadata)
        metadata.setdefault(self._REMOTE_ID_META_KEY, spec.layer_id)
        init_kwargs = self._build_init_kwargs(spec, metadata)
        super().__init__(data, **init_kwargs)
        self._allow_data_update = False
        self._remote_spec = spec
        self._remote_id = spec.layer_id
        self.editable = False
        self._keep_auto_contrast = False
        self._install_empty_slice()
        fallback_preview = self._extract_preview() if preview is None else preview
        self.update_preview(fallback_preview)
        self._apply_controls(spec.controls)

    # ------------------------------------------------------------------
    def _build_init_kwargs(self, spec: LayerSpec, metadata: dict) -> dict:
        render = spec.render
        kwargs: dict = {
            "name": spec.name,
            "metadata": metadata,
            "multiscale": bool(spec.multiscale),
            "axis_labels": tuple(spec.axis_labels) if spec.axis_labels else None,
            "scale": tuple(spec.scale) if spec.scale else None,
            "translate": tuple(spec.translate) if spec.translate else None,
            "contrast_limits": tuple(spec.contrast_limits) if spec.contrast_limits else None,
        }
        if render is not None:
            kwargs.update(self._render_kwargs(render))
        return {k: v for k, v in kwargs.items() if v is not None}

    def _render_kwargs(self, hints: LayerRenderHints) -> dict:
        mapping: dict = {}
        if hints.mode:
            mapping["rendering"] = hints.mode
        if hints.colormap:
            mapping["colormap"] = hints.colormap
        if hints.shading:
            mapping["shading"] = hints.shading
        return mapping

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
    def update_from_spec(self, spec: LayerSpec) -> None:
        self._remote_spec = spec
        self._remote_id = spec.layer_id
        metadata = dict(spec.metadata or {})
        preview = self._preview_from_metadata(metadata)
        metadata.setdefault(self._REMOTE_ID_META_KEY, spec.layer_id)
        self._update_remote_arrays(spec)
        self._allow_data_update = True
        try:
            if not self.multiscale and self._remote_arrays:
                self.data = self._remote_arrays[0]
        finally:
            self._allow_data_update = False
        self.name = spec.name
        if spec.axis_labels:
            self.axis_labels = tuple(spec.axis_labels)
        if spec.scale:
            self.scale = tuple(spec.scale)
        if spec.translate:
            self.translate = tuple(spec.translate)
        self.metadata = metadata
        self._apply_render(spec.render)
        self._apply_controls(spec.controls)
        self._install_empty_slice()
        fallback_preview = self._extract_preview() if preview is None else preview
        self.update_preview(fallback_preview)

    def _update_remote_arrays(self, spec: LayerSpec) -> None:
        extras = spec.extras or {}
        data_id = extras.get("data_id") if isinstance(extras, dict) else None
        cache_version = extras.get("cache_version") if isinstance(extras, dict) else None
        try:
            cache_version = int(cache_version) if cache_version is not None else None
        except Exception:
            cache_version = None
        if spec.multiscale and isinstance(spec.multiscale, MultiscaleSpec):
            levels = spec.multiscale.levels
        else:
            levels = None
        if levels:
            if not self.multiscale or len(levels) != len(self._remote_arrays):
                self._rebuild_data(spec)
                return
            for remote, level in zip(self._remote_arrays, levels):
                remote.update(shape=level.shape or spec.shape, dtype=spec.dtype, data_id=data_id, cache_version=cache_version)
            return
        if not self._remote_arrays or self.multiscale:
            self._rebuild_data(spec)
            return
        self._remote_arrays[0].update(shape=spec.shape, dtype=spec.dtype, data_id=data_id, cache_version=cache_version)

    def _rebuild_data(self, spec: LayerSpec) -> None:
        data, multiscale = build_remote_data(spec)
        self.multiscale = bool(spec.multiscale)
        if isinstance(multiscale, RemoteMultiscale):
            arrays = multiscale.arrays
            new_data = multiscale.as_multiscale()
        elif isinstance(data, RemoteArray):
            arrays = (data,)
            new_data = data
        else:
            arrays = ()
            new_data = data
        self._remote_arrays = arrays
        self._allow_data_update = True
        try:
            Image.data.fset(self, new_data)
        finally:
            self._allow_data_update = False

    def _apply_render(self, hints: Optional[LayerRenderHints]) -> None:
        if hints is None:
            return
        if hints.mode:
            try:
                self.rendering = hints.mode
            except Exception:
                logger.debug("RemoteImageLayer: render mode %s unsupported", hints.mode, exc_info=True)
        if hints.colormap:
            try:
                self.colormap = hints.colormap
            except Exception:
                logger.debug("RemoteImageLayer: colormap %s unsupported", hints.colormap, exc_info=True)
        if hints.shading:
            try:
                setattr(self, "shading", hints.shading)
            except Exception:
                logger.debug("RemoteImageLayer: shading update failed", exc_info=True)

    def _apply_controls(self, controls: Optional[Mapping[str, Any]]) -> None:
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
        return getattr(self, "_remote_id", "")

    @Image.data.setter  # type: ignore[misc]
    def data(self, data):  # type: ignore[override]
        if getattr(self, "_allow_data_update", False):
            Image.data.fset(self, data)
            return
        logger.debug("RemoteImageLayer: ignoring data mutation attempt")

    # ------------------------------------------------------------------
    def update_preview(self, preview: np.ndarray | None) -> None:
        """Update the cached preview (scalar or RGB data in [0, 1])."""

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
        if QtCore is None:  # No Qt -> assume current thread is acceptable
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
            preview = self._remote_preview.as_thumbnail(self.rgb, (thumb_shape[0], thumb_shape[1]))
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
