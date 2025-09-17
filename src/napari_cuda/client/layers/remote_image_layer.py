"""napari Image layer subclass backed by remote metadata only."""

from __future__ import annotations

import logging
from typing import Optional

from napari.layers.image.image import Image
from napari.layers._scalar_field._slice import _ScalarFieldSliceResponse

from napari_cuda.protocol.messages import LayerRenderHints, LayerSpec, MultiscaleSpec

from .remote_data import RemoteArray, RemoteMultiscale, build_remote_data

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
        metadata = dict(spec.metadata or {})
        metadata.setdefault(self._REMOTE_ID_META_KEY, spec.layer_id)
        init_kwargs = self._build_init_kwargs(spec, metadata)
        super().__init__(data, **init_kwargs)
        self._allow_data_update = False
        self._remote_spec = spec
        self._remote_id = spec.layer_id
        self.editable = False
        self._keep_auto_contrast = False
        self.visible = bool(spec.render.visibility) if spec.render and spec.render.visibility is not None else False
        self._install_empty_slice()

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
        if hints.opacity is not None:
            mapping["opacity"] = float(hints.opacity)
        if hints.visibility is not None:
            mapping["visible"] = bool(hints.visibility)
        if hints.colormap:
            mapping["colormap"] = hints.colormap
        if hints.mode:
            mapping["rendering"] = hints.mode
        if hints.gamma is not None:
            mapping["gamma"] = float(hints.gamma)
        if hints.iso_threshold is not None:
            mapping["iso_threshold"] = float(hints.iso_threshold)
        if hints.attenuation is not None:
            mapping["attenuation"] = float(hints.attenuation)
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
        if spec.contrast_limits:
            self.contrast_limits = tuple(spec.contrast_limits)
        self.metadata = metadata
        self._apply_render(spec.render)
        self._install_empty_slice()

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
        if hints.visibility is not None:
            self.visible = bool(hints.visibility)
        if hints.opacity is not None:
            self.opacity = float(hints.opacity)
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
        if hints.gamma is not None:
            self.gamma = float(hints.gamma)
        if hints.iso_threshold is not None:
            try:
                self.iso_threshold = float(hints.iso_threshold)
            except Exception:
                logger.debug("RemoteImageLayer: iso threshold update failed", exc_info=True)
        if hints.attenuation is not None:
            try:
                self.attenuation = float(hints.attenuation)
            except Exception:
                logger.debug("RemoteImageLayer: attenuation update failed", exc_info=True)

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
