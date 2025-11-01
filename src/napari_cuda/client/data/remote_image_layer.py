"""napari Image layer subclass backed by remote metadata only."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from napari.layers.image.image import Image
from napari.layers._scalar_field._slice import _ScalarFieldSliceResponse, _ScalarFieldView
from napari.layers.image._image_constants import Interpolation as NapariInterpolation
from napari.utils.events import EventEmitter
try:
    from qtpy import QtCore  # type: ignore
except Exception:  # pragma: no cover - best effort import
    QtCore = None

from .remote_data import RemoteArray, RemoteMultiscale, RemotePreview, build_remote_data

try:
    from napari_cuda.client.runtime.client_loop.scheduler import CallProxy
except Exception:  # pragma: no cover - import is expected to succeed in client runtime
    CallProxy = None

logger = logging.getLogger(__name__)

_CONTROL_ONLY_BLOCK_KEYS = {
    "controls",
    "contrast_limits",
    "render",
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
        self._control_event_proxy: Optional[CallProxy] = None

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
        default_preview = self._extract_preview() if preview is None else preview
        self.update_preview(default_preview)
        controls_block = self._remote_block["controls"]
        assert isinstance(controls_block, Mapping), "remote layer missing controls mapping"
        self._apply_controls(controls_block)

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
                # Ensure napari treats data as multiscale before setting it and
                # pre-size corner_pixels to the incoming ndim to avoid early vispy indexing.
                self.multiscale = multiscale_flag
                target_ndim = int(new_block.get("ndim") or 0) or (
                    len(arrays[-1].shape) if arrays else len(new_block.get("shape") or ())
                ) or getattr(self, "ndim", 2)
                self.corner_pixels = np.zeros((2, int(target_ndim)), dtype=int)
                Image.data.fset(self, new_data)
            finally:
                self._allow_data_update = False

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
            default_preview = self._extract_preview() if preview is None else preview
            self.update_preview(default_preview)
        else:
            if metadata:
                try:
                    self.metadata = metadata
                except Exception:
                    self.metadata.update(metadata)
            default_preview = self._extract_preview() if preview is None else preview
            self.update_preview(default_preview)

        self._apply_render(self._remote_block.get("render"))
        controls_block = self._remote_block["controls"]
        assert isinstance(controls_block, Mapping), "remote layer missing controls mapping"
        self._apply_controls(controls_block)
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
        # On dim/ndisplay changes, re-apply the current preview into the
        # layer slice so thumbnails reflect the new view mode immediately
        # (e.g., 3D↔2D). Avoid clearing to an empty slice which would blank
        # the thumbnail until the next preview metadata arrives.
        self._apply_preview_to_slice()
        # Schedule a thumbnail refresh on the GUI thread when available.
        self._schedule_thumbnail_refresh()

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

        known_keys = {
            "visible",
            "opacity",
            "blending",
            "interpolation",
            "colormap",
            "rendering",
            "gamma",
            "contrast_limits",
            "iso_threshold",
            "attenuation",
        }
        unexpected = set(controls) - known_keys
        assert not unexpected, f"Unsupported control keys for remote layer: {sorted(unexpected)}"

        pending_events: list[tuple[EventEmitter, Any]] = []

        if "visible" in controls:
            value = controls["visible"]
            assert value is not None, "visible control cannot be None"
            new_val = bool(value)
            if bool(self.visible) is not new_val:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("RemoteImageLayer[%s]: set visible -> %s", self._remote_id, new_val)
                with self.events.visible.blocker():
                    self.visible = new_val
                pending_events.append((self.events.visible, new_val))

        if "opacity" in controls:
            value = controls["opacity"]
            assert value is not None, "opacity control cannot be None"
            new_val = float(value)
            current = float(self.opacity)
            if current != new_val:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("RemoteImageLayer[%s]: set opacity -> %s", self._remote_id, new_val)
                with self.events.opacity.blocker():
                    self.opacity = new_val
                pending_events.append((self.events.opacity, new_val))

        if "blending" in controls:
            new_val = str(controls["blending"])
            if self.blending != new_val:
                with self.events.blending.blocker():
                    self.blending = new_val
                pending_events.append((self.events.blending, new_val))

        if "interpolation" in controls:
            new_val = str(controls["interpolation"])
            change_2d = str(self.interpolation2d) != new_val
            change_3d = str(self.interpolation3d) != new_val
            if change_2d or change_3d:
                if change_2d:
                    with self.events.interpolation2d.blocker():
                        self.interpolation2d = new_val
                    pending_events.append((self.events.interpolation2d, new_val))
                if change_3d:
                    with self.events.interpolation3d.blocker():
                        self.interpolation3d = new_val
                    pending_events.append((self.events.interpolation3d, new_val))

        if "colormap" in controls:
            new_val = str(controls["colormap"])
            if str(self.colormap) != new_val:
                with self.events.colormap.blocker():
                    self.colormap = new_val
                pending_events.append((self.events.colormap, new_val))

        if "rendering" in controls:
            new_val = str(controls["rendering"])
            if self.rendering != new_val:
                with self.events.rendering.blocker():
                    self.rendering = new_val
                pending_events.append((self.events.rendering, new_val))

        if "gamma" in controls:
            value = controls["gamma"]
            assert value is not None, "gamma control cannot be None"
            new_val = float(value)
            if float(self.gamma) != new_val:
                with self.events.gamma.blocker():
                    self.gamma = new_val
                pending_events.append((self.events.gamma, new_val))

        if "contrast_limits" in controls:
            pair = controls["contrast_limits"]
            assert isinstance(pair, (list, tuple)) and len(pair) == 2, "contrast_limits requires 2 values"
            new_limits = (float(pair[0]), float(pair[1]))
            current = tuple(float(v) for v in self.contrast_limits)
            if current != new_limits:
                with self.events.contrast_limits.blocker():
                    self.contrast_limits = new_limits
                pending_events.append((self.events.contrast_limits, new_limits))

        if "iso_threshold" in controls:
            value = controls["iso_threshold"]
            assert value is not None, "iso_threshold control cannot be None"
            new_val = float(value)
            if float(self.iso_threshold) != new_val:
                with self.events.iso_threshold.blocker():
                    self.iso_threshold = new_val
                pending_events.append((self.events.iso_threshold, new_val))

        if "attenuation" in controls:
            value = controls["attenuation"]
            assert value is not None, "attenuation control cannot be None"
            new_val = float(value)
            if float(self.attenuation) != new_val:
                with self.events.attenuation.blocker():
                    self.attenuation = new_val
                pending_events.append((self.events.attenuation, new_val))

        if pending_events:
            self._dispatch_control_events(pending_events)

    def _emit_control_event(self, emitter: EventEmitter, *, value: Any) -> None:
        assert isinstance(emitter, EventEmitter), "RemoteImageLayer missing expected event emitter"
        if QtCore is None:
            emitter(value=value)
            return
        app = QtCore.QCoreApplication.instance()
        if app is None or QtCore.QThread.currentThread() is app.thread():
            emitter(value=value)
            return
        proxy = self._control_event_proxy
        if proxy is None:
            assert CallProxy is not None, "CallProxy unavailable for RemoteImageLayer control events"
            proxy = CallProxy(parent=app)
            self._control_event_proxy = proxy
        proxy.call.emit(lambda: emitter(value=value))

    def _dispatch_control_events(self, events: Sequence[tuple[EventEmitter, Any]]) -> None:
        for emitter, value in events:
            self._emit_control_event(emitter, value=value)

    def apply_control_changes(self, block: Mapping[str, Any], changes: Mapping[str, Any]) -> None:
        self._remote_block = dict(block)
        controls = self._remote_block["controls"]
        assert isinstance(controls, Mapping), "control block missing 'controls' mapping"
        subset: dict[str, Any] = {}
        for key in changes:
            if key == "removed":
                continue
            assert key in controls, f"control delta missing value for {key}"
            subset[key] = controls[key]
        if subset:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "RemoteImageLayer[%s]: apply_control_changes keys=%s",
                    self._remote_id,
                    list(subset.keys()),
                )
            self._apply_controls(subset)

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
