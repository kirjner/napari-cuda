"""Mirror confirmed layer state from the client ledger into napari."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.data.registry import (
    RegistrySnapshot,
    RemoteLayerRegistry,
)
from napari_cuda.client.data.remote_image_layer import RemoteImageLayer
from napari_cuda.client.rendering.presenter_facade import PresenterFacade
from napari_cuda.client._qt.thumbnail_injector import (
    emit_layer_thumbnail_data_changed,
    flush_all_layers,
)
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.protocol.messages import NotifyLayersFrame, NotifySceneFrame
from napari_cuda.protocol.snapshots import (
    layer_delta_from_payload,
    scene_snapshot_from_payload,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from napari_cuda.client.control.emitters import NapariLayerIntentEmitter


logger = logging.getLogger(__name__)


class NapariLayerMirror:
    """Subscribe to layer ledger events and mirror them into napari."""

    def __init__(
        self,
        *,
        ledger: ClientStateLedger,
        state: ControlStateContext,
        loop_state: ClientLoopState,
        registry: RemoteLayerRegistry,
        presenter: PresenterFacade,
        viewer_ref,
        ui_call,
        log_layers_info: bool,
    ) -> None:
        assert ui_call is not None, "NapariLayerMirror requires a GUI call proxy"
        app = QtCore.QCoreApplication.instance()
        assert app is not None, "Qt application instance must exist"
        self._ui_thread = app.thread()
        self._ledger = ledger
        self._state = state
        self._loop_state = loop_state
        self._registry = registry
        self._presenter = presenter
        self._viewer_ref = viewer_ref
        self._log_layers_info = bool(log_layers_info)
        self._emitter: NapariLayerIntentEmitter | None = None
        self._last_snapshot: RegistrySnapshot | None = None
        self._last_scene_metadata: dict[str, Any] | None = None
        self._welcome_visible: Optional[bool] = None
        self._attached_layers: set[str] = set()

        registry.add_listener(self._on_registry_snapshot)

    # ------------------------------------------------------------------ configuration
    def set_logging(self, enabled: bool) -> None:
        self._log_layers_info = bool(enabled)

    def attach_emitter(self, emitter: NapariLayerIntentEmitter) -> None:
        self._assert_gui_thread()
        self._emitter = emitter
        snapshot = self._registry.snapshot()
        for record in snapshot.iter():
            if record.layer_id in self._attached_layers:
                continue
            emitter.attach_layer(record.layer)
            with emitter.suppressing():
                emitter.prime_from_block(record.layer_id, record.block)
            self._attached_layers.add(record.layer_id)

    # ------------------------------------------------------------------ ingest API
    def ingest_scene_snapshot(self, frame: NotifySceneFrame) -> None:
        self._assert_gui_thread()
        snapshot = scene_snapshot_from_payload(frame.payload)
        self._last_scene_metadata = dict(snapshot.metadata) if snapshot.metadata else None
        with self._suppress_emitter():
            self._registry.apply_snapshot(snapshot)
        self._update_welcome_overlay(snapshot.metadata, len(snapshot.layers))
        # Optionally force layer-list thumbnails to repaint immediately
        try:
            injected = flush_all_layers(self._current_viewer())
            if injected and logger.isEnabledFor(logging.DEBUG):
                logger.debug("notify.scene applied: forced %s thumbnail repaints", injected)
        except Exception:
            logger.debug("notify.scene: thumbnail injector failed", exc_info=True)
        logger.debug("notify.scene applied: layers=%s", len(snapshot.layers))

    def ingest_layer_delta(self, frame: NotifyLayersFrame) -> None:
        self._assert_gui_thread()
        delta = layer_delta_from_payload(frame.payload)
        with self._suppress_emitter():
            self._registry.apply_delta(delta)
            emitter = self._emitter
            if emitter is not None and delta.controls:
                emitter.apply_remote_values(delta.layer_id, delta.controls)
        self._presenter.apply_layer_delta(delta)
        # Optionally force a repaint of the specific layer row to reflect new thumbnail
        try:
            # Resolve the updated layer instance from the registry snapshot
            record = None
            snap = self._registry.snapshot()
            for r in snap.iter():
                if r.layer_id == delta.layer_id:
                    record = r
                    break
            if record is not None:
                viewer = self._current_viewer()
                if viewer is not None:
                    emit_layer_thumbnail_data_changed(viewer, record.layer)
        except Exception:
            logger.debug("notify.layers: thumbnail injector failed", exc_info=True)
        keys = tuple((delta.controls or {}).keys())
        logger.debug("notify.layers applied: id=%s controls=%s", delta.layer_id, keys)

    def replay_last_payload(self) -> None:
        self._assert_gui_thread()
        snapshot = self._last_snapshot
        if snapshot is None:
            return
        with self._suppress_emitter():
            self._sync_viewer_layers(snapshot)

    # ------------------------------------------------------------------ registry listener
    def _on_registry_snapshot(self, snapshot: RegistrySnapshot) -> None:
        self._assert_gui_thread()
        self._last_snapshot = snapshot
        emitter = self._emitter
        if emitter is not None:
            snapshot_ids = {record.layer_id for record in snapshot.iter()}
            for record in snapshot.iter():
                if record.layer_id in self._attached_layers:
                    continue
                emitter.attach_layer(record.layer)
                with emitter.suppressing():
                    emitter.prime_from_block(record.layer_id, record.block)
                self._attached_layers.add(record.layer_id)
            for layer_id in tuple(self._attached_layers):
                if layer_id not in snapshot_ids:
                    emitter.detach_layer(layer_id)
                    self._attached_layers.discard(layer_id)
        with self._suppress_emitter():
            self._sync_viewer_layers(snapshot)
        # Optionally ensure UI repaints thumbnails for all rows
        try:
            injected = flush_all_layers(self._current_viewer())
            if injected and logger.isEnabledFor(logging.DEBUG):
                logger.debug("registry snapshot: forced %s thumbnail repaints", injected)
        except Exception:
            logger.debug("registry snapshot: thumbnail injector failed", exc_info=True)
        self._update_welcome_overlay(self._last_scene_metadata, len(tuple(snapshot.iter())))

    # ------------------------------------------------------------------ helpers
    def _sync_viewer_layers(self, snapshot: RegistrySnapshot) -> None:
        viewer = self._current_viewer()
        if viewer is None:
            return
        layers_obj = viewer.layers
        self._assert_gui_thread()
        desired_records = list(snapshot.iter())
        desired_ids = [record.layer_id for record in desired_records]

        layers_events = layers_obj.events
        blocker = layers_events.blocker()
        previous_flag = getattr(viewer, "_suppress_forward_flag", False)
        viewer._suppress_forward_flag = True
        try:
            with blocker:
                existing_layers = list(layers_obj)
                for layer in existing_layers:
                    remote_id = self._remote_id(layer)
                    if remote_id is not None and remote_id not in desired_ids:
                        layers_obj.remove(layer)
                for idx, record in enumerate(desired_records):
                    layer = record.layer
                    if layer not in layers_obj:
                        layers_obj.insert(idx, layer)
                    current_index = layers_obj.index(layer)
                    if current_index != idx:
                        try:
                            layers_obj.move(current_index, idx)
                        except Exception:
                            layers_obj.pop(current_index)
                            layers_obj.insert(idx, layer)
                    self._apply_controls(layer, record.block)
        finally:
            viewer._suppress_forward_flag = previous_flag

    def _apply_controls(self, layer: RemoteImageLayer, block: Mapping[str, Any]) -> None:
        controls = block.get("controls")
        if not isinstance(controls, Mapping):
            return
        if "visible" in controls:
            target = bool(controls["visible"])
            if target != bool(layer.visible):
                emitter = layer.events.visible
                with emitter.blocker():
                    layer.visible = target

    def _current_viewer(self):
        return self._viewer_ref() if callable(self._viewer_ref) else None

    def _suppress_emitter(self):
        emitter = self._emitter
        if emitter is None:
            return nullcontext()
        return emitter.suppressing()

    @staticmethod
    def _remote_id(layer: object) -> Optional[str]:
        if isinstance(layer, RemoteImageLayer):
            return layer.remote_id
        return None

    def _update_welcome_overlay(
        self,
        metadata: Optional[Mapping[str, Any]],
        layer_count: int,
    ) -> None:
        """Toggle napari's welcome overlay to mirror upstream idle UX."""

        show = self._should_show_welcome(metadata, layer_count)
        if self._welcome_visible is not None and self._welcome_visible == show:
            return
        self._set_welcome_visible(show)
        self._welcome_visible = show

    def _set_welcome_visible(self, visible: bool) -> None:
        viewer = self._current_viewer()
        if viewer is None:
            return
        window = getattr(viewer, "window", None)
        qt_viewer = getattr(window, "_qt_viewer", None)
        if qt_viewer is None:
            return
        setter = getattr(qt_viewer, "set_welcome_visible", None)
        if callable(setter):
            setter(bool(visible))
            return
        overlay = getattr(qt_viewer, "_welcome_widget", None)
        toggler = getattr(overlay, "set_welcome_visible", None)
        if callable(toggler):
            toggler(bool(visible))

    @staticmethod
    def _should_show_welcome(
        metadata: Optional[Mapping[str, Any]],
        layer_count: int,
    ) -> bool:
        if metadata is not None:
            status = str(metadata.get("status", "")).lower()
            if status == "idle":
                return True
            if status == "ready":
                return False
        return layer_count <= 0

    def _assert_gui_thread(self) -> None:
        current = QtCore.QThread.currentThread()
        assert (
            current is self._ui_thread
        ), "NapariLayerMirror methods must run on the Qt GUI thread"


__all__ = ["NapariLayerMirror"]
