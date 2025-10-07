"""Mirror confirmed layer state from the client ledger into napari."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Mapping, Optional, TYPE_CHECKING

from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.data.registry import RegistrySnapshot, RemoteLayerRegistry
from napari_cuda.client.data.remote_image_layer import RemoteImageLayer
from napari_cuda.client.rendering.presenter_facade import PresenterFacade
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.protocol.messages import NotifySceneFrame, NotifyLayersFrame
from napari_cuda.protocol.snapshots import layer_delta_from_payload, scene_snapshot_from_payload


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
        self._emitter: "NapariLayerIntentEmitter" | None = None
        self._last_snapshot: RegistrySnapshot | None = None
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
        with self._suppress_emitter():
            self._registry.apply_snapshot(snapshot)
        logger.debug("notify.scene applied: layers=%s", len(snapshot.layers))

    def ingest_layer_delta(self, frame: NotifyLayersFrame) -> None:
        self._assert_gui_thread()
        delta = layer_delta_from_payload(frame.payload)
        with self._suppress_emitter():
            self._registry.apply_delta(delta)
            emitter = self._emitter
            if emitter is not None:
                emitter.apply_remote_values(delta.layer_id, delta.changes)
        self._presenter.apply_layer_delta(delta)
        logger.debug("notify.layers applied: id=%s keys=%s", delta.layer_id, tuple(delta.changes.keys()))

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

    def _assert_gui_thread(self) -> None:
        current = QtCore.QThread.currentThread()
        assert (
            current is self._ui_thread
        ), "NapariLayerMirror methods must run on the Qt GUI thread"


__all__ = ["NapariLayerMirror"]
