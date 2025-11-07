"""Mirror confirmed camera state from the ledger into presentation hooks."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.rendering.presenter_facade import PresenterFacade
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.protocol import NotifyCamera

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.control.emitters.napari_camera_intent_emitter import (
        NapariCameraIntentEmitter,
    )

logger = logging.getLogger(__name__)


class NapariCameraMirror:
    """Ingest `notify.camera` frames and keep local state in sync."""

    def __init__(
        self,
        *,
        ledger: ClientStateLedger,
        state: ControlStateContext,
        loop_state: ClientLoopState,
        presenter: PresenterFacade,
        ui_call: Any,
        log_camera_info: bool,
    ) -> None:
        app = QtCore.QCoreApplication.instance()
        assert app is not None, "Qt application instance must exist"
        self._ui_thread = app.thread()
        self._ledger = ledger
        self._state = state
        self._loop_state = loop_state
        self._presenter = presenter
        self._ui_call = ui_call
        self._log_camera_info = bool(log_camera_info)
        self._emitter: NapariCameraIntentEmitter | None = None
        self._last_deltas: dict[str, Mapping[str, Any]] = {}
        self._last_states: dict[str, Mapping[str, Any]] = {}

    # ------------------------------------------------------------------ configuration
    def set_logging(self, enabled: bool) -> None:
        self._log_camera_info = bool(enabled)

    def attach_emitter(self, emitter: NapariCameraIntentEmitter) -> None:
        self._assert_gui_thread()
        self._emitter = emitter
        for mode, delta in self._last_deltas.items():
            emitter.record_confirmed(mode, delta)

    # ------------------------------------------------------------------ ingest API
    def ingest_notify_camera(self, frame: NotifyCamera) -> None:
        self._assert_gui_thread()
        payload = frame.payload
        mode = str(payload.mode or "")
        mode_key = mode if mode else "main"
        timestamp = frame.envelope.timestamp

        if payload.state is not None:
            state_payload = dict(payload.state) if isinstance(payload.state, Mapping) else payload.state
            self._ledger.record_confirmed(
                "camera",
                "main",
                mode_key,
                state_payload,
                timestamp=timestamp,
            )
            self._state.camera_state[mode_key] = state_payload
            self._last_states[mode_key] = state_payload
            self._presenter.apply_camera_update(mode=mode_key, payload=state_payload)
            if self._log_camera_info:
                logger.info(
                    "notify.camera applied: mode=%s intent=%s state=%s",
                    mode_key,
                    frame.envelope.intent_id,
                    state_payload,
                )
            return

        if payload.delta is None:
            raise ValueError("notify.camera delta payload missing")
        delta_payload = payload.delta
        self._ledger.record_confirmed(
            "camera",
            "main",
            mode_key,
            delta_payload,
            timestamp=timestamp,
        )
        self._state.camera_state[mode_key] = delta_payload
        self._last_deltas[mode_key] = delta_payload
        emitter = self._emitter
        if emitter is not None:
            emitter.record_confirmed(mode_key, delta_payload)
        if self._log_camera_info:
            logger.info(
                "notify.camera applied: mode=%s intent=%s delta=%s",
                mode_key,
                frame.envelope.intent_id,
                delta_payload,
            )
        self._presenter.apply_camera_update(mode=mode_key, payload=delta_payload)

    def replay_last_payload(self) -> None:
        self._assert_gui_thread()
        for mode, state_payload in self._last_states.items():
            self._presenter.apply_camera_update(mode=mode, payload=state_payload)
        for mode, delta_payload in self._last_deltas.items():
            self._presenter.apply_camera_update(mode=mode, payload=delta_payload)

    # ------------------------------------------------------------------ helpers
    def _assert_gui_thread(self) -> None:
        current = QtCore.QThread.currentThread()
        assert current is self._ui_thread, "NapariCameraMirror methods must run on the Qt GUI thread"


__all__ = ["NapariCameraMirror"]
