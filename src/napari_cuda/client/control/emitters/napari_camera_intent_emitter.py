"""Emit camera intents derived from local interactions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import AckReconciliation, ClientStateLedger, IntentRecord
from napari_cuda.client.control.state_update_actions import (
    ControlStateContext,
    _emit_state_update,
    _normalize_camera_state_value,
    _update_runtime_from_ack_outcome,
)
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState

logger = logging.getLogger(__name__)


class NapariCameraIntentEmitter:
    """Translate viewer camera updates into coordinator intents."""

    def __init__(
        self,
        *,
        ledger: ClientStateLedger,
        state: ControlStateContext,
        loop_state: ClientLoopState,
        dispatch_state_update: Callable[[IntentRecord, str], bool],
        ui_call: Any | None,
        log_camera_info: bool,
    ) -> None:
        app = QtCore.QCoreApplication.instance()
        assert app is not None, "Qt application instance must exist"
        self._ui_thread = app.thread()
        self._ledger = ledger
        self._state = state
        self._loop_state = loop_state
        self._dispatch_state_update = dispatch_state_update
        self._ui_call = ui_call
        self._log_camera_info = bool(log_camera_info)
        self._confirmed: dict[str, Mapping[str, Any]] = {}

    # ------------------------------------------------------------------ configuration
    def set_logging(self, enabled: bool) -> None:
        self._log_camera_info = bool(enabled)

    # ------------------------------------------------------------------ public API
    def set(
        self,
        *,
        center: Optional[Sequence[float]] = None,
        zoom: Optional[float] = None,
        angles: Optional[Sequence[float]] = None,
        origin: str,
    ) -> bool:
        """Publish an absolute camera snapshot via `camera.set` intent."""

        payload: dict[str, Any] = {}
        if center is not None:
            payload["center"] = [float(value) for value in center]
        if zoom is not None:
            payload["zoom"] = float(zoom)
        if angles is not None:
            payload["angles"] = [float(value) for value in angles]
        if not payload:
            return False
        return self._emit_camera_update(
            key="set",
            value=payload,
            origin=origin,
            metadata={
                "mode": "set",
                "origin": origin,
                "state": dict(payload),
                "update_kind": "absolute",
            },
        )

    def handle_ack(self, outcome: AckReconciliation) -> None:
        self._assert_gui_thread()
        if outcome.scope != "camera" or outcome.target != "main" or outcome.key is None:
            return

        _update_runtime_from_ack_outcome(self._state, outcome)

        key = str(outcome.key)
        if outcome.status == "accepted":
            applied = outcome.applied_value or outcome.confirmed_value or outcome.pending_value
            if applied is not None:
                normalized = _normalize_camera_state_value(applied)
                self._state.camera_state[key] = normalized
                self._confirmed[key] = normalized
            if self._log_camera_info:
                logger.info(
                    "camera intent accepted: key=%s pending=%d",
                    key,
                    outcome.pending_len,
                )
            return

        revert_value = outcome.confirmed_value if outcome.confirmed_value is not None else outcome.pending_value
        if revert_value is None:
            return
        normalized = _normalize_camera_state_value(revert_value)
        self._state.camera_state[key] = normalized
        self._confirmed[key] = normalized
        if self._log_camera_info:
            logger.warning("camera intent reverted: key=%s", key)

    def record_confirmed(self, key: str, value: Mapping[str, Any]) -> None:
        self._assert_gui_thread()
        normalized = _normalize_camera_state_value(value)
        self._state.camera_state[str(key)] = normalized
        self._confirmed[str(key)] = normalized

    # ------------------------------------------------------------------ helpers
    def _emit_camera_update(
        self,
        *,
        key: str,
        value: Mapping[str, Any],
        origin: str,
        metadata: Mapping[str, Any] | None,
    ) -> bool:
        self._assert_gui_thread()
        ok, projection = _emit_state_update(
            self._state,
            self._loop_state,
            self._ledger,
            self._dispatch_state_update,
            scope="camera",
            target="main",
            key=key,
            value=dict(value),
            origin=origin,
            metadata=metadata,
        )
        if not ok:
            return False
        projection_value = projection if projection is not None else dict(value)
        normalized = _normalize_camera_state_value(projection_value)
        self._state.camera_state[str(key)] = normalized
        self._confirmed[str(key)] = normalized
        if self._log_camera_info:
            logger.info(
                "camera intent -> state.update key=%s origin=%s value=%s",
                key,
                origin,
                normalized,
            )
        return True

    def _assert_gui_thread(self) -> None:
        current = QtCore.QThread.currentThread()
        assert current is self._ui_thread, "NapariCameraIntentEmitter methods must run on the Qt GUI thread"


__all__ = ["NapariCameraIntentEmitter"]
