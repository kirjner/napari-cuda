"""Emit camera intents derived from local interactions."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import (
    AckReconciliation,
    ClientStateLedger,
    IntentRecord,
)
from napari_cuda.client.control.state_update_actions import (
    ControlStateContext,
    _emit_state_update,
    _update_runtime_from_ack_outcome,
)
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState

logger = logging.getLogger(__name__)


class NapariCameraIntentEmitter:
    """Translate viewer camera interactions into coordinator intents."""

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
    def zoom(
        self,
        *,
        factor: float,
        anchor_px: tuple[float, float],
        origin: str,
    ) -> bool:
        sanitized = {
            "factor": float(factor),
            "anchor_px": [float(anchor_px[0]), float(anchor_px[1])],
        }
        return self._emit_camera_delta(
            key="zoom",
            value=sanitized,
            origin=origin,
            metadata={
                "mode": "zoom",
                "origin": origin,
                "delta": dict(sanitized),
                "update_kind": "delta",
            },
        )

    def pan(
        self,
        *,
        dx_px: float,
        dy_px: float,
        origin: str,
    ) -> bool:
        sanitized = {"dx_px": float(dx_px), "dy_px": float(dy_px)}
        return self._emit_camera_delta(
            key="pan",
            value=sanitized,
            origin=origin,
            metadata={
                "mode": "pan",
                "origin": origin,
                "delta": dict(sanitized),
                "update_kind": "delta",
            },
        )

    def orbit(
        self,
        *,
        d_az_deg: float,
        d_el_deg: float,
        origin: str,
    ) -> bool:
        sanitized = {"d_az_deg": float(d_az_deg), "d_el_deg": float(d_el_deg)}
        return self._emit_camera_delta(
            key="orbit",
            value=sanitized,
            origin=origin,
            metadata={
                "mode": "orbit",
                "origin": origin,
                "delta": dict(sanitized),
                "update_kind": "delta",
            },
        )

    def reset(
        self,
        *,
        reason: str,
        origin: str,
    ) -> bool:
        sanitized = {"reason": str(reason)}
        return self._emit_camera_delta(
            key="reset",
            value=sanitized,
            origin=origin,
            metadata={
                "mode": "reset",
                "origin": origin,
                "delta": dict(sanitized),
                "update_kind": "delta",
            },
        )

    def set(
        self,
        *,
        center: Optional[Sequence[float]] = None,
        zoom: Optional[float] = None,
        angles: Optional[Sequence[float]] = None,
        origin: str,
    ) -> bool:
        payload: dict[str, Any] = {}
        if center is not None:
            payload["center"] = [float(value) for value in center]
        if zoom is not None:
            payload["zoom"] = float(zoom)
        if angles is not None:
            payload["angles"] = [float(value) for value in angles]
        if not payload:
            return False
        return self._emit_camera_delta(
            key="set",
            value=payload,
            origin=origin,
            metadata={
                "mode": "set",
                "origin": origin,
                "delta": dict(payload),
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
                self._state.camera_state[key] = applied
                self._confirmed[key] = applied
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
        self._state.camera_state[key] = revert_value
        self._confirmed[key] = revert_value
        if self._log_camera_info:
            logger.warning("camera intent reverted: key=%s", key)

    def record_confirmed(self, key: str, value: Mapping[str, Any]) -> None:
        self._assert_gui_thread()
        self._state.camera_state[str(key)] = dict(value)
        self._confirmed[str(key)] = dict(value)

    # ------------------------------------------------------------------ helpers
    def _emit_camera_delta(
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
        self._state.camera_state[str(key)] = projection_value
        self._confirmed[str(key)] = projection_value
        if self._log_camera_info:
            logger.info(
                "camera intent -> state.update key=%s origin=%s value=%s",
                key,
                origin,
                projection_value,
            )
        return True

    def _assert_gui_thread(self) -> None:
        current = QtCore.QThread.currentThread()
        assert current is self._ui_thread, "NapariCameraIntentEmitter methods must run on the Qt GUI thread"


__all__ = ["NapariCameraIntentEmitter"]
