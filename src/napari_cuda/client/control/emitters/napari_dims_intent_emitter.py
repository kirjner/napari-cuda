"""Emit dims intents derived from local UI interactions."""

from __future__ import annotations

import logging
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from qtpy import QtCore

from napari_cuda.client.control import state_update_actions as control_actions
from napari_cuda.client.control.client_state_ledger import (
    ClientStateLedger,
    IntentRecord,
)
from napari_cuda.client.control.state_update_actions import (
    ControlStateContext,
    _emit_state_update,
    _rate_gate_settings,
    current_ndisplay,
)
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingCoalesce:
    axis: int
    value: int


class NapariDimsIntentEmitter:
    """Translate viewer dims interactions into coordinator intents."""

    def __init__(
        self,
        *,
        ledger: ClientStateLedger,
        state: ControlStateContext,
        loop_state: ClientLoopState,
        dispatch_state_update: Callable[[IntentRecord, str], bool],
        ui_call: Optional[Any],
        log_dims_info: bool,
        tx_interval_ms: int,
    ) -> None:
        self._ledger = ledger
        self._state = state
        self._loop_state = loop_state
        self._dispatch_state_update = dispatch_state_update
        self._ui_call = ui_call
        self._log_dims_info = bool(log_dims_info)
        self._tx_interval_ms = max(0, int(tx_interval_ms))

        self._viewer_ref: weakref.ReferenceType[Any] | None = None
        self._viewer_dims = None
        self._dims_tx_timer: QtCore.QTimer | None = None
        self._pending: _PendingCoalesce | None = None
        self._last_step_ui: tuple[int, ...] | None = None
        self._suppress_count = 0
        self._is_playing = False
        self._play_axis: int | None = None
        self._last_margin_left: tuple[float, ...] | None = None
        self._last_margin_right: tuple[float, ...] | None = None

    # --------------------------------------------------------------------- lifecycle
    def attach_viewer(self, viewer: Any) -> None:
        """Attach the viewer whose dims we will observe."""

        self.detach_viewer()

        dims = getattr(viewer, "dims", None)
        if dims is None:
            raise AssertionError("viewer missing dims")
        events = getattr(dims, "events", None)
        if events is None:
            raise AssertionError("viewer dims missing events")

        events.current_step.connect(self._on_dims_change)
        events.ndisplay.connect(self._on_ndisplay_change)
        # Sync slab margins for 2D projection thickness
        if hasattr(events, 'margin_left') and hasattr(events, 'margin_right'):
            events.margin_left.connect(self._on_margins_change)
            events.margin_right.connect(self._on_margins_change)

        self._viewer_ref = weakref.ref(viewer)
        self._viewer_dims = dims
        self._sync_last_step_from_viewer()

    def detach_viewer(self) -> None:
        dims = self._viewer_dims
        if dims is not None:
            events = getattr(dims, "events", None)
            if events is not None:
                events.current_step.disconnect(self._on_dims_change)
                events.ndisplay.disconnect(self._on_ndisplay_change)
                events.margin_left.disconnect(self._on_margins_change)
                events.margin_right.disconnect(self._on_margins_change)
        if self._dims_tx_timer is not None:
            self._dims_tx_timer.stop()
            self._dims_tx_timer.deleteLater()
            self._dims_tx_timer = None
        self._viewer_ref = None
        self._viewer_dims = None
        self._pending = None
        self._last_step_ui = None
        self._is_playing = False
        self._play_axis = None
        self._last_margin_left = None
        self._last_margin_right = None

    def shutdown(self) -> None:
        self.detach_viewer()

    # ---------------------------------------------------------------- suppression API
    def suppress_forward(self) -> None:
        self._suppress_count += 1
        if self._suppress_count == 1:
            self._pending = None

    def resume_forward(self) -> None:
        if self._suppress_count == 0:
            return
        self._suppress_count -= 1
        if self._suppress_count == 0:
            self._sync_last_step_from_viewer()

    @contextmanager
    def suppressing(self):
        self.suppress_forward()
        try:
            yield
        finally:
            self.resume_forward()

    # --------------------------------------------------------------------- settings
    def set_logging(self, enabled: bool) -> None:
        self._log_dims_info = bool(enabled)

    def set_tx_interval_ms(self, value: int) -> None:
        self._tx_interval_ms = max(0, int(value))

    # ------------------------------------------------------------------- public API
    def dims_step(self, axis: int | str, delta: int, *, origin: str = "ui") -> bool:
        assert self._state.dims_ready, "dims intents require initial notify"
        idx = control_actions._axis_to_index(self._state, axis)
        if idx is None:
            return False
        viewer_obj = self._viewer_ref() if self._viewer_ref is not None else None
        if control_actions._is_axis_playing(viewer_obj, idx) and origin != "play":
            return False
        now = time.perf_counter()
        target_label = control_actions._axis_target_label(self._state, idx)
        if logger.isEnabledFor(logging.DEBUG):
            confirmed = self._ledger.confirmed_value('dims', target_label, 'index')
            logger.debug(
                "dims_step request: axis=%s target=%s delta=%s origin=%s confirmed_index=%s pending_step=%s",
                idx,
                target_label,
                delta,
                origin,
                confirmed,
                self._ledger.has_pending('dims', target_label, 'step'),
            )
        ok, _ = _emit_state_update(
            self._state,
            self._loop_state,
            self._ledger,
            self._dispatch_state_update,
            scope="dims",
            target=target_label,
            key="step",
            value=int(delta),
            origin=origin,
            metadata={
                "axis_index": idx,
                "axis_target": target_label,
                "update_kind": "step",
            },
        )
        if not ok:
            return False
        self._state.last_dims_send = now
        return True

    def dims_set_index(self, axis: int | str, value: int, *, origin: str = "ui") -> bool:
        assert self._state.dims_ready, "dims intents require initial notify"
        idx = control_actions._axis_to_index(self._state, axis)
        if idx is None:
            return False
        viewer_obj = self._viewer_ref() if self._viewer_ref is not None else None
        if control_actions._is_axis_playing(viewer_obj, idx) and origin != "play":
            return False
        now = time.perf_counter()
        target_label = control_actions._axis_target_label(self._state, idx)
        if logger.isEnabledFor(logging.DEBUG):
            confirmed = self._ledger.confirmed_value('dims', target_label, 'index')
            logger.debug(
                "dims_set_index request: axis=%s target=%s value=%s origin=%s confirmed_index=%s pending_index=%s",
                idx,
                target_label,
                value,
                origin,
                confirmed,
                self._ledger.has_pending('dims', target_label, 'index'),
            )
        ok, _ = _emit_state_update(
            self._state,
            self._loop_state,
            self._ledger,
            self._dispatch_state_update,
            scope="dims",
            target=target_label,
            key="index",
            value=int(value),
            origin=origin,
            metadata={
                "axis_index": idx,
                "axis_target": target_label,
                "update_kind": "index",
            },
        )
        if not ok:
            return False
        self._state.last_dims_send = now
        return True

    # ---------------------------------------------------------------- margins sync
    def _on_margins_change(self, _event=None) -> None:
        dims = self._viewer_dims
        assert dims is not None, "viewer dims missing"
        left = tuple(float(v) for v in getattr(dims, 'margin_left'))
        right = tuple(float(v) for v in getattr(dims, 'margin_right'))
        # Initialize cache
        if self._last_margin_left is None:
            self._last_margin_left = left
        if self._last_margin_right is None:
            self._last_margin_right = right

        nd = len(left)
        # Emit per-axis updates for changed entries
        for idx in range(nd):
            if idx < len(self._last_margin_left) and left[idx] != self._last_margin_left[idx]:
                self._emit_margin_update(idx, 'margin_left', left[idx])
            if idx < len(self._last_margin_right) and right[idx] != self._last_margin_right[idx]:
                self._emit_margin_update(idx, 'margin_right', right[idx])
        self._last_margin_left = left
        self._last_margin_right = right

    def _emit_margin_update(self, axis_idx: int, key: str, value: float) -> None:
        assert self._state.dims_ready, "dims intents require initial notify"
        target_label = control_actions._axis_target_label(self._state, axis_idx)
        ok, _ = _emit_state_update(
            self._state,
            self._loop_state,
            self._ledger,
            self._dispatch_state_update,
            scope="dims",
            target=target_label,
            key=key,
            value=float(value),
            origin="ui",
            metadata={
                "axis_index": axis_idx,
                "axis_target": target_label,
                "update_kind": "margin",
            },
        )
        if ok and self._log_dims_info:
            logger.info("dims margin update: axis=%s key=%s value=%s", axis_idx, key, value)

    def handle_wheel(self, data: Mapping[str, Any]) -> bool:
        ay = int(data.get('angle_y') or 0)
        py = int(data.get('pixel_y') or 0)
        step = 0
        if ay != 0:
            step = (1 if ay > 0 else -1) * int(self._state.wheel_step or 1)
        elif py != 0:
            self._state.wheel_px_accum += float(py)
            thr = 30.0
            while self._state.wheel_px_accum >= thr:
                step += int(self._state.wheel_step or 1)
                self._state.wheel_px_accum -= thr
            while self._state.wheel_px_accum <= -thr:
                step -= int(self._state.wheel_step or 1)
                self._state.wheel_px_accum += thr
        if step == 0:
            return False
        sent = self.dims_step('primary', int(step), origin='wheel')
        if self._log_dims_info:
            logger.info("wheel->state.update dims.step d=%+d sent=%s", int(step), bool(sent))
        else:
            logger.debug("wheel->state.update dims.step d=%+d sent=%s", int(step), bool(sent))
        return sent

    def view_set_ndisplay(self, ndisplay: int, *, origin: str = "ui") -> bool:
        assert self._state.dims_ready, "ndisplay intents require dims ready"
        if _rate_gate_settings(self._state, origin):
            return False
        nd_value = int(ndisplay)
        nd_target = 3 if nd_value >= 3 else 2
        cur = control_actions.current_ndisplay(self._state, self._ledger)
        if cur is not None and int(cur) == nd_target:
            return True
        ok, _ = _emit_state_update(
            self._state,
            self._loop_state,
            self._ledger,
            self._dispatch_state_update,
            scope="view",
            target="main",
            key="ndisplay",
            value=int(nd_target),
            origin=origin,
        )
        return ok

    def toggle_ndisplay(self, *, origin: str = "ui") -> bool:
        assert self._state.dims_ready, "ndisplay intents require dims ready"
        current = current_ndisplay(self._state, self._ledger)
        target = 2 if current == 3 else 3
        return self.view_set_ndisplay(target, origin=origin)

    # ------------------------------------------------------------------ event hooks
    def _on_dims_change(self, event: Any | None = None) -> None:
        if self._suppress_count > 0:
            return
        viewer = self._viewer_ref() if self._viewer_ref is not None else None
        if getattr(viewer, '_suppress_forward', False):
            return
        dims = self._viewer_dims
        if dims is None:
            return
        current = self._coerce_step(getattr(dims, "current_step", ()))
        if self._last_step_ui is not None and current == self._last_step_ui:
            return

        changed_axis = self._detect_changed_axis(current)
        self._update_play_state(changed_axis)

        if changed_axis is None:
            if self._last_step_ui is None:
                changed_axis = self._primary_axis_index()
            else:
                self._last_step_ui = current
                return

        value = int(current[changed_axis])
        prev_step = self._last_step_ui
        self._last_step_ui = current

        if self._is_playing and self._play_axis is not None and int(changed_axis) == int(self._play_axis):
            prev_value = 0 if prev_step is None or changed_axis >= len(prev_step) else int(prev_step[changed_axis])
            delta = value - prev_value
            if delta != 0:
                if self._log_dims_info:
                    logger.info(
                        "dims.play -> state.update step axis=%d delta=%+d",
                        changed_axis,
                        delta,
                    )
                if not self.dims_step(changed_axis, delta, origin="play"):
                    logger.debug("dims play tick rejected axis=%d", changed_axis)
            return

        if self._tx_interval_ms <= 0:
            if self._log_dims_info:
                logger.info(
                    "slider -> state.update index axis=%d value=%d",
                    changed_axis,
                    value,
                )
            if not self.dims_set_index(changed_axis, value, origin="ui"):
                logger.debug("dims index rejected axis=%d value=%d", changed_axis, value)
            return

        self._queue_coalesced_update(changed_axis, value)

    def _on_ndisplay_change(self, event: Any | None = None) -> None:
        if self._suppress_count > 0:
            return
        viewer = self._viewer_ref() if self._viewer_ref is not None else None
        if getattr(viewer, '_suppress_forward', False):
            return
        if not self._viewer_dims:
            return
        raw_value = getattr(event, "value", None)
        if raw_value is None:
            raw_value = getattr(self._viewer_dims, "ndisplay", None)
        if raw_value is None:
            return
        value = int(raw_value)
        if self._log_dims_info:
            logger.info("viewer ndisplay -> state.update %s", value)
        if not self.view_set_ndisplay(value, origin="ui"):
            logger.debug("ndisplay intent rejected value=%d", value)

    # --------------------------------------------------------------------- helpers
    def _queue_coalesced_update(self, axis: int, value: int) -> None:
        self._pending = _PendingCoalesce(axis=axis, value=value)
        timer = self._dims_tx_timer
        if timer is None:
            timer = QtCore.QTimer()
            timer.setSingleShot(True)
            timer.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]
            timer.timeout.connect(self._flush_pending)
            self._dims_tx_timer = timer
        timer.start(max(1, self._tx_interval_ms))

    def _flush_pending(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        if self._log_dims_info:
            logger.info(
                "slider (coalesced) -> state.update index axis=%d value=%d",
                pending.axis,
                pending.value,
            )
        if not self.dims_set_index(pending.axis, pending.value, origin="ui"):
            logger.debug(
                "coalesced dims index rejected axis=%d value=%d",
                pending.axis,
                pending.value,
            )

    def _sync_last_step_from_viewer(self) -> None:
        dims = self._viewer_dims
        if dims is None:
            self._last_step_ui = None
            return
        current_step = getattr(dims, "current_step", None)
        if current_step is None:
            self._last_step_ui = None
            return
        try:
            self._last_step_ui = self._coerce_step(current_step)
        except Exception:
            self._last_step_ui = None

    def _coerce_step(self, raw: Sequence[object]) -> tuple[int, ...]:
        return tuple(int(value) for value in raw)

    def _detect_changed_axis(self, current: tuple[int, ...]) -> int | None:
        prev = self._last_step_ui
        if prev is None or len(prev) != len(current):
            return None
        for idx, (before, after) in enumerate(zip(prev, current, strict=False)):
            if before != after:
                return idx
        return None

    def _primary_axis_index(self) -> int:
        primary = self._state.primary_axis_index
        return int(primary) if primary is not None else 0

    def _update_play_state(self, changed_axis: int | None) -> None:
        viewer = self._viewer_ref() if self._viewer_ref is not None else None
        if viewer is None:
            self._is_playing = False
            self._play_axis = None
            return
        window = getattr(viewer, "window", None)
        qdims = getattr(getattr(window, "_qt_viewer", None), "dims", None)
        if qdims is not None and hasattr(qdims, "is_playing"):
            try:
                self._is_playing = bool(qdims.is_playing)
            except Exception:
                logger.debug("dims play state probe failed", exc_info=True)
                self._is_playing = False
        else:
            self._is_playing = False
        if not self._is_playing:
            self._play_axis = None
        elif self._play_axis is None and changed_axis is not None:
            self._play_axis = int(changed_axis)

__all__ = ["NapariDimsIntentEmitter"]
