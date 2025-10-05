"""Helpers for mirroring dims state between the server and the local viewer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple

from qtpy import QtCore

try:  # pragma: no cover - optional type support
    from napari.components.viewer_model import ViewerModel
except Exception:  # pragma: no cover
    ViewerModel = Any  # type: ignore


@dataclass(slots=True)
class _PendingCoalesce:
    axis: int
    value: int


class DimsBridge:
    """Project server-controlled dims into a local napari viewer and emit intents."""

    def __init__(
        self,
        viewer: ViewerModel,
        *,
        logger: logging.Logger,
        tx_interval_ms: int = 10,
        log_dims_info: bool = False,
    ) -> None:
        self._viewer = viewer
        self._logger = logger
        self._state_sender: Any | None = None
        self._suppress_forward = False
        self._last_step_ui: Tuple[int, ...] | None = None
        self._dims_tx_interval_ms = max(0, int(tx_interval_ms))
        self._dims_tx_timer: QtCore.QTimer | None = None
        self._dims_tx_pending: _PendingCoalesce | None = None
        self._is_playing = False
        self._play_axis: int | None = None
        self._log_dims_info = bool(log_dims_info)
        self._logger.debug(
            "DimsBridge: initialised interval=%d log=%s",
            self._dims_tx_interval_ms,
            self._log_dims_info,
        )

        events = getattr(getattr(viewer, "dims", None), "events", None)
        if events is None:
            raise RuntimeError("Viewer missing dims events; cannot attach DimsBridge")
        events.current_step.connect(self._on_dims_change)
        events.ndisplay.connect(self._on_ndisplay_change)

        self._sync_last_step_from_viewer()

    # ------------------------------------------------------------------ configuration
    @property
    def suppress_forward(self) -> bool:
        return self._suppress_forward

    @suppress_forward.setter
    def suppress_forward(self, value: bool) -> None:
        self._suppress_forward = bool(value)

    def set_logging(self, enabled: bool) -> None:
        self._log_dims_info = bool(enabled)

    def set_tx_interval_ms(self, value: int) -> None:
        self._dims_tx_interval_ms = max(0, int(value))
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("DimsBridge: tx interval set to %d ms", self._dims_tx_interval_ms)

    def attach_state_sender(self, sender: Any) -> None:
        self._state_sender = sender
        # Once attached to the coordinator we rely on its scheduling; disable local coalescing.
        self._dims_tx_interval_ms = 0
        if self._last_step_ui is None:
            self._sync_last_step_from_viewer()

    def detach_state_sender(self) -> None:
        self._state_sender = None

    # ------------------------------------------------------------------ UI event handling
    def handle_dims_change(self, event: Any | None = None) -> None:
        self._on_dims_change(event)

    def handle_ndisplay_change(self, event: Any | None = None) -> None:
        self._on_ndisplay_change(event)


    def _on_dims_change(self, event: Any | None = None) -> None:
        dims = self._viewer.dims
        current = self._coerce_step(dims.current_step)

        if self._suppress_forward:
            self._last_step_ui = current
            if self._dims_tx_timer is not None and self._dims_tx_timer.isActive():
                self._dims_tx_timer.stop()
            self._dims_tx_pending = None
            return

        sender = self._state_sender
        if sender is None:
            self._last_step_ui = current
            return

        if self._last_step_ui is not None and current == self._last_step_ui:
            return

        changed_axis = self._detect_changed_axis(current)
        self._update_play_state(changed_axis)

        if changed_axis is None:
            if self._last_step_ui is None:
                changed_axis = self._primary_axis_index(sender)
            else:
                self._last_step_ui = current
                return

        value = int(current[changed_axis])

        if self._is_playing and self._play_axis is not None and int(changed_axis) == int(self._play_axis):
            self._handle_play_tick(sender, changed_axis, value)
            self._last_step_ui = current
            return

        if self._dims_tx_interval_ms <= 0:
            self._dispatch_index(sender, changed_axis, value)
        else:
            self._queue_coalesced_update(changed_axis, value)
        self._last_step_ui = current

    def _on_ndisplay_change(self, event: Any | None = None) -> None:
        if self._suppress_forward:
            return
        sender = self._state_sender
        if sender is None:
            return
        raw_value = getattr(event, "value", None)
        if raw_value is None:
            raw_value = self._viewer.dims.ndisplay
        ndisplay = int(raw_value)
        fn = getattr(sender, "view_set_ndisplay", None)
        if callable(fn) and not fn(ndisplay, origin="ui"):
            self._logger.debug("DimsBridge: ndisplay intent rejected by coordinator")
        if self._log_dims_info:
            self._logger.info("ProxyViewer ndisplay change -> %s", ndisplay)

    # ------------------------------------------------------------------ remote apply
    def apply_remote(
        self,
        *,
        current_step: Sequence[int] | None = None,
        ndisplay: int | None = None,
        ndim: int | None = None,
        dims_range: Sequence[Sequence[float]] | None = None,
        order: Sequence[int | str] | None = None,
        axis_labels: Sequence[str] | None = None,
        sizes: Sequence[int] | None = None,
        displayed: Sequence[int] | None = None,
    ) -> None:
        prev = self._suppress_forward
        self._suppress_forward = True
        dims = self._viewer.dims
        try:
            if ndim is not None:
                dims.ndim = int(ndim)
            if dims_range is not None:
                coerced = tuple(self._coerce_range(entry) for entry in dims_range)
                dims.range = coerced
            elif sizes is not None:
                dims.range = tuple((0.0, float(int(size) - 1), 1.0) for size in sizes)
            if order is not None:
                dims.order = self._coerce_order(order, axis_labels)
            if axis_labels is not None:
                dims.axis_labels = tuple(str(lbl) for lbl in axis_labels)
            if ndisplay is not None:
                dims.ndisplay = int(ndisplay)
            if displayed is not None:
                self._apply_displayed_axes(displayed)
            if current_step is not None:
                step_tuple = self._coerce_step(current_step)
                dims.current_step = step_tuple
                dims.point = tuple(float(x) for x in current_step)
                if self._dims_tx_timer is not None and self._dims_tx_timer.isActive():
                    self._dims_tx_timer.stop()
                self._dims_tx_pending = None
                self._last_step_ui = step_tuple
            if self._log_dims_info:
                self._logger.info(
                    "DimsBridge applied: ndim=%s ndisplay=%s order=%s step=%s range=%s",
                    dims.ndim,
                    dims.ndisplay,
                    dims.order,
                    getattr(dims, "current_step", None),
                    getattr(dims, "range", None),
                )
        except Exception:
            self._logger.debug("DimsBridge: apply_remote failed", exc_info=True)
        finally:
            if prev:
                self._suppress_forward = prev
            else:
                self._clear_suppression()

    def _clear_suppression(self) -> None:
        self._suppress_forward = False

    # ------------------------------------------------------------------ helpers
    def _sync_last_step_from_viewer(self) -> None:
        dims = getattr(self._viewer, "dims")
        step = self._coerce_step(dims.current_step)
        self._last_step_ui = step if step else None

    def _coerce_step(self, raw: Sequence[object]) -> Tuple[int, ...]:
        return tuple(int(value) for value in raw)

    def _detect_changed_axis(self, current: Tuple[int, ...]) -> int | None:
        prev = self._last_step_ui
        if prev is None or len(prev) != len(current):
            return None
        for idx, (before, after) in enumerate(zip(prev, current)):
            if before != after:
                return idx
        return None

    @staticmethod
    def _primary_axis_index(sender: Any) -> int:
        primary = getattr(sender, "_primary_axis_index", 0)
        return int(primary)

    def _update_play_state(self, changed_axis: int | None) -> None:
        window = getattr(self._viewer, "window", None)
        qdims = getattr(getattr(window, "_qt_viewer", None), "dims", None)
        if qdims is not None and hasattr(qdims, "is_playing"):
            try:
                self._is_playing = bool(qdims.is_playing)
            except Exception:
                self._logger.debug("DimsBridge: play state probe failed", exc_info=True)
        if not self._is_playing:
            self._play_axis = None
        elif self._play_axis is None and changed_axis is not None:
            self._play_axis = int(changed_axis)

    def _handle_play_tick(self, sender: Any, axis: int, value: int) -> None:
        prev = self._last_step_ui
        if prev is None or axis >= len(prev):
            return
        previous_value = prev[axis]
        delta = value - previous_value
        if delta == 0:
            return
        if self._log_dims_info:
            self._logger.info(
                "play tick -> state.update dims.step axis=%d delta=%+d",
                axis,
                delta,
            )
        if not sender.dims_step(axis, delta, origin="play"):
            self._logger.debug("DimsBridge: dims_step rejected (axis=%d delta=%d)", axis, delta)

    def _dispatch_index(self, sender: Any, axis: int, value: int) -> None:
        if self._log_dims_info:
            self._logger.info(
                "slider -> state.update dims.index axis=%d value=%d",
                axis,
                value,
            )
        if not sender.dims_set_index(axis, value, origin="ui"):
            self._logger.debug("DimsBridge: dims_set_index rejected (axis=%d value=%d)", axis, value)

    def _queue_coalesced_update(self, axis: int, value: int) -> None:
        self._dims_tx_pending = _PendingCoalesce(axis=axis, value=value)
        timer = self._dims_tx_timer
        if timer is None:
            parent = getattr(getattr(self._viewer, "window", None), "_qt_viewer", None)
            timer = QtCore.QTimer(parent)
            timer.setSingleShot(True)
            timer.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]
            timer.timeout.connect(self._flush_pending)
            self._dims_tx_timer = timer
        timer.start(max(1, self._dims_tx_interval_ms))

    def _flush_pending(self) -> None:
        pending = self._dims_tx_pending
        self._dims_tx_pending = None
        sender = self._state_sender
        if pending is None or sender is None:
            return
        if self._log_dims_info:
            self._logger.info(
                "slider (coalesced) -> state.update dims.index axis=%d value=%d",
                pending.axis,
                pending.value,
            )
        if not sender.dims_set_index(pending.axis, pending.value, origin="ui"):
            self._logger.debug(
                "DimsBridge: coalesced dims_set_index rejected (axis=%d value=%d)",
                pending.axis,
                pending.value,
            )

    def _coerce_order(self, order: Sequence[int | str], axis_labels: Sequence[str] | None) -> Tuple[int, ...]:
        if not order:
            return tuple()
        try:
            return tuple(int(x) for x in order)
        except Exception:
            if axis_labels is None:
                raise
            label_to_idx = {str(lbl): idx for idx, lbl in enumerate(axis_labels)}
            return tuple(int(label_to_idx[str(lbl)]) for lbl in order)

    def _apply_displayed_axes(self, displayed: Sequence[int]) -> None:
        dims = self._viewer.dims
        values = [int(x) for x in displayed]
        values = [x for x in values if 0 <= x < dims.ndim]
        if not values:
            return
        current_order = list(dims.order) or list(range(dims.ndim))
        base = [axis for axis in current_order if axis not in values]
        base.extend(axis for axis in values if axis not in base)
        for axis in range(dims.ndim):
            if axis not in base:
                base.insert(0, axis)
        if len(base) == len(current_order):
            dims.order = tuple(base)

    @staticmethod
    def _coerce_range(entry: Sequence[float]) -> Tuple[float, float, float]:
        if len(entry) == 2:
            start, stop = entry
            step = 1.0
        elif len(entry) >= 3:
            start, stop, step = entry[:3]
        else:
            raise ValueError("range entry must contain at least 2 values")
        return (float(start), float(stop), float(step))

    # ------------------------------------------------------------------ teardown
    def shutdown(self) -> None:
        if self._dims_tx_timer is not None:
            self._dims_tx_timer.stop()
            self._dims_tx_timer.deleteLater()
            self._dims_tx_timer = None
        self._dims_tx_pending = None
        self._state_sender = None
        self._is_playing = False
        self._play_axis = None

__all__ = ["DimsBridge"]
