from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from qtpy import QtCore, QtWidgets  # type: ignore

logger = logging.getLogger(__name__)


class _EventFilter(QtCore.QObject):  # type: ignore[misc]
    """Qt event filter to capture wheel and resize events and forward them.

    - Coalesces high-rate wheel events to a target max rate.
    - Debounces resize events to avoid thrash while the user drags.
    - Sends messages via the provided send_json callable.
    """

    def __init__(
        self,
        widget: QtWidgets.QWidget,  # type: ignore[valid-type]
        send_json: Callable[[dict], bool],
        *,
        max_rate_hz: float = 120.0,
        resize_debounce_ms: int = 80,
        on_wheel: Optional[Callable[[dict], None]] = None,
        log_info: bool = False,
    ) -> None:
        super().__init__(widget)
        self._widget = widget
        self._send_json = send_json
        self._max_rate_hz = float(max(1.0, max_rate_hz))
        self._min_dt = 1.0 / self._max_rate_hz
        self._last_wheel_send: float = 0.0
        self._pending_resize: Optional[tuple[int, int, float]] = None
        self._resize_timer = QtCore.QTimer(self)  # type: ignore[no-untyped-call]
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._flush_resize)
        self._on_wheel = on_wheel
        self._log_info = bool(log_info)

    def eventFilter(self, obj, event):  # type: ignore[no-untyped-def]
        try:
            et = event.type()
            if et == QtCore.QEvent.Wheel:  # type: ignore[attr-defined]
                return self._handle_wheel(event)
            elif et == QtCore.QEvent.Resize:  # type: ignore[attr-defined]
                return self._handle_resize(event)
            elif et == QtCore.QEvent.KeyPress or et == QtCore.QEvent.KeyRelease:  # type: ignore[attr-defined]
                return self._handle_key(event, down=(et == QtCore.QEvent.KeyPress))
        except Exception:
            logger.debug("InputSender eventFilter error", exc_info=True)
        return False

    # --- Wheel ---------------------------------------------------------------------
    def _handle_wheel(self, ev) -> bool:  # type: ignore[no-untyped-def]
        now = time.perf_counter()
        if (now - float(self._last_wheel_send or 0.0)) < self._min_dt:
            # Drop excessive wheel rate
            return False
        try:
            # angleDelta is in 1/8 deg units, typical step is 120 per notch
            a = ev.angleDelta()
            ax = int(getattr(a, 'x')()) if hasattr(a, 'x') else int(a.x())
            ay = int(getattr(a, 'y')()) if hasattr(a, 'y') else int(a.y())
        except Exception:
            ax = 0
            ay = 0
        try:
            p = ev.pixelDelta()
            px = int(getattr(p, 'x')()) if hasattr(p, 'x') else int(p.x())
            py = int(getattr(p, 'y')()) if hasattr(p, 'y') else int(p.y())
        except Exception:
            px = 0
            py = 0
        # Position in widget-local pixels
        try:
            # Qt6: position() returns QPointF; Qt5: pos() returns QPoint
            if hasattr(ev, 'position'):
                pos = ev.position()
                x = float(pos.x())
                y = float(pos.y())
            else:
                pos = ev.pos()
                x = float(pos.x())
                y = float(pos.y())
        except Exception:
            x = y = 0.0
        try:
            mods = int(ev.modifiers())
        except Exception:
            mods = 0
        try:
            dpr = float(self._widget.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        msg = {
            'type': 'input.wheel',
            'angle_x': int(ax),
            'angle_y': int(ay),
            'pixel_x': int(px),
            'pixel_y': int(py),
            'x_px': float(x),
            'y_px': float(y),
            'mods': int(mods),
            'width_px': int(self._widget.width()),
            'height_px': int(self._widget.height()),
            'dpr': float(dpr),
            'ts': float(time.time()),
        }
        ok = self._send_json(msg)
        if self._log_info:
            logger.info(
                "input.wheel sent: ay=%d py=%d mods=%d pos=(%.1f,%.1f)",
                int(ay), int(py), int(mods), float(x), float(y),
            )
        # Notify optional wheel callback (e.g., for dims.set mapping)
        try:
            if self._on_wheel is not None:
                self._on_wheel(msg)
        except Exception:
            logger.debug("on_wheel callback failed", exc_info=True)
        if ok:
            self._last_wheel_send = now
        return False

    # --- Resize --------------------------------------------------------------------
    def _handle_resize(self, ev) -> bool:  # type: ignore[no-untyped-def]
        try:
            w = int(self._widget.width())
            h = int(self._widget.height())
            try:
                dpr = float(self._widget.devicePixelRatioF())
            except Exception:
                dpr = 1.0
            self._pending_resize = (w, h, dpr)
            # Debounce (restart timer)
            try:
                # Default 80 ms unless overridden by property on timer (set by wrapper)
                interval = int(self._resize_timer.interval()) or 80
            except Exception:
                interval = 80
            self._resize_timer.start(interval)
        except Exception:
            logger.debug("Resize handling failed", exc_info=True)
        return False

    def _flush_resize(self) -> None:
        pr = self._pending_resize
        if not pr:
            return
        w, h, dpr = pr
        self._pending_resize = None
        msg = {
            'type': 'view.resize',
            'width_px': int(w),
            'height_px': int(h),
            'dpr': float(dpr),
            'ts': float(time.time()),
        }
        _ = self._send_json(msg)
        if self._log_info:
            logger.info("view.resize sent: %dx%d dpr=%.2f", int(w), int(h), float(dpr))

    # --- Key events ----------------------------------------------------------------
    def _handle_key(self, ev, *, down: bool) -> bool:  # type: ignore[no-untyped-def]
        try:
            key = int(ev.key())
        except Exception:
            key = 0
        try:
            txt = str(ev.text()) if hasattr(ev, 'text') else ''
        except Exception:
            txt = ''
        try:
            mods = int(ev.modifiers())
        except Exception:
            mods = 0
        try:
            auto = bool(getattr(ev, 'isAutoRepeat')() if hasattr(ev, 'isAutoRepeat') else False)
        except Exception:
            auto = False
        msg = {
            'type': 'input.key',
            'phase': 'down' if down else 'up',
            'key': int(key),
            'text': txt,
            'mods': int(mods),
            'auto': bool(auto),
            'ts': float(time.time()),
        }
        ok = self._send_json(msg)
        if self._log_info and down:
            logger.info("input.key sent: key=%d auto=%s mods=%d", int(key), bool(auto), int(mods))
        # Optional: coordinator mapping callback is invoked via constructor hook (on_key)
        return False


class InputSender:
    """Attach input forwarding to a canvas widget and a StateChannel.

    Minimal MVP: wheel + resize only (mouse/key to be added in later phases).
    """

    def __init__(
        self,
        widget: QtWidgets.QWidget,  # type: ignore[valid-type]
        send_json: Callable[[dict], bool],
        *,
        max_rate_hz: float = 120.0,
        resize_debounce_ms: int = 80,
        on_wheel: Optional[Callable[[dict], None]] = None,
        log_info: bool = False,
    ) -> None:
        self._widget = widget
        self._filter = _EventFilter(
            widget,
            send_json,
            max_rate_hz=max_rate_hz,
            resize_debounce_ms=resize_debounce_ms,
            on_wheel=on_wheel,
            log_info=log_info,
        )
        # Set debounce interval explicitly on the timer
        try:
            self._filter._resize_timer.setInterval(int(max(0, resize_debounce_ms)))  # type: ignore[attr-defined]
        except Exception:
            pass

    def start(self) -> None:
        try:
            self._widget.installEventFilter(self._filter)
        except Exception:
            logger.debug("Failed to install input event filter", exc_info=True)
        # Send initial resize snapshot so server knows canvas size at start
        try:
            w = int(self._widget.width())
            h = int(self._widget.height())
            try:
                dpr = float(self._widget.devicePixelRatioF())
            except Exception:
                dpr = 1.0
            msg = {'type': 'view.resize', 'width_px': w, 'height_px': h, 'dpr': dpr, 'ts': float(time.time())}
            _ = self._filter._send_json(msg)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Initial resize send failed", exc_info=True)
