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
    - Sends messages via the provided `post` callable.
    """

    def __init__(
        self,
        widget: QtWidgets.QWidget,  # type: ignore[valid-type]
        post: Callable[[dict], bool],
        *,
        max_rate_hz: float = 120.0,
        resize_debounce_ms: int = 80,
        on_wheel: Optional[Callable[[dict], None]] = None,
        on_pointer: Optional[Callable[[dict], None]] = None,
        on_key: Optional[Callable[[dict], None]] = None,
        log_info: bool = False,
    ) -> None:
        super().__init__(widget)
        self._widget = widget
        self._post = post
        self._max_rate_hz = float(max(1.0, max_rate_hz))
        self._min_dt = 1.0 / self._max_rate_hz
        self._last_wheel_send: float = 0.0
        self._pending_resize: Optional[tuple[int, int, float]] = None
        self._resize_timer = QtCore.QTimer(self)  # type: ignore[no-untyped-call]
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._flush_resize)
        self._on_wheel = on_wheel
        self._on_pointer = on_pointer
        self._on_key = on_key
        self._log_info = bool(log_info)
        # Cache top-level window for gating app-level events
        try:
            self._top_window = widget.window()  # type: ignore[attr-defined]
        except Exception:
            self._top_window = None

    def eventFilter(self, obj, event):  # type: ignore[no-untyped-def]
        try:
            et = event.type()
            # Only handle wheel/resize/mouse for the canvas widget itself.
            try:
                target_is_canvas = (obj is self._widget)
            except Exception:
                target_is_canvas = False
            if et == QtCore.QEvent.Wheel and target_is_canvas:  # type: ignore[attr-defined]
                return self._handle_wheel(event)
            elif et == QtCore.QEvent.KeyPress:  # type: ignore[attr-defined]
                # Gate to our top-level window to avoid cross-window noise
                try:
                    if obj is not None and hasattr(obj, 'window') and self._top_window is not None:
                        if obj.window() is not self._top_window:  # type: ignore[attr-defined]
                            return False
                except Exception:
                    pass
                try:
                    key = int(event.key())
                except Exception:
                    key = -1
                try:
                    mods = int(event.modifiers())
                except Exception:
                    mods = 0
                try:
                    txt = event.text()
                except Exception:
                    txt = ""
                if self._log_info:
                    logger.info("key press: key=%s mods=%s text=%r", key, mods, txt)
                # Notify optional key callback and allow it to consume
                try:
                    if self._on_key is not None:
                        consumed = bool(self._on_key({'type': 'input.key', 'key': int(key), 'mods': int(mods), 'text': str(txt)}))
                        if consumed:
                            try:
                                ev.accept()
                            except Exception:
                                pass
                            return True
                except Exception:
                    logger.debug("on_key callback failed", exc_info=True)
                return False
            elif et == QtCore.QEvent.Resize and target_is_canvas:  # type: ignore[attr-defined]
                return self._handle_resize(event)
            elif et == QtCore.QEvent.MouseButtonPress and target_is_canvas:  # type: ignore[attr-defined]
                return self._handle_mouse_down(event)
            elif et == QtCore.QEvent.MouseMove and target_is_canvas:  # type: ignore[attr-defined]
                return self._handle_mouse_move(event)
            elif et == QtCore.QEvent.MouseButtonRelease and target_is_canvas:  # type: ignore[attr-defined]
                return self._handle_mouse_up(event)
        except Exception:
            logger.debug("InputSender eventFilter error", exc_info=True)
        return False

    # --- Wheel ---------------------------------------------------------------------
    def _handle_wheel(self, ev) -> bool:  # type: ignore[no-untyped-def]
        now = time.perf_counter()
        if (now - float(self._last_wheel_send or 0.0)) < self._min_dt:
            # Coalesce: consume event to avoid parent handling/focus highlight,
            # but skip sending to server when over rate limit
            try:
                ev.accept()
            except Exception:
                pass
            try:
                # Ensure canvas doesn't get/keep focus
                self._widget.clearFocus()
            except Exception:
                pass
            return True
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
        ok = self._post(msg)
        if self._log_info:
            logger.info(
                "input.wheel sent: ay=%d py=%d mods=%d pos=(%.1f,%.1f)",
                int(ay), int(py), int(mods), float(x), float(y),
            )
        # Notify optional wheel callback (e.g., for dims intent mapping)
        try:
            if self._on_wheel is not None:
                self._on_wheel(msg)
        except Exception:
            logger.debug("on_wheel callback failed", exc_info=True)
        if ok:
            self._last_wheel_send = now
        # Consume to avoid selection highlighting in parent widgets
        try:
            ev.accept()
        except Exception:
            pass
        return True

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
        _ = self._post(msg)
        if self._log_info:
            logger.info("view.resize sent: %dx%d dpr=%.2f", int(w), int(h), float(dpr))

    # (no key handling)

    # --- Mouse handling (forward to on_pointer) ----------------------------------
    def _handle_mouse_down(self, ev) -> bool:  # type: ignore[no-untyped-def]
        try:
            if self._on_pointer is not None:
                try:
                    pos = ev.position() if hasattr(ev, 'position') else ev.pos()
                    x = float(pos.x()); y = float(pos.y())
                except Exception:
                    x = y = 0.0
                try:
                    btn = int(ev.button())
                    btns = int(ev.buttons())
                    mods = int(ev.modifiers())
                except Exception:
                    btn = 0; btns = 0; mods = 0
                self._on_pointer({
                    'type': 'input.pointer',
                    'phase': 'down',
                    'x_px': float(x),
                    'y_px': float(y),
                    'button': int(btn),
                    'buttons': int(btns),
                    'mods': int(mods),
                    'width_px': int(self._widget.width()),
                    'height_px': int(self._widget.height()),
                    'ts': float(time.time()),
                })
        except Exception:
            logger.debug("on_pointer (down) failed", exc_info=True)
        try:
            ev.accept()
        except Exception:
            pass
        return True

    def _handle_mouse_move(self, ev) -> bool:  # type: ignore[no-untyped-def]
        try:
            if self._on_pointer is not None:
                try:
                    pos = ev.position() if hasattr(ev, 'position') else ev.pos()
                    x = float(pos.x()); y = float(pos.y())
                except Exception:
                    x = y = 0.0
                try:
                    btns = int(ev.buttons())
                    mods = int(ev.modifiers())
                except Exception:
                    btns = 0; mods = 0
                self._on_pointer({
                    'type': 'input.pointer',
                    'phase': 'move',
                    'x_px': float(x),
                    'y_px': float(y),
                    'buttons': int(btns),
                    'mods': int(mods),
                    'width_px': int(self._widget.width()),
                    'height_px': int(self._widget.height()),
                    'ts': float(time.time()),
                })
        except Exception:
            logger.debug("on_pointer (move) failed", exc_info=True)
        try:
            ev.accept()
        except Exception:
            pass
        return True

    def _handle_mouse_up(self, ev) -> bool:  # type: ignore[no-untyped-def]
        try:
            if self._on_pointer is not None:
                try:
                    pos = ev.position() if hasattr(ev, 'position') else ev.pos()
                    x = float(pos.x()); y = float(pos.y())
                except Exception:
                    x = y = 0.0
                try:
                    btn = int(ev.button())
                    btns = int(ev.buttons())
                    mods = int(ev.modifiers())
                except Exception:
                    btn = 0; btns = 0; mods = 0
                self._on_pointer({
                    'type': 'input.pointer',
                    'phase': 'up',
                    'x_px': float(x),
                    'y_px': float(y),
                    'button': int(btn),
                    'buttons': int(btns),
                    'mods': int(mods),
                    'width_px': int(self._widget.width()),
                    'height_px': int(self._widget.height()),
                    'ts': float(time.time()),
                })
        except Exception:
            logger.debug("on_pointer (up) failed", exc_info=True)
        try:
            ev.accept()
        except Exception:
            pass
        return True


class InputSender:
    """Attach input forwarding to a canvas widget and a StateChannel.

    Handles: wheel, resize, mouse, and optional key observation via callback.
    """

    def __init__(
        self,
        widget: QtWidgets.QWidget,  # type: ignore[valid-type]
        post: Callable[[dict], bool],
        *,
        max_rate_hz: float = 120.0,
        resize_debounce_ms: int = 80,
        on_wheel: Optional[Callable[[dict], None]] = None,
        on_pointer: Optional[Callable[[dict], None]] = None,
        on_key: Optional[Callable[[dict], None]] = None,
        log_info: bool = False,
    ) -> None:
        self._widget = widget
        self._filter = _EventFilter(
            widget,
            post,
            max_rate_hz=max_rate_hz,
            resize_debounce_ms=resize_debounce_ms,
            on_wheel=on_wheel,
            on_pointer=on_pointer,
            on_key=on_key,
            log_info=log_info,
        )
        # Pointer callback wiring
        self._on_pointer = on_pointer
        self._on_key = on_key
        # Set debounce interval explicitly on the timer
        try:
            self._filter._resize_timer.setInterval(int(max(0, resize_debounce_ms)))  # type: ignore[attr-defined]
        except Exception:
            pass

    def start(self) -> None:
        # Install event filter
        try:
            self._widget.installEventFilter(self._filter)
        except Exception:
            logger.debug("Failed to install input event filter", exc_info=True)
        # Install at the application level when input logging or on_key observation is enabled
        try:
            app = QtWidgets.QApplication.instance()
            if app is not None and (bool(getattr(self._filter, '_log_info', False)) or self._on_key is not None):
                app.installEventFilter(self._filter)
        except Exception:
            logger.debug("Failed to install app-level event filter", exc_info=True)
        # Send initial resize snapshot so server knows canvas size at start
        try:
            w = int(self._widget.width())
            h = int(self._widget.height())
            try:
                dpr = float(self._widget.devicePixelRatioF())
            except Exception:
                dpr = 1.0
            msg = {'type': 'view.resize', 'width_px': w, 'height_px': h, 'dpr': dpr, 'ts': float(time.time())}
            _ = self._filter._post(msg)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Initial resize send failed", exc_info=True)

    # --- Mouse/Pan ---------------------------------------------------------------
    
    # No additional focus suppression helpers needed; coalesced wheel consumption
    # in _handle_wheel is sufficient to prevent highlight during fast scroll.
