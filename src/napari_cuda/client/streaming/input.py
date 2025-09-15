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
        on_pointer: Optional[Callable[[dict], None]] = None,
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
        self._on_pointer = on_pointer
        self._log_info = bool(log_info)

    def eventFilter(self, obj, event):  # type: ignore[no-untyped-def]
        try:
            et = event.type()
            # Prevent focus highlight on focus changes
            if et == QtCore.QEvent.FocusIn:  # type: ignore[attr-defined]
                # Only act for our widget or its descendants
                try:
                    is_target = (obj is self._widget) or (
                        isinstance(obj, QtWidgets.QWidget) and hasattr(self._widget, 'isAncestorOf') and self._widget.isAncestorOf(obj)  # type: ignore[attr-defined]
                    )
                except Exception:
                    is_target = False
                if not is_target:
                    return False
                try:
                    if hasattr(obj, 'clearFocus'):
                        obj.clearFocus()
                except Exception:
                    pass
                return True
            if et == QtCore.QEvent.Wheel:  # type: ignore[attr-defined]
                # Only act for our widget or its descendants
                try:
                    is_target = (obj is self._widget) or (
                        isinstance(obj, QtWidgets.QWidget) and hasattr(self._widget, 'isAncestorOf') and self._widget.isAncestorOf(obj)  # type: ignore[attr-defined]
                    )
                except Exception:
                    is_target = False
                if not is_target:
                    return False
                try:
                    if hasattr(obj, 'clearFocus'):
                        obj.clearFocus()
                    if hasattr(self._widget, 'clearFocus'):
                        self._widget.clearFocus()
                except Exception:
                    pass
                return self._handle_wheel(event)
            elif et == QtCore.QEvent.Resize:  # type: ignore[attr-defined]
                return self._handle_resize(event)
            elif et == QtCore.QEvent.MouseButtonPress:  # type: ignore[attr-defined]
                return self._handle_mouse_down(event)
            elif et == QtCore.QEvent.MouseMove:  # type: ignore[attr-defined]
                return self._handle_mouse_move(event)
            elif et == QtCore.QEvent.MouseButtonRelease:  # type: ignore[attr-defined]
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
        # Consume to avoid selection highlighting in parent widgets
        try:
            ev.accept()
        except Exception:
            pass
        # Ensure canvas does not appear focused after wheel
        try:
            self._widget.clearFocus()
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
        _ = self._send_json(msg)
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
        on_pointer: Optional[Callable[[dict], None]] = None,
        log_info: bool = False,
    ) -> None:
        self._widget = widget
        self._filter = _EventFilter(
            widget,
            send_json,
            max_rate_hz=max_rate_hz,
            resize_debounce_ms=resize_debounce_ms,
            on_wheel=on_wheel,
            on_pointer=on_pointer,
            log_info=log_info,
        )
        # Pointer callback wiring
        self._on_pointer = on_pointer
        # Set debounce interval explicitly on the timer
        try:
            self._filter._resize_timer.setInterval(int(max(0, resize_debounce_ms)))  # type: ignore[attr-defined]
        except Exception:
            pass

    def start(self) -> None:
        # Best-effort: remove focus highlight without deep nesting
        try:
            self._disable_focus_highlight()
            self._configure_focus_policies()
            self._apply_focus_rules_recursively()
            # Install app-level filter as a safety net to catch late focus/wheel
            try:
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    app.installEventFilter(self._filter)
            except Exception:
                pass
        except Exception:
            logger.debug("Failed to disable focus highlight", exc_info=True)
        # Install event filter last
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

    # --- Mouse/Pan ---------------------------------------------------------------
    
    def _disable_focus_highlight(self) -> None:
        # Disable macOS focus ring when possible
        if hasattr(QtCore.Qt, 'WA_MacShowFocusRect'):
            self._widget.setAttribute(QtCore.Qt.WA_MacShowFocusRect, False)  # type: ignore[attr-defined]
        # Suppress outline on focus via stylesheet
        cur = self._widget.styleSheet() or ""
        rules = [
            "QWidget:focus { outline: none; border: 0px; }",
            "QOpenGLWidget:focus { outline: none; border: 0px; }",
            "*:focus { outline: none; }",
        ]
        extra = "\n".join([r for r in rules if r not in cur])
        if extra:
            self._widget.setStyleSheet((cur + ("\n" if cur else "")) + extra)

    def _configure_focus_policies(self) -> None:
        """Ensure wheel events do not force focus and trigger highlights.

        - Prefer ClickFocus on the canvas widget so wheel does not grant focus.
        - Set NoFocus on child widgets (e.g., GL subwidgets) to avoid focus rings.
        - Apply macOS focus ring suppression recursively.
        """
        try:
            self._widget.setFocusPolicy(QtCore.Qt.NoFocus)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Recursively apply to child widgets (best-effort)
        try:
            children = self._widget.findChildren(QtWidgets.QWidget)  # type: ignore[valid-type]
        except Exception:
            children = []
        for ch in children:
            try:
                ch.setFocusPolicy(QtCore.Qt.NoFocus)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                if hasattr(QtCore.Qt, 'WA_MacShowFocusRect'):
                    ch.setAttribute(QtCore.Qt.WA_MacShowFocusRect, False)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _apply_focus_rules_recursively(self) -> None:
        """Apply NoFocus, disable Mac focus ring, and install event filter on children.

        Called at start and once more shortly after to catch late-created GL widgets.
        """
        try:
            children = self._widget.findChildren(QtWidgets.QWidget)  # type: ignore[valid-type]
        except Exception:
            children = []
        for ch in children:
            try:
                ch.setFocusPolicy(QtCore.Qt.NoFocus)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                if hasattr(QtCore.Qt, 'WA_MacShowFocusRect'):
                    ch.setAttribute(QtCore.Qt.WA_MacShowFocusRect, False)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                ch.installEventFilter(self._filter)
            except Exception:
                pass
        # Re-apply shortly to catch dynamic children
        try:
            t = QtCore.QTimer(self._widget)
            t.setSingleShot(True)
            t.setInterval(200)
            def _again() -> None:
                try:
                    self._apply_focus_rules_recursively()
                except Exception:
                    pass
            t.timeout.connect(_again)
            t.start()
            self._focus_timer = t  # type: ignore[attr-defined]
        except Exception:
            pass
