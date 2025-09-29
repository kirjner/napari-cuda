from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from qtpy import QtCore, QtWidgets  # type: ignore

logger = logging.getLogger(__name__)

def _pointer_xy(event) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    """Return pointer coordinates, asserting the Qt event exposes them."""

    if hasattr(event, 'position'):
        pos = event.position()
        assert pos is not None, "pointer event returned None from position()"
        return float(pos.x()), float(pos.y())
    assert hasattr(event, 'pos'), "pointer event missing position()/pos()"
    pos = event.pos()
    assert pos is not None, "pointer event returned None from pos()"
    return float(pos.x()), float(pos.y())


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
        assert hasattr(widget, 'window'), "widget must expose window()"
        self._top_window = widget.window()

    def eventFilter(self, obj, event):  # type: ignore[no-untyped-def]
        event_type = event.type()
        target_is_canvas = obj is self._widget

        if event_type == QtCore.QEvent.Wheel and target_is_canvas:  # type: ignore[attr-defined]
            return self._handle_wheel(event)

        if event_type == QtCore.QEvent.KeyPress:  # type: ignore[attr-defined]
            if self._top_window is not None:
                if not hasattr(obj, 'window'):
                    logger.debug("key event target missing window(): %r", obj)
                    return False
                other_window = obj.window()
                if other_window is not None and other_window is not self._top_window:
                    return False
            key = int(event.key())
            mods = int(event.modifiers())
            text = str(event.text())
            if self._log_info:
                logger.info("key press: key=%s mods=%s text=%r", key, mods, text)
            if self._on_key is not None:
                consumed = bool(self._on_key({'type': 'input.key', 'key': key, 'mods': mods, 'text': text}))
                if consumed:
                    event.accept()
                    return True
            return False

        if event_type == QtCore.QEvent.Resize and target_is_canvas:  # type: ignore[attr-defined]
            return self._handle_resize(event)

        if event_type == QtCore.QEvent.MouseButtonPress and target_is_canvas:  # type: ignore[attr-defined]
            return self._handle_mouse_down(event)

        if event_type == QtCore.QEvent.MouseMove and target_is_canvas:  # type: ignore[attr-defined]
            return self._handle_mouse_move(event)

        if event_type == QtCore.QEvent.MouseButtonRelease and target_is_canvas:  # type: ignore[attr-defined]
            return self._handle_mouse_up(event)

        return False

    # --- Wheel ---------------------------------------------------------------------
    def _handle_wheel(self, ev) -> bool:  # type: ignore[no-untyped-def]
        now = time.perf_counter()
        if (now - self._last_wheel_send) < self._min_dt:
            # Coalesce: consume event to avoid parent handling/focus highlight,
            # but skip sending to server when over rate limit
            ev.accept()
            self._widget.clearFocus()
            return True
        # angleDelta is in 1/8 deg units, typical step is 120 per notch
        angle_delta = ev.angleDelta()
        assert angle_delta is not None, "wheel event missing angleDelta()"
        ax_val = angle_delta.x()
        ay_val = angle_delta.y()
        pixel_delta = ev.pixelDelta()
        px_val = pixel_delta.x()
        py_val = pixel_delta.y()
        x, y = _pointer_xy(ev)
        mods = int(ev.modifiers())
        dpr = float(self._widget.devicePixelRatioF())
        width_px = int(self._widget.width())
        height_px = int(self._widget.height())
        ax = int(ax_val)
        ay = int(ay_val)
        px = int(px_val)
        py = int(py_val)
        msg = {
            'type': 'input.wheel',
            'angle_x': ax,
            'angle_y': ay,
            'pixel_x': px,
            'pixel_y': py,
            'x_px': float(x),
            'y_px': float(y),
            'mods': mods,
            'width_px': width_px,
            'height_px': height_px,
            'dpr': dpr,
            'ts': float(time.time()),
        }
        ok = self._post(msg)
        assert isinstance(ok, bool), "post must return bool"
        if self._log_info:
            logger.info(
                "input.wheel sent: ay=%d py=%d mods=%d pos=(%.1f,%.1f)",
                ay, py, mods, float(x), float(y),
            )
        # Notify optional wheel callback (e.g., for dims state.update mapping)
        if self._on_wheel is not None:
            self._on_wheel(msg)
        if ok:
            self._last_wheel_send = now
        # Consume to avoid selection highlighting in parent widgets
        ev.accept()
        return True

    # --- Resize --------------------------------------------------------------------
    def _handle_resize(self, ev) -> bool:  # type: ignore[no-untyped-def]
        w = int(self._widget.width())
        h = int(self._widget.height())
        dpr = float(self._widget.devicePixelRatioF())
        self._pending_resize = (w, h, dpr)
        # Debounce (restart timer)
        interval = int(self._resize_timer.interval())
        assert interval >= 0, "QTimer returned negative interval"
        self._resize_timer.start(interval)
        return False

    def _flush_resize(self) -> None:
        pr = self._pending_resize
        if not pr:
            return
        w, h, dpr = pr
        self._pending_resize = None
        msg = {
            'type': 'view.resize',
            'width_px': w,
            'height_px': h,
            'dpr': dpr,
            'ts': float(time.time()),
        }
        ok = self._post(msg)
        assert isinstance(ok, bool), "post must return bool"
        if self._log_info:
            logger.info("view.resize sent: %dx%d dpr=%.2f", w, h, dpr)

    # (no key handling)

    # --- Mouse handling (forward to on_pointer) ----------------------------------
    def _handle_mouse_down(self, ev) -> bool:  # type: ignore[no-untyped-def]
        if self._on_pointer is not None:
            x, y = _pointer_xy(ev)
            btn_val = int(ev.button())
            btns_val = int(ev.buttons())
            mods_val = int(ev.modifiers())
            payload = {
                'type': 'input.pointer',
                'phase': 'down',
                'x_px': float(x),
                'y_px': float(y),
                'button': btn_val,
                'buttons': btns_val,
                'mods': mods_val,
                'width_px': int(self._widget.width()),
                'height_px': int(self._widget.height()),
                'ts': float(time.time()),
            }
            self._on_pointer(payload)
        ev.accept()
        return True

    def _handle_mouse_move(self, ev) -> bool:  # type: ignore[no-untyped-def]
        if self._on_pointer is not None:
            x, y = _pointer_xy(ev)
            btns_val = int(ev.buttons())
            mods_val = int(ev.modifiers())
            payload = {
                'type': 'input.pointer',
                'phase': 'move',
                'x_px': float(x),
                'y_px': float(y),
                'buttons': btns_val,
                'mods': mods_val,
                'width_px': int(self._widget.width()),
                'height_px': int(self._widget.height()),
                'ts': float(time.time()),
            }
            self._on_pointer(payload)
        ev.accept()
        return True

    def _handle_mouse_up(self, ev) -> bool:  # type: ignore[no-untyped-def]
        if self._on_pointer is not None:
            x, y = _pointer_xy(ev)
            btn_val = int(ev.button())
            btns_val = int(ev.buttons())
            mods_val = int(ev.modifiers())
            payload = {
                'type': 'input.pointer',
                'phase': 'up',
                'x_px': float(x),
                'y_px': float(y),
                'button': btn_val,
                'buttons': btns_val,
                'mods': mods_val,
                'width_px': int(self._widget.width()),
                'height_px': int(self._widget.height()),
                'ts': float(time.time()),
            }
            self._on_pointer(payload)
        ev.accept()
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
        # Set debounce interval explicitly on the timer
        interval = int(max(0, resize_debounce_ms))
        self._filter._resize_timer.setInterval(interval)

    def start(self) -> None:
        # Install event filter
        self._widget.installEventFilter(self._filter)
        # Install at the application level when input logging or on_key observation is enabled
        app = QtWidgets.QApplication.instance()
        if app is not None and (self._filter._log_info or self._filter._on_key is not None):  # type: ignore[attr-defined]
            app.installEventFilter(self._filter)
        # Send initial resize snapshot so server knows canvas size at start
        w = int(self._widget.width())
        h = int(self._widget.height())
        dpr = float(self._widget.devicePixelRatioF())
        msg = {'type': 'view.resize', 'width_px': w, 'height_px': h, 'dpr': dpr, 'ts': float(time.time())}
        ok = self._filter._post(msg)
        assert isinstance(ok, bool), "post must return bool"

    # --- Mouse/Pan ---------------------------------------------------------------
    
    # No additional focus suppression helpers needed; coalesced wheel consumption
    # in _handle_wheel is sufficient to prevent highlight during fast scroll.
