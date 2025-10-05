"""
ProxyViewer - Thin client viewer that mirrors server state via the coordinator.

All authoritative state is routed through the ClientStreamLoop over a single
state channel. No direct sockets or legacy dims.set paths remain here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import nullcontext
from typing import Optional, TYPE_CHECKING

import napari
from qtpy import QtCore  # type: ignore
from qtpy.QtWidgets import QApplication
from napari.components import Dims, LayerList
from napari.components.viewer_model import ViewerModel

try:
    # pydantic v1 compatibility (used by napari EventedModel)
    from pydantic.v1 import PrivateAttr
except Exception:  # pragma: no cover
    from pydantic import PrivateAttr  # type: ignore

from napari_cuda.client.data import RegistrySnapshot

if TYPE_CHECKING:
    from napari._qt.qt_main_window import Window

logger = logging.getLogger(__name__)


def _maybe_enable_debug_logger() -> None:
    """Enable DEBUG logs for this module only when env is set."""

    try:
        flag = (os.getenv("NAPARI_CUDA_CLIENT_DEBUG") or os.getenv("NAPARI_CUDA_DEBUG") or "").lower()
        if flag not in {"1", "true", "yes", "on", "dbg", "debug"}:
            return
        if any(getattr(handler, "_napari_cuda_local", False) for handler in logger.handlers):
            return
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s"))
        handler.setLevel(logging.DEBUG)
        setattr(handler, "_napari_cuda_local", True)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    except Exception:
        pass


class ProxyViewer(ViewerModel):
    """ViewerModel subclass that mirrors remote state via the coordinator."""

    _server_host: str = PrivateAttr(default="localhost")
    _server_port: int = PrivateAttr(default=8081)
    _window = PrivateAttr(default=None)
    _connection_pending: bool = PrivateAttr(default=False)

    _state_sender = PrivateAttr(default=None)
    _suppress_forward: bool = PrivateAttr(default=False)
    _last_step_ui = PrivateAttr(default=None)
    _dims_tx_timer = PrivateAttr(default=None)
    _dims_tx_pending = PrivateAttr(default=None)
    _dims_tx_interval_ms: int = PrivateAttr(default=10)
    _is_playing: bool = PrivateAttr(default=False)
    _play_axis: Optional[int] = PrivateAttr(default=None)
    _log_dims_info: bool = PrivateAttr(default=False)

    def __init__(self, server_host: str = "localhost", server_port: int = 8081, offline: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        _maybe_enable_debug_logger()

        self._server_host = server_host
        self._server_port = server_port
        self._window = None  # type: ignore[assignment]
        self._connection_pending = False

        self.camera.events.center.connect(self._on_camera_change)
        self.camera.events.zoom.connect(self._on_camera_change)
        self.camera.events.angles.connect(self._on_camera_change)
        self.dims.events.current_step.connect(self._on_dims_change)
        self.dims.events.ndisplay.connect(self._on_ndisplay_change)

        logger.info("ProxyViewer: thin client mode (coordinator-driven)")

        try:
            interval = os.getenv("NAPARI_CUDA_SLIDER_TX_MS")
            if interval and interval.strip():
                self._dims_tx_interval_ms = max(0, int(interval))
        except Exception:
            logger.debug("ProxyViewer: slider interval parse failed", exc_info=True)

        try:
            flag = (os.getenv("NAPARI_CUDA_LOG_DIMS_INFO") or "").lower()
            self._log_dims_info = flag in {"1", "true", "yes", "on", "dbg", "debug"}
        except Exception:
            self._log_dims_info = False

        logger.info("ProxyViewer initialized for %s:%s", server_host, server_port)

    # ------------------------------------------------------------------ camera + dims handlers
    def _on_camera_change(self, event=None) -> None:  # pragma: no cover - no local camera forwarding
        return

    def _on_dims_change(self, event=None) -> None:
        if self._suppress_forward:
            return
        sender = self._state_sender
        if sender is None:
            return

        try:
            cur = tuple(int(x) for x in self.dims.current_step)
        except Exception:
            cur = self.dims.current_step

        changed_axis: Optional[int] = None
        try:
            prev = tuple(int(x) for x in (self._last_step_ui or ()))
            if isinstance(cur, tuple) and isinstance(prev, tuple) and len(prev) == len(cur):
                for idx, (before, after) in enumerate(zip(prev, cur)):
                    if before != after:
                        changed_axis = idx
                        break
        except Exception:
            changed_axis = None

        logger.debug("dims.change: cur=%s prev=%s -> changed_axis=%s", cur, prev if "prev" in locals() else None, changed_axis)

        qdims = getattr(getattr(self.window, "_qt_viewer", None), "dims", None) if self.window is not None else None
        if qdims is not None and hasattr(qdims, "is_playing"):
            try:
                self._is_playing = bool(qdims.is_playing)
            except Exception:
                logger.debug("ProxyViewer: probe play state failed", exc_info=True)
        if not self._is_playing:
            self._play_axis = None
        elif self._play_axis is None and changed_axis is not None:
            self._play_axis = int(changed_axis)

        logger.debug("play state: playing=%s play_axis=%s", self._is_playing, self._play_axis)

        if changed_axis is None:
            try:
                changed_axis = int(getattr(sender, "_primary_axis_index", 0) or 0)
            except Exception:
                changed_axis = 0

        try:
            val = int(cur[changed_axis]) if isinstance(cur, (list, tuple)) else int(cur)
        except Exception:
            val = None
        if val is None:
            return

        if self._is_playing and self._play_axis is not None and int(changed_axis) == int(self._play_axis):
            prev_val = None
            try:
                prev_val = int(prev[changed_axis]) if isinstance(prev, tuple) else None
            except Exception:
                prev_val = None
            if prev_val is not None:
                delta = int(val) - int(prev_val)
                if delta != 0:
                    logger.debug("play tick -> state.update dims.step axis=%d delta=%+d", int(changed_axis), delta)
                    sender.dims_step(int(changed_axis), int(delta), origin="play")
            self._last_step_ui = tuple(cur) if isinstance(cur, tuple) else cur
            return

        if self._dims_tx_interval_ms <= 0:
            if self._log_dims_info:
                logger.info("slider -> state.update dims.index axis=%d value=%d", int(changed_axis), int(val))
            logger.debug("slider -> state.update dims.index axis=%d value=%d (immediate)", int(changed_axis), int(val))
            sender.dims_set_index(int(changed_axis), int(val), origin="ui")
        else:
            self._dims_tx_pending = (int(changed_axis), int(val))
            if self._dims_tx_timer is None:
                qt_parent = getattr(self.window, "_qt_viewer", None)
                timer = QtCore.QTimer(qt_parent)
                timer.setSingleShot(True)
                timer.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]

                def _fire() -> None:
                    try:
                        pair = self._dims_tx_pending
                        self._dims_tx_pending = None
                        if pair is None:
                            return
                        ax, vv = pair
                        if self._log_dims_info:
                            logger.info("slider (coalesced) -> state.update dims.index axis=%d value=%d", ax, vv)
                        logger.debug("slider (coalesced) -> state.update dims.index axis=%d value=%d", ax, vv)
                        sender.dims_set_index(int(ax), int(vv), origin="ui")
                    except Exception:
                        logger.debug("ProxyViewer dims intent send failed", exc_info=True)

                timer.timeout.connect(_fire)
                self._dims_tx_timer = timer
            self._dims_tx_timer.start(max(1, int(self._dims_tx_interval_ms)))

        self._last_step_ui = tuple(cur) if isinstance(cur, tuple) else cur

    def _on_ndisplay_change(self, event=None) -> None:
        if self._suppress_forward:
            if self._log_dims_info:
                logger.info("ProxyViewer ndisplay change suppressed: value=%s", getattr(event, "value", None))
            return
        sender = self._state_sender
        if sender is None:
            if self._log_dims_info:
                logger.info("ProxyViewer ndisplay change (no sender): value=%s", getattr(event, "value", None))
            return
        raw_value = getattr(event, "value", None)
        if raw_value is None:
            try:
                raw_value = self.dims.ndisplay
            except Exception:
                logger.debug("ProxyViewer: dims.ndisplay access failed", exc_info=True)
                return
        try:
            ndisplay = int(raw_value)
        except (TypeError, ValueError):
            logger.debug("ProxyViewer: ndisplay parse failed (%r)", raw_value, exc_info=True)
            return
        fn = getattr(sender, "view_set_ndisplay", None)
        if callable(fn) and not fn(ndisplay, origin="ui"):
            logger.debug("ProxyViewer: ndisplay intent rejected")

    # ------------------------------------------------------------------ streaming bridge
    def attach_state_sender(self, sender) -> None:
        self._state_sender = sender
        assert hasattr(sender, "_control_state"), "state sender missing control_state"
        self._dims_tx_interval_ms = 0

    def _apply_remote_dims_update(
        self,
        *,
        current_step=None,
        ndisplay=None,
        ndim=None,
        dims_range=None,
        order=None,
        axis_labels=None,
        sizes=None,
        displayed=None,
    ) -> None:
        prev_suppress = self._suppress_forward
        self._suppress_forward = True
        try:
            if ndim is not None:
                self.dims.ndim = int(ndim)
            if dims_range is not None:
                rng_list = []
                for entry in dims_range:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        start = float(entry[0])
                        stop = float(entry[1])
                        step = float(entry[2]) if len(entry) >= 3 else 1.0
                        rng_list.append((start, stop, step))
                    else:
                        raise ValueError("range entry must have 2 or 3 values")
                self.dims.range = tuple(rng_list)
            elif sizes is not None:
                self.dims.range = tuple((0.0, float(int(size) - 1), 1.0) for size in sizes)

            if order is not None:
                try:
                    if all(isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()) for x in order):
                        self.dims.order = tuple(int(x) for x in order)
                    elif axis_labels is not None and all(isinstance(x, str) for x in order):
                        label_to_idx = {str(lbl): idx for idx, lbl in enumerate(axis_labels)}
                        numeric = [label_to_idx.get(str(lbl), idx) for idx, lbl in enumerate(order)]
                        self.dims.order = tuple(int(x) for x in numeric)
                except Exception:
                    logger.debug("apply dims.order failed", exc_info=True)

            if axis_labels is not None:
                try:
                    self.dims.axis_labels = tuple(str(lbl) for lbl in axis_labels)
                except Exception:
                    logger.debug("apply axis_labels failed", exc_info=True)

            if ndisplay is not None:
                try:
                    self.dims.ndisplay = int(ndisplay)
                except Exception:
                    pass

            if displayed is not None:
                try:
                    disp = [int(x) for x in displayed]
                    disp = [x for x in disp if 0 <= x < self.dims.ndim]
                    if disp:
                        current_order = list(self.dims.order) or list(range(self.dims.ndim))
                        base = [ax for ax in current_order if ax not in disp]
                        base.extend(ax for ax in disp if ax not in base)
                        for ax in range(self.dims.ndim):
                            if ax not in base:
                                base.insert(0, ax)
                        if len(base) == len(current_order):
                            self.dims.order = tuple(base)
                except Exception:
                    logger.debug("apply dims.displayed failed", exc_info=True)

            if current_step is not None:
                step_tuple = tuple(int(x) for x in current_step)
                self.dims.current_step = step_tuple
                try:
                    self.dims.point = tuple(float(x) for x in current_step)
                except Exception:
                    logger.debug("apply dims.point failed", exc_info=True)
                self._last_step_ui = step_tuple

            if self._log_dims_info:
                logger.info(
                    "ProxyViewer dims applied: ndim=%s ndisplay=%s displayed=%s order=%s step=%s range=%s",
                    self.dims.ndim,
                    self.dims.ndisplay,
                    self.dims.displayed,
                    self.dims.order,
                    getattr(self.dims, "current_step", None),
                    getattr(self.dims, "range", None),
                )
        except Exception:
            logger.debug("ProxyViewer mirror dims apply failed", exc_info=True)
        finally:
            self._suppress_forward = prev_suppress

    def _sync_remote_layers(self, snapshot: RegistrySnapshot) -> None:
        blocker = getattr(getattr(self.layers, "events", None), "blocker", None)
        ctx = blocker() if callable(blocker) else nullcontext()
        self._suppress_forward = True
        try:
            with ctx:
                desired_ids = list(snapshot.ids())
                existing_layers = list(self.layers)
                for layer in existing_layers:
                    remote_id = getattr(layer, "remote_id", None)
                    if remote_id and remote_id not in desired_ids:
                        try:
                            self.layers.remove(layer)
                        except ValueError:
                            continue
                for idx, record in enumerate(snapshot.iter()):
                    layer = record.layer
                    if layer not in self.layers:
                        self.layers.insert(idx, layer)
                    current_index = self.layers.index(layer)
                    if current_index != idx:
                        try:
                            self.layers.move(current_index, idx)
                        except Exception:
                            self.layers.pop(current_index)
                            self.layers.insert(idx, layer)
                    controls = record.block.get("controls") if isinstance(record.block.get("controls"), dict) else None
                    target = controls.get("visible") if isinstance(controls, dict) else None
                    if target is not None and bool(layer.visible) is not bool(target):
                        emitter = getattr(getattr(layer, "events", None), "visible", None)
                        block_ctx = emitter.blocker() if hasattr(emitter, "blocker") else nullcontext()
                        with block_ctx:
                            layer.visible = bool(target)
        finally:
            self._suppress_forward = False

    # ------------------------------------------------------------------ proxy passthroughs
    @property
    def server_host(self) -> str:
        return self._server_host

    @property
    def server_port(self) -> int:
        return self._server_port

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        self._window = value

    def _connect_to_server(self) -> None:  # legacy shim retained for launcher compatibility
        return

    def close(self):
        if self._state_sender:
            try:
                asyncio.create_task(self._state_sender.close())
            except Exception:
                pass
        if self._window:
            self._window.close()
        logger.info("ProxyViewer closed")
