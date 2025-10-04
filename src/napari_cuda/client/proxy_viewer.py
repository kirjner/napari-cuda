"""
ProxyViewer - Thin client viewer that mirrors server state via the coordinator.

All authoritative state is routed through the ClientStreamLoop over a single
state channel. No direct sockets or legacy dims.set paths remain here.
"""

import logging
import math
import os
from contextlib import nullcontext
from typing import Optional, TYPE_CHECKING

import napari
from qtpy.QtWidgets import QApplication
from napari.components.viewer_model import ViewerModel
try:
    # pydantic v1 compatibility (used by napari EventedModel)
    from pydantic.v1 import PrivateAttr
except Exception:  # pragma: no cover
    from pydantic import PrivateAttr  # type: ignore
from napari.components import LayerList, Dims, Camera
from qtpy import QtCore  # type: ignore

from napari_cuda.client.layers import RegistrySnapshot

if TYPE_CHECKING:
    # This import reveals napari's intentional architectural coupling between
    # Viewer and Window. While theoretically the model shouldn't know about
    # the GUI, napari deliberately couples them for a cohesive application.
    # This coupling is actually beneficial for our proxy pattern - it means
    # ViewerModel already has the exact "shape" the GUI expects, making our
    # ProxyViewer a perfect drop-in replacement that can intercept all 
    # viewer operations while maintaining API compatibility.
    from napari._qt.qt_main_window import Window

logger = logging.getLogger(__name__)


def _maybe_enable_debug_logger() -> None:
    """Enable DEBUG logs for this module only when env is set.

    - Attaches a on-module handler at DEBUG.
    - Disables propagation to avoid global DEBUG flood.
    - Controlled by NAPARI_CUDA_CLIENT_DEBUG or NAPARI_CUDA_DEBUG envs.
    """
    try:
        import os as _os
        flag = (_os.getenv('NAPARI_CUDA_CLIENT_DEBUG') or _os.getenv('NAPARI_CUDA_DEBUG') or '').lower()
        if flag not in ('1', 'true', 'yes', 'on', 'dbg', 'debug'):
            return
        has_local = any(getattr(h, '_napari_cuda_local', False) for h in logger.handlers)
        if has_local:
            return
        h = logging.StreamHandler()
        fmt = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        h.setFormatter(logging.Formatter(fmt))
        h.setLevel(logging.DEBUG)
        setattr(h, '_napari_cuda_local', True)
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    except Exception:
        pass


class ProxyViewer(ViewerModel):
    """
    A viewer that forwards all commands to a remote server instead of 
    rendering locally.
    
    This inherits from ViewerModel (not Viewer) to avoid creating a Window.
    The Window and QtViewer will be created separately by the launcher.
    """
    
    # Private attributes (not part of the pydantic/EventedModel schema)
    _server_host: str = PrivateAttr(default='localhost')
    _server_port: int = PrivateAttr(default=8081)
    _window = PrivateAttr(default=None)
    _connection_pending: bool = PrivateAttr(default=False)
    # Bridge to coordinator (thin client path)
    _state_sender = PrivateAttr(default=None)
    _suppress_forward: bool = PrivateAttr(default=False)
    _last_step_ui = PrivateAttr(default=None)
    _dims_tx_timer = PrivateAttr(default=None)
    _dims_tx_pending = PrivateAttr(default=None)
    _dims_tx_interval_ms: int = PrivateAttr(default=10)
    _is_playing: bool = PrivateAttr(default=False)
    _play_axis: Optional[int] = PrivateAttr(default=None)
    _log_dims_info: bool = PrivateAttr(default=False)

    def __init__(self, server_host='localhost', server_port=8081, offline: bool = False, **kwargs):
        """
        Initialize proxy viewer connected to remote server.
        
        Parameters
        ----------
        server_host : str
            Remote server hostname
        server_port : int
            Remote server state sync port
        """
        # Initialize ViewerModel without creating a Window
        super().__init__(**kwargs)
        _maybe_enable_debug_logger()
        
        # Initialize private attributes (avoid pydantic field errors)
        self._server_host = server_host
        self._server_port = server_port
        self._window = None  # type: ignore[assignment]
        self._connection_pending = False
        
        # Connect local events to forward to server
        self.camera.events.center.connect(self._on_camera_change)
        self.camera.events.zoom.connect(self._on_camera_change)
        self.camera.events.angles.connect(self._on_camera_change)
        self.dims.events.current_step.connect(self._on_dims_change)
        self.dims.events.ndisplay.connect(self._on_ndisplay_change)
        
        # In streaming client, delegate state to coordinator (no direct sockets)
        logger.info("ProxyViewer: thin client mode (coordinator-driven)")
        # Slider transmit interval (ms); 0 disables coalescing and sends immediately
        try:
            import os as _os
            iv = _os.getenv('NAPARI_CUDA_SLIDER_TX_MS')
            if iv is not None and str(iv).strip() != '':
                self._dims_tx_interval_ms = max(0, int(iv))
        except Exception:
            pass

        try:
            flag = (os.getenv('NAPARI_CUDA_LOG_DIMS_INFO') or '').lower()
            self._log_dims_info = flag in ('1', 'true', 'yes', 'on', 'dbg', 'debug')
        except Exception:
            self._log_dims_info = False
        
        logger.info(f"ProxyViewer initialized for {server_host}:{server_port}")

    # Expose read-only properties for host/port
    @property
    def server_host(self) -> str:
        return self._server_host

    @property
    def server_port(self) -> int:
        return self._server_port

    # Provide explicit property for window so launcher can set it
    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):  # value: 'Window'
        self._window = value
    
    # No direct socket handling in thin client mode
    
    def _on_camera_change(self, event=None):
        """Forward camera change.

        In streaming client mode (state sender attached), do not forward
        local camera changes, as local model lacks authoritative scene extents
        and camera is server-controlled. Keyboard/UI handlers send explicit
        ops through the coordinator.
        """
        return
    
    def _on_dims_change(self, event=None):
        """Forward dimension change via coordinator when attached.

        Avoid loops by suppressing forwarding during mirror updates.
        """
        if self._suppress_forward:
            return
        if self._state_sender is not None:
            # Compute which axis changed (best-effort)
            try:
                cur = tuple(int(x) for x in self.dims.current_step)
            except Exception:
                cur = self.dims.current_step
            changed_axis = None
            try:
                prev = tuple(int(x) for x in (self._last_step_ui or ()))
                if isinstance(cur, tuple) and isinstance(prev, tuple) and len(prev) == len(cur):
                    for i, (a, b) in enumerate(zip(prev, cur)):
                        if a != b:
                            changed_axis = i
                            break
            except Exception:
                changed_axis = None
            logger.debug("dims.change: cur=%s prev=%s -> changed_axis=%s", cur, prev if 'prev' in locals() else None, changed_axis)
            # Probe play state from QtDims
            try:
                qdims = getattr(getattr(self.window, '_qt_viewer', None), 'dims', None) if self.window is not None else None
                if qdims is not None and hasattr(qdims, 'is_playing'):
                    self._is_playing = bool(qdims.is_playing)
                if not self._is_playing:
                    self._play_axis = None
            except Exception:
                logger.debug("ProxyViewer: probe play state failed", exc_info=True)
            if self._is_playing and self._play_axis is None and changed_axis is not None:
                self._play_axis = int(changed_axis)
            logger.debug("play state: playing=%s play_axis=%s", self._is_playing, self._play_axis)
            if changed_axis is None:
                # Fallback to coordinator's primary axis if available
                try:
                    changed_axis = int(getattr(self._state_sender, '_primary_axis_index', 0) or 0)
                except Exception:
                    changed_axis = 0
            # Coalesce rapid slider changes via single-shot timer
            try:
                val = int(cur[changed_axis]) if isinstance(cur, (list, tuple)) else int(cur)
            except Exception:
                val = None
            if val is None:
                return
            # If the playing axis ticked, send step delta; suppress other local emitters on that axis
            if self._is_playing and self._play_axis is not None and int(changed_axis) == int(self._play_axis):
                try:
                    prev_val = int(prev[changed_axis]) if isinstance(prev, tuple) else None
                except Exception:
                    prev_val = None
                if prev_val is not None:
                    delta = int(val) - int(prev_val)
                    if delta != 0:
                        logger.debug(
                            "play tick -> state.update dims.step axis=%d delta=%+d",
                            int(changed_axis),
                            int(delta),
                        )
                        _ = self._state_sender.dims_step(int(changed_axis), int(delta), origin='play')
                # Update last snapshot and stop here (no set_index on play axis)
                self._last_step_ui = tuple(cur) if isinstance(cur, tuple) else cur
                return
            # Otherwise, send set_index for non-play axis changes (slider drags, etc.)
            # Send immediately when coalescing is disabled, otherwise coalesce
            try:
                if int(getattr(self, '_dims_tx_interval_ms', 10) or 0) <= 0:
                    if bool(getattr(self, '_log_dims_info', False)):
                        logger.info(
                            "slider -> state.update dims.index axis=%d value=%d",
                            int(changed_axis),
                            int(val),
                        )
                    logger.debug(
                        "slider -> state.update dims.index axis=%d value=%d (immediate)",
                        int(changed_axis),
                        int(val),
                    )
                    _ = self._state_sender.dims_set_index(int(changed_axis), int(val), origin='ui')
                else:
                    self._dims_tx_pending = (int(changed_axis), int(val))
                    if self._dims_tx_timer is None:
                        t = QtCore.QTimer(self.window._qt_viewer) if self.window is not None else QtCore.QTimer()
                        t.setSingleShot(True)
                        t.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]
                        def _fire() -> None:
                            try:
                                pair = self._dims_tx_pending
                                self._dims_tx_pending = None
                                if pair is None:
                                    return
                                ax, vv = pair
                                if bool(getattr(self, '_log_dims_info', False)):
                                    logger.info(
                                        "slider (coalesced) -> state.update dims.index axis=%d value=%d",
                                        int(ax),
                                        int(vv),
                                    )
                                logger.debug(
                                    "slider (coalesced) -> state.update dims.index axis=%d value=%d",
                                    int(ax),
                                    int(vv),
                                )
                                _ = self._state_sender.dims_set_index(int(ax), int(vv), origin='ui')
                            except Exception:
                                logger.debug("ProxyViewer dims control send failed", exc_info=True)
                        t.timeout.connect(_fire)
                        self._dims_tx_timer = t
                    # restart timer with configured interval
                    self._dims_tx_timer.start(max(1, int(self._dims_tx_interval_ms)))
                # Update local snapshot so subsequent deltas reflect the UI state
                try:
                    self._last_step_ui = tuple(cur) if isinstance(cur, tuple) else cur
                except Exception:
                    self._last_step_ui = cur
            except Exception:
                logger.debug("ProxyViewer dims control send failed", exc_info=True)
            return
        return

    def _on_ndisplay_change(self, event=None):
        if self._suppress_forward:
            if bool(getattr(self, '_log_dims_info', False)):
                try:
                    logger.info("ProxyViewer ndisplay change suppressed: value=%s", getattr(event, 'value', None))
                except Exception:
                    pass
            return
        sender = self._state_sender
        if sender is None:
            if bool(getattr(self, '_log_dims_info', False)):
                try:
                    logger.info("ProxyViewer ndisplay change (no sender): value=%s", getattr(event, 'value', None))
                except Exception:
                    pass
            return
        if event is not None and hasattr(event, 'value'):
            raw_value = event.value
        else:
            try:
                raw_value = self.dims.ndisplay
            except IndexError:
                logger.debug("ProxyViewer: dims.ndisplay access failed", exc_info=True)
                return
        if bool(getattr(self, '_log_dims_info', False)):
            try:
                logger.info("ProxyViewer ndisplay change -> %s", raw_value)
            except Exception:
                pass
        try:
            ndisplay = int(raw_value)
        except (TypeError, ValueError):
            logger.debug("ProxyViewer: ndisplay value parse failed (%r)", raw_value, exc_info=True)
            return
        fn = getattr(sender, 'view_set_ndisplay', None)
        if callable(fn) and not fn(ndisplay, origin='ui'):
            logger.debug("ProxyViewer: ndisplay control send failed")


    # --- Streaming client bridge -------------------------------------------------
    def attach_state_sender(self, sender) -> None:
        """Attach a coordinator-like sender for thin client state forwarding."""
        self._state_sender = sender
        assert hasattr(sender, '_control_state'), "state sender missing control_state"
        control_state = sender._control_state  # type: ignore[attr-defined]
        interval_ms = int(math.ceil(control_state.dims_min_dt * 1000.0))
        self._dims_tx_interval_ms = max(self._dims_tx_interval_ms, interval_ms)

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
        """Apply server-driven dims metadata and step into local UI without re-forwarding.

        Accepts optional fields piggybacked on dims_update: ndim, range, order, axis_labels, sizes.
        """
        prev_suppress = self._suppress_forward
        self._suppress_forward = True
        try:
            # Apply metadata first so sliders exist before setting current_step
            if ndim is not None:
                try:
                    self.dims.ndim = int(ndim)
                except Exception:
                    pass
            # Prefer explicit range; accept (start, stop) or (start, stop, step).
            # If not valid, synthesize from sizes when available.
            applied_range = False
            if dims_range is not None:
                try:
                    rng_list = []
                    for entry in dims_range:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            a = float(entry[0]); b = float(entry[1])
                            c = float(entry[2]) if len(entry) >= 3 else 1.0
                            rng_list.append((a, b, c))
                        else:
                            raise ValueError("range entry must have 2 or 3 values")
                    self.dims.range = tuple(rng_list)
                    applied_range = True
                except Exception:
                    logger.debug("apply dims.range failed", exc_info=True)
            if not applied_range and sizes is not None:
                try:
                    rng = tuple((0.0, float(int(s) - 1), 1.0) for s in sizes)
                    self.dims.range = rng
                    applied_range = True
                except Exception:
                    logger.debug("apply sizes->range failed", exc_info=True)
            # Apply order; accept ints or labels matching axis_labels
            if order is not None:
                try:
                    if all(isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()) for x in order):
                        self.dims.order = tuple(int(x) for x in order)
                    elif axis_labels is not None and all(isinstance(x, str) for x in order):
                        label_to_idx = {str(lbl): i for i, lbl in enumerate(axis_labels)}
                        numeric = [label_to_idx.get(str(lbl), i) for i, lbl in enumerate(order)]
                        self.dims.order = tuple(int(x) for x in numeric)
                except Exception:
                    logger.debug("apply dims.order failed", exc_info=True)
            if axis_labels is not None:
                try:
                    self.dims.axis_labels = tuple(str(x) for x in axis_labels)
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
                        current_order = list(self.dims.order)
                        if not current_order:
                            current_order = list(range(self.dims.ndim))
                        base = [ax for ax in current_order if ax not in disp]
                        base.extend(ax for ax in disp if ax not in base)
                        for ax in range(self.dims.ndim):
                            if ax not in base:
                                base.insert(0, ax)
                        if len(base) == len(current_order):
                            self.dims.order = tuple(base)
                except Exception:
                    if bool(getattr(self, '_log_dims_info', False)):
                        logger.info("ProxyViewer dims order apply failed", exc_info=True)
                    else:
                        logger.debug("apply dims.displayed failed", exc_info=True)
            if current_step is not None:
                # Do not block events here: allow napari UI (sliders) to update.
                # Loopback is prevented by _suppress_forward above.
                step_tuple = tuple(int(x) for x in current_step)
                self.dims.current_step = step_tuple
                try:
                    self.dims.point = tuple(float(x) for x in current_step)
                except Exception:
                    logger.debug("apply dims.point failed", exc_info=True)
                try:
                    self._last_step_ui = step_tuple
                except Exception:
                    self._last_step_ui = current_step
            if bool(getattr(self, '_log_dims_info', False)):
                logger.info(
                    "ProxyViewer dims applied: ndim=%s ndisplay=%s displayed=%s order=%s step=%s range=%s",
                    self.dims.ndim,
                    self.dims.ndisplay,
                    self.dims.displayed,
                    self.dims.order,
                    getattr(self.dims, 'current_step', None),
                    getattr(self.dims, 'range', None),
                )
        except Exception:
            logger.debug("ProxyViewer mirror dims apply failed", exc_info=True)
        finally:
            self._suppress_forward = prev_suppress

    def _sync_remote_layers(self, snapshot: RegistrySnapshot) -> None:
        """Synchronize remote layer mirrors with the latest registry snapshot."""
        blocker = getattr(getattr(self.layers, 'events', None), 'blocker', None)
        ctx = blocker() if callable(blocker) else nullcontext()
        self._suppress_forward = True
        try:
            with ctx:
                desired_ids = list(snapshot.ids())
                existing_layers = list(self.layers)
                for layer in existing_layers:
                    remote_id = getattr(layer, 'remote_id', None)
                    if remote_id and remote_id not in desired_ids:
                        try:
                            self.layers.remove(layer)
                        except ValueError:
                            continue
                # Ensure mapping reflects removals
                for idx, record in enumerate(snapshot.iter()):
                    layer = record.layer
                    if layer not in self.layers:
                        self.layers.insert(idx, layer)
                    current_index = self.layers.index(layer)
                    if current_index != idx:
                        try:
                            self.layers.move(current_index, idx)
                        except Exception:
                            # Fallback to manual reposition
                            self.layers.pop(current_index)
                            self.layers.insert(idx, layer)
                    try:
                        controls = record.block.get("controls") if isinstance(record.block.get("controls"), dict) else None
                        target = controls.get("visible") if isinstance(controls, dict) else None
                        if target is not None and bool(layer.visible) is not bool(target):
                            emitter = getattr(getattr(layer, "events", None), "visible", None)
                            blocker = emitter.blocker() if hasattr(emitter, "blocker") else nullcontext()
                            with blocker:
                                layer.visible = bool(target)
                    except Exception:
                        pass
        finally:
            self._suppress_forward = False

    
    
    # Override methods that would try to render locally
    def add_image(self, *args, **kwargs):
        """Block local image addition."""
        logger.warning("ProxyViewer: add_image blocked (server-side only)")
        return None
    
    def add_labels(self, *args, **kwargs):
        """Block local labels addition."""
        logger.warning("ProxyViewer: add_labels blocked (server-side only)")
        return None
    
    def add_points(self, *args, **kwargs):
        """Block local points addition."""
        logger.warning("ProxyViewer: add_points blocked (server-side only)")
        return None
    
    def add_shapes(self, *args, **kwargs):
        """Block local shapes addition."""
        logger.warning("ProxyViewer: add_shapes blocked (server-side only)")
        return None
    
    def screenshot(self, *args, **kwargs):
        """Screenshot happens server-side."""
        logger.info("Screenshot requested from server")
        # Could implement by requesting high-quality frame from server
        return None
    
    @property
    def window(self):
        """Return the window if set by launcher."""
        return self._window
    
    @window.setter
    def window(self, value):
        """Allow launcher to set the window."""
        self._window = value
    
    def close(self):
        """Close connection to server and window if exists."""
        if self._window:
            self._window.close()
        logger.info("ProxyViewer closed")
