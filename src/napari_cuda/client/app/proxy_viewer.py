"""
ProxyViewer - Thin client viewer that mirrors server state via the coordinator.

All authoritative state is routed through the ClientStreamLoop over a single
state channel. No direct sockets or legacy dims.set paths remain here.
"""

import logging
import os
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
from napari_cuda.client.state import DimsBridge

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
    _dims_bridge: DimsBridge | None = PrivateAttr(default=None)
    _log_dims_info: bool = PrivateAttr(default=False)

    def __init__(self, server_host: str = 'localhost', server_port: int = 8081, offline: bool = False, **kwargs):
        """Initialize proxy viewer connected to the remote coordinator."""
        super().__init__(**kwargs)
        _maybe_enable_debug_logger()

        self._server_host = server_host
        self._server_port = server_port
        self._window = None  # type: ignore[assignment]
        self._connection_pending = False

        self.camera.events.center.connect(self._on_camera_change)
        self.camera.events.zoom.connect(self._on_camera_change)
        self.camera.events.angles.connect(self._on_camera_change)

        logger.info('ProxyViewer: thin client mode (coordinator-driven)')

        try:
            flag = (os.getenv('NAPARI_CUDA_LOG_DIMS_INFO') or '').lower()
            self._log_dims_info = flag in ('1', 'true', 'yes', 'on', 'dbg', 'debug')
        except Exception:
            self._log_dims_info = False

        tx_interval = 10
        try:
            raw_interval = os.getenv('NAPARI_CUDA_SLIDER_TX_MS')
            if raw_interval is not None and raw_interval.strip():
                tx_interval = max(0, int(raw_interval))
        except Exception:
            logger.debug('ProxyViewer: slider interval parse failed', exc_info=True)

        self._dims_bridge = DimsBridge(
            viewer=self,
            logger=logger.getChild('dims'),
            tx_interval_ms=tx_interval,
            log_dims_info=self._log_dims_info,
        )

        logger.info('ProxyViewer initialized for %s:%s', server_host, server_port)
    def __setattr__(self, name: str, value: object) -> None:
        if name == '_log_dims_info':
            dims_projection = getattr(self, '_dims_bridge', None)
            if dims_projection is not None:  # pragma: no branch - guard only
                dims_projection.set_logging(bool(value))
        super().__setattr__(name, value)


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
        projection = self._dims_bridge
        if projection is None:
            return
        projection.handle_dims_change(event)
        return

    def _on_ndisplay_change(self, event=None):
        projection = self._dims_bridge
        if projection is None:
            return
        projection.handle_ndisplay_change(event)


    # --- Streaming client bridge -------------------------------------------------
    def attach_state_sender(self, sender) -> None:
        """Attach a coordinator-like sender for thin client state forwarding."""
        self._state_sender = sender
        assert hasattr(sender, '_control_state'), "state sender missing control_state"
        if self._dims_bridge is not None:
            self._dims_bridge.attach_state_sender(sender)

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
        """Delegate server-driven dims updates to the projection helper."""
        if self._dims_bridge is None:
            return
        self._dims_bridge.apply_remote(
            current_step=current_step,
            ndisplay=ndisplay,
            ndim=ndim,
            dims_range=dims_range,
            order=order,
            axis_labels=axis_labels,
            sizes=sizes,
            displayed=displayed,
        )



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
        if self._dims_bridge is not None:
            self._dims_bridge.shutdown()
        if self._window:
            self._window.close()
        logger.info("ProxyViewer closed")

    @property
    def _suppress_forward(self) -> bool:  # noqa: D401 - legacy compatibility
        """Expose suppression flag for legacy callers."""
        if self._dims_bridge is None:
            return False
        return self._dims_bridge.suppress_forward

    @_suppress_forward.setter
    def _suppress_forward(self, value: bool) -> None:
        logger.debug("ProxyViewer: _suppress_forward -> %s", value)
        if self._dims_bridge is not None:
            self._dims_bridge.suppress_forward = bool(value)
