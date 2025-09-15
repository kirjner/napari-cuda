"""
ProxyViewer - Client-side viewer that forwards commands to remote server.

Critical: This must prevent local Window creation while maintaining
the Viewer API for the QtViewer to interact with.
"""

import asyncio
import json
import logging
import websockets
from typing import Optional, TYPE_CHECKING

import napari
import qasync
from qtpy.QtWidgets import QApplication
from napari.components.viewer_model import ViewerModel
try:
    # pydantic v1 compatibility (used by napari EventedModel)
    from pydantic.v1 import PrivateAttr
except Exception:  # pragma: no cover
    from pydantic import PrivateAttr  # type: ignore
from napari.components import LayerList, Dims, Camera

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
    _state_websocket = PrivateAttr(default=None)
    _window = PrivateAttr(default=None)
    _connection_pending: bool = PrivateAttr(default=False)
    # Bridge to coordinator (thin client path)
    _state_sender = PrivateAttr(default=None)
    _suppress_forward: bool = PrivateAttr(default=False)

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
        
        # Initialize private attributes (avoid pydantic field errors)
        self._server_host = server_host
        self._server_port = server_port
        self._state_websocket = None
        self._window = None  # type: ignore[assignment]
        self._connection_pending = False
        
        # Connect local events to forward to server
        self.camera.events.center.connect(self._on_camera_change)
        self.camera.events.zoom.connect(self._on_camera_change)
        self.camera.events.angles.connect(self._on_camera_change)
        self.dims.events.current_step.connect(self._on_dims_change)
        self.dims.events.ndisplay.connect(self._on_dims_change)
        
        # In streaming client, we run offline and delegate state to coordinator
        if not bool(offline):
            self._connect_to_server()
        else:
            logger.info("ProxyViewer offline mode: skipping server connection")
        
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
    
    def _connect_to_server(self):
        """Establish WebSocket connection to server."""
        # Get or create qasync event loop (we require qasync)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            app = QApplication.instance()
            if app:
                # Create qasync event loop for Qt/asyncio integration
                loop = qasync.QEventLoop(app)
                asyncio.set_event_loop(loop)
            else:
                # This shouldn't happen in normal usage
                raise RuntimeError("No Qt application found - ProxyViewer requires Qt context")
        
        # Start connection in background; prefer loop.create_task to avoid requiring a running loop
        try:
            loop.create_task(self._maintain_connection())
            self._connection_pending = False
        except Exception as e:
            logger.error(f"Failed to create connection task: {e}")
            # Set flag to retry later when event loop is ready
            self._connection_pending = True
    
    async def _maintain_connection(self):
        """Maintain persistent connection to server."""
        url = f"ws://{self.server_host}:{self.server_port}"
        logger.info(f"Connecting to server at {url}")
        
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    self._state_websocket = websocket
                    logger.info("Connected to server")
                    
                    # Listen for state updates from server
                    async for message in websocket:
                        data = json.loads(message)
                        await self._handle_server_message(data)
                        
            except Exception as e:
                logger.error(f"Connection lost: {e}")
                self._state_websocket = None
                
                # Reconnect after delay
                await asyncio.sleep(5)
                logger.info("Attempting reconnection...")
    
    async def _handle_server_message(self, data):
        """Process state update from server."""
        msg_type = data.get('type')
        
        if msg_type == 'camera_update':
            # Update local camera without triggering events
            self.camera.events.center.block()
            self.camera.events.zoom.block()
            
            self.camera.center = data.get('center', self.camera.center)
            self.camera.zoom = data.get('zoom', self.camera.zoom)
            
            self.camera.events.center.unblock()
            self.camera.events.zoom.unblock()
            
        elif msg_type == 'dims_update':
            # Update local dims without triggering events
            self.dims.events.current_step.block()
            self.dims.current_step = data.get('current_step', self.dims.current_step)
            self.dims.events.current_step.unblock()
    
    def _on_camera_change(self, event=None):
        """Forward camera change.

        In streaming client mode (state sender attached), do not forward
        local camera changes, as local model lacks authoritative scene extents
        and camera is server-controlled. Keyboard/UI handlers send explicit
        ops through the coordinator.
        """
        if self._state_sender is not None:
            return
        if self._state_websocket:
            command = {
                'type': 'set_camera',
                'center': list(self.camera.center),
                'zoom': float(self.camera.zoom)
            }
            asyncio.create_task(self._send_command(command))
    
    def _on_dims_change(self, event=None):
        """Forward dimension change via coordinator when attached.

        Avoid loops by suppressing forwarding during mirror updates.
        """
        if self._suppress_forward:
            return
        if self._state_sender is not None:
            try:
                step = list(self.dims.current_step)
            except Exception:
                step = self.dims.current_step
            try:
                ndisp = int(self.dims.ndisplay)
            except Exception:
                ndisp = None
            try:
                self._state_sender.set_dims(current_step=step, ndisplay=ndisp, origin='ui')
            except Exception:
                logger.debug("ProxyViewer dims forward failed", exc_info=True)
            return
        if self._state_websocket:
            command = {
                'type': 'dims.set',
                'current_step': list(self.dims.current_step),
                'ndisplay': int(self.dims.ndisplay),
            }
            asyncio.create_task(self._send_command(command))
    
    async def _send_command(self, command):
        """Send command to server."""
        if self._state_websocket:
            try:
                await self._state_websocket.send(json.dumps(command))
            except Exception as e:
                logger.error(f"Failed to send command: {e}")

    # --- Streaming client bridge -------------------------------------------------
    def attach_state_sender(self, sender) -> None:
        """Attach a coordinator-like sender for thin client state forwarding."""
        self._state_sender = sender

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
    ) -> None:
        """Apply server-driven dims metadata and step into local UI without re-forwarding.

        Accepts optional fields piggybacked on dims_update: ndim, range, order, axis_labels, sizes.
        """
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
            if current_step is not None:
                try:
                    self.dims.events.current_step.block()
                except Exception:
                    pass
                try:
                    self.dims.current_step = tuple(int(x) for x in current_step)
                finally:
                    try:
                        self.dims.events.current_step.unblock()
                    except Exception:
                        pass
        except Exception:
            logger.debug("ProxyViewer mirror dims apply failed", exc_info=True)
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
        if self._state_websocket:
            asyncio.create_task(self._state_websocket.close())
        if self._window:
            self._window.close()
        logger.info("ProxyViewer closed")
