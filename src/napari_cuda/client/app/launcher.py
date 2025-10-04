"""
Launcher for napari-cuda streaming client.

This creates the client application that connects to a remote napari-cuda server.
"""

import sys
import os
import logging
import argparse

import napari
from napari._qt.qt_main_window import Window
from napari.components.viewer_model import ViewerModel
from napari.utils.action_manager import action_manager
from napari.utils.translations import trans

from .proxy_viewer import ProxyViewer
from .streaming_canvas import StreamingCanvas

logger = logging.getLogger(__name__)


def launch_streaming_client(server_host='localhost', 
                          state_port=8081,
                          pixel_port=8082,
                          debug=False):
    """
    Launch napari client connected to remote server.
    
    Parameters
    ----------
    server_host : str
        Remote server hostname/IP
    state_port : int
        Port for state synchronization
    pixel_port : int
        Port for pixel/video stream
    debug : bool
        Enable debug logging
    """
    # Setup logging - check env var too
    if os.getenv('NAPARI_CUDA_DEBUG', '').lower() in ('1', 'true', 'yes'):
        debug = True
    # Keep root at INFO to avoid third-party DEBUG flood; enable module DEBUG via env
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    # If --debug, enable selective module debugging via env flags
    if debug:
        os.environ['NAPARI_CUDA_CLIENT_DEBUG'] = os.environ.get('NAPARI_CUDA_CLIENT_DEBUG', '1')
        # Keep websockets quiet unless explicitly enabled
        if os.environ.get('NAPARI_CUDA_WEBSOCKETS_DEBUG', '').lower() not in ('1','true','yes','on','dbg','debug'):
            try:
                logging.getLogger('websockets').setLevel(logging.INFO)
                logging.getLogger('websockets.client').setLevel(logging.INFO)
                logging.getLogger('websockets.server').setLevel(logging.INFO)
                logging.getLogger('websockets.protocol').setLevel(logging.INFO)
            except Exception:
                pass
    
    logger.info(f"Launching streaming client for {server_host}")
    
    # Create ProxyViewer (inherits from ViewerModel, no Window created)
    # In the streaming client, keep ProxyViewer offline and forward state
    # via the ClientStreamLoop to avoid dual sockets/drift.
    proxy_viewer = ProxyViewer(
        server_host=server_host,
        server_port=state_port,
        offline=True,
    )
    
    # Create Window manually with our ProxyViewer
    # The Window constructor expects a Viewer, but ViewerModel works too
    window = Window(proxy_viewer, show=False)
    
    # Store window reference in the ProxyViewer
    proxy_viewer.window = window
    
    # Replace the canvas with our streaming canvas
    qt_viewer = window._qt_viewer
    
    # Create streaming canvas
    streaming_canvas = StreamingCanvas(
        proxy_viewer,
        server_host=server_host,
        server_port=pixel_port,
        key_map_handler=getattr(qt_viewer, '_key_map_handler', None),
        parent=qt_viewer
    )
    
    # Replace the default canvas with streaming canvas
    old_canvas = qt_viewer.canvas
    qt_viewer.canvas = streaming_canvas

    # If ProxyViewer connection was deferred, try again now that Qt loop exists
    if getattr(proxy_viewer, '_connection_pending', False):
        try:
            proxy_viewer._connect_to_server()
        except Exception as e:
            logger.debug("Deferred server connect failed (will retry later)", exc_info=True)
    
    # Replace canvas inside the QtViewer welcome overlay stack (current napari layout)
    try:
        ow = qt_viewer._welcome_widget  # QtWidgetOverlay(QStackedWidget)
        # If the old canvas.native is present in the overlay, replace it with the new one
        try:
            for i in range(ow.count()):
                if ow.widget(i) is old_canvas.native:
                    # Remove the old widget and insert our streaming canvas at index 0
                    ow.removeWidget(old_canvas.native)
                    break
        except Exception:
            logger.debug("launcher: failed to remove old canvas from overlay", exc_info=True)
        # Ensure our streaming canvas is the primary widget in the overlay
        try:
            # Avoid duplicate insert if already present
            present = any(ow.widget(i) is streaming_canvas.native for i in range(ow.count()))
            if not present:
                ow.insertWidget(0, streaming_canvas.native)
        except Exception:
            logger.debug("launcher: failed to insert streaming canvas into overlay", exc_info=True)
        # Hide the welcome screen so only the video canvas is visible
        try:
            qt_viewer.set_welcome_visible(False)
        except Exception:
            try:
                ow.set_welcome_visible(False)
            except Exception:
                logger.debug("launcher: failed to hide welcome overlay (overlay widget)", exc_info=True)
    except Exception:
        logger.debug("launcher: overlay manipulation failed", exc_info=True)
    
    # Clean up old canvas widget
    try:
        old_canvas.delete()
    except Exception:
        # Fallback: delete native widget if present
        try:
            old_canvas.native.deleteLater()
        except Exception:
            logger.debug("launcher: cleanup old canvas native failed", exc_info=True)
    
    # Delay showing the window until the first authoritative dims.update arrives.
    try:
        streaming_canvas.defer_window_show(window)
    except Exception:
        logger.debug("launcher: deferred window show failed; showing immediately", exc_info=True)
        window.show()

    # Wire the Home button to remote camera.reset via coordinator and
    # override the 2D/3D toggle to send intents rather than mutate locally.
    mgr = getattr(streaming_canvas, '_manager', None)
    if mgr is not None:
        try:
            rvb = window._qt_viewer.viewerButtons.resetViewButton
            # Avoid duplicate connections by lambdas with default arg
            rvb.clicked.connect(lambda _=False, m=mgr: m.reset_camera(origin='ui'))
        except Exception:
            logger.debug(
                "launcher: failed to bind Home button to loop.reset_camera",
                exc_info=True,
            )

        try:
            ndb = window._qt_viewer.viewerButtons.ndisplayButton

            def _remote_toggle_ndisplay(viewer: ViewerModel) -> None:
                """Toggle 2D/3D mode by forwarding an intent to the server."""

                current = mgr.current_ndisplay()
                if current is None:
                    try:
                        current = viewer.dims.ndisplay
                    except Exception:
                        current = 2
                target = 2 if current == 3 else 3
                if not mgr.view_set_ndisplay(target, origin='ui'):
                    logger.info(
                        "toggle_ndisplay: remote intent rejected (dims not ready or rate limited)"
                    )
                    return

                suppress_token = None
                if hasattr(viewer, '_suppress_forward'):
                    suppress_token = getattr(viewer, '_suppress_forward')
                    setattr(viewer, '_suppress_forward', True)
                try:
                    viewer.dims.ndisplay = target
                except Exception:
                    logger.debug(
                        "toggle_ndisplay: local mirror update failed", exc_info=True
                    )
                finally:
                    if suppress_token is not None:
                        setattr(viewer, '_suppress_forward', suppress_token)

                try:
                    ndb.setChecked(target == 3)
                except Exception:
                    pass

            action_manager.register_action(
                name='napari:toggle_ndisplay',
                command=_remote_toggle_ndisplay,
                description=trans._('Toggle 2D/3D view'),
                keymapprovider=ViewerModel,
            )
        except Exception:
            logger.debug(
                "launcher: failed to override toggle_ndisplay action", exc_info=True
            )

    logger.info("Client launched successfully")
    
    # Run Qt event loop
    napari.run()

    # Cleanup
    if mgr is not None:
        mgr.stop()
    proxy_viewer.close()
    logger.info("Client closed")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='napari-cuda streaming client'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Remote server hostname/IP (default: localhost)'
    )
    
    parser.add_argument(
        '--state-port',
        type=int,
        default=8081,
        help='State synchronization port (default: 8081)'
    )
    
    parser.add_argument(
        '--pixel-port',
        type=int,
        default=8082,
        help='Pixel stream port (default: 8082)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--metrics',
        action='store_true',
        help='Enable client metrics collection (HUD can display decode/submit timings)'
    )
    parser.add_argument(
        '--metrics-out',
        default=None,
        help='Write periodic client metrics CSV to this path'
    )
    parser.add_argument(
        '--metrics-interval',
        type=float,
        default=None,
        help='Client metrics dump interval in seconds (CSV), default 1.0'
    )
    # Latency controls
    parser.add_argument(
        '--latency-ms',
        type=float,
        default=None,
        help='Target VT latency (ms) (sets NAPARI_CUDA_CLIENT_VT_LATENCY_MS)'
    )
    parser.add_argument(
        '--pyav-latency-ms',
        type=float,
        default=None,
        help='Target PyAV latency (ms) used when falling back to CPU decode (sets NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS)'
    )
    # For SSH tunnel setup hint
    parser.add_argument(
        '--tunnel-hint',
        action='store_true',
        help='Show SSH tunnel setup command'
    )
    
    args = parser.parse_args()
    
    if args.tunnel_hint:
        print("SSH Tunnel Setup:")
        print(f"ssh -L {args.state_port}:localhost:{args.state_port} \\")
        print(f"    -L {args.pixel_port}:localhost:{args.pixel_port} \\")
        print(f"    user@{args.host}")
        print("\nThen run client with: --host localhost")
        sys.exit(0)
    
    # Client metrics envs
    if args.metrics:
        os.environ['NAPARI_CUDA_CLIENT_METRICS'] = '1'
    if args.metrics_out:
        os.environ['NAPARI_CUDA_CLIENT_METRICS_OUT'] = str(args.metrics_out)
    if args.metrics_interval is not None:
        os.environ['NAPARI_CUDA_CLIENT_METRICS_INTERVAL'] = str(float(args.metrics_interval))
    # Latency envs (CLI wins over existing)
    if args.latency_ms is not None:
        os.environ['NAPARI_CUDA_CLIENT_VT_LATENCY_MS'] = str(float(args.latency_ms))
    if args.pyav_latency_ms is not None:
        os.environ['NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS'] = str(float(args.pyav_latency_ms))

    # Launch client
    launch_streaming_client(
        server_host=args.host,
        state_port=args.state_port,
        pixel_port=args.pixel_port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
