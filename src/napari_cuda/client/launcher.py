"""
Launcher for napari-cuda streaming client.

This creates the client application that connects to a remote napari-cuda server.
"""

import sys
import os
import logging
import argparse

import napari
from napari._qt.qt_viewer import QtViewer
from napari._qt.qt_main_window import Window

from .proxy_viewer import ProxyViewer
from .streaming_canvas import StreamingCanvas

logger = logging.getLogger(__name__)


def launch_streaming_client(server_host='localhost', 
                          state_port=8081,
                          pixel_port=8082,
                          debug=False,
                          vt_smoke: bool = False):
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
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Launching streaming client for {server_host}")
    
    # Create ProxyViewer (inherits from ViewerModel, no Window created)
    proxy_viewer = ProxyViewer(
        server_host=server_host,
        server_port=state_port,
        offline=bool(vt_smoke),
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
        vt_smoke=vt_smoke,
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
    
    # Now show the window
    window.show()
    
    logger.info("Client launched successfully")
    
    # Run Qt event loop
    napari.run()
    
    # Cleanup
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
    # Unified smoke flag (offline; no server)
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run client-side smoke test (offline)'
    )
    parser.add_argument(
        '--preset',
        default=None,
        help='Smoke preset (e.g., 4k60)'
    )
    parser.add_argument(
        '--preencode',
        action='store_true',
        help='Enable preencode smoke mode (encode once, then replay)'
    )
    parser.add_argument(
        '--pre-mb',
        type=int,
        default=None,
        help='Preencode memory cap in MB (0 = unlimited)'
    )
    parser.add_argument(
        '--pre-path',
        default=None,
        help='Directory path for disk-backed preencode cache'
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
        help='Target VT latency (ms) for ARRIVAL mode (sets NAPARI_CUDA_CLIENT_VT_LATENCY_MS)'
    )
    parser.add_argument(
        '--pyav-latency-ms',
        type=float,
        default=None,
        help='Target PyAV latency (ms) used when falling back to CPU decode (sets NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS)'
    )
    # Jitter presets
    parser.add_argument(
        '--jitter',
        action='store_true',
        help='Enable jitter with a mild default preset (overridden by explicit envs)'
    )
    parser.add_argument(
        '--jitter-preset',
        default=None,
        help='Jitter preset name (off,mild,heavy,cap4mbps,wifi30). Explicit envs override preset.'
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
    
    # Apply smoke-related CLI envs before launch
    if args.smoke:
        os.environ.setdefault('NAPARI_CUDA_SMOKE', '1')
    if args.preset:
        os.environ['NAPARI_CUDA_SMOKE_PRESET'] = str(args.preset)
    if args.preencode:
        os.environ['NAPARI_CUDA_SMOKE_PREENCODE'] = '1'
    if args.pre_mb is not None:
        os.environ['NAPARI_CUDA_SMOKE_PRE_MB'] = str(int(args.pre_mb))
    if args.pre_path:
        os.environ['NAPARI_CUDA_SMOKE_PRE_PATH'] = str(args.pre_path)
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

    # Apply jitter presets (user envs win)
    try:
        if args.jitter and not args.jitter_preset:
            os.environ.setdefault('NAPARI_CUDA_JIT_PRESET', 'mild')
        if args.jitter_preset:
            # Set a marker env so downstream components can apply if launched differently
            os.environ['NAPARI_CUDA_JIT_PRESET'] = str(args.jitter_preset)
        # Apply now in this process for consistency
        from .streaming.pipelines.jitter_presets import apply_preset as _apply_jit
        preset_name = os.getenv('NAPARI_CUDA_JIT_PRESET')
        if preset_name:
            try:
                applied = _apply_jit(preset_name)
                if applied:
                    logger.info("launcher: applied jitter preset='%s' (set %d vars; existing envs preserved)", preset_name, len(applied))
            except Exception:
                logger.exception("launcher: failed to apply jitter preset '%s'", preset_name)
    except Exception:
        logger.exception('launcher: jitter preset handling failed')

    # Launch client
    smoke = bool(args.smoke)
    launch_streaming_client(
        server_host=args.host,
        state_port=args.state_port,
        pixel_port=args.pixel_port,
        debug=args.debug,
        vt_smoke=smoke
    )


if __name__ == "__main__":
    main()
