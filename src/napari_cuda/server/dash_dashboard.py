"""
Dash/Plotly dashboard server for napari-cuda.

Runs a Dash app (Flask/Werkzeug) on a dedicated port. It exposes:
- "/" and "/dashboard/" : interactive dashboard
- "/metrics.json"         : JSON snapshot passthrough from Metrics

This is independent from the asyncio loop used by the main server.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, Dict, Tuple

from .metrics import Metrics


def start_dash_dashboard(host: str, port: int, metrics: Metrics, refresh_ms: int = 1000) -> threading.Thread:
    import dash  # type: ignore
    from dash import dcc, html  # type: ignore
    from dash.dependencies import Input, Output, State  # type: ignore
    import plotly.graph_objs as go  # type: ignore

    # Reduce werkzeug request logs
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    app = dash.Dash(__name__, url_base_pathname='/dash/')
    app.index_string = (
        """
        <!DOCTYPE html>
        <html>
        <head>
            {%metas%}
            <title>napari-cuda Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                body { background:#0b0f19; color:#e5e7eb; font: 14px/1.4 system-ui, sans-serif; }
                .tiles { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; }
                .tile { background:#111827; border-radius:10px; padding:14px; min-height:96px; }
                .label { color:#9ca3af; font-size:12px; }
                .value { font-size:36px; font-weight:700; margin-top:6px; line-height:1.0; }
                .unit { color:#9ca3af; font-size:12px; margin-top:6px; }
                .charts { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-top:16px; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
        </html>
        """
    )
    server = app.server  # Flask

    # Flask route for JSON snapshot
    @server.route('/metrics.json')
    def metrics_json():  # type: ignore[no-redef]
        try:
            return metrics.snapshot()
        except Exception:
            return {
                "version": "v1", "ts": 0.0, "gauges": {}, "counters": {}, "histograms": {}, "derived": {}
            }

    # Client-side state buffers (kept server-side in Dash via dcc.Store)
    def make_fig(y_vals, title, color, budget_ms: float | None = None):
        fig = go.Figure(
            data=[go.Scatter(y=y_vals, mode='lines', line={'color': color, 'width': 2})],
            layout=go.Layout(
                title=title,
                margin={'l': 30, 'r': 10, 't': 28, 'b': 24},
                height=220,
                template='plotly_dark',
            ),
        )
        if budget_ms is not None:
            # Add a horizontal dashed line for target budget
            x1 = max(len(y_vals) - 1, 10)
            fig.add_shape(
                type='line', xref='x', yref='y',
                x0=0, x1=x1, y0=budget_ms, y1=budget_ms,
                line={'color': '#9ca3af', 'dash': 'dash', 'width': 1},
            )
            fig.add_annotation(
                x=1, y=budget_ms, xref='paper', yref='y',
                text=f"budget {budget_ms:.1f} ms", showarrow=False,
                font={'color': '#9ca3af', 'size': 10},
                xanchor='left', yanchor='bottom'
            )
        return fig

    app.layout = html.Div([
        html.H2('napari-cuda Dashboard'),
        html.Div([
            html.Div([
                html.Div('FPS', className='label'),
                html.Div(id='fps', className='value'),
                html.Div('frames/sec', className='unit'),
            ], className='tile'),
            html.Div([
                html.Div('Queue Depth', className='label'),
                html.Div(id='queue', className='value'),
                html.Div('items', className='unit'),
            ], className='tile'),
            html.Div([
                html.Div('Frames', className='label'),
                html.Div(id='frames', className='value'),
                html.Div('count', className='unit'),
            ], className='tile'),
            html.Div([
                html.Div('Bitrate', className='label'),
                html.Div(id='bitrate', className='value'),
                html.Div('Mbps', className='unit'),
            ], className='tile'),
            html.Div([
                html.Div('Dropped', className='label'),
                html.Div(id='dropped', className='value'),
                html.Div('frames', className='unit'),
            ], className='tile'),
            html.Div([
                html.Div('Pixel Clients', className='label'),
                html.Div(id='clients', className='value'),
                html.Div('clients', className='unit'),
            ], className='tile'),
        ], className='tiles'),
        html.Div([
            dcc.Graph(id='g_total'),
            dcc.Graph(id='g_encode'),
            dcc.Graph(id='g_map'),
        ], className='charts'),
        dcc.Store(id='store_total', data=[]),
        dcc.Store(id='store_encode', data=[]),
        dcc.Store(id='store_map', data=[]),
        dcc.Store(id='store_meta', data={'last_b': 0, 'last_ts': 0.0}),
        dcc.Interval(id='tick', interval=refresh_ms, n_intervals=0),
    ])

    @app.callback(
        Output('fps', 'children'),
        Output('queue', 'children'),
        Output('frames', 'children'),
        Output('bitrate', 'children'),
        Output('dropped', 'children'),
        Output('clients', 'children'),
        Output('store_total', 'data'),
        Output('store_encode', 'data'),
        Output('store_map', 'data'),
        Output('store_meta', 'data'),
        Input('tick', 'n_intervals'),
        State('store_total', 'data'),
        State('store_encode', 'data'),
        State('store_map', 'data'),
        State('store_meta', 'data'),
    )
    def on_tick(_n, s_total, s_encode, s_map, s_meta):  # type: ignore
        snap = metrics.snapshot()
        g = snap.get('gauges', {})
        c = snap.get('counters', {})
        h = snap.get('histograms', {})
        d = snap.get('derived', {})

        def _append(store, val):
            arr = list(store or [])
            arr.append(float(val) if val is not None else 0.0)
            if len(arr) > 180:
                arr = arr[-180:]
            return arr

        total = (h.get('napari_cuda_total_ms') or h.get('napari_cuda_end_to_end_ms') or {}).get('mean_ms', 0.0)
        encode = (h.get('napari_cuda_encode_ms') or {}).get('mean_ms', 0.0)
        map_ms = (h.get('napari_cuda_map_ms') or {}).get('mean_ms', 0.0)

        s_total = _append(s_total, total)
        s_encode = _append(s_encode, encode)
        s_map = _append(s_map, map_ms)

        fps = float(d.get('fps', 0.0))
        qd = float(g.get('napari_cuda_frame_queue_depth', g.get('napari_cuda_capture_queue_depth', 0.0)))
        frames = int(c.get('napari_cuda_frames_total', 0))
        bytes_total = float(c.get('napari_cuda_bytes_total', 0.0))
        dropped = int(c.get('napari_cuda_frames_dropped', 0))
        clients = int(float(g.get('napari_cuda_pixel_clients', 0.0)))

        # Compute instantaneous bitrate (Mbps) from byte delta over tick interval
        last_b = float((s_meta or {}).get('last_b', 0.0))
        last_ts = float((s_meta or {}).get('last_ts', 0.0))
        ts = float(snap.get('ts', 0.0))
        dt = max(1e-3, ts - last_ts) if last_ts else 0.0
        delta_b = max(0.0, bytes_total - last_b) if last_ts else 0.0
        mbps = (delta_b * 8.0) / (1e6 * dt) if dt > 0 else 0.0
        s_meta = {'last_b': bytes_total, 'last_ts': ts}

        def fmt_int(v):
            try:
                return f"{int(v):,}"
            except Exception:
                return "—"

        def fmt_float(v, d=1):
            try:
                return f"{float(v):.{d}f}"
            except Exception:
                return "—"

        return (
            fmt_float(fps, 1),
            fmt_int(qd),
            fmt_int(frames),
            fmt_float(mbps, 1),
            fmt_int(dropped),
            fmt_int(clients),
            s_total,
            s_encode,
            s_map,
            s_meta,
        )

    @app.callback(
        Output('g_total', 'figure'),
        Output('g_encode', 'figure'),
        Output('g_map', 'figure'),
        Input('store_total', 'data'),
        Input('store_encode', 'data'),
        Input('store_map', 'data'),
    )
    def update_figs(s_total, s_encode, s_map):  # type: ignore
        budget = 1000.0 / 60.0  # 16.7 ms target for 60 FPS
        return (
            make_fig(s_total or [], 'Total GPU (ms)', '#4f46e5', budget),
            make_fig(s_encode or [], 'Encode (ms)', '#16a34a'),
            make_fig(s_map or [], 'Map (ms)', '#dc2626'),
        )

    def _run():
        # Dash 2.16+: run_server is deprecated; use run()
        app.run(host=host, port=port, debug=False)

    th = threading.Thread(target=_run, name='dash-dashboard', daemon=True)
    th.start()
    return th
