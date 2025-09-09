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
import os
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

    # Serve dashboard at a stable path; hard-refresh to clear client cache
    base_path = "/dash/"
    app = dash.Dash(__name__, url_base_pathname=base_path)
    app.config.suppress_callback_exceptions = True
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
                .charts { display:grid; grid-template-columns: repeat(5, 1fr); gap:14px; margin-top:16px; }
                .sumrow { margin-top:16px; }
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
    logging.getLogger(__name__).info("Dash dashboard path: %s (port %s)", base_path, port)

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
                uirevision='dash-metrics-v1',  # preserve zoom/pan/legend state across updates
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
        html.Div(id='heartbeat', className='meta'),
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
            dcc.Graph(id='g_convert'),
            dcc.Graph(id='g_bitrate'),
            dcc.Graph(id='g_pack'),
        ], className='charts'),
        html.Div([
            dcc.Graph(id='g_sumcheck'),
        ], className='sumrow'),
        dcc.Store(id='store_total', data=[]),
        dcc.Store(id='store_encode', data=[]),
        dcc.Store(id='store_map', data=[]),
        dcc.Store(id='store_convert', data=[]),
        dcc.Store(id='store_render', data=[]),
        dcc.Store(id='store_blitcpu', data=[]),
        dcc.Store(id='store_copy', data=[]),
        dcc.Store(id='store_pack', data=[]),
        dcc.Store(id='store_bitrate', data=[]),
        dcc.Store(id='store_meta', data={'last_b': 0, 'last_ts': 0.0}),
        dcc.Interval(id='tick', interval=refresh_ms, n_intervals=0),
    ])

    # Simple heartbeat to verify callbacks are running
    @app.callback(Output('heartbeat', 'children'), Input('tick', 'n_intervals'))
    def _hb(n):  # type: ignore
        import time as _t
        return f"Updated: {_t.strftime('%H:%M:%S')}"

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
        Output('store_convert', 'data'),
        Output('store_render', 'data'),
        Output('store_blitcpu', 'data'),
        Output('store_copy', 'data'),
        Output('store_pack', 'data'),
        Output('store_bitrate', 'data'),
        Input('tick', 'n_intervals'),
        State('store_total', 'data'),
        State('store_encode', 'data'),
        State('store_map', 'data'),
        State('store_meta', 'data'),
        State('store_convert', 'data'),
        State('store_render', 'data'),
        State('store_blitcpu', 'data'),
        State('store_copy', 'data'),
        State('store_pack', 'data'),
        State('store_bitrate', 'data'),
    )
    def on_tick(_n, s_total, s_encode, s_map, s_meta, s_convert, s_render, s_blit, s_copy, s_pack, s_bitrate):  # type: ignore
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
        conv = (h.get('napari_cuda_convert_ms') or {}).get('mean_ms', 0.0)
        rend = (h.get('napari_cuda_render_ms') or {}).get('mean_ms', 0.0)
        blitc = (h.get('napari_cuda_capture_blit_cpu_ms') or {}).get('mean_ms', 0.0)
        copy = (h.get('napari_cuda_copy_ms') or {}).get('mean_ms', 0.0)
        pack = (h.get('napari_cuda_pack_ms') or {}).get('mean_ms', 0.0)

        s_total = _append(s_total, total)
        s_encode = _append(s_encode, encode)
        s_map = _append(s_map, map_ms)
        s_convert = _append(s_convert, conv)
        s_render = _append(s_render, rend)
        s_blit = _append(s_blit, blitc)
        s_copy = _append(s_copy, copy)
        s_pack = _append(s_pack, pack)

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
        s_bitrate = _append(s_bitrate, mbps)

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
            s_convert,
            s_render,
            s_blit,
            s_copy,
            s_pack,
            s_bitrate,
        )

    # Note: a previous compatibility callback duplicated Outputs and blocked updates.
    # We removed it to ensure a single source of truth for all tiles and stores.

    @app.callback(
        Output('g_total', 'figure'),
        Output('g_encode', 'figure'),
        Output('g_map', 'figure'),
        Output('g_convert', 'figure'),
        Output('g_bitrate', 'figure'),
        Output('g_sumcheck', 'figure'),
        Output('g_pack', 'figure'),
        Input('store_total', 'data'),
        Input('store_encode', 'data'),
        Input('store_map', 'data'),
        Input('store_convert', 'data'),
        Input('store_render', 'data'),
        Input('store_blitcpu', 'data'),
        Input('store_copy', 'data'),
        Input('store_pack', 'data'),
        Input('store_bitrate', 'data'),
    )
    def update_figs(s_total, s_encode, s_map, s_convert, s_render, s_blit, s_copy, s_pack, s_bitrate):  # type: ignore
        budget = 1000.0 / 60.0  # 16.7 ms target for 60 FPS
        # Include repacking cost in displayed total by summing total + pack series
        def _sum_series(a, b):
            la = len(a or [])
            lb = len(b or [])
            if la == 0:
                return list(b or [])
            if lb == 0:
                return list(a or [])
            n = min(la, lb)
            out = []
            off_a = la - n
            off_b = lb - n
            for i in range(n):
                va = float((a or [])[off_a + i])
                vb = float((b or [])[off_b + i])
                out.append(va + vb)
            return out
        s_total_eff = _sum_series(s_total, s_pack)
        # Build sum check stacked area vs total line
        def make_sum_fig():
            x = list(range(len(s_total_eff or [])))
            comp = [
                ('Render', s_render or [], '#60a5fa'),
                ('BlitCPU', s_blit or [], '#93c5fd'),
                ('Map', s_map or [], '#22d3ee'),
                ('Copy', s_copy or [], '#10b981'),
                ('Convert', s_convert or [], '#f59e0b'),
                ('Encode', s_encode or [], '#16a34a'),
                ('Pack', s_pack or [], '#d946ef'),
            ]
            fig = go.Figure()
            cum = None
            for i, (name, y, color) in enumerate(comp):
                if not y:
                    y = []
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='lines', name=name,
                    line={'width': 1.5, 'color': color},
                    fill='tonexty' if i > 0 else 'none',
                    stackgroup='one',
                ))
            # Total overlay
            fig.add_trace(go.Scatter(
                x=x, y=(s_total_eff or []), mode='lines', name='Total',
                line={'width': 2, 'color': '#ffffff'},
            ))
            fig.update_layout(
                title='Sum vs Total (ms)', template='plotly_dark', height=260,
                margin={'l': 30, 'r': 10, 't': 28, 'b': 24},
                uirevision='dash-metrics-v1',  # preserve UI state across updates
            )
            return fig

        return (
            make_fig(s_total_eff or [], 'Total GPU (ms)', '#4f46e5', budget),
            make_fig(s_encode or [], 'Encode (ms)', '#16a34a'),
            make_fig(s_map or [], 'Map (ms)', '#dc2626'),
            make_fig(s_convert or [], 'Convert (ms)', '#f59e0b'),
            make_fig(s_bitrate or [], 'Bitrate (Mbps)', '#22c55e'),
            make_sum_fig(),
            make_fig(s_pack or [], 'Pack (ms)', '#d946ef'),
        )

    def _run():
        # Dash 2.16+: run_server is deprecated; use run()
        app.run(host=host, port=port, debug=False)

    th = threading.Thread(target=_run, name='dash-dashboard', daemon=True)
    th.start()
    return th
