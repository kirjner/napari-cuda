from __future__ import annotations

"""Client presenter façade implementation.

Owns draw-event wiring, optional display-loop management, and the HUD overlay
for the streaming client. The façade keeps the Qt-specific plumbing out of
``StreamingCanvas`` so presenter changes can land without repeatedly patching
Qt code.
"""

import contextlib
import logging
import os
import sys
import time
import weakref
from typing import Any, Callable, Optional, TYPE_CHECKING

from qtpy import QtCore, QtWidgets

if TYPE_CHECKING:  # pragma: no cover - narrow compile-time imports only
    from qtpy.QtCore import QObject

    from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop
    from napari_cuda.client.streaming.config import ClientConfig
    from napari_cuda.client.streaming.presenter import FixedLatencyPresenter
    from napari_cuda.client.streaming.renderer import GLRenderer


logger = logging.getLogger(__name__)


class PresenterFacade:
    """Coordinate presenter-side wiring for the streaming client."""

    def __init__(self) -> None:
        # Primary collaborators
        self._scene_canvas: Any = None
        self._scene_native: Any = None
        self._loop: Optional["ClientStreamLoop"] = None
        self._presenter: Optional["FixedLatencyPresenter"] = None
        self._renderer: Optional["GLRenderer"] = None
        self._client_cfg: Optional["ClientConfig"] = None

        # GUI wiring
        self._draw_events: Any = None
        self._enable_dims_play: Optional[Callable[[Any], None]] = None
        self._legacy_draw: Optional[Callable[[Any], None]] = None
        self._loop_draw_wrapper: Optional[Callable[[Any], None]] = None
        self._draw_bound: bool = False

        # Display loop & HUD
        self._use_display_loop: bool = False
        self._display_loop: Optional["QObject"] = None
        self._hud_enabled: bool = False
        self._fps_label: Optional[QtWidgets.QLabel] = None
        self._fps_timer: Optional[QtCore.QTimer] = None
        self._hud_prev_time: float = 0.0
        self._hud_prev_submit: dict[str, int] = {'vt': 0, 'pyav': 0}
        self._hud_prev_out: dict[str, int] = {'vt': 0, 'pyav': 0}
        self._hud_prev_preview: dict[str, int] = {'vt': 0, 'pyav': 0}
        self._hud_prev_jit_time: float = 0.0
        self._hud_prev_jitter: dict[str, int] = {
            'delivered': 0,
            'dropped': 0,
            'reordered': 0,
            'duplicated': 0,
        }

        # Misc state
        self._viewer_ref: Optional["weakref.ReferenceType[Any]"] = None
        self._last_dims_payload: Optional[dict[str, Any]] = None
        self._intent_dispatcher: Optional[Callable[[str, dict[str, Any]], None]] = None

    # ------------------------------------------------------------------ lifecycle
    def start_presenting(
        self,
        *,
        scene_canvas: Any,
        loop: "ClientStreamLoop",
        presenter: "FixedLatencyPresenter",
        renderer: "GLRenderer",
        client_cfg: Optional["ClientConfig"],
        use_display_loop: bool,
    ) -> None:
        """Record base collaborators and config."""

        self._scene_canvas = scene_canvas
        self._scene_native = getattr(scene_canvas, 'native', None)
        self._loop = loop
        self._presenter = presenter
        self._renderer = renderer
        self._client_cfg = client_cfg
        self._use_display_loop = bool(use_display_loop)
        logger.debug(
            "PresenterFacade.start_presenting: display_loop=%s",
            self._use_display_loop,
        )
        self._compute_hud_enabled()

    def bind_canvas(
        self,
        *,
        enable_dims_play: Callable[[Any], None],
        legacy_draw: Callable[[Any], None],
    ) -> None:
        """Attach draw handlers and HUD once the canvas is ready."""

        if self._scene_canvas is None or self._loop is None:
            logger.debug("PresenterFacade.bind_canvas skipped (scene/loop missing)")
            return
        draw_events = getattr(self._scene_canvas, 'events', None)
        if draw_events is None or not hasattr(draw_events, 'draw'):
            logger.warning('PresenterFacade.bind_canvas: scene has no draw events')
            return
        self._draw_events = draw_events.draw
        self._enable_dims_play = enable_dims_play
        self._legacy_draw = legacy_draw

        with contextlib.suppress(ValueError, RuntimeError):
            self._draw_events.disconnect(legacy_draw)
        with contextlib.suppress(ValueError, RuntimeError):
            self._draw_events.disconnect(self._loop.draw)
        self._draw_events.connect(enable_dims_play, position='first')
        def _loop_draw_wrapper(event: Any) -> None:
            if self._loop is not None:
                self._loop.draw()

        self._loop_draw_wrapper = _loop_draw_wrapper
        self._draw_events.connect(_loop_draw_wrapper, position='last')
        self._draw_bound = True
        logger.info('PresenterFacade: draw wiring bound to ClientStreamLoop.draw')

        self._ensure_display_loop()
        if self._hud_enabled:
            self._init_fps_hud()

    def shutdown(self) -> None:
        """Release Qt resources and detach handlers."""

        if self._draw_bound and self._draw_events is not None:
            with contextlib.suppress(ValueError, RuntimeError):
                if self._enable_dims_play is not None:
                    self._draw_events.disconnect(self._enable_dims_play)
            with contextlib.suppress(ValueError, RuntimeError):
                if self._loop_draw_wrapper is not None:
                    self._draw_events.disconnect(self._loop_draw_wrapper)
            if self._legacy_draw is not None:
                with contextlib.suppress(Exception):
                    self._draw_events.connect(self._legacy_draw, position='last')
        self._draw_bound = False
        self._draw_events = None
        self._enable_dims_play = None
        self._legacy_draw = None
        self._loop_draw_wrapper = None

        if self._display_loop is not None and hasattr(self._display_loop, 'stop'):
            try:
                self._display_loop.stop()
            except Exception:
                logger.debug('PresenterFacade: display loop stop failed', exc_info=True)
        self._display_loop = None

        if self._fps_timer is not None:
            self._fps_timer.stop()
            self._fps_timer.deleteLater()
        self._fps_timer = None
        if self._fps_label is not None:
            self._fps_label.deleteLater()
        self._fps_label = None
        self._hud_prev_time = 0.0
        self._hud_prev_jit_time = 0.0
        self._hud_prev_submit = {'vt': 0, 'pyav': 0}
        self._hud_prev_out = {'vt': 0, 'pyav': 0}
        self._hud_prev_preview = {'vt': 0, 'pyav': 0}
        self._hud_prev_jitter = {
            'delivered': 0,
            'dropped': 0,
            'reordered': 0,
            'duplicated': 0,
        }

        self._scene_canvas = None
        self._scene_native = None
        self._loop = None
        self._presenter = None
        self._renderer = None
        self._client_cfg = None
        self._viewer_ref = None
        self._last_dims_payload = None
        self._intent_dispatcher = None

    # ----------------------------------------------------------------- façade API
    def apply_dims_update(self, payload: dict[str, Any]) -> None:
        """Cache dims payloads and forward to any registered dispatcher."""

        self._last_dims_payload = dict(payload)
        if self._intent_dispatcher is None:
            return
        try:
            self._intent_dispatcher('dims', payload)
        except Exception:
            logger.exception('PresenterFacade intent dispatcher failed')

    def set_viewer_mirror(self, viewer: object) -> None:
        """Record a weakref to the viewer mirror for future mirroring work."""

        try:
            self._viewer_ref = weakref.ref(viewer)  # type: ignore[arg-type]
        except TypeError:
            self._viewer_ref = None
            logger.debug('PresenterFacade: viewer does not support weakref')
        else:
            logger.debug('PresenterFacade: viewer mirror registered')

    def set_intent_dispatcher(
        self, dispatcher: Optional[Callable[[str, Any], None]]
    ) -> Optional[Callable[[str, Any], None]]:
        prev = self._intent_dispatcher
        self._intent_dispatcher = dispatcher
        logger.debug('PresenterFacade: intent dispatcher set=%s', bool(dispatcher))
        return prev

    def cached_dims_payload(self) -> Optional[dict[str, Any]]:
        return dict(self._last_dims_payload) if self._last_dims_payload is not None else None

    def current_viewer(self) -> Optional[object]:
        return self._viewer_ref() if self._viewer_ref is not None else None

    def apply_layer_update(self, message: Any) -> None:
        if self._intent_dispatcher is None:
            return
        try:
            self._intent_dispatcher('layer-update', message)
        except Exception:
            logger.exception('PresenterFacade layer dispatcher failed')

    # ------------------------------------------------------------------ internals
    def _compute_hud_enabled(self) -> None:
        hud_env = os.getenv('NAPARI_CUDA_FPS_HUD') or ''
        view_hud_env = (os.getenv('NAPARI_CUDA_VIEW_HUD') or '0').lower()
        metrics_env = (os.getenv('NAPARI_CUDA_CLIENT_METRICS') or '0').lower()
        self._hud_enabled = (
            hud_env == '1'
            or view_hud_env in ('1', 'true', 'yes', 'on')
            or metrics_env in ('1', 'true', 'yes', 'on')
        )

    def _ensure_display_loop(self) -> None:
        if not self._use_display_loop or self._display_loop is not None:
            return
        if self._scene_canvas is None:
            return
        fps_val: Optional[float] = None
        if self._client_cfg is not None and hasattr(self._client_cfg, 'draw_fps'):
            fps_val = float(self._client_cfg.draw_fps)
        try:
            from napari_cuda.client.streaming.display_loop import DisplayLoop
        except Exception:
            logger.exception('PresenterFacade: DisplayLoop import failed')
            return
        self._display_loop = DisplayLoop(
            scene_canvas=self._scene_canvas,
            fps=fps_val,
            prefer_vispy=True,
        )
        try:
            self._display_loop.start()
        except Exception:
            logger.exception('PresenterFacade: DisplayLoop start failed')
            self._display_loop = None
        else:
            logger.info('PresenterFacade: DisplayLoop enabled (fps=%s)', fps_val)

    # ---------------------------- HUD helpers ------------------------
    def _init_fps_hud(self) -> None:
        if self._scene_native is None:
            logger.debug('PresenterFacade: no native widget; skipping HUD init')
            return
        label = QtWidgets.QLabel(self._scene_native)
        label.setObjectName('napari_cuda_fps_hud')
        label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        label.setStyleSheet(
            "#napari_cuda_fps_hud {"
            "  background: rgba(0,0,0,120);"
            "  color: #fff;"
            "  border-radius: 4px;"
            "  padding: 3px 6px;"
            "  font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;"
            "  font-size: 11px;"
            "}"
        )
        label.setText('fps: --')
        label.setWordWrap(False)
        label.adjustSize()
        label.setMinimumWidth(260)
        label.move(8, 8)
        label.raise_()
        label.show()
        self._fps_label = label
        self._hud_prev_time = 0.0
        self._hud_prev_submit = {'vt': 0, 'pyav': 0}
        self._hud_prev_out = {'vt': 0, 'pyav': 0}
        self._hud_prev_preview = {'vt': 0, 'pyav': 0}
        self._hud_prev_jit_time = 0.0
        self._hud_prev_jitter = {
            'delivered': 0,
            'dropped': 0,
            'reordered': 0,
            'duplicated': 0,
        }
        timer = QtCore.QTimer(self._scene_native)
        timer.setTimerType(QtCore.Qt.PreciseTimer)
        timer.setInterval(1000)
        timer.timeout.connect(self._update_fps_hud)
        timer.start()
        self._fps_timer = timer
        logger.info('PresenterFacade: FPS HUD initialised')

    def _update_fps_hud(self) -> None:
        if self._loop is None or self._presenter is None or self._fps_label is None:
            return
        stats = self._presenter.stats()
        sub = stats.get('submit', {})
        out = stats.get('out', {})
        prev = stats.get('preview', {})
        vt_sub = int(sub.get('vt', 0))
        py_sub = int(sub.get('pyav', 0))
        vt_out = int(out.get('vt', 0))
        py_out = int(out.get('pyav', 0))
        vt_prev = int(prev.get('vt', 0))
        py_prev = int(prev.get('pyav', 0))
        now = time.time()
        last = self._hud_prev_time
        if last <= 0.0:
            self._hud_prev_time = now
            self._hud_prev_submit = {'vt': vt_sub, 'pyav': py_sub}
            self._hud_prev_out = {'vt': vt_out, 'pyav': py_out}
            self._hud_prev_preview = {'vt': vt_prev, 'pyav': py_prev}
            return
        dt = max(1e-3, now - last)
        fps_sub_vt = (vt_sub - self._hud_prev_submit['vt']) / dt
        fps_sub_py = (py_sub - self._hud_prev_submit['pyav']) / dt
        fps_out_vt = (vt_out - self._hud_prev_out['vt']) / dt
        fps_out_py = (py_out - self._hud_prev_out['pyav']) / dt
        fps_prev_vt = (vt_prev - self._hud_prev_preview['vt']) / dt
        fps_prev_py = (py_prev - self._hud_prev_preview['pyav']) / dt
        self._hud_prev_time = now
        self._hud_prev_submit = {'vt': vt_sub, 'pyav': py_sub}
        self._hud_prev_out = {'vt': vt_out, 'pyav': py_out}
        self._hud_prev_preview = {'vt': vt_prev, 'pyav': py_prev}

        active = getattr(self._loop._source_mux, 'active', None)
        active_str = getattr(active, 'value', str(active)) if active is not None else '-'
        lat_ms = int(stats.get('latency_ms', 0) or 0)
        sub_fps = fps_sub_vt if active_str == 'vt' else fps_sub_py
        out_fps = fps_out_vt if active_str == 'vt' else fps_out_py
        prev_fps = fps_prev_vt if active_str == 'vt' else fps_prev_py
        buf = stats.get('buf', {}) or {}
        buf_vt = int(buf.get('vt', 0))
        buf_py = int(buf.get('pyav', 0))

        state = self._loop._loop_state  # type: ignore[attr-defined]
        vt_pipeline = state.vt_pipeline
        vt_q_len = None
        if vt_pipeline is not None and hasattr(vt_pipeline, 'counts'):
            with contextlib.suppress(Exception):
                counts = vt_pipeline.counts()
                if counts is not None:
                    vt_q_len = int(counts[2])
        q_vt = 0
        if vt_pipeline is not None and hasattr(vt_pipeline, 'qsize'):
            with contextlib.suppress(Exception):
                q_vt = int(vt_pipeline.qsize())
        pyav_pipeline = state.pyav_pipeline
        q_py = 0
        if pyav_pipeline is not None and hasattr(pyav_pipeline, 'qsize'):
            with contextlib.suppress(Exception):
                q_py = int(pyav_pipeline.qsize())

        metrics = state.metrics
        dec_py_ms = vt_dec_ms = vt_submit_ms = None
        render_vt_ms = render_pyav_ms = None
        draw_mean_ms = draw_last_ms = None
        present_fps = None
        jit_q = jit_deliv_rate = jit_drop_rate = jit_sched_mean = None
        late_last_ms = late_mean_ms = late_p90_ms = None
        if metrics is not None and hasattr(metrics, 'snapshot'):
            with contextlib.suppress(Exception):
                snap = metrics.snapshot()
                h = snap.get('histograms', {}) or {}
                dec_py_ms = (h.get('napari_cuda_client_pyav_decode_ms') or {}).get('mean_ms')
                vt_dec_ms = (h.get('napari_cuda_client_vt_decode_ms') or {}).get('mean_ms')
                vt_submit_ms = (h.get('napari_cuda_client_vt_submit_ms') or {}).get('mean_ms')
                render_vt_ms = (h.get('napari_cuda_client_render_vt_ms') or {}).get('mean_ms')
                render_pyav_ms = (h.get('napari_cuda_client_render_pyav_ms') or {}).get('mean_ms')
                d_hist = h.get('napari_cuda_client_draw_interval_ms') or {}
                draw_mean_ms = d_hist.get('mean_ms')
                draw_last_ms = d_hist.get('last_ms')
                g = snap.get('gauges', {}) or {}
                c = snap.get('counters', {}) or {}
                jit_q = g.get('napari_cuda_jit_qdepth')
                now_j = time.time()
                prev_t = self._hud_prev_jit_time
                if prev_t <= 0.0:
                    self._hud_prev_jit_time = now_j
                    self._hud_prev_jitter['delivered'] = int(c.get('napari_cuda_jit_delivered', 0) or 0)
                    self._hud_prev_jitter['dropped'] = int(c.get('napari_cuda_jit_dropped', 0) or 0)
                    self._hud_prev_jitter['reordered'] = int(c.get('napari_cuda_jit_reordered', 0) or 0)
                    self._hud_prev_jitter['duplicated'] = int(c.get('napari_cuda_jit_duplicated', 0) or 0)
                else:
                    dtj = max(1e-3, now_j - prev_t)
                    delivered = int(c.get('napari_cuda_jit_delivered', 0) or 0)
                    dropped = int(c.get('napari_cuda_jit_dropped', 0) or 0)
                    jit_deliv_rate = max(0.0, (delivered - self._hud_prev_jitter['delivered']) / dtj)
                    jit_drop_rate = max(0.0, (dropped - self._hud_prev_jitter['dropped']) / dtj)
                    self._hud_prev_jit_time = now_j
                    self._hud_prev_jitter['delivered'] = delivered
                    self._hud_prev_jitter['dropped'] = dropped
                    self._hud_prev_jitter['reordered'] = int(c.get('napari_cuda_jit_reordered', 0) or 0)
                    self._hud_prev_jitter['duplicated'] = int(c.get('napari_cuda_jit_duplicated', 0) or 0)
                jit_sched = h.get('napari_cuda_jit_sched_delay_ms') or {}
                jit_sched_mean = jit_sched.get('mean_ms')
                late_hist = h.get('napari_cuda_client_present_lateness_ms') or {}
                late_last_ms = late_hist.get('last_ms')
                late_mean_ms = late_hist.get('mean_ms')
                late_p90_ms = late_hist.get('p90_ms')
                gauges = snap.get('gauges', {}) or {}
                present_fps = gauges.get('napari_cuda_client_presented_fps')

        txt = (
            f"src:{active_str} lat:{lat_ms}ms "
            f"submit:{sub_fps:.1f} draw:{out_fps:.1f} prev:{prev_fps:.1f}"
        )
        txt += f"\nqueues: presenter[vt:{buf_vt} py:{buf_py}] pipeline[vt:{q_vt} py:{q_py}]"
        if vt_q_len is not None:
            txt += f" vt_count:{vt_q_len}"
        loop_bits: list[str] = []
        if isinstance(dec_py_ms, (int, float)):
            loop_bits.append(f"pyav_dec:{dec_py_ms:.2f}ms")
        if isinstance(vt_dec_ms, (int, float)):
            loop_bits.append(f"vt_dec:{vt_dec_ms:.2f}ms")
        if isinstance(vt_submit_ms, (int, float)):
            loop_bits.append(f"vt_submit:{vt_submit_ms:.2f}ms")
        if isinstance(render_vt_ms, (int, float)):
            loop_bits.append(f"vt_render:{render_vt_ms:.2f}ms")
        if isinstance(render_pyav_ms, (int, float)):
            loop_bits.append(f"py_render:{render_pyav_ms:.2f}ms")
        if isinstance(draw_last_ms, (int, float)) and draw_last_ms > 0:
            loop_bits.append(f"loop:{draw_last_ms:.2f}ms")
        if (os.getenv('NAPARI_CUDA_HUD_LOOP_SHOW_MEAN') or '0').lower() in ('1', 'true', 'yes'):
            if isinstance(draw_mean_ms, (int, float)) and draw_mean_ms > 0:
                loop_bits.append(f"mean:{draw_mean_ms:.2f}ms")
        if isinstance(late_last_ms, (int, float)) and isinstance(late_mean_ms, (int, float)):
            loop_bits.append(f"late:{late_last_ms:.1f}/{late_mean_ms:.1f}ms")
        if isinstance(late_p90_ms, (int, float)):
            loop_bits.append(f"p90:{late_p90_ms:.1f}ms")
        if isinstance(present_fps, (int, float)):
            loop_bits.append(f"present:{present_fps:.1f}fps")
        mem_env = (os.getenv('NAPARI_CUDA_HUD_SHOW_MEM') or '0').lower()
        if mem_env in ('1', 'true', 'yes'):
            cur_mb = max_mb = None
            with contextlib.suppress(Exception):
                import psutil  # type: ignore

                cur_mb = float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
            with contextlib.suppress(Exception):
                import resource

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == 'darwin':
                    max_mb = float(rss) / (1024.0 * 1024.0)
                else:
                    max_mb = float(rss) / 1024.0
            if isinstance(cur_mb, float) and isinstance(max_mb, float):
                loop_bits.append(f"mem:{cur_mb:.0f}/{max_mb:.0f}MB")
            elif isinstance(cur_mb, float):
                loop_bits.append(f"mem:{cur_mb:.0f}MB")
            elif isinstance(max_mb, float):
                loop_bits.append(f"mem_max:{max_mb:.0f}MB")
        if loop_bits:
            txt += "\n" + "  ".join(loop_bits)

        view_hud_enabled = (os.getenv('NAPARI_CUDA_VIEW_HUD') or '0').lower() in ('1', 'true', 'yes', 'on')
        if view_hud_enabled:
            view_txt = self._build_view_hud_text()
            if view_txt:
                txt += "\n" + view_txt

        show_jitter = any(val is not None for val in (jit_q, jit_deliv_rate, jit_drop_rate, jit_sched_mean))
        if show_jitter:
            parts: list[str] = []
            if isinstance(jit_q, (int, float)):
                parts.append(f"jitq:{int(jit_q)}")
            if isinstance(jit_deliv_rate, float):
                parts.append(f"jit:{jit_deliv_rate:.1f}/s")
            if isinstance(jit_drop_rate, float) and jit_drop_rate > 0:
                parts.append(f"drop:{jit_drop_rate:.1f}/s")
            if isinstance(jit_sched_mean, (int, float)):
                parts.append(f"sched:{jit_sched_mean:.1f}ms")
            if parts:
                txt += "\n" + "  ".join(parts)

        self._fps_label.setText(txt)
        self._fps_label.adjustSize()

    def _build_view_hud_text(self) -> Optional[str]:
        if self._loop is None or not hasattr(self._loop, 'view_hud_snapshot'):
            return None
        with contextlib.suppress(Exception):
            vs = self._loop.view_hud_snapshot()  # type: ignore[attr-defined]
        if not isinstance(vs, dict) or not vs:
            return None
        vol_flag = 'vol' if bool(vs.get('volume')) else 'img'
        ndisp = vs.get('ndisplay')
        vm = '3D' if bool(vs.get('vol_mode')) else '2D'
        zb = vs.get('zoom_base')
        zf = vs.get('last_zoom_factor')
        pdx = vs.get('last_pan_dx')
        pdy = vs.get('last_pan_dy')
        line_v1 = (
            f"view:{vol_flag}/{vm} ndisp:{ndisp} zbase:{zb} zfac:{zf:.3f}"
            if isinstance(zf, (int, float))
            else f"view:{vol_flag}/{vm} ndisp:{ndisp} zbase:{zb}"
        )
        if isinstance(pdx, (int, float)) and isinstance(pdy, (int, float)):
            line_v1 += f" pan:({pdx:.1f},{pdy:.1f})"
        rbits: list[str] = []
        rmode = vs.get('render_mode')
        if rmode:
            rbits.append(f"mode:{rmode}")
        clo = vs.get('clim_lo')
        chi = vs.get('clim_hi')
        if isinstance(clo, (int, float)) and isinstance(chi, (int, float)):
            rbits.append(f"clim:[{clo:.2f},{chi:.2f}]")
        cmap = vs.get('colormap')
        if cmap:
            rbits.append(f"map:{cmap}")
        opa = vs.get('opacity')
        if isinstance(opa, (int, float)):
            rbits.append(f"opac:{opa:.2f}")
        sst = vs.get('sample_step')
        if isinstance(sst, (int, float)):
            rbits.append(f"step:{sst:.2f}")
        line_v2 = "render: " + " ".join(rbits) if rbits else "render: -"
        msbits: list[str] = []
        ms_pol = vs.get('ms_policy')
        if ms_pol:
            msbits.append(f"pol:{ms_pol}")
        ms_lvl = vs.get('ms_level')
        ms_n = vs.get('ms_levels')
        if isinstance(ms_lvl, int) and isinstance(ms_n, int) and ms_n > 0:
            display_lvl = max(0, ms_lvl) + 1
            msbits.append(f"lvl:{display_lvl}/{ms_n}")
        ms_path = vs.get('ms_path')
        if isinstance(ms_path, str) and ms_path:
            msbits.append(f"path:{ms_path}")
        if msbits:
            line_v2 += "  ms: " + " ".join(msbits)
        line_v3 = None
        aw = vs.get('last_zoom_widget_px')
        av = vs.get('last_zoom_video_px')
        asv = vs.get('last_zoom_anchor_px')
        if (
            isinstance(aw, (list, tuple))
            and isinstance(av, (list, tuple))
            and isinstance(asv, (list, tuple))
            and len(aw) >= 2
            and len(av) >= 2
            and len(asv) >= 2
        ):
            line_v3 = (
                f"anchor: w:({aw[0]:.1f},{aw[1]:.1f}) "
                f"v:({av[0]:.1f},{av[1]:.1f}) s:({asv[0]:.1f},{asv[1]:.1f})"
            )
        out = line_v1 + "\n" + line_v2
        if line_v3:
            out += "\n" + line_v3
        return out


__all__ = ['PresenterFacade']
