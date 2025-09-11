from __future__ import annotations

import logging
import os
import queue
import time
from threading import Thread
from typing import Optional

import numpy as np
from qtpy import QtCore

from napari_cuda.client.streaming.presenter import FixedLatencyPresenter, SourceMux
from napari_cuda.client.streaming.receiver import PixelReceiver, Packet
from napari_cuda.client.streaming.state import StateChannel
from napari_cuda.client.streaming.types import Source, SubmittedFrame, TimestampMode
from napari_cuda.client.streaming.renderer import GLRenderer
from napari_cuda.client.streaming.decoders.pyav import PyAVDecoder
from napari_cuda.client.streaming.decoders.vt import VTLiveDecoder
from napari_cuda.codec.avcc import (
    annexb_to_avcc,
    is_annexb,
    split_annexb,
    split_avcc_by_len,
    build_avcc,
    find_sps_pps,
)
from napari_cuda.codec.h264_encoder import H264Encoder, EncoderConfig
from napari_cuda.utils.env import env_float, env_str
from napari_cuda.client.streaming.smoke.generators import make_generator
from napari_cuda.client.streaming.smoke.submit import submit_vt, submit_pyav

logger = logging.getLogger(__name__)


class StreamManager:
    """Orchestrates receiver, state, decoders, presenter, and renderer.

    This is a slim re-hosting of the logic previously embedded in
    StreamingCanvas, with behavior unchanged.
    """

    def __init__(
        self,
        scene_canvas,
        server_host: str,
        server_port: int,
        state_port: int,
        vt_latency_s: float,
        vt_buffer_limit: int,
        pyav_latency_s: Optional[float] = None,
        vt_ts_mode: str = 'arrival',
        stream_format: str = 'avcc',
        vt_backlog_trigger: int = 16,
        pyav_backlog_trigger: int = 16,
        vt_smoke: bool = False,
    ) -> None:
        self._scene_canvas = scene_canvas
        self.server_host = server_host
        self.server_port = int(server_port)
        self.state_port = int(state_port)
        self._stream_format = stream_format
        self._stream_format_set = False

        # Presenter + source mux
        ts_mode = TimestampMode.ARRIVAL if (vt_ts_mode or 'arrival') == 'arrival' else TimestampMode.SERVER
        # Preview guard small to avoid choppiness in SERVER mode
        self._presenter = FixedLatencyPresenter(
            latency_s=float(vt_latency_s),
            buffer_limit=int(vt_buffer_limit),
            ts_mode=ts_mode,
            preview_guard_s=1.0/60.0,
        )
        self._source_mux = SourceMux(Source.PYAV)
        self._vt_latency_s = float(vt_latency_s)
        self._pyav_latency_s = float(pyav_latency_s) if pyav_latency_s is not None else max(0.06, float(vt_latency_s))
        # Default to PyAV latency until VT is proven ready
        try:
            self._presenter.set_latency(self._pyav_latency_s)
        except Exception:
            pass
        # Startup warmup (arrival mode): temporarily increase latency then ramp down
        # Auto-sized to roughly exceed one frame interval (assume 60 Hz if FPS unknown)
        self._warmup_ms_override = env_str('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MS', None)
        self._warmup_window_s = env_float('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_WINDOW_S', 0.75)
        self._warmup_margin_ms = env_float('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MARGIN_MS', 2.0)
        self._warmup_max_ms = env_float('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MAX_MS', 24.0)
        self._warmup_until: float = 0.0
        self._warmup_extra_active_s: float = 0.0
        self._fps: Optional[float] = None

        # Renderer
        self._renderer = GLRenderer(self._scene_canvas)

        # Decoders
        self.decoder: Optional[PyAVDecoder] = None
        self._vt_decoder: Optional[VTLiveDecoder] = None
        self._vt_backend: Optional[str] = None
        self._vt_cfg_key = None
        self._vt_wait_keyframe: bool = False
        self._vt_gate_lift_time: float = 0.0
        self._vt_ts_offset: Optional[float] = None
        self._vt_errors = 0
        # Small negative bias so frames tend to be due slightly earlier in SERVER mode
        self._server_bias_s = env_float('NAPARI_CUDA_SERVER_TS_BIAS_MS', 5.0) / 1000.0
        # avcC nal length size (default 4 if unknown)
        self._nal_length_size: int = 4

        # Queues + workers
        self._vt_in_q: "queue.Queue[tuple[bytes, float | None]]" = queue.Queue(maxsize=64)
        self._vt_backlog_trigger = int(vt_backlog_trigger)
        self._vt_enqueued = 0
        self._pyav_in_q: "queue.Queue[tuple[bytes, float | None]]" = queue.Queue(maxsize=64)
        self._pyav_backlog_trigger = int(pyav_backlog_trigger)
        self._pyav_enqueued = 0
        self._pyav_wait_keyframe: bool = False

        # Last VT payload for redraw fallback (offline VT-source or VT decode)
        self._last_vt_payload = None  # type: ignore[var-annotated]
        self._last_vt_persistent = False
        # Last PyAV RGB frame for redraw fallback (avoid flicker when no new frame ready)
        self._last_pyav_frame = None  # type: ignore[var-annotated]

        # Frame queue for renderer (latest-wins)
        self._frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=3)

        # Receiver/state
        self._stream_seen_keyframe = False
        self._state_channel: Optional[StateChannel] = None
        self._receiver: Optional[PixelReceiver] = None
        self._threads: list[Thread] = []
        self._vt_smoke = bool(vt_smoke)

        # Stats logging (1 Hz) controlled by env NAPARI_CUDA_VT_STATS
        lvl_env = (env_str('NAPARI_CUDA_VT_STATS', '') or '').lower()
        if lvl_env in ('1', 'true', 'yes', 'info'):
            self._stats_level = logging.INFO
        elif lvl_env in ('debug', 'dbg'):
            self._stats_level = logging.DEBUG
        else:
            self._stats_level = None
        self._last_stats_time: float = 0.0
        self._stats_timer = None
        # Adaptive VT scheduling state
        self._preview_streak: int = 0
        self._arrival_fallback: bool = False
        self._relearn_logged: bool = False
        self._arrival_logged: bool = False
        # Debounce duplicate video_config
        self._last_vcfg_key = None
        # Stream continuity tracking
        self._last_seq: Optional[int] = None
        self._disco_gated: bool = False
        self._last_disco_log: float = 0.0
        self._last_key_logged: Optional[int] = None
        self._keyframes_seen: int = 0

    def start(self) -> None:
        # State channel thread (disabled in offline VT smoke mode)
        def _state_worker() -> None:
            def _on_video_config(data: dict) -> None:
                try:
                    width = int(data.get('width') or 0)
                    height = int(data.get('height') or 0)
                    fps = float(data.get('fps') or 0.0)
                    if fps > 0:
                        self._fps = fps
                    fmt = (data.get('format') or '').lower() or 'avcc'
                    self._stream_format = 'annexb' if fmt.startswith('annex') else 'avcc'
                    self._stream_format_set = True
                    avcc_b64 = data.get('data')
                    if width > 0 and height > 0 and avcc_b64:
                        self._init_vt_from_avcc(avcc_b64, width, height)
                except Exception:
                    logger.debug("video_config handling failed", exc_info=True)

            self._state_channel = StateChannel(self.server_host, self.state_port, on_video_config=_on_video_config)
            self._state_channel.run()

        if not self._vt_smoke:
            t_state = Thread(target=_state_worker, daemon=True)
            t_state.start()
            self._threads.append(t_state)

        # Decoupled VT submit worker
        def _vt_submit_worker() -> None:
            while True:
                try:
                    data, ts = self._vt_in_q.get()
                except Exception:
                    logger.debug("VT submit worker queue.get failed", exc_info=True)
                    continue
                try:
                    self._decode_vt_live(data, ts)
                except Exception as e:
                    logger.debug("VT submit worker error: %s", e)

        Thread(target=_vt_submit_worker, daemon=True).start()

        # VT drain worker: decouple VT output from UI draw loop
        def _vt_drain_worker() -> None:
            while True:
                try:
                    dec = self._vt_decoder
                    if dec is None:
                        time.sleep(0.005)
                        continue
                    drained = False
                    while True:
                        item = dec.get_frame_nowait()
                        if not item:
                            break
                        drained = True
                        img_buf, pts = item
                        if self._vt_wait_keyframe:
                            # Drain but don't submit while waiting for keyframe
                            try:
                                from napari_cuda import _vt as vt  # type: ignore
                                vt.release_frame(img_buf)
                            except Exception:
                                pass
                            continue
                        try:
                            from napari_cuda import _vt as vt  # type: ignore
                            # Retain a copy for draw-last-frame fallback
                            try:
                                # Take an extra retain so we can safely reuse as last-frame fallback
                                vt.retain_frame(img_buf)
                                # Release previously cached decode PB retain if any
                                if self._last_vt_payload is not None and not getattr(self, '_last_vt_persistent', False):
                                    try:
                                        vt.release_frame(self._last_vt_payload)
                                    except Exception:
                                        pass
                                self._last_vt_payload = img_buf
                                self._last_vt_persistent = True
                            except Exception:
                                self._last_vt_persistent = False
                            self._presenter.submit(
                                SubmittedFrame(
                                    source=Source.VT,
                                    server_ts=float(pts) if pts is not None else None,
                                    arrival_ts=time.time(),
                                    payload=img_buf,
                                    release_cb=vt.release_frame,
                                )
                            )
                        except Exception:
                            logger.debug("Presenter submit (VT) failed", exc_info=True)
                    if drained:
                        try:
                            QtCore.QTimer.singleShot(0, self._scene_canvas.native.update)
                        except Exception:
                            pass
                    if not drained:
                        time.sleep(0.002)
                except Exception:
                    logger.debug("VT drain worker error", exc_info=True)

        Thread(target=_vt_drain_worker, daemon=True).start()

        # PyAV worker
        def _pyav_worker() -> None:
            while True:
                try:
                    b, ts = self._pyav_in_q.get()
                except Exception:
                    continue
                try:
                    arr = None
                    try:
                        arr = self.decoder.decode(b) if self.decoder else None
                    except Exception as e:
                        logger.debug("PyAV worker decode error: %s", e)
                        arr = None
                    if arr is None:
                        continue
                    try:
                        self._presenter.submit(
                            SubmittedFrame(
                                source=Source.PYAV,
                                server_ts=float(ts) if ts is not None else None,
                                arrival_ts=time.time(),
                                payload=arr,
                                release_cb=None,
                            )
                        )
                    except Exception:
                        logger.debug("Presenter submit (PyAV) failed", exc_info=True)
                    QtCore.QTimer.singleShot(0, self._scene_canvas.native.update)
                except Exception:
                    logger.debug("PyAV worker error", exc_info=True)

        Thread(target=_pyav_worker, daemon=True).start()

        # Receiver thread or VT smoke
        if self._vt_smoke:
            logger.info("StreamManager in VT smoke test mode (offline)")
            smoke_source = (os.getenv('NAPARI_CUDA_VT_SMOKE_SOURCE') or 'vt').lower()
            # Offline VT-source generator: produce CVPixelBuffer BGRA frames and render zero-copy
            def _vt_synthetic_worker() -> None:
                try:
                    sw = int(os.getenv('NAPARI_CUDA_VT_SMOKE_W', '1280'))
                except Exception:
                    sw = 1280
                try:
                    sh = int(os.getenv('NAPARI_CUDA_VT_SMOKE_H', '720'))
                except Exception:
                    sh = 720
                try:
                    fps = float(os.getenv('NAPARI_CUDA_VT_SMOKE_FPS', '60'))
                except Exception:
                    fps = 60.0
                interval = max(1.0 / max(1.0, fps), 1.0 / 240.0)
                import numpy as _np
                # Precompute gradient bases
                x = _np.linspace(0, 255, sw, dtype=_np.uint8)[None, :]
                y = _np.linspace(0, 255, sh, dtype=_np.uint8)[:, None]
                # Precompute turntable geometry (all per‑pixel math factored out)
                aspect = sw / float(max(1, sh))
                xs = _np.linspace(-1.0, 1.0, sw, dtype=_np.float32) * aspect
                ys = _np.linspace(-1.0, 1.0, sh, dtype=_np.float32)
                X = _np.broadcast_to(xs[None, :], (sh, sw))
                Y = _np.broadcast_to(ys[:, None], (sh, sw))
                R2 = X * X + Y * Y
                MASK = R2 <= 1.0
                # Static light direction
                L = _np.array([0.35, 0.6, 1.0], dtype=_np.float32)
                L = L / _np.linalg.norm(L)
                # Precompute invariant sphere data
                Z0 = _np.zeros_like(X)
                Z0[MASK] = _np.sqrt(_np.maximum(0.0, 1.0 - R2[MASK]))
                Nx0 = X
                Ny0 = Y
                Nz0 = Z0
                # Base long/lat for checker albedo; Ny0 is static so lat0 is static
                lon0 = _np.arctan2(Nz0, Nx0)
                lat0 = _np.arcsin(_np.clip(Ny0, -1.0, 1.0))
                # Precompute scaled u0 and integer v-cells for 8x8 tiling
                u0_scaled = (lon0 / (2.0 * _np.pi) + 0.5) * 8.0
                v_cells = _np.floor((lat0 / _np.pi + 0.5) * 8.0).astype(_np.int32)
                # Preallocate work arrays to avoid per-frame allocations
                u_work = _np.empty_like(u0_scaled, dtype=_np.float32)
                lam = _np.empty_like(Nx0, dtype=_np.float32)
                check_f = _np.empty_like(Nx0, dtype=_np.float32)
                # Allocate double-buffered persistent CVPixelBuffers
                try:
                    from napari_cuda import _vt as vt  # type: ignore
                except Exception as e:
                    logger.error("VT shim not available for offline VT-source smoke: %s", e)
                    return
                pb0 = vt.alloc_pixelbuffer_bgra(sw, sh, True)
                pb1 = vt.alloc_pixelbuffer_bgra(sw, sh, True)
                pbs = [pb0, pb1]
                if pb0 is None or pb1 is None:
                    logger.error("alloc_pixelbuffer_bgra failed; falling back to RGB generator")
                    # Fall back to RGB generator path
                    while True:
                        t = time.perf_counter()
                        ox = int(((_np.sin(t * 2.0) * 0.5 + 0.5) * 64))
                        oy = int(((_np.cos(t * 1.7) * 0.5 + 0.5) * 64))
                        xx = (_np.arange(sw, dtype=_np.int32)[None, :] + ox) // 32
                        yy = (_np.arange(sh, dtype=_np.int32)[:, None] + oy) // 32
                        mask = ((xx ^ yy) & 1).astype(_np.uint8)
                        r = mask * 255
                        g = _np.broadcast_to(x, (sh, sw))
                        b = _np.broadcast_to(y, (sh, sw))
                        rgb = _np.dstack([r, g, b])
                        try:
                            self._presenter.submit(SubmittedFrame(Source.PYAV, None, time.time(), rgb, None))
                            QtCore.QTimer.singleShot(0, self._scene_canvas.native.update)
                        except Exception:
                            logger.debug("Presenter submit (fallback RGB) failed", exc_info=True)
                        time.sleep(interval)
                next_t = time.perf_counter()
                idx = 0
                mode = (os.getenv('NAPARI_CUDA_VT_SMOKE_MODE', 'checker') or 'checker').lower()
                logger.info("VT smoke (vt-source): %dx%d @ %.1f fps (mode=%s)", sw, sh, fps, mode)
                # Drive presenter in ARRIVAL mode; keep configured latency
                try:
                    self._presenter.set_mode(TimestampMode.ARRIVAL)
                except Exception:
                    pass
                # Ensure VT is the active source
                try:
                    self._source_mux.set_active(Source.VT)
                except Exception:
                    pass
                while True:
                    pb = pbs[idx & 1]
                    idx += 1
                    # Lock base and fill BGRA
                    try:
                        addr, bpr, width_i, height_i = vt.pb_lock_base(pb)
                    except Exception:
                        addr = None
                    if addr:
                        import ctypes as _ctypes
                        width_i = int(width_i); height_i = int(height_i); bpr = int(bpr)
                        size = bpr * height_i
                        raw = (_ctypes.c_ubyte * size).from_address(int(addr))
                        arr = _np.ctypeslib.as_array(raw)
                        img = arr.reshape((height_i, bpr // 4, 4))
                        t = time.perf_counter()
                        if mode == 'checker':
                            ox = int(((_np.sin(t * 2.0) * 0.5 + 0.5) * 64))
                            oy = int(((_np.cos(t * 1.7) * 0.5 + 0.5) * 64))
                            xx = (_np.arange(width_i, dtype=_np.int32)[None, :] + ox) // 32
                            yy = (_np.arange(height_i, dtype=_np.int32)[:, None] + oy) // 32
                            mask = ((xx ^ yy) & 1).astype(_np.uint8)
                            r = mask * 255
                            g = _np.broadcast_to(x, (height_i, width_i))
                            b = _np.broadcast_to(y, (height_i, width_i))
                            img[:height_i, :width_i, 0] = b
                            img[:height_i, :width_i, 1] = g
                            img[:height_i, :width_i, 2] = r
                            img[:height_i, :width_i, 3] = 255
                        elif mode == 'turntable':
                            # Lambert-shaded sphere with checker albedo; fully vectorized, preallocated temporaries
                            theta = t * 0.8
                            c = _np.cos(theta); s = _np.sin(theta)
                            # lam = max(0, Nx0*(c*Lx - s*Lz) + Ny0*Ly + Nz0*(s*Lx + c*Lz))
                            _np.multiply(Nx0, (c * L[0] - s * L[2]), out=lam)
                            lam += Ny0 * L[1]
                            lam += Nz0 * (s * L[0] + c * L[2])
                            _np.maximum(lam, 0.0, out=lam)
                            # Checker: floor((u0_scaled - theta*8)) + v_cells mod 2
                            u_work[...] = u0_scaled
                            u_work -= (theta * 8.0)
                            u_cells = _np.floor(u_work, out=u_work).astype(_np.int32, copy=False)
                            check = (u_cells + v_cells) & 1
                            # Convert to float once
                            _np.multiply(check, 1.0, out=check_f, dtype=_np.float32)
                            # Base albedo per channel from check
                            base_r = 0.2 + 0.8 * check_f
                            base_g = 1.0 - 0.7 * check_f
                            base_b = 0.2 + 0.8 * check_f
                            shade = 0.2 + 0.8 * lam
                            # Compute colors in-place, write only inside sphere mask
                            col_r = (shade * base_r * 255.0).astype(_np.uint8, copy=False)
                            col_g = (shade * base_g * 255.0).astype(_np.uint8, copy=False)
                            col_b = (shade * base_b * 255.0).astype(_np.uint8, copy=False)
                            img[MASK, 0] = col_b[MASK]
                            img[MASK, 1] = col_g[MASK]
                            img[MASK, 2] = col_r[MASK]
                            img[:height_i, :width_i, 3] = 255
                        else:
                            g = _np.broadcast_to(x, (height_i, width_i))
                            b = _np.broadcast_to(y, (height_i, width_i))
                            img[:height_i, :width_i, 0] = b
                            img[:height_i, :width_i, 1] = g
                            img[:height_i, :width_i, 2] = 255 - b
                            img[:height_i, :width_i, 3] = 255
                        try:
                            vt.pb_unlock_base(pb)
                        except Exception:
                            pass
                    # Submit CVPixelBuffer for zero-copy draw; keep PBs persistent (no release_cb)
                    try:
                        self._presenter.submit(
                            SubmittedFrame(
                                source=Source.VT,
                                server_ts=None,
                                arrival_ts=time.time(),
                                payload=pb,
                                release_cb=None,
                            )
                        )
                        # Cache last VT payload for draw-last-frame fallback to avoid flicker
                        self._last_vt_payload = pb
                        self._last_vt_persistent = True  # vt-source owns persistent PBs (no release_cb)
                        try:
                            QtCore.QTimer.singleShot(0, self._scene_canvas.native.update)
                        except Exception:
                            pass
                    except Exception:
                        logger.debug("Presenter submit (vt-source) failed", exc_info=True)
                    next_t += interval
                    sleep = next_t - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_t = time.perf_counter()
            def _vt_encode_worker() -> None:
                try:
                    sw = int(os.getenv('NAPARI_CUDA_VT_SMOKE_W', '1280'))
                except Exception:
                    sw = 1280
                try:
                    sh = int(os.getenv('NAPARI_CUDA_VT_SMOKE_H', '720'))
                except Exception:
                    sh = 720
                try:
                    fps = float(os.getenv('NAPARI_CUDA_VT_SMOKE_FPS', '60'))
                except Exception:
                    fps = 60.0
                smoke_mode = (os.getenv('NAPARI_CUDA_VT_SMOKE_MODE', 'checker') or 'checker').lower()
                from napari_cuda.client.streaming.smoke.runner import run_encode_smoke as _run_encode_smoke
                _run_encode_smoke(
                    width=sw,
                    height=sh,
                    fps=fps,
                    smoke_mode=smoke_mode,
                    presenter=self._presenter,
                    source_mux=self._source_mux,
                    vt_in_q=self._vt_in_q,
                    vt_backlog_trigger=self._vt_backlog_trigger,
                    vt_latency_s=self._vt_latency_s,
                    init_vt_from_avcc=lambda avcc_b64, w, h: self._init_vt_from_avcc(avcc_b64, w, h),
                    after_vt_init=lambda: setattr(self, '_vt_wait_keyframe', False),
                    on_enqueued=lambda n: setattr(self, '_vt_enqueued', self._vt_enqueued + int(n)),
                    logger=logger,
                )

            if smoke_source == 'encode':
                Thread(target=_vt_encode_worker, daemon=True).start()
            elif smoke_source == 'pyav':
                # PyAV decode smoke: encode with H264Encoder and decode via PyAV
                def _pyav_encode_worker() -> None:
                    try:
                        self._init_decoder()
                    except Exception:
                        logger.debug('pyav-smoke: init decoder failed', exc_info=True)
                    try:
                        sw = int(os.getenv('NAPARI_CUDA_VT_SMOKE_W', '1280'))
                    except Exception:
                        sw = 1280
                    try:
                        sh = int(os.getenv('NAPARI_CUDA_VT_SMOKE_H', '720'))
                    except Exception:
                        sh = 720
                    try:
                        fps = float(os.getenv('NAPARI_CUDA_VT_SMOKE_FPS', '60'))
                    except Exception:
                        fps = 60.0
                    smoke_mode = (os.getenv('NAPARI_CUDA_VT_SMOKE_MODE', 'checker') or 'checker').lower()
                    from napari_cuda.client.streaming.smoke.runner import run_pyav_smoke as _run_pyav_smoke
                    _run_pyav_smoke(
                        width=sw,
                        height=sh,
                        fps=fps,
                        smoke_mode=smoke_mode,
                        presenter=self._presenter,
                        source_mux=self._source_mux,
                        pyav_in_q=self._pyav_in_q,
                        pyav_backlog_trigger=self._pyav_backlog_trigger,
                        pyav_latency_s=self._pyav_latency_s,
                        on_enqueued=lambda n: setattr(self, '_pyav_enqueued', self._pyav_enqueued + int(n)),
                        logger=logger,
                    )
                Thread(target=_pyav_encode_worker, daemon=True).start()
            else:
                Thread(target=_vt_synthetic_worker, daemon=True).start()
        else:
            def _on_connected() -> None:
                try:
                    self._init_decoder()
                except Exception:
                    logger.debug("init decoder on connect failed", exc_info=True)

            def _on_frame(pkt: Packet) -> None:
                try:
                    # Log keyframe detection to verify server behavior
                    # try:
                    #     if (pkt.flags & 0x01) != 0 or self._is_keyframe(pkt.payload, pkt.codec):
                    #         s = int(pkt.seq)
                    #         if self._last_key_logged != s:
                    #             logger.info("Keyframe detected (seq=%d)", s)
                    #             self._last_key_logged = s
                    #             self._keyframes_seen += 1
                    # except Exception:
                    #     pass
                    # Detect stream discontinuity by sequence number
                    try:
                        cur = int(pkt.seq)
                        if self._last_seq is not None and not (self._vt_wait_keyframe or self._pyav_wait_keyframe):
                            expected = (self._last_seq + 1) & 0xFFFFFFFF
                            if cur != expected:
                                # Log at most ~5 Hz to avoid spam
                                now = time.time()
                                if (now - self._last_disco_log) > 0.2:
                                    logger.warning(
                                        "Pixel stream discontinuity: expected=%d got=%d; gating until keyframe",
                                        expected,
                                        cur,
                                    )
                                    self._last_disco_log = now
                                # Gate VT path
                                if self._vt_decoder is not None:
                                    self._vt_wait_keyframe = True
                                    while self._vt_in_q.qsize() > 0:
                                        _ = self._vt_in_q.get_nowait()
                                    self._presenter.clear(Source.VT)
                                    self._request_keyframe_once()
                                # Gate PyAV path
                                self._pyav_wait_keyframe = True
                                while self._pyav_in_q.qsize() > 0:
                                    _ = self._pyav_in_q.get_nowait()
                                self._presenter.clear(Source.PYAV)
                                try:
                                    self._init_decoder()
                                except Exception:
                                    logger.debug("PyAV decoder reinit after discontinuity failed", exc_info=True)
                                self._disco_gated = True
                    finally:
                        # Always update last_seq
                        try:
                            self._last_seq = int(pkt.seq)
                        except Exception:
                            self._last_seq = None
                    # Global initial gate (first frame must be keyframe)
                    if not self._stream_seen_keyframe:
                        if self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01):
                            self._stream_seen_keyframe = True
                        else:
                            return
                    # VT-specific gate: after VT (re)init, require a fresh keyframe
                    if self._vt_decoder is not None and self._vt_wait_keyframe:
                        if self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01):
                            self._vt_wait_keyframe = False
                            self._vt_gate_lift_time = time.time()
                            try:
                                self._vt_ts_offset = float(self._vt_gate_lift_time - float(pkt.ts)) - float(self._server_bias_s)
                            except Exception:
                                self._vt_ts_offset = None
                            self._source_mux.set_active(Source.VT)
                            self._presenter.set_offset(self._vt_ts_offset)
                            # Restore low latency for VT
                            try:
                                self._presenter.set_latency(self._vt_latency_s)
                            except Exception:
                                pass
                            self._presenter.clear(Source.PYAV)
                            logger.info("VT gate lifted on keyframe (seq=%d); presenter=VT", int(pkt.seq))
                            self._disco_gated = False
                            # Engage warmup ramp in ARRIVAL mode to reduce startup ticks
                            try:
                                if self._presenter.clock.mode == TimestampMode.ARRIVAL and self._warmup_window_s > 0:
                                    # Determine extra: explicit override (ms) or auto size to exceed one frame interval
                                    if self._warmup_ms_override:
                                        extra_ms = max(0.0, float(self._warmup_ms_override))
                                    else:
                                        frame_ms = 1000.0 / (self._fps if (self._fps and self._fps > 0) else 60.0)
                                        target_ms = frame_ms + float(self._warmup_margin_ms)
                                        base_ms = float(self._vt_latency_s) * 1000.0
                                        extra_ms = max(0.0, min(float(self._warmup_max_ms), target_ms - base_ms))
                                    extra_s = extra_ms / 1000.0
                                    if extra_s > 0.0:
                                        self._warmup_extra_active_s = extra_s
                                        self._warmup_until = time.time() + float(self._warmup_window_s)
                                        self._presenter.set_latency(self._vt_latency_s + extra_s)
                            except Exception:
                                pass
                        else:
                            # While VT is gated, avoid decoding previews; just request keyframe
                            self._request_keyframe_once()
                            return

                    # Prefer VT if ready; otherwise PyAV
                    if self._vt_decoder is not None and not self._vt_wait_keyframe:
                        try:
                            ts_float = float(pkt.ts)
                        except Exception:
                            ts_float = None
                        b = bytes(pkt.payload)
                        try:
                            if self._vt_in_q.qsize() >= max(2, self._vt_backlog_trigger - 1):
                                self._vt_wait_keyframe = True
                                logger.info("VT backlog detected (q=%d); requesting keyframe and resync", self._vt_in_q.qsize())
                                self._request_keyframe_once()
                                while self._vt_in_q.qsize() > 0:
                                    _ = self._vt_in_q.get_nowait()
                                self._presenter.clear(Source.VT)
                            self._vt_in_q.put_nowait((b, ts_float))
                            self._vt_enqueued += 1
                        except queue.Full:
                            self._vt_wait_keyframe = True
                            self._request_keyframe_once()
                    else:
                        # PyAV path with backlog-safe gating
                        try:
                            ts_float = float(pkt.ts)
                        except Exception:
                            ts_float = None
                        b = bytes(pkt.payload)
                        # If backlog builds, reset decoder and wait for a fresh keyframe
                        if self._pyav_in_q.qsize() >= max(2, self._pyav_backlog_trigger - 1):
                            self._pyav_wait_keyframe = True
                            # Drain queued items and clear presenter buffer for a clean start
                            while self._pyav_in_q.qsize() > 0:
                                _ = self._pyav_in_q.get_nowait()
                            self._presenter.clear(Source.PYAV)
                            # Recreate decoder fresh to drop reference history
                            try:
                                self._init_decoder()
                            except Exception:
                                logger.debug("PyAV decoder reinit failed after backlog", exc_info=True)
                        # If waiting for keyframe, skip until flags indicate a keyframe
                        if self._pyav_wait_keyframe:
                            if not (self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01)):
                                return
                            # Got keyframe: clear gate
                            self._pyav_wait_keyframe = False
                            # Reinitialize decoder one more time right at the keyframe
                            try:
                                self._init_decoder()
                            except Exception:
                                logger.debug("PyAV decoder reinit at keyframe failed", exc_info=True)
                            self._disco_gated = False
                        # Enqueue for decode
                        try:
                            self._pyav_in_q.put_nowait((b, ts_float))
                            self._pyav_enqueued += 1
                        except queue.Full:
                            # Best-effort: drop one and retry; if still full, drop this frame
                            try:
                                _ = self._pyav_in_q.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self._pyav_in_q.put_nowait((b, ts_float))
                            except queue.Full:
                                logger.debug("PyAV queue full; dropping frame", exc_info=True)
                except Exception:
                    logger.debug("stream frame handling failed", exc_info=True)

            def _on_disconnect(exc: Exception | None) -> None:
                return

            self._receiver = PixelReceiver(
                self.server_host,
                self.server_port,
                on_connected=_on_connected,
                on_frame=_on_frame,
                on_disconnect=_on_disconnect,
            )
            Thread(target=self._receiver.run, daemon=True).start()

        # Dedicated 1 Hz presenter stats timer (decoupled from draw loop)
        if self._stats_level is not None:
            try:
                self._stats_timer = QtCore.QTimer(self._scene_canvas.native)
                self._stats_timer.setTimerType(QtCore.Qt.PreciseTimer)
                self._stats_timer.setInterval(1000)
                self._stats_timer.timeout.connect(self._log_stats)
                self._stats_timer.start()
            except Exception:
                logger.debug("Failed to start stats timer; falling back to draw-based logging", exc_info=True)

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                logger.debug("Renderer queue drain race", exc_info=True)
        self._frame_q.put(frame)

    def draw(self) -> None:
        # Arrival-mode startup warmup: ramp latency from (base+extra) down to base
        if self._warmup_until > 0 and self._presenter.clock.mode == TimestampMode.ARRIVAL:
            now = time.time()
            if now >= self._warmup_until:
                try:
                    self._presenter.set_latency(self._vt_latency_s)
                except Exception:
                    pass
                self._warmup_until = 0.0
            else:
                remain = max(0.0, self._warmup_until - now)
                frac = remain / max(1e-6, self._warmup_window_s)
                cur = self._vt_latency_s + self._warmup_extra_active_s * frac
                try:
                    self._presenter.set_latency(cur)
                except Exception:
                    pass
        # Stats are reported via a dedicated timer now

        # VT output is drained continuously by worker; draw focuses on presenting

        ready = self._presenter.pop_due(time.time(), self._source_mux.active)
        if ready is not None:
            src_val = getattr(ready.source, 'value', str(ready.source))
            # Adaptive scheduling for VT: handle sustained preview-only condition
            if src_val == 'vt':
                if getattr(ready, 'preview', False):
                    self._preview_streak += 1
                    if self._preview_streak == 20 and not self._relearn_logged:
                        off = self._presenter.relearn_offset(Source.VT)
                        logger.info(
                            "VT preview streak; relearned offset=%s",
                            f"{off:.3f}s" if off is not None else "n/a",
                        )
                        self._relearn_logged = True
                    if self._preview_streak >= 40 and not self._arrival_fallback:
                        self._presenter.set_mode(TimestampMode.ARRIVAL)
                        self._arrival_fallback = True
                        self._arrival_logged = True
                        logger.info("VT adaptive fallback: ARRIVAL mode engaged")
                else:
                    if self._arrival_fallback:
                        # Consuming VT frames again; restore SERVER mode
                        self._presenter.set_mode(TimestampMode.SERVER)
                        self._arrival_fallback = False
                        self._relearn_logged = False
                        if self._arrival_logged:
                            logger.info("VT adaptive fallback: SERVER mode restored")
                            self._arrival_logged = False
                    self._preview_streak = 0
            if src_val == 'vt':
                # Zero-copy path: pass CVPixelBuffer + release_cb through to GLRenderer
                try:
                    release_cb = None if getattr(ready, 'preview', False) else ready.release_cb
                    self._enqueue_frame((ready.payload, release_cb))
                except Exception:
                    logger.debug("enqueue VT payload failed", exc_info=True)
            else:
                # Cache for last-frame fallback and enqueue
                try:
                    self._last_pyav_frame = ready.payload
                except Exception:
                    pass
                self._enqueue_frame(ready.payload)

        frame = None
        # Draw-last-frame fallbacks to avoid black frames on GUI clears
        if ready is None:
            if self._source_mux.active == Source.VT and self._last_vt_payload is not None and getattr(self, '_last_vt_persistent', False):
                try:
                    self._enqueue_frame((self._last_vt_payload, None))
                except Exception:
                    logger.debug("enqueue last VT payload failed", exc_info=True)
            elif self._source_mux.active == Source.PYAV and self._last_pyav_frame is not None:
                try:
                    self._enqueue_frame(self._last_pyav_frame)
                except Exception:
                    logger.debug("enqueue last PyAV frame failed", exc_info=True)
        while not self._frame_q.empty():
            try:
                frame = self._frame_q.get_nowait()
            except queue.Empty:
                break
        self._renderer.draw(frame)

    def _log_stats(self) -> None:
        if self._stats_level is None:
            return
        try:
            pres_stats = self._presenter.stats()
            vt_counts = None
            try:
                if self._vt_decoder is not None:
                    vt_counts = self._vt_decoder.counts()
            except Exception:
                vt_counts = None
            logger.log(
                self._stats_level,
                "presenter=%s vt_counts=%s keyframes_seen=%d",
                pres_stats,
                vt_counts,
                self._keyframes_seen,
            )
        except Exception:
            logger.debug("stats logging failed", exc_info=True)

    def _init_decoder(self) -> None:
        import os as _os
        swap_rb = (_os.getenv('NAPARI_CUDA_CLIENT_SWAP_RB', '0') or '0') in ('1',)
        pf = (_os.getenv('NAPARI_CUDA_CLIENT_PIXEL_FMT', 'rgb24') or 'rgb24').lower()
        self.decoder = PyAVDecoder(self._stream_format, pixfmt=pf, swap_rb=swap_rb)

    def _init_vt_from_avcc(self, avcc_b64: str, width: int, height: int) -> None:
        import base64
        import sys
        try:
            avcc = base64.b64decode(avcc_b64)
            cfg_key = (int(width), int(height), avcc)
            # Ignore exact duplicate configs
            if self._last_vcfg_key == cfg_key:
                logger.debug("Duplicate video_config ignored (%dx%d)", width, height)
                return
            if self._vt_decoder is not None and self._vt_cfg_key == cfg_key:
                logger.debug("VT already initialized; ignoring duplicate video_config")
                return
            # Respect NAPARI_CUDA_VT_BACKEND env: off/0/false disables VT
            backend_env = (os.getenv('NAPARI_CUDA_VT_BACKEND', 'shim') or 'shim').lower()
            vt_disabled = backend_env in ("off", "0", "false", "no")
            if sys.platform == 'darwin' and not vt_disabled:
                try:
                    self._vt_decoder = VTLiveDecoder(avcc, width, height)
                    self._vt_backend = 'shim'
                except Exception as e:
                    logger.warning("VT shim unavailable: %s; falling back to PyAV", e)
                    self._vt_decoder = None
            self._vt_cfg_key = cfg_key
            self._last_vcfg_key = cfg_key
            self._vt_wait_keyframe = True
            # Parse nal length size from avcC if available (5th byte low 2 bits + 1)
            try:
                if len(avcc) >= 5:
                    self._nal_length_size = int((avcc[4] & 0x03) + 1)
                    if self._nal_length_size not in (1, 2, 3, 4):
                        self._nal_length_size = 4
            except Exception:
                self._nal_length_size = 4
            logger.info("VideoToolbox live decoder initialized: %dx%d", width, height)
        except Exception as e:
            logger.error("VT live init failed: %s", e)

    def _request_keyframe_once(self) -> None:
        ch = self._state_channel
        if ch is not None:
            ch.request_keyframe_once()

    def _decode_frame(self, h264_data: bytes) -> None:
        if self.decoder:
            try:
                frame = self.decoder.decode(h264_data)
            except Exception as e:
                logger.debug("PyAV decode error; skipping frame: %s", e)
                frame = None
            if frame is not None:
                self._enqueue_frame(frame)

    def _annexb_to_avcc(self, data: bytes) -> bytes:
        out = bytearray()
        n = len(data)
        idx: list[int] = []
        i = 0
        while i + 3 <= n:
            if data[i:i+3] == b"\x00\x00\x01":
                idx.append(i)
                i += 3
            elif i + 4 <= n and data[i:i+4] == b"\x00\x00\x00\x01":
                idx.append(i)
                i += 4
            else:
                i += 1
        idx.append(n)
        for a, b in zip(idx, idx[1:]):
            j = a
            while j < b and data[j] == 0:
                j += 1
            if j + 3 <= b and data[j:j+3] == b"\x00\x00\x01":
                j += 3
            elif j + 4 <= b and data[j:j+4] == b"\x00\x00\x00\x01":
                j += 4
            nal = data[j:b]
            if not nal:
                continue
            out.extend(len(nal).to_bytes(4, "big"))
            out.extend(nal)
        return bytes(out)

    def _decode_vt_live(self, data: bytes, ts: float | None) -> None:
        try:
            if not data or self._vt_decoder is None:
                return
            # Normalize to AVCC with declared NAL length size (default 4)
            target_len = int(self._nal_length_size or 4)
            if is_annexb(data):
                avcc_au = annexb_to_avcc(data)
            else:
                # Try to repackage AVCC if length size differs
                nals = split_avcc_by_len(data, 4)
                if not nals:
                    nals = split_avcc_by_len(data, 2)
                if nals:
                    out = bytearray()
                    for n in nals:
                        out.extend(len(n).to_bytes(target_len, 'big'))
                        out.extend(n)
                    avcc_au = bytes(out)
                else:
                    # Fallback to raw data
                    avcc_au = data
            ok = self._vt_decoder.decode(avcc_au, ts)
            if not ok:
                self._vt_errors += 1
                if self._vt_errors <= 3 or (self._vt_errors % 50 == 0):
                    logger.warning("VT decode submit failed (errors=%d)", self._vt_errors)
                return
            self._vt_errors = 0
            try:
                if self._vt_enqueued <= 3:
                    self._vt_decoder.flush()
            except Exception:
                logger.debug("VT flush failed", exc_info=True)
        except Exception as e:
            self._vt_errors += 1
            logger.exception("VT live decode/map failed (%d): %s", self._vt_errors, e)
            if self._vt_errors >= 3:
                logger.error("Disabling VT after repeated errors; falling back to PyAV")
                self._vt_decoder = None
                try:
                    self._source_mux.set_active(Source.PYAV)
                    # Increase latency for PyAV fallback to smooth playback
                    self._presenter.set_latency(self._pyav_latency_s)
                    self._presenter.clear(Source.VT)
                except Exception:
                    pass

    # Keyframe detection that doesn't rely solely on header flags
    def _is_keyframe(self, payload: bytes, codec: int) -> bool:
        try:
            c = int(codec)
        except Exception:
            c = 0
        # Fast path: Annex B start codes present
        if payload[:4] == b"\x00\x00\x00\x01" or payload[:3] == b"\x00\x00\x01":
            return self._is_keyframe_annexb(payload, c)
        # Otherwise, assume AVCC 4-byte length delimited
        return self._is_keyframe_avcc(payload, c)

    def _is_keyframe_annexb(self, data: bytes, codec: int) -> bool:
        i = 0
        n = len(data)
        # Iterate NAL units separated by 00 00 01 or 00 00 00 01
        while i < n:
            # Find next start code
            sc = -1
            sc_len = 0
            j = i
            while j + 3 <= n:
                if data[j:j+3] == b"\x00\x00\x01":
                    sc = j
                    sc_len = 3
                    break
                if j + 4 <= n and data[j:j+4] == b"\x00\x00\x00\x01":
                    sc = j
                    sc_len = 4
                    break
                j += 1
            if sc == -1:
                break
            # Move to NAL header
            j = sc + sc_len
            if j >= n:
                break
            nal_hdr0 = data[j]
            if codec == 2:  # HEVC/H.265
                # HEVC nal_unit_type: bits 1..6 (after forbidden_zero_bit)
                # Two-byte header in HEVC
                if j + 1 >= n:
                    break
                nal_unit_type = (data[j] >> 1) & 0x3F
                # IRAP pictures: 16..21
                if 16 <= nal_unit_type <= 21:
                    return True
            else:  # H.264 default
                nal_unit_type = nal_hdr0 & 0x1F
                # IDR slice
                if nal_unit_type == 5:
                    return True
            i = j + 1
        return False

    def _is_keyframe_avcc(self, data: bytes, codec: int) -> bool:
        i = 0
        n = len(data)
        # Assume 4-byte lengths (common for avcC)
        while i + 4 <= n:
            ln = int.from_bytes(data[i:i+4], 'big', signed=False)
            i += 4
            if ln <= 0 or i + ln > n:
                break
            nal_hdr0 = data[i]
            if codec == 2:  # HEVC/H.265
                if i + 1 >= n:
                    break
                nal_unit_type = (data[i] >> 1) & 0x3F
                if 16 <= nal_unit_type <= 21:
                    return True
            else:  # H.264
                nal_unit_type = nal_hdr0 & 0x1F
                if nal_unit_type == 5:
                    return True
            i += ln
        return False

# No alias export; prefer StreamManager
