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
from napari_cuda.client.streaming.controllers import StateController, ReceiveController
from napari_cuda.client.streaming.types import Source, SubmittedFrame, TimestampMode
from napari_cuda.client.streaming.renderer import GLRenderer
from napari_cuda.client.streaming.decoders.pyav import PyAVDecoder
from napari_cuda.client.streaming.decoders.vt import VTLiveDecoder
from napari_cuda.client.streaming.pipelines.pyav_pipeline import PyAVPipeline
from napari_cuda.client.streaming.pipelines.vt_pipeline import VTPipeline
from napari_cuda.client.streaming.pipelines.smoke_pipeline import SmokePipeline, SmokeConfig
from napari_cuda.codec.avcc import (
    annexb_to_avcc,
    is_annexb,
    split_annexb,
    split_avcc_by_len,
    build_avcc,
    find_sps_pps,
)
from napari_cuda.codec.h264 import contains_idr_annexb, contains_idr_avcc
from napari_cuda.codec.h264_encoder import H264Encoder, EncoderConfig
from napari_cuda.utils.env import env_float, env_str
from napari_cuda.client.streaming.smoke.generators import make_generator
from napari_cuda.client.streaming.smoke.submit import submit_vt, submit_pyav
from napari_cuda.client.streaming.config import extract_video_config

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
        self._presenter.set_latency(self._pyav_latency_s)
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

        # VT pipeline replaces inline queues/workers
        self._vt_backlog_trigger = int(vt_backlog_trigger)
        self._vt_enqueued = 0
        # Callbacks for VT pipeline coordination
        def _is_vt_gated() -> bool:
            return bool(self._vt_wait_keyframe)
        def _on_vt_backlog_gate() -> None:
            self._vt_wait_keyframe = True
        def _req_keyframe() -> None:
            self._request_keyframe_once()
        def _on_cache_last(pb: object, persistent: bool) -> None:
            try:
                # Keep local cache for draw fallback
                self._last_vt_payload = pb
                self._last_vt_persistent = bool(persistent)
            except Exception:
                logger.debug("cache last VT payload callback failed", exc_info=True)
        self._vt_pipeline = VTPipeline(
            presenter=self._presenter,
            source_mux=self._source_mux,
            scene_canvas=self._scene_canvas,
            backlog_trigger=self._vt_backlog_trigger,
            is_gated=_is_vt_gated,
            on_backlog_gate=_on_vt_backlog_gate,
            request_keyframe=_req_keyframe,
            on_cache_last=_on_cache_last,
        )
        self._pyav_backlog_trigger = int(pyav_backlog_trigger)
        self._pyav_enqueued = 0
        # PyAV pipeline replaces inline queue/worker
        self._pyav_pipeline = PyAVPipeline(
            presenter=self._presenter,
            source_mux=self._source_mux,
            scene_canvas=self._scene_canvas,
            backlog_trigger=self._pyav_backlog_trigger,
            latency_s=self._pyav_latency_s,
        )
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
        if not self._vt_smoke:
            def _on_video_config(data: dict) -> None:
                w, h, fps, fmt, avcc_b64 = extract_video_config(data)
                if fps > 0:
                    self._fps = fps
                self._stream_format = fmt
                self._stream_format_set = True
                if w > 0 and h > 0 and avcc_b64:
                    self._init_vt_from_avcc(avcc_b64, w, h)

            st = StateController(self.server_host, self.state_port, _on_video_config)
            self._state_channel, t_state = st.start()
            self._threads.append(t_state)

        # Start VT pipeline workers
        self._vt_pipeline.start()

        # Start PyAV pipeline worker
        self._pyav_pipeline.start()

        # Receiver thread or smoke mode
        if self._vt_smoke:
            logger.info("StreamManager in smoke test mode (offline)")
            # Ensure PyAV decoder is ready if targeting pyav
            smoke_source = (os.getenv('NAPARI_CUDA_SMOKE_SOURCE') or 'vt').lower()
            if smoke_source == 'pyav':
                self._init_decoder()
                dec = self.decoder.decode if self.decoder else None
                self._pyav_pipeline.set_decoder(dec)
            # Read config
            try:
                sw = int(os.getenv('NAPARI_CUDA_SMOKE_W', '1280'))
            except Exception:
                sw = 1280
            try:
                sh = int(os.getenv('NAPARI_CUDA_SMOKE_H', '720'))
            except Exception:
                sh = 720
            try:
                fps = float(os.getenv('NAPARI_CUDA_SMOKE_FPS', '60'))
            except Exception:
                fps = 60.0
            smoke_mode = (os.getenv('NAPARI_CUDA_SMOKE_MODE', 'checker') or 'checker').lower()
            preencode = (os.getenv('NAPARI_CUDA_SMOKE_PREENCODE', '0') or '0') in ('1', 'true', 'yes')
            try:
                pre_frames = int(os.getenv('NAPARI_CUDA_SMOKE_PRE_FRAMES', str(int(fps) * 3)))
            except Exception:
                pre_frames = int(fps) * 3
            cfg = SmokeConfig(
                width=sw,
                height=sh,
                fps=fps,
                smoke_mode=smoke_mode,
                preencode=preencode,
                pre_frames=pre_frames,
                backlog_trigger=self._vt_backlog_trigger if smoke_source == 'vt' else self._pyav_backlog_trigger,
                target='pyav' if smoke_source == 'pyav' else 'vt',
                vt_latency_s=self._vt_latency_s,
                pyav_latency_s=self._pyav_latency_s,
            )
            def _init_and_clear(avcc_b64: str, w: int, h: int) -> None:
                self._init_vt_from_avcc(avcc_b64, w, h)
                # In smoke mode, lift VT gate immediately after init
                self._vt_wait_keyframe = False

            self._smoke = SmokePipeline(
                config=cfg,
                presenter=self._presenter,
                source_mux=self._source_mux,
                pipeline=self._pyav_pipeline if smoke_source == 'pyav' else self._vt_pipeline,
                init_vt_from_avcc=_init_and_clear,
            )
            self._smoke.start()
        else:
            def _on_connected() -> None:
                self._init_decoder()
                # Bind decoder to pipeline
                dec = self.decoder.decode if self.decoder else None
                self._pyav_pipeline.set_decoder(dec)

            def _on_frame(pkt: Packet) -> None:
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
                                self._vt_pipeline.clear()
                                self._presenter.clear(Source.VT)
                                self._request_keyframe_once()
                            # Gate PyAV path
                            self._pyav_wait_keyframe = True
                            self._pyav_pipeline.clear()
                            self._presenter.clear(Source.PYAV)
                            self._init_decoder()
                            dec = self.decoder.decode if self.decoder else None
                            self._pyav_pipeline.set_decoder(dec)
                            self._disco_gated = True
                    # Always update last_seq
                    self._last_seq = int(pkt.seq)
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
                                logger.debug("Presenter set_latency restore for VT failed", exc_info=True)
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
                                logger.debug("Warmup latency set failed", exc_info=True)
                        else:
                            # While VT is gated, avoid decoding previews; just request keyframe
                            self._request_keyframe_once()
                            return

                    # Prefer VT if ready; otherwise PyAV
                    if self._vt_decoder is not None and not self._vt_wait_keyframe:
                        ts_float = float(pkt.ts)
                        b = bytes(pkt.payload)
                        try:
                            self._vt_pipeline.enqueue(b, ts_float)
                            self._vt_enqueued += 1
                        except Exception:
                            logger.debug("VT pipeline enqueue failed", exc_info=True)
                    else:
                        # PyAV path with backlog-safe gating
                        try:
                            ts_float = float(pkt.ts)
                        except Exception:
                            ts_float = None
                        b = bytes(pkt.payload)
                        # If backlog builds, reset decoder and wait for a fresh keyframe
                        if self._pyav_pipeline.qsize() >= max(2, self._pyav_backlog_trigger - 1):
                            self._pyav_wait_keyframe = True
                            # Drain queued items and clear presenter buffer for a clean start
                            self._pyav_pipeline.clear()
                            self._presenter.clear(Source.PYAV)
                            # Recreate decoder fresh to drop reference history
                            self._init_decoder()
                            dec = self.decoder.decode if self.decoder else None
                            self._pyav_pipeline.set_decoder(dec)
                        # If waiting for keyframe, skip until flags indicate a keyframe
                        if self._pyav_wait_keyframe:
                            if not (self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01)):
                                return
                            # Got keyframe: clear gate
                            self._pyav_wait_keyframe = False
                            # Reinitialize decoder one more time right at the keyframe
                            self._init_decoder()
                            self._disco_gated = False
                        # Enqueue for decode
                        self._pyav_pipeline.enqueue(b, ts_float)
                        self._pyav_enqueued += 1

            def _on_disconnect(exc: Exception | None) -> None:
                return

            rc = ReceiveController(
                self.server_host,
                self.server_port,
                on_connected=_on_connected,
                on_frame=_on_frame,
                on_disconnect=_on_disconnect,
            )
            self._receiver, t_rx = rc.start()
            self._threads.append(t_rx)

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
                self._presenter.set_latency(self._vt_latency_s)
                self._warmup_until = 0.0
            else:
                remain = max(0.0, self._warmup_until - now)
                frac = remain / max(1e-6, self._warmup_window_s)
                cur = self._vt_latency_s + self._warmup_extra_active_s * frac
                self._presenter.set_latency(cur)
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
                    logger.debug("Cache last PyAV frame failed", exc_info=True)
                self._enqueue_frame(ready.payload)

        frame = None
        # Draw-last-frame fallbacks to avoid black frames on GUI clears
        if ready is None:
            if self._source_mux.active == Source.VT:
                try:
                    pb, persistent = self._vt_pipeline.last_payload_info()
                except Exception:
                    pb, persistent = (None, False)
                if pb is not None and persistent:
                    try:
                        self._enqueue_frame((pb, None))
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
                vt_counts = self._vt_pipeline.counts()
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
                    # Bind decoder to VT pipeline
                    self._vt_pipeline.set_decoder(self._vt_decoder)
                except Exception as e:
                    logger.warning("VT shim unavailable: %s; falling back to PyAV", e)
                    self._vt_decoder = None
            self._vt_cfg_key = cfg_key
            self._last_vcfg_key = cfg_key
            self._vt_wait_keyframe = True
            # Parse nal length size from avcC if available (5th byte low 2 bits + 1)
            if len(avcc) >= 5:
                nsz = int((avcc[4] & 0x03) + 1)
                if nsz in (1, 2, 3, 4):
                    self._nal_length_size = nsz
                else:
                    logger.warning("Invalid avcC nal_length_size=%s; defaulting to 4", nsz)
                    self._nal_length_size = 4
            else:
                logger.warning("avcC too short (%d); defaulting nal_length_size=4", len(avcc))
                self._nal_length_size = 4
            self._vt_pipeline.update_nal_length_size(self._nal_length_size)
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

    # VT decode/submit is handled by VTPipeline

    # Keyframe detection via shared helpers (AnnexB/AVCC)
    def _is_keyframe(self, payload: bytes, codec: int) -> bool:
        try:
            hevc = int(codec) == 2
        except Exception:
            hevc = False
        if is_annexb(payload):
            return contains_idr_annexb(payload, hevc=hevc)
        # Use parsed nal_length_size when available (defaults to 4)
        nsz = int(self._nal_length_size or 4)
        return contains_idr_avcc(payload, nal_len_size=nsz, hevc=hevc)

# No alias export; prefer StreamManager
