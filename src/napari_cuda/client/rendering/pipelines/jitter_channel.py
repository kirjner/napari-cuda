from __future__ import annotations

import heapq
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


@dataclass
class JitterConfig:
    enable: bool = False
    # Base latency and jitter distribution
    base_ms: float = 0.0
    mode: str = "uniform"  # uniform|normal|pareto
    jitter_ms: float = 20.0  # uniform half-range or normal sigma
    pareto_alpha: float = 2.5
    pareto_scale_ms: float = 5.0
    # Bursts
    burst_p: float = 0.03
    burst_min_ms: Optional[float] = None
    burst_max_ms: Optional[float] = None
    burst_ms: float = 80.0  # used when min/max not set
    # Loss / reorder / duplication
    loss_p: float = 0.0
    reorder_p: float = 0.0
    reorder_advance_ms: float = 5.0
    dup_p: float = 0.0
    dup_ms: float = 10.0
    # Bandwidth limiter (token bucket)
    bw_kbps: int = 0  # 0 = disabled
    burst_bytes: int = 32768
    # PTS policy
    affect_pts: bool = False  # when True and ts_source='encode', shift PTS by delay
    ts_source: str = 'encode'  # 'encode' (use incoming PTS) or 'send' (stamp at submit time)
    ts_bias_ms: float = 0.0    # additional bias added to PTS
    # Internal queue caps
    queue_cap: int = 512
    # Seed for determinism
    seed: Optional[int] = 1234


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).lower() in ("1", "true", "yes", "on")


def from_env() -> JitterConfig:
    return JitterConfig(
        enable=_env_bool("NAPARI_CUDA_JIT_ENABLE", False),
        base_ms=_env_float("NAPARI_CUDA_JIT_BASE_MS", 0.0),
        mode=os.getenv("NAPARI_CUDA_JIT_MODE", "uniform").lower(),
        jitter_ms=_env_float("NAPARI_CUDA_JIT_JITTER_MS", 20.0),
        pareto_alpha=_env_float("NAPARI_CUDA_JIT_PARETO_ALPHA", 2.5),
        pareto_scale_ms=_env_float("NAPARI_CUDA_JIT_PARETO_SCALE", 5.0),
        burst_p=_env_float("NAPARI_CUDA_JIT_BURST_P", 0.03),
        burst_min_ms=os.getenv("NAPARI_CUDA_JIT_BURST_MIN_MS"),
        burst_max_ms=os.getenv("NAPARI_CUDA_JIT_BURST_MAX_MS"),
        burst_ms=_env_float("NAPARI_CUDA_JIT_BURST_MS", 80.0),
        loss_p=_env_float("NAPARI_CUDA_JIT_LOSS_P", 0.0),
        reorder_p=_env_float("NAPARI_CUDA_JIT_REORDER_P", 0.0),
        reorder_advance_ms=_env_float("NAPARI_CUDA_JIT_REORDER_ADV_MS", 5.0),
        dup_p=_env_float("NAPARI_CUDA_JIT_DUP_P", 0.0),
        dup_ms=_env_float("NAPARI_CUDA_JIT_DUP_MS", 10.0),
        bw_kbps=_env_int("NAPARI_CUDA_JIT_BW_KBPS", 0),
        burst_bytes=_env_int("NAPARI_CUDA_JIT_BURST_BYTES", 32768),
        affect_pts=_env_bool("NAPARI_CUDA_JIT_AFFECT_PTS", False),
        ts_source=os.getenv("NAPARI_CUDA_JIT_TS_SOURCE", "encode").lower(),
        ts_bias_ms=_env_float("NAPARI_CUDA_JIT_TS_BIAS_MS", 0.0),
        queue_cap=_env_int("NAPARI_CUDA_JIT_QUEUE_CAP", 512),
        seed=_env_int("NAPARI_CUDA_JIT_SEED", 1234),
    )


class JitterChannel:
    """Scheduler that simulates network jitter and loss before enqueuing AUs.

    Downstream must implement `.enqueue(bytes, ts)` and `.qsize()`. Optional
    `.clear()` is used when clearing the channel.
    """

    def __init__(self, pipeline, metrics=None, config: Optional[JitterConfig] = None) -> None:
        self._log = logging.getLogger(__name__)
        self._down = pipeline
        self._metrics = metrics
        self.cfg = config or JitterConfig()
        # Heap of (deliver_time, seq, payload, pts, is_key)
        self._heap: list[Tuple[float, int, bytes, float, bool]] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._rng = random.Random(self.cfg.seed if self.cfg.seed is not None else None)
        self._have_key = False
        # Token bucket
        self._rate_bps = max(0, int(self.cfg.bw_kbps)) * 1000
        self._bucket = float(self.cfg.burst_bytes)
        self._bucket_cap = float(self.cfg.burst_bytes)
        self._last_refill = time.perf_counter()

    # --- Public API ---
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        t = self._thread
        if t is not None:
            t.join(timeout=0.2)
        self._thread = None

    def clear(self) -> None:
        with self._cv:
            self._heap.clear()
            self._have_key = False
            self._cv.notify_all()

    def qsize(self) -> int:
        with self._lock:
            return len(self._heap)

    def submit(self, payload: bytes, pts: float, is_key: bool) -> None:
        if not payload:
            return
        # Loss
        if self._rng.random() < max(0.0, float(self.cfg.loss_p)):
            self._inc("napari_cuda_jit_dropped", 1.0)
            return
        # Schedule delivery time
        now = time.perf_counter()
        delay_ms = self._compute_delay_ms()
        # Reorder: advance by a small amount so it may overtake a previous packet
        if self._rng.random() < max(0.0, float(self.cfg.reorder_p)):
            delay_ms = max(0.0, delay_ms - float(self.cfg.reorder_advance_ms))
            self._inc("napari_cuda_jit_reordered", 1.0)
        deliver_t = now + (delay_ms / 1000.0)
        # PTS policy
        if self.cfg.ts_source == 'send':
            deliver_pts = time.time() + (float(self.cfg.ts_bias_ms) / 1000.0)
        else:
            deliver_pts = float(pts)
            if self.cfg.affect_pts:
                deliver_pts = deliver_pts + (delay_ms / 1000.0)
            if self.cfg.ts_bias_ms:
                deliver_pts = deliver_pts + (float(self.cfg.ts_bias_ms) / 1000.0)
        try:
            self._log.debug(
                "JIT_SUB seq=%d delay_ms=%.3f deliver_t=%.6f pts_wall=%.6f flags(loss=%.2f, reo=%.2f, dup=%.2f, burst_p=%.2f)",
                int(self._seq + 1), float(delay_ms), float(deliver_t), float(deliver_pts),
                float(self.cfg.loss_p), float(self.cfg.reorder_p), float(self.cfg.dup_p), float(self.cfg.burst_p),
            )
        except Exception:
            pass
        # Bounded queue
        with self._cv:
            cap = max(1, int(self.cfg.queue_cap))
            if len(self._heap) >= cap:
                # Latest-wins: drop the oldest scheduled item to keep the new one
                try:
                    heapq.heappop(self._heap)
                except Exception:
                    self._heap.clear()
                self._inc("napari_cuda_jit_dropped", 1.0)
            self._push(deliver_t, payload, deliver_pts, is_key)
            # Duplicate (second copy delayed by dup_ms)
            if self._rng.random() < max(0.0, float(self.cfg.dup_p)):
                dup_t = deliver_t + (float(self.cfg.dup_ms) / 1000.0)
                # Keep the same stamped PTS for duplicate packets
                self._push(dup_t, payload, deliver_pts, is_key)
                self._inc("napari_cuda_jit_duplicated", 1.0)
            self._cv.notify_all()
        self._inc("napari_cuda_jit_submitted", 1.0)
        self._set("napari_cuda_jit_qdepth", float(self.qsize()))

    # --- Internals ---
    def _push(self, deliver_t: float, payload: bytes, pts: float, is_key: bool) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (deliver_t, self._seq, payload, pts, is_key))

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                if not self._heap:
                    self._cv.wait(timeout=0.05)
                    if self._stop:
                        return
                    continue
                deliver_t, seq, payload, pts, is_key = self._heap[0]
                now = time.perf_counter()
                wait = deliver_t - now
                if wait > 0:
                    self._cv.wait(timeout=wait)
                    continue
                heapq.heappop(self._heap)
            # Bandwidth limiter
            need = len(payload)
            self._refill_tokens()
            if self._rate_bps > 0:
                while self._bucket < need and not self._stop:
                    # Sleep until enough tokens accumulate
                    deficit = need - self._bucket
                    sec = deficit / float(self._rate_bps)
                    time.sleep(min(0.02, max(0.0, sec)))
                    self._refill_tokens()
                self._bucket -= need
            # Keyframe gate
            if not self._have_key:
                if not is_key:
                    self._inc("napari_cuda_jit_dropped", 1.0)
                    self._set("napari_cuda_jit_qdepth", float(self.qsize()))
                    continue
                self._have_key = True
            # Deliver
            token_before = float(self._bucket)
            t0 = time.perf_counter()
            try:
                self._down.enqueue(payload, pts)
            except Exception:
                # Swallow errors; count as dropped, but log for visibility
                self._log.debug("jitter: downstream enqueue failed", exc_info=True)
                self._inc("napari_cuda_jit_dropped", 1.0)
                continue
            t1 = time.perf_counter()
            try:
                self._log.debug(
                    "JIT_DELIVER seq=%d now=%.6f sched_t=%.6f wait_ms=%.3f bytes=%d tokens_before=%.1f tokens_after=%.1f",
                    int(seq), float(t1), float(deliver_t), (t1 - float(deliver_t)) * 1000.0, int(need), token_before, float(self._bucket)
                )
            except Exception:
                pass
            self._observe_ms("napari_cuda_jit_sched_delay_ms", (t1 - t0) * 1000.0)
            self._inc("napari_cuda_jit_delivered", 1.0)
            self._set("napari_cuda_jit_qdepth", float(self.qsize()))

    def _refill_tokens(self) -> None:
        if self._rate_bps <= 0:
            return
        now = time.perf_counter()
        dt = now - self._last_refill
        if dt <= 0:
            return
        self._last_refill = now
        self._bucket = min(self._bucket_cap, self._bucket + dt * self._rate_bps)
        self._set("napari_cuda_jit_bw_tokens", float(self._bucket))

    def _compute_delay_ms(self) -> float:
        d = float(self.cfg.base_ms)
        m = self.cfg.mode
        if m == "normal":
            # Zero-mean normal with sigma=jitter_ms
            d += self._rng.gauss(0.0, float(self.cfg.jitter_ms))
        elif m == "pareto":
            # Heavy tail; draw from Pareto then subtract mean to make zero-ish mean
            alpha = max(0.1, float(self.cfg.pareto_alpha))
            scale = max(0.0, float(self.cfg.pareto_scale_ms))
            x = (self._rng.paretovariate(alpha) - 1.0) * scale
            # Center roughly by subtracting median of Pareto(alpha)
            # median = scale * (2**(1/alpha) - 1)
            median = scale * ((2 ** (1.0 / alpha)) - 1.0)
            d += (x - median)
        else:  # uniform
            j = float(self.cfg.jitter_ms)
            d += self._rng.uniform(-j, j)
        # Bursts
        if self._rng.random() < max(0.0, float(self.cfg.burst_p)):
            if self.cfg.burst_min_ms is not None and self.cfg.burst_max_ms is not None:
                try:
                    bmin = float(self.cfg.burst_min_ms)
                    bmax = float(self.cfg.burst_max_ms)
                    d += self._rng.uniform(bmin, bmax)
                except Exception:
                    d += float(self.cfg.burst_ms)
            else:
                d += float(self.cfg.burst_ms)
        return max(0.0, d)

    # --- Metrics helpers ---
    def _inc(self, name: str, v: float) -> None:
        m = self._metrics
        if m is not None:
            try:
                m.inc(name, float(v))
            except Exception:
                self._log.debug("jitter: metrics inc failed (%s)", name, exc_info=True)

    def _set(self, name: str, v: float) -> None:
        m = self._metrics
        if m is not None:
            try:
                m.set(name, float(v))
            except Exception:
                self._log.debug("jitter: metrics set failed (%s)", name, exc_info=True)

    def _observe_ms(self, name: str, v: float) -> None:
        m = self._metrics
        if m is not None:
            try:
                m.observe_ms(name, float(v))
            except Exception:
                self._log.debug("jitter: metrics observe_ms failed (%s)", name, exc_info=True)
