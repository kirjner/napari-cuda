"""
EGLHeadlessServer - Async server harness for headless EGL rendering + NVENC streaming.
"""

from __future__ import annotations

import asyncio
import json
import base64
import logging
import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Deque, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed
import importlib.resources as ilr
import socket

# Encoder profile presets for convenient NVENC tuning
def _apply_encoder_profile(profile: str) -> None:
    profiles: dict[str, dict[str, str]] = {
        'latency': {
            'NAPARI_CUDA_RC': 'cbr',
        },
        'quality': {
            'NAPARI_CUDA_RC': 'vbr',
            'NAPARI_CUDA_BITRATE': '35000000',
            'NAPARI_CUDA_MAXBITRATE': '45000000',
            'NAPARI_CUDA_LOOKAHEAD': '10',
            'NAPARI_CUDA_AQ': '1',
            'NAPARI_CUDA_TEMPORALAQ': '1',
            'NAPARI_CUDA_BFRAMES': '2',
            'NAPARI_CUDA_PRESET': 'P5',
            'NAPARI_CUDA_IDR_PERIOD': '120',
        },
    }
    settings = profiles.get((profile or '').lower())
    if not settings:
        return
    for key, value in settings.items():
        if key and value and key not in os.environ:
            os.environ[key] = value
    try:
        logger.info("Applied encoder profile '%s' (override via env vars as needed)", profile)
    except Exception:
        logger.debug("Failed to log encoder profile %s", profile, exc_info=True)

from .egl_worker import CameraCommand, EGLRendererWorker, ServerSceneState
from .layer_manager import ViewerSceneManager
from .bitstream import ParamCache, pack_to_avcc, build_avcc_config
from .metrics import Metrics
from napari_cuda.utils.env import env_bool
from .zarr_source import ZarrSceneSource, ZarrSceneSourceError
from napari_cuda.server.config import load_server_config, ServerConfig

logger = logging.getLogger(__name__)


@dataclass
class EncodeConfig:
    fps: int = 60
    codec: int = 1  # 1=h264, 2=hevc, 3=av1
    bitrate: int = 10_000_000
    keyint: int = 120


 


class EGLHeadlessServer:
    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False,
                 host: str = '0.0.0.0', state_port: int = 8081, pixel_port: int = 8082, fps: int = 60,
                 animate: bool = False, animate_dps: float = 30.0, log_sends: bool = False,
                 zarr_path: str | None = None, zarr_level: str | None = None,
                 zarr_axes: str | None = None, zarr_z: int | None = None,
                 debug: bool = False) -> None:
        self.width = width
        self.height = height
        self.use_volume = use_volume
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        self.cfg = EncodeConfig(fps=fps)
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception:
            self._animate_dps = 30.0

        self.metrics = Metrics()
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._state_clients: Set[websockets.WebSocketServerProtocol] = set()
        try:
            # Keep queue size at 1 for latest-wins, never-block behavior
            qsize = int(os.getenv('NAPARI_CUDA_FRAME_QUEUE', '1'))
        except Exception:
            qsize = 1
        # Queue holds tuples of (payload_bytes, flags, seq, stamp_ts)
        # stamp_ts is minted post-pack in on_frame to keep encode/pack jitter common-mode
        self._frame_q: asyncio.Queue[tuple[bytes, int, int, float]] = asyncio.Queue(maxsize=max(1, qsize))
        self._seq = 0
        self._stop = threading.Event()
        

        self._worker: Optional[EGLRendererWorker] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._latest_state = ServerSceneState()
        self._camera_commands: Deque[CameraCommand] = deque()
        # Bitstream parameter cache and config tracking (server-side)
        self._param_cache = ParamCache()
        self._needs_config = True
        self._last_avcc: Optional[bytes] = None
        # Optional bitstream dump for validation
        try:
            self._dump_remaining = int(os.getenv('NAPARI_CUDA_DUMP_BITSTREAM', '0'))
        except Exception:
            self._dump_remaining = 0
        self._dump_dir = os.getenv('NAPARI_CUDA_DUMP_DIR', 'benchmarks/bitstreams')
        self._dump_path: Optional[str] = None
        # Track last keyframe for metrics only
        self._last_key_seq: Optional[int] = None
        self._last_key_ts: Optional[float] = None
        # Watchdog task handle (cancel when keyframe arrives)
        self._kf_watchdog_task: Optional[asyncio.Task] = None
        # Broadcaster pacing: bypass once until keyframe for immediate start
        self._bypass_until_key: bool = False
        # Keyframe watchdog cooldown to avoid rapid encoder resets
        self._kf_last_reset_ts: Optional[float] = None
        _cool = os.getenv('NAPARI_CUDA_KF_WATCHDOG_COOLDOWN')
        try:
            self._kf_watchdog_cooldown_s = float(_cool) if _cool else 2.0
        except ValueError:
            self._kf_watchdog_cooldown_s = 2.0
        # State access synchronization for latest-wins camera op coalescing
        self._state_lock = threading.Lock()
        # Logging controls for camera ops
        self._log_cam_info: bool = env_bool('NAPARI_CUDA_LOG_CAMERA_INFO', False)
        self._log_cam_debug: bool = env_bool('NAPARI_CUDA_LOG_CAMERA_DEBUG', False)
        logger.info(
            "camera logging flags info=%s debug=%s",
            self._log_cam_info,
            self._log_cam_debug,
        )
        # State trace toggles (per-message start/end logs)
        self._log_state_traces: bool = env_bool('NAPARI_CUDA_LOG_STATE_TRACES', False)
        # Logging controls for volume/multiscale intents
        self._log_volume_info: bool = env_bool('NAPARI_CUDA_LOG_VOLUME_INFO', False)
        # Force IDR on reset (default True)
        self._idr_on_reset: bool = env_bool('NAPARI_CUDA_IDR_ON_RESET', True)

        # Drop/send tracking
        self._drops_total: int = 0
        self._last_send_ts: Optional[float] = None
        self._send_count: int = 0
        # Optional detailed per-send logging (seq, send_ts, stamp_ts, delta)
        try:
            self._log_sends = bool(log_sends or int(os.getenv('NAPARI_CUDA_LOG_SENDS', '0') or '0'))
        except Exception:
            self._log_sends = bool(log_sends)

        # Data configuration (optional OME-Zarr dataset for real data)
        self._zarr_path = zarr_path or os.getenv('NAPARI_CUDA_ZARR_PATH') or None
        self._zarr_level = zarr_level or os.getenv('NAPARI_CUDA_ZARR_LEVEL') or None
        self._zarr_axes = zarr_axes or os.getenv('NAPARI_CUDA_ZARR_AXES') or None
        try:
            _z = zarr_z if zarr_z is not None else int(os.getenv('NAPARI_CUDA_ZARR_Z', '-1'))
            self._zarr_z = _z if _z >= 0 else None
        except Exception:
            self._zarr_z = None
        self._policy_metrics_snapshot: Dict[str, object] = {}
        # Policy event writer state (single-writer JSONL)
        try:
            self._last_written_decision_seq: int = 0
        except Exception:
            self._last_written_decision_seq = 0
        self._policy_event_path: Path = Path(os.getenv('NAPARI_CUDA_POLICY_EVENT_PATH', 'tmp/policy_events.jsonl'))
        # Verbose dims logging control: default debug, upgrade to info with flag
        self._log_dims_info: bool = env_bool('NAPARI_CUDA_LOG_DIMS_INFO', False)
        # Dedicated debug control for this module only (no dependency loggers)
        self._debug_only_this_logger: bool = bool(debug) or env_bool('NAPARI_CUDA_DEBUG', False)
        # Dims intent/update tracking (server-authoritative dims seq and last client)
        self._dims_seq: int = 0
        self._last_dims_client_id: Optional[str] = None
        # Volume and multiscale server-side state (reported via dims.update meta)
        self._volume_state: dict = {
            'mode': 'mip',           # 'mip' | 'translucent' | 'iso'
            'colormap': 'gray',      # name string
            'clim': [0.0, 1.0],      # [lo, hi]
            'opacity': 1.0,          # 0..1
            'sample_step': 1.0,      # relative multiplier
        }
        self._ms_state: dict = {
            'levels': [],            # [{shape: [...], downsample: [...]}]
            'current_level': 0,
            'policy': 'oversampling',  # simplified policy; legacy 'fixed' removed
            'index_space': 'base',
        }
        self._allowed_render_modes = {'mip', 'translucent', 'iso'}
        # Populate multiscale description from NGFF metadata if available
        try:
            self._populate_multiscale_state()
        except Exception:
            logger.debug("populate multiscale state failed", exc_info=True)

        self._scene_manager = ViewerSceneManager((self.width, self.height))

    # --- Logging + Broadcast helpers --------------------------------------------
    def _enqueue_camera_command(self, cmd: CameraCommand) -> None:
        with self._state_lock:
            self._camera_commands.append(cmd)
            queue_len = len(self._camera_commands)
        if self._log_cam_info or self._log_cam_debug:
            logger.info(
                "enqueue camera command kind=%s queue_len=%d payload=%s",
                cmd.kind,
                queue_len,
                cmd,
            )

    def _log_volume_intent(self, fmt: str, *args) -> None:
        try:
            if self._log_volume_info:
                logger.info(fmt, *args)
            else:
                logger.debug(fmt, *args)
        except Exception:
            logger.debug("volume intent log failed", exc_info=True)

    def _schedule_coro(self, coro: Awaitable[None], label: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        task = loop.create_task(coro)

        def _log_task_result(t: asyncio.Task) -> None:
            try:
                t.result()
            except Exception:
                logger.exception("Scheduled task '%s' failed", label)

        task.add_done_callback(_log_task_result)

    async def _await_adapter_level_ready(self, timeout_s: float = 0.5) -> None:
        """Wait briefly for the adapter's multiscale level to stabilize.

        This reduces races where the first dims.update after connect is built
        before the adapter selects the intended level. Returns after timeout or
        when two consecutive reads of (current_level, level_shape) match.
        """
        start = time.perf_counter()
        last: tuple[int | None, tuple[int, ...] | None] | None = None
        # Wait for worker and scene source to appear, then for level/shape to stabilize
        while (time.perf_counter() - start) < max(0.0, float(timeout_s)):
            try:
                if self._worker is None:
                    await asyncio.sleep(0.01)
                    continue
                src = getattr(self._worker, '_scene_source', None)
                if src is None:
                    await asyncio.sleep(0.01)
                    continue
                lvl = None
                try:
                    lvl = int(getattr(src, 'current_level', 0))
                except Exception:
                    lvl = None
                shp: tuple[int, ...] | None = None
                try:
                    descs = getattr(src, 'level_descriptors', [])
                    if isinstance(descs, list) and descs and lvl is not None and 0 <= lvl < len(descs):
                        s = getattr(descs[int(lvl)], 'shape', None)
                        if isinstance(s, (list, tuple)):
                            shp = tuple(int(x) for x in s)
                except Exception:
                    shp = None
                cur: tuple[int | None, tuple[int, ...] | None] = (lvl, shp)
                if last is not None and cur == last:
                    return
                last = cur
            except Exception:
                break
            await asyncio.sleep(0.01)
        return

    async def _rebroadcast_meta(self, client_id: Optional[str]) -> None:
        """Re-broadcast a dims.update with current_step and updated meta.

        Safe to call after mutating volume/multiscale state. Never raises.
        """
        try:
            cur = None
            with self._state_lock:
                cur = getattr(self._latest_state, 'current_step', None)
            step_list = list(cur) if isinstance(cur, (list, tuple)) else [0]
            await self._broadcast_dims_update(step_list, last_client_id=client_id)
        except Exception as e:
            logger.debug("rebroadcast meta failed: %s", e)

    async def _ensure_keyframe(self) -> None:
        """Request a clean keyframe and set up watchdog + pacing bypass.

        Tries a lightweight IDR request first; falls back to encoder reset.
        Also rebroadcasts current video_config if available.
        """
        if self._worker is None:
            return
        # Try force IDR; fall back to reset without nesting
        forced = False
        try:
            self._worker.force_idr()
            forced = True
        except Exception:
            logger.debug("force_idr request failed; will reset encoder", exc_info=True)
        if not forced:
            try:
                self._worker.reset_encoder()
            except Exception:
                logger.exception("Encoder reset failed in ensure_keyframe")
                return
            self._kf_last_reset_ts = time.time()
        # Bypass pacing once to deliver next keyframe immediately
        self._bypass_until_key = True
        # Count encoder force/reset; log failure instead of passing
        try:
            self.metrics.inc('napari_cuda_encoder_resets')
        except Exception:
            logger.debug("metrics inc failed: napari_cuda_encoder_resets", exc_info=True)
        # Start/restart watchdog to hard reset if no keyframe within 300 ms
        self._start_kf_watchdog()
        # Re-broadcast current video config to tighten resync window
        if self._last_avcc is not None:
            msg = {
                'type': 'video_config',
                'codec': 'h264',
                'format': 'avcc',
                'data': base64.b64encode(self._last_avcc).decode('ascii'),
                'width': self.width,
                'height': self.height,
                'fps': self.cfg.fps,
            }
            await self._broadcast_state_json(msg)
        else:
            self._needs_config = True

    async def _handle_set_ndisplay(self, ndisplay: int, client_id: Optional[str], client_seq: Optional[int]) -> None:
        """Apply a 2D/3D view toggle request.

        - Normalizes `ndisplay` to 2 or 3.
        - Requests the worker to switch pipelines if supported.
        - Forces a keyframe for immediate visual change.
        - Rebroadcasts dims meta to keep clients synchronized.
        """
        try:
            ndisp = 3 if int(ndisplay) >= 3 else 2
        except Exception:
            ndisp = 2
        try:
            if self._log_dims_info:
                logger.info("intent: view.set_ndisplay ndisplay=%d client_id=%s seq=%s", int(ndisp), client_id, client_seq)
            else:
                logger.debug("intent: view.set_ndisplay ndisplay=%d client_id=%s seq=%s", int(ndisp), client_id, client_seq)
        except Exception:
            pass
        # Ask worker to apply the mode switch on the render thread
        if self._worker is not None and hasattr(self._worker, 'request_ndisplay'):
            try:
                self._worker.request_ndisplay(int(ndisp))  # type: ignore[attr-defined]
                # Force a keyframe and bypass pacing so the switch is immediate
                try:
                    self._worker.force_idr()
                    self._bypass_until_key = True
                except Exception:
                    logger.debug("view.set_ndisplay: force_idr failed", exc_info=True)
            except Exception:
                logger.exception("view.set_ndisplay: worker request failed")
        # Re-broadcast dims meta so clients reflect the authoritative state
        try:
            await self._rebroadcast_meta(client_id)
        except Exception:
            logger.debug("view.set_ndisplay: rebroadcast failed", exc_info=True)

    def _start_kf_watchdog(self) -> None:
        async def _kf_watchdog(last_key_seq: Optional[int]):
            await asyncio.sleep(0.30)
            if self._last_key_seq == last_key_seq and self._worker is not None:
                now = time.time()
                if self._kf_last_reset_ts is not None and (now - self._kf_last_reset_ts) < self._kf_watchdog_cooldown_s:
                    rem = self._kf_watchdog_cooldown_s - (now - self._kf_last_reset_ts)
                    logger.debug("Keyframe watchdog cooldown active (%.2fs remaining); skip reset", rem)
                    return
                logger.warning("Keyframe watchdog fired; resetting encoder")
                try:
                    self._worker.reset_encoder()
                except Exception:
                    logger.exception("Encoder reset failed during keyframe watchdog")
                    return
                self._bypass_until_key = True
                self._kf_last_reset_ts = now
        try:
            if self._kf_watchdog_task is not None and not self._kf_watchdog_task.done():
                self._kf_watchdog_task.cancel()
            self._kf_watchdog_task = asyncio.create_task(_kf_watchdog(self._last_key_seq))
        except Exception:
            logger.debug("start watchdog failed", exc_info=True)

    # --- Meta builders ------------------------------------------------------------
    def _populate_multiscale_state(self) -> None:
        """Populate self._ms_state['levels'] from NGFF multiscales if available.

        Stores minimal fields needed by clients and worker switching:
        - path: dataset subpath (e.g., 'level_03')
        - downsample: per-axis scale factors ordered as z,y,x when axes present; else [1,1,1]
        - shape: optional, if inexpensive to obtain (omitted here for speed)
        """
        root = self._zarr_path
        if not root:
            return
        axes_override = tuple(self._zarr_axes) if isinstance(self._zarr_axes, str) else None
        try:
            source = ZarrSceneSource(root, preferred_level=self._zarr_level, axis_override=axes_override)
        except ZarrSceneSourceError:
            logger.debug("multiscale state: invalid Zarr scene source", exc_info=True)
            return
        except Exception:
            logger.debug("multiscale state: unexpected error building Zarr scene source", exc_info=True)
            return

        levels: list[dict] = []
        for desc in source.level_descriptors:
            levels.append(
                {
                    'path': desc.path,
                    'downsample': list(desc.downsample),
                    'shape': [int(x) for x in desc.shape],
                }
            )

        if levels:
            self._ms_state['levels'] = levels
            self._ms_state['current_level'] = int(source.current_level)
            self._zarr_axes = ''.join(source.axes)
            current_desc = source.level_descriptors[source.current_level]
            if current_desc.path:
                self._zarr_level = current_desc.path

    def _maybe_enable_debug_logger(self) -> None:
        """Enable DEBUG logs for this module only, leaving root/others unchanged.

        - Attaches a dedicated StreamHandler at DEBUG to our module logger.
        - Disables propagation to avoid duplicate INFO/WARNING via root handlers.
        - Does nothing unless NAPARI_CUDA_DEBUG or --debug is provided.
        """
        if not getattr(self, '_debug_only_this_logger', False):
            return
        try:
            # Avoid stacking handlers on repeated calls
            has_ours = any(isinstance(h, logging.StreamHandler) and getattr(h, '_napari_cuda_local', False)
                           for h in logger.handlers)
            if has_ours:
                return
            h = logging.StreamHandler()
            fmt = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            h.setFormatter(logging.Formatter(fmt))
            h.setLevel(logging.DEBUG)
            # Tag so we don't add duplicates later
            setattr(h, '_napari_cuda_local', True)
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)
            # Ensure our records don't also bubble to root (which may be INFO)
            logger.propagate = False
        except Exception:
            # Fail silent; we don't want logging issues to break the server
            pass

    # --- Helper to compute dims metadata for piggyback on dims_update ---------
    def _dims_metadata(self) -> dict:
        try:
            self._update_scene_manager()
        except Exception:
            logger.debug("scene manager update failed during dims metadata", exc_info=True)
        # Use centralized helper for HUD dims/meta
        from napari_cuda.server import scene_spec as _scn
        scene = self._scene_manager.scene_spec()
        return _scn.dims_metadata(scene)

    def _update_scene_manager(self) -> None:
        current_step = None
        with self._state_lock:
            if getattr(self._latest_state, 'current_step', None) is not None:
                current_step = list(self._latest_state.current_step)  # type: ignore[arg-type]
        extras = {
            'zarr_axes': (self._worker._zarr_axes if self._worker is not None else None),
            'zarr_level': self._zarr_level,
        }
        if self._policy_metrics_snapshot:
            extras['policy_metrics'] = self._policy_metrics_snapshot
        if self._worker is not None:
            source = getattr(self._worker, '_scene_source', None)
            if source is not None:
                extras['zarr_scale'] = list(source.level_scale(source.current_level))
                extras['multiscale_levels'] = [
                    {
                        'index': desc.index,
                        'path': desc.path,
                        'shape': [int(x) for x in desc.shape],
                        'downsample': list(desc.downsample),
                    }
                    for desc in source.level_descriptors
                ]
                extras['multiscale_current_level'] = int(source.current_level)
                # Keep multiscale server state in sync so client HUD reflects live level
                try:
                    # Advertise adaptive policy; reflect zoom-driven switches
                    self._ms_state['policy'] = 'auto'
                    self._ms_state['current_level'] = int(source.current_level)
                    # Refresh levels if descriptor count changed (defensive)
                    levels = self._ms_state.get('levels') or []
                    if not isinstance(levels, list) or len(levels) != len(source.level_descriptors):
                        self._ms_state['levels'] = [
                            {
                                'path': desc.path,
                                'downsample': list(desc.downsample),
                                'shape': [int(x) for x in desc.shape],
                            }
                            for desc in source.level_descriptors
                        ]
                except Exception:
                    logger.debug('ms_state sync failed', exc_info=True)
        viewer_model = None
        if self._worker is not None:
            try:
                viewer_model = self._worker.viewer_model()
                if viewer_model is not None:
                    extras.setdefault('adapter_engine', 'napari-vispy')
            except Exception:
                logger.debug('worker viewer_model fetch failed', exc_info=True)
        self._scene_manager.update_from_sources(
            worker=self._worker,
            scene_state=self._latest_state,
            multiscale_state=dict(self._ms_state),
            volume_state=dict(self._volume_state),
            current_step=current_step,
            ndisplay=self._current_ndisplay(),
            zarr_path=self._zarr_path,
            viewer_model=viewer_model,
            extras=extras,
        )

    def _current_ndisplay(self) -> int:
        if self._worker is not None and bool(getattr(self._worker, 'use_volume', False)):
            return 3
        if bool(self.use_volume):
            return 3
        return 2

    # --- Validation helpers -------------------------------------------------------
    def _is_valid_render_mode(self, mode: str) -> bool:
        return str(mode or '').lower() in self._allowed_render_modes

    def _normalize_clim(self, lo: object, hi: object) -> Optional[tuple[float, float]]:
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception:
            return None
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        return (lo_f, hi_f)

    def _clamp_opacity(self, a: object) -> Optional[float]:
        try:
            aa = float(a)
        except Exception:
            return None
        return max(0.0, min(1.0, aa))

    def _clamp_sample_step(self, rel: object) -> Optional[float]:
        try:
            rr = float(rel)
        except Exception:
            return None
        return max(0.1, min(4.0, rr))

    def _clamp_level(self, lvl: object) -> Optional[int]:
        try:
            iv = int(lvl)
        except Exception:
            return None
        levels = self._ms_state.get('levels') or []
        n = len(levels)
        if n > 0:
            return max(0, min(n - 1, iv))
        return max(0, iv)

    # --- Dims intent helpers --------------------------------------------------
    def _resolve_axis_index(self, axis: object, meta: dict, cur_len: int) -> Optional[int]:
        """Resolve an axis specifier to a concrete index in current_step.

        Accepts an integer index, a numeric string, or a label present in
        meta['order'] or meta['axis_labels']. Returns None if unresolved.
        """
        try:
            # Direct integer index
            if isinstance(axis, int):
                idx = int(axis)
                return idx if 0 <= idx < max(0, int(cur_len)) else None
            # Numeric string index
            if isinstance(axis, str) and axis.strip().isdigit():
                idx2 = int(axis.strip())
                return idx2 if 0 <= idx2 < max(0, int(cur_len)) else None
            # Label lookup: order then axis_labels
            if isinstance(axis, str):
                ax = axis.strip().lower()
                order = meta.get('order') or []
                if isinstance(order, (list, tuple)):
                    try:
                        pos = [str(x).lower() for x in order].index(ax)
                        return pos if 0 <= pos < max(0, int(cur_len)) else None
                    except Exception:
                        pass
                labels = meta.get('axis_labels') or []
                if isinstance(labels, (list, tuple)):
                    try:
                        pos = [str(x).lower() for x in labels].index(ax)
                        return pos if 0 <= pos < max(0, int(cur_len)) else None
                    except Exception:
                        pass
        except Exception:
            logger.debug("resolve axis failed", exc_info=True)
        return None

    def _build_dims_update_message(
        self,
        step_list: list[int],
        last_client_id: Optional[str],
        *,
        ack: bool = False,
        intent_seq: Optional[int] = None,
    ) -> dict:
        """Build dims.update payload with nested metadata.

        {'type':'dims.update', 'seq':..., 'last_client_id':..., 'current_step':[...], 'meta':{...}}
        """
        meta = self._dims_metadata() or {}
        # Always include ndisplay in meta for new, and as top-level for legacy
        try:
            # If 3D and volume mode, prefer ndisplay=3; else 2
            ndisp = 3 if int(meta.get('ndim') or 2) >= 3 and bool(meta.get('volume')) else 2
        except Exception:
            ndisp = 2
        # Inflate/align current_step to full length and order
        try:
            ndim = int(meta.get('ndim') or 0)
        except Exception:
            ndim = 0
        if ndim <= 0:
            ndim = len(step_list) if step_list else 1
        order = meta.get('order') if isinstance(meta.get('order'), (list, tuple)) else []
        rng = meta.get('range') if isinstance(meta.get('range'), (list, tuple)) else None
        # Start with zeros, then place provided indices
        full = [0 for _ in range(int(ndim))]
        if len(step_list) >= int(ndim):
            full = [int(x) for x in step_list[:int(ndim)]]
        elif len(step_list) == 1:
            val = int(step_list[0])
            try:
                if order and 'z' in [str(c).lower() for c in order]:
                    # Place value into the 'z' axis index if present
                    lower = [str(c).lower() for c in order]
                    zi = lower.index('z')
                    if 0 <= zi < len(full):
                        full[zi] = val
                    else:
                        full[0] = val
                else:
                    full[0] = val
            except Exception:
                full[0] = val
        else:
            # Partial list: place sequentially starting from 0
            for i, v in enumerate(step_list):
                if i < len(full):
                    full[i] = int(v)
        # Clamp to meta range if available
        if rng:
            try:
                for i in range(min(len(full), len(rng))):
                    lohi = rng[i]
                    if isinstance(lohi, (list, tuple)) and len(lohi) >= 2:
                        lo, hi = int(lohi[0]), int(lohi[1])
                        if hi < lo:
                            lo, hi = hi, lo
                        full[i] = max(lo, min(hi, int(full[i])))
            except Exception:
                logger.debug("dims.update clamp in builder failed", exc_info=True)
        # New shape with nested meta
        seq_val = int(self._dims_seq) & 0x7FFFFFFF
        self._dims_seq = (int(self._dims_seq) + 1) & 0x7FFFFFFF
        # Build base message
        new_msg: dict = {
            'type': 'dims.update',
            'seq': seq_val,
            'last_client_id': last_client_id,
            'current_step': list(int(x) for x in full),
            'meta': {**meta, 'ndisplay': int(ndisp)},
        }
        # Enrich meta with flattened, per-level details when available
        try:
            src = getattr(self._worker, '_scene_source', None) if self._worker is not None else None
            if src is not None:
                try:
                    lvl = int(getattr(src, 'current_level', 0))
                    new_msg['meta']['level'] = lvl
                except Exception:
                    pass
                try:
                    descs = getattr(src, 'level_descriptors', [])
                    if isinstance(descs, list) and descs:
                        cur = lvl if (0 <= lvl < len(descs)) else 0
                        shape = getattr(descs[cur], 'shape', None)
                        if shape is not None:
                            new_msg['meta']['level_shape'] = [int(s) for s in shape]
                except Exception:
                    pass
                try:
                    dt = getattr(src, 'dtype', None)
                    if dt is not None:
                        new_msg['meta']['dtype'] = str(dt)
                except Exception:
                    pass
                # For 2D slice path we normalize slabs to [0,1]
                try:
                    normalized = (not bool(getattr(self._worker, 'use_volume', False)))
                    new_msg['meta']['normalized'] = bool(normalized)
                except Exception:
                    pass
        except Exception:
            logger.debug('dims.update meta enrichment failed', exc_info=True)
        # Ensure sizes/range reflect the active adapter-selected level, overriding any stale cache.
        # We derive a fresh multiscale descriptor from the live scene source when available.
        try:
            meta_obj = new_msg.get('meta') if isinstance(new_msg.get('meta'), dict) else None
            if isinstance(meta_obj, dict):
                eff_shape = None
                # Hard source of truth: live adapter source
                src = getattr(self._worker, '_scene_source', None) if self._worker is not None else None
                if src is not None:
                    try:
                        cur = int(getattr(src, 'current_level', 0))
                    except Exception:
                        cur = 0
                    try:
                        descs = getattr(src, 'level_descriptors', [])
                    except Exception:
                        descs = []
                    # Overwrite/construct multiscale meta from descriptors
                    if isinstance(descs, list) and descs:
                        levels_meta: list[dict] = []
                        for d in descs:
                            try:
                                levels_meta.append({
                                    'shape': [int(s) for s in (getattr(d, 'shape', []) or [])],
                                    'downsample': list(getattr(d, 'downsample', []) or []),
                                    'path': getattr(d, 'path', None),
                                })
                            except Exception:
                                levels_meta.append({'shape': list(getattr(d, 'shape', []) or [])})
                        meta_obj['multiscale'] = {
                            'levels': levels_meta,
                            'current_level': cur,
                            'policy': (self._ms_state.get('policy') if isinstance(self._ms_state, dict) else None) or 'auto',
                            'index_space': 'base',
                        }
                        if 0 <= cur < len(levels_meta):
                            sh = levels_meta[cur].get('shape')
                            if isinstance(sh, (list, tuple)):
                                eff_shape = [int(max(1, int(s))) for s in sh]
                # Secondary: explicit level_shape from earlier enrichment
                if eff_shape is None:
                    lvl_shape = meta_obj.get('level_shape')
                    if isinstance(lvl_shape, (list, tuple)) and len(lvl_shape) > 0:
                        eff_shape = [int(max(1, int(s))) for s in lvl_shape]
                # Tertiary: cached multiscale block, if consistent
                if eff_shape is None:
                    ms = meta_obj.get('multiscale')
                    if isinstance(ms, dict):
                        cur = int(ms.get('current_level', 0))
                        levels = ms.get('levels')
                        if isinstance(levels, (list, tuple)) and 0 <= cur < len(levels):
                            entry = levels[cur]
                            if isinstance(entry, dict):
                                sh = entry.get('shape')
                                if isinstance(sh, (list, tuple)):
                                    eff_shape = [int(max(1, int(s))) for s in sh]
                if eff_shape:
                    meta_obj['sizes'] = list(eff_shape)
                    meta_obj['range'] = [[0, max(0, int(s) - 1)] for s in eff_shape]
        except Exception:
            logger.debug('dims.update: live level reconciliation failed', exc_info=True)
        # Optional verbose meta log to diagnose slider bounds/ranges
        try:
            if getattr(self, '_log_dims_info', False):
                meta_dbg = new_msg.get('meta', {}) if isinstance(new_msg.get('meta'), dict) else {}
                logger.info(
                    "dims.update meta: level=%s level_shape=%s sizes=%s range=%s",
                    meta_dbg.get('level'),
                    meta_dbg.get('level_shape'),
                    meta_dbg.get('sizes'),
                    meta_dbg.get('range'),
                )
        except Exception:
            logger.debug('dims.update: meta logging failed', exc_info=True)
        # Provide both 'range' and 'ranges' aliases for client normalization
        try:
            rng = new_msg.get('meta', {}).get('range')
            if rng is not None and isinstance(new_msg.get('meta'), dict):
                new_msg['meta']['ranges'] = rng
        except Exception:
            logger.debug('dims.update ranges alias failed', exc_info=True)
        # ACK fields for deterministic client reconciliation
        try:
            if ack:
                new_msg['ack'] = True
                if intent_seq is not None:
                    new_msg['intent_seq'] = int(intent_seq)
        except Exception:
            logger.debug('dims.update ack/intent_seq set failed', exc_info=True)
        return new_msg

    async def _broadcast_dims_update(self, step_list: list[int], last_client_id: Optional[str], *, ack: bool = False, intent_seq: Optional[int] = None) -> None:
        """Broadcast dims update in both new and legacy formats.

        Never raises; logs and continues on failure.
        """
        try:
            self._last_dims_client_id = last_client_id
            obj = self._build_dims_update_message(step_list, last_client_id, ack=ack, intent_seq=intent_seq)
            await self._broadcast_state_json(obj)
            # Intentionally skip scene.spec here to avoid client layer churn on
            # pure dims changes. Scene spec is sent on init and when topology/
            # render configuration changes, not on slider steps.
        except Exception:
            logger.debug("dims update broadcast failed", exc_info=True)

    def _apply_dims_intent(self, axis: object, step_delta: Optional[int], set_value: Optional[int]) -> Optional[list[int]]:
        """Apply a dims intent to server state and return the new step list.

        Axis may be label or index. Clamps result to meta range when available.
        """
        # Capture current step and infer length
        with self._state_lock:
            cur = getattr(self._latest_state, 'current_step', None)
        meta = self._dims_metadata() or {}
        try:
            ndim = int(meta.get('ndim') or (len(cur) if cur is not None else 0))
        except Exception:
            ndim = len(cur) if cur is not None else 0
        if ndim <= 0:
            # Fallback to at least 1D if we can infer from cur
            ndim = len(cur) if cur is not None else 1
        # Build a working list from current or zeros
        step = list(int(x) for x in (list(cur) if cur is not None else [0] * int(ndim)))
        if len(step) < int(ndim):
            step.extend([0] * (int(ndim) - len(step)))
        idx = self._resolve_axis_index(axis, meta, len(step))
        if idx is None:
            # Default to first axis
            idx = 0 if len(step) > 0 else None
        if idx is None:
            return None
        # Compute target value
        target = step[idx]
        if step_delta is not None:
            try:
                target = int(target) + int(step_delta)
            except Exception:
                target = int(target)
        if set_value is not None:
            try:
                target = int(set_value)
            except Exception:
                target = int(target)
        # Clamp by range if available
        try:
            rng = meta.get('range')
            if isinstance(rng, (list, tuple)) and idx < len(rng):
                lohi = rng[idx]
                if isinstance(lohi, (list, tuple)) and len(lohi) >= 2:
                    lo = int(lohi[0]); hi = int(lohi[1])
                    if hi < lo:
                        lo, hi = hi, lo
                    target = max(lo, min(hi, int(target)))
        except Exception:
            logger.debug("clamp to range failed", exc_info=True)
        # Update state
        step[idx] = int(target)
        with self._state_lock:
            s = self._latest_state
            self._latest_state = ServerSceneState(
                center=s.center,
                zoom=s.zoom,
                angles=s.angles,
                current_step=tuple(step),
            )
        return step

    async def start(self) -> None:
        # Keep global logging at INFO; optionally enable module-only DEBUG logging
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
        logger.info(
            "camera logging flags info=%s debug=%s",
            self._log_cam_info,
            self._log_cam_debug,
        )
        # Apply module-only debug after basicConfig so our handler takes precedence
        self._maybe_enable_debug_logger()
        # Observe-only: resolve and log ServerConfig with visibility
        try:
            _cfg: ServerConfig = load_server_config()
            if self._debug_only_this_logger:
                logger.info("Resolved ServerConfig: %s", _cfg)
            else:
                logger.debug("Resolved ServerConfig: %s", _cfg)
        except Exception:
            logger.debug("ServerConfig load/log failed", exc_info=True)
        logger.info("Starting EGLHeadlessServer %dx%d @ %dfps", self.width, self.height, self.cfg.fps)
        loop = asyncio.get_running_loop()
        self._start_worker(loop)
        # Start websocket servers; disable permessage-deflate to avoid CPU and latency on large frames
        state_server = await websockets.serve(
            self._handle_state, self.host, self.state_port, compression=None
        )
        pixel_server = await websockets.serve(
            self._handle_pixel, self.host, self.pixel_port, compression=None
        )
        metrics_server = await self._start_metrics_server()
        logger.info(
            "WS listening on %s:%d (state), %s:%d (pixel) | Dashboard: http://%s:%s/dash/ JSON: http://%s:%s/metrics.json",
            self.host,
            self.state_port,
            self.host,
            self.pixel_port,
            self.host,
            os.getenv('NAPARI_CUDA_METRICS_PORT', '8083'),
            self.host,
            os.getenv('NAPARI_CUDA_METRICS_PORT', '8083'),
        )
        broadcaster = asyncio.create_task(self._broadcast_loop())
        try:
            await asyncio.Future()
        finally:
            broadcaster.cancel()
            state_server.close(); await state_server.wait_closed()
            pixel_server.close(); await pixel_server.wait_closed()
            await self._stop_metrics_server(metrics_server)
            self._stop_worker()

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        def on_frame(payload_obj, _flags: int, capture_wall_ts: Optional[float] = None, seq: Optional[int] = None) -> None:
            # Hold pixel emission until worker orientation is ready to avoid a momentary inverted frame
            if self._worker is not None:
                if getattr(self._worker, '_orientation_ready', True) is False:
                    return
            # Convert encoder AU (Annex B or AVCC) to AVCC and detect keyframe, and record pack time
            # Optional: raw NAL summary before packing for diagnostics
            try:
                if int(os.getenv('NAPARI_CUDA_LOG_NALS', '0')):
                    from .bitstream import parse_nals
                    raw_bytes: bytes
                    if isinstance(payload_obj, (bytes, bytearray, memoryview)):
                        raw_bytes = bytes(payload_obj)
                    elif isinstance(payload_obj, (list, tuple)):
                        raw_bytes = b''.join([bytes(x) for x in payload_obj if x is not None])
                    else:
                        raw_bytes = bytes(payload_obj) if payload_obj is not None else b''
                    nals = parse_nals(raw_bytes)
                    has_sps = any(((n[0] & 0x1F) == 7) or (((n[0] >> 1) & 0x3F) == 33) for n in nals if n)
                    has_pps = any(((n[0] & 0x1F) == 8) or (((n[0] >> 1) & 0x3F) == 34) for n in nals if n)
                    has_idr = any(((n[0] & 0x1F) == 5) or (((n[0] >> 1) & 0x3F) in (19, 20, 21)) for n in nals if n)
                    logger.debug("Raw NALs: count=%d sps=%s pps=%s idr=%s", len(nals), has_sps, has_pps, has_idr)
            except Exception as e:
                logger.debug("Raw NAL summary failed: %s", e)
            t_p0 = time.perf_counter()
            avcc_pkt, is_key = pack_to_avcc(payload_obj, self._param_cache)
            t_p1 = time.perf_counter()
            try:
                self.metrics.observe_ms('napari_cuda_pack_ms', (t_p1 - t_p0) * 1000.0)
            except Exception:
                pass
            try:
                if int(os.getenv('NAPARI_CUDA_LOG_NALS', '0')) and avcc_pkt is not None:
                    from .bitstream import parse_nals
                    nals2 = parse_nals(avcc_pkt)
                    has_idr2 = any(((n[0] & 0x1F) == 5) or (((n[0] >> 1) & 0x3F) in (19, 20, 21)) for n in nals2 if n)
                    if bool(has_idr2) != bool(is_key):
                        logger.warning("Keyframe detect mismatch: parse=%s is_key=%s nals_after=%d", has_idr2, is_key, len(nals2))
            except Exception as e:
                logger.debug("Post-pack NAL summary failed: %s", e)
            if not avcc_pkt:
                return
            # Build and send video_config if needed or changed
            avcc_cfg = build_avcc_config(self._param_cache)
            if avcc_cfg is not None and (self._needs_config or self._last_avcc != avcc_cfg):
                try:
                    msg = {
                        'type': 'video_config',
                        'codec': 'h264',
                        'format': 'avcc',
                        'data': base64.b64encode(avcc_cfg).decode('ascii'),
                        'width': self.width,
                        'height': self.height,
                        'fps': self.cfg.fps,
                    }
                    loop.call_soon_threadsafe(lambda: asyncio.create_task(self._broadcast_state_json(msg)))
                    self._last_avcc = avcc_cfg
                    self._needs_config = False
                    try:
                        self.metrics.inc('napari_cuda_video_config_sends')
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug("Failed to schedule video_config broadcast: %s", e)
            # Optional payload dump (AVCC payload)
            if self._dump_remaining > 0:
                try:
                    os.makedirs(self._dump_dir, exist_ok=True)
                    if not self._dump_path:
                        ts = int(time.time())
                        self._dump_path = os.path.join(self._dump_dir, f"dump_{self.width}x{self.height}_{ts}.h264")
                    with open(self._dump_path, 'ab') as f:
                        f.write(avcc_pkt)
                    self._dump_remaining -= 1
                    if self._dump_remaining == 0:
                        logger.info("Bitstream dump complete: %s", self._dump_path)
                except Exception as e:
                    logger.debug("Bitstream dump error: %s", e)
            # Mint the header timestamp at post-pack time to keep encode/pack jitter common-mode
            stamp_ts = time.time()
            flags = 0x01 if is_key else 0
            # Mint sequence at pack/enqueue time to match previous semantics
            seq_val = self._seq & 0xFFFFFFFF
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            # Enqueue via callback that handles QueueFull inside the event loop thread
            def _enqueue():
                try:
                    self._frame_q.put_nowait((avcc_pkt, flags, seq_val, stamp_ts))
                    try:
                        self.metrics.set('napari_cuda_frame_queue_depth', float(self._frame_q.qsize()))
                    except Exception:
                        pass
                except asyncio.QueueFull:
                    try:
                        self._drain_and_put((avcc_pkt, flags, seq_val, stamp_ts))
                        try:
                            self.metrics.inc('napari_cuda_frames_dropped')
                            self._drops_total += 1
                            if (self._drops_total % 100) == 1:
                                logger.info("Pixel queue full: dropped oldest (total drops=%d)", self._drops_total)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug("Failed to drain and enqueue frame: %s", e)
            loop.call_soon_threadsafe(_enqueue)

        def worker_loop() -> None:
            try:
                # Scene refresh callback: rebroadcast current dims/meta on worker-driven changes
                def _on_scene_refresh() -> None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.get_event_loop()
                    # Fire-and-forget dims rebroadcast with latest state
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._rebroadcast_meta(client_id=None))
                    )

                self._worker = EGLRendererWorker(
                    width=self.width,
                    height=self.height,
                    use_volume=self.use_volume,
                    fps=self.cfg.fps,
                    animate=self._animate,
                    animate_dps=self._animate_dps,
                    zarr_path=self._zarr_path,
                    zarr_level=self._zarr_level,
                    zarr_axes=self._zarr_axes,
                    zarr_z=self._zarr_z,
                    policy_name=self._ms_state.get('policy'),
                    scene_refresh_cb=_on_scene_refresh,
                )
                # After worker init, capture initial Z (if any) and broadcast a baseline dims_update.
                z0 = getattr(self._worker, '_z_index', None)
                if z0 is not None:
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=(int(z0),),
                        )
                    # Build the authoritative dims.update once so logs match the sent payload
                    step_list = [int(z0)]
                    obj = self._build_dims_update_message(step_list, last_client_id=None)
                    # Schedule broadcast of this exact object on the asyncio loop thread
                    loop.call_soon_threadsafe(
                        lambda o=obj: asyncio.create_task(self._broadcast_state_json(o))
                    )
                    if self._log_dims_info:
                        logger.info("init: dims.update current_step=%s", obj.get('current_step'))
                    else:
                        logger.debug("init: dims.update current_step=%s", obj.get('current_step'))
                else:
                    # Pure 3D volume startup: send baseline dims so client enters volume mode
                    meta = self._dims_metadata() or {}
                    nd = int(meta.get('ndim') or 3)
                    step_list = [0 for _ in range(max(1, nd))]
                    # Do not mutate state.current_step here; worker has no discrete Z
                    obj = self._build_dims_update_message(step_list, last_client_id=None)
                    loop.call_soon_threadsafe(
                        lambda o=obj: asyncio.create_task(self._broadcast_state_json(o))
                    )
                    if self._log_dims_info:
                        logger.info("init: dims.update current_step=%s (baseline volume)", obj.get('current_step'))
                    else:
                        logger.debug("init: dims.update current_step=%s (baseline volume)", obj.get('current_step'))

                tick = 1.0 / max(1, self.cfg.fps)
                next_t = time.perf_counter()
                
                while not self._stop.is_set():
                    # Snapshot state and queued camera commands atomically
                    with self._state_lock:
                        queued = self._latest_state
                        commands = list(self._camera_commands)
                        self._camera_commands.clear()
                        # Clear one-shot fields so subsequent intents accumulate deltas until next frame
                        self._latest_state = ServerSceneState(
                            center=queued.center,
                            zoom=queued.zoom,
                            angles=queued.angles,
                            current_step=queued.current_step,
                        )
                    if commands:
                        logger.debug("frame commands snapshot count=%d", len(commands))
                    state = ServerSceneState(
                        center=queued.center,
                        zoom=queued.zoom,
                        angles=queued.angles,
                        current_step=queued.current_step,
                        volume_mode=getattr(queued, 'volume_mode', None),
                        volume_colormap=getattr(queued, 'volume_colormap', None),
                        volume_clim=getattr(queued, 'volume_clim', None),
                        volume_opacity=getattr(queued, 'volume_opacity', None),
                        volume_sample_step=getattr(queued, 'volume_sample_step', None),
                    )
                    if commands and (self._log_cam_info or self._log_cam_debug):
                        summaries: List[str] = []
                        for cmd in commands:
                            if cmd.kind == 'zoom':
                                factor = cmd.factor if cmd.factor is not None else 0.0
                                if cmd.anchor_px is not None:
                                    ax, ay = cmd.anchor_px
                                    summaries.append(
                                        f"zoom factor={factor:.4f} anchor=({ax:.1f},{ay:.1f})"
                                    )
                                else:
                                    summaries.append(f"zoom factor={factor:.4f}")
                            elif cmd.kind == 'pan':
                                summaries.append(f"pan dx={cmd.dx_px:.2f} dy={cmd.dy_px:.2f}")
                            elif cmd.kind == 'orbit':
                                summaries.append(f"orbit daz={cmd.d_az_deg:.2f} del={cmd.d_el_deg:.2f}")
                            elif cmd.kind == 'reset':
                                summaries.append('reset')
                            else:
                                summaries.append(cmd.kind)
                        msg = "apply: cam cmds=" + "; ".join(summaries)
                        if self._log_cam_info:
                            logger.info(msg)
                        else:
                            logger.debug(msg)
                    self._worker.apply_state(state)
                    if commands:
                        self._worker.process_camera_commands(commands)

                    timings, pkt, flags, seq = self._worker.capture_and_encode_packet()
                    # Observe timings (ms)
                    try:
                        self.metrics.observe_ms('napari_cuda_render_ms', timings.render_ms)
                        if timings.blit_gpu_ns is not None:
                            self.metrics.observe_ms('napari_cuda_capture_blit_ms', timings.blit_gpu_ns / 1e6)
                        # CPU wall time for blit (adds to additivity)
                        self.metrics.observe_ms('napari_cuda_capture_blit_cpu_ms', getattr(timings, 'blit_cpu_ms', 0.0))
                        self.metrics.observe_ms('napari_cuda_map_ms', timings.map_ms)
                        self.metrics.observe_ms('napari_cuda_copy_ms', timings.copy_ms)
                        self.metrics.observe_ms('napari_cuda_convert_ms', getattr(timings, 'convert_ms', 0.0))
                        self.metrics.observe_ms('napari_cuda_encode_ms', timings.encode_ms)
                        self.metrics.observe_ms('napari_cuda_pack_ms', getattr(timings, 'pack_ms', 0.0))
                        self.metrics.observe_ms('napari_cuda_total_ms', timings.total_ms)
                    except Exception:
                        pass
                    # Track last keyframe timestamp/seq for metrics if needed
                    # Keyframe tracking is handled after AVCC packing in on_frame
                    # Forward the encoder output along with the capture wall timestamp
                    cap_ts = getattr(timings, 'capture_wall_ts', None)
                    on_frame(pkt, flags, cap_ts, seq)
                    # Refresh policy metrics + append event if new decision was recorded
                    try:
                        self._publish_policy_metrics()
                        snap = self._policy_metrics_snapshot if isinstance(self._policy_metrics_snapshot, dict) else {}
                        last = snap.get('last_decision') if isinstance(snap, dict) else None
                        if isinstance(last, dict):
                            seq_dec = int(last.get('seq') or 0)
                            if seq_dec > self._last_written_decision_seq:
                                self._policy_event_path.parent.mkdir(parents=True, exist_ok=True)
                                with self._policy_event_path.open('a', encoding='utf-8') as f:
                                    f.write(json.dumps(last) + "\n")
                                self._last_written_decision_seq = seq_dec
                    except Exception:
                        logger.debug('policy metrics/event publish failed', exc_info=True)
                    next_t += tick
                    sleep = next_t - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_t = time.perf_counter()
            except Exception as e:
                logger.exception("Render worker error: %s", e)
            finally:
                try:
                    if self._worker:
                        self._worker.cleanup()
                except Exception as e:
                    logger.debug("Worker cleanup error: %s", e)

        # Start render thread outside of worker_loop definition
        self._worker_thread = threading.Thread(target=worker_loop, name="egl-render", daemon=True)
        self._worker_thread.start()

    # No per-frame header color hints; reserved set to 0

    def _stop_worker(self) -> None:
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

    def _drain_and_put(self, data: tuple[bytes, int, int, float]) -> None:
        try:
            while not self._frame_q.empty():
                self._frame_q.get_nowait()
        except Exception as e:
            logger.debug("Queue drain error: %s", e)
        try:
            self._frame_q.put_nowait(data)
        except Exception as e:
            logger.debug("Frame enqueue error: %s", e)

    async def _handle_state(self, ws: websockets.WebSocketServerProtocol):
        self._state_clients.add(ws)
        self.metrics.inc('napari_cuda_state_connects')
        try:
            self._update_client_gauges()
            # Reduce latency: disable Nagle for control channel
            try:
                sock = ws.transport.get_extra_info('socket')  # type: ignore[attr-defined]
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            # Send current dims baseline FIRST so client can gate inputs and inflate UI deterministically
            try:
                await self._await_adapter_level_ready(0.5)
                cur = None
                with self._state_lock:
                    cur = getattr(self._latest_state, 'current_step', None)
                if cur is not None:
                    # Build once so logs match sent payload
                    obj = self._build_dims_update_message(list(cur), last_client_id=None)
                else:
                    # No current_step yet (e.g., pure 3D volume). Send baseline dims.
                    meta = self._dims_metadata() or {}
                    try:
                        nd = int(meta.get('ndim') or 3)
                    except Exception:
                        nd = 3
                    step_list = [0 for _ in range(max(1, nd))]
                    obj = self._build_dims_update_message(step_list, last_client_id=None)
                # Send authoritative dims.update with nested metadata
                await ws.send(json.dumps(obj))
                if self._log_dims_info:
                    logger.info("connect: dims.update -> current_step=%s", obj.get('current_step'))
                else:
                    logger.debug("connect: dims.update -> current_step=%s", obj.get('current_step'))
            except Exception:
                logger.exception("Initial dims baseline send failed")
            # Send the current scene specification so clients hydrate remote layers before video_config
            try:
                await self._send_scene_spec(ws, reason="connect")
            except Exception:
                logger.exception("Initial scene.spec send failed")
            # Then send latest video config if available
            try:
                if self._last_avcc is not None:
                    msg = {
                        'type': 'video_config',
                        'codec': 'h264',
                        'format': 'avcc',
                        'data': base64.b64encode(self._last_avcc).decode('ascii'),
                        'width': self.width,
                        'height': self.height,
                        'fps': self.cfg.fps,
                    }
                    await ws.send(json.dumps(msg))
            except Exception:
                logger.exception("Initial state config send failed")
            remote = getattr(ws, 'remote_address', None)
            (logger.info if self._log_state_traces else logger.debug)(
                "state client loop start remote=%s id=%s", remote, id(ws)
            )
            try:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    await self._process_state_message(data, ws)
            except ConnectionClosed as exc:
                logger.info(
                    "state client closed remote=%s id=%s code=%s reason=%s",
                    remote,
                    id(ws),
                    getattr(exc, 'code', None),
                    getattr(exc, 'reason', None),
                )
            except Exception:
                logger.exception("state client error remote=%s id=%s", remote, id(ws))
        finally:
            try:
                await ws.close()
            except Exception as e:
                logger.debug("State WS close error: %s", e)
            self._state_clients.discard(ws)
            self._update_client_gauges()

    async def _process_state_message(self, data: dict, ws: websockets.WebSocketServerProtocol) -> None:
        t = data.get('type')
        seq = data.get('client_seq')
        (logger.info if self._log_state_traces else logger.debug)(
            "state message start type=%s seq=%s", t, seq
        )
        handled = False
        try:
            if t == 'set_camera':
                center = data.get('center')
                zoom = data.get('zoom')
                angles = data.get('angles')
                try:
                    if self._log_cam_info:
                        logger.info("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
                    elif self._log_cam_debug:
                        logger.debug("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
                except Exception:
                    pass
                with self._state_lock:
                    self._latest_state = ServerSceneState(
                        center=tuple(center) if center else None,
                        zoom=float(zoom) if zoom is not None else None,
                        angles=tuple(angles) if angles else None,
                        current_step=self._latest_state.current_step,
                    )
                handled = True
                return

            if t == 'dims.set' or t == 'set_dims':
                try:
                    logger.info("state: dims.set ignored (use dims.intent.*)")
                except Exception:
                    pass
                handled = True
                return

            if t == 'dims.intent.step':
                axis = data.get('axis')
                delta = int(data.get('delta') or 0)
                client_seq = data.get('client_seq')
                client_id = data.get('client_id') or None
                try:
                    if self._log_dims_info:
                        logger.info("intent: step axis=%r delta=%d client_id=%s seq=%s", axis, delta, client_id, client_seq)
                    else:
                        logger.debug("intent: step axis=%r delta=%d client_id=%s seq=%s", axis, delta, client_id, client_seq)
                except Exception:
                    pass
                try:
                    new_step = self._apply_dims_intent(axis=axis, step_delta=delta, set_value=None)
                    if new_step is not None:
                        try:
                            intent_i = int(client_seq) if client_seq is not None else None
                        except Exception:
                            intent_i = None
                        self._schedule_coro(
                            self._broadcast_dims_update(new_step, last_client_id=client_id, ack=True, intent_seq=intent_i),
                            'dims_update-step',
                        )
                except Exception as e:
                    logger.debug("dims.intent.step handling failed: %s", e)
                handled = True
                return

            if t == 'dims.intent.set_index':
                axis = data.get('axis')
                try:
                    value = int(data.get('value'))
                except Exception:
                    value = 0
                client_seq = data.get('client_seq')
                client_id = data.get('client_id') or None
                try:
                    if self._log_dims_info:
                        logger.info("intent: set_index axis=%r value=%d client_id=%s seq=%s", axis, value, client_id, client_seq)
                    else:
                        logger.debug("intent: set_index axis=%r value=%d client_id=%s seq=%s", axis, value, client_id, client_seq)
                except Exception:
                    pass
                try:
                    new_step = self._apply_dims_intent(axis=axis, step_delta=None, set_value=value)
                    if new_step is not None:
                        try:
                            intent_i = int(client_seq) if client_seq is not None else None
                        except Exception:
                            intent_i = None
                        self._schedule_coro(
                            self._broadcast_dims_update(new_step, last_client_id=client_id, ack=True, intent_seq=intent_i),
                            'dims_update-set_index',
                        )
                except Exception as e:
                    logger.debug("dims.intent.set_index handling failed: %s", e)
                handled = True
                return

            if t == 'volume.intent.set_render_mode':
                mode = str(data.get('mode') or '').lower()
                client_seq = data.get('client_seq')
                client_id = data.get('client_id') or None
                if self._is_valid_render_mode(mode):
                    self._volume_state['mode'] = mode
                    self._log_volume_intent("intent: volume.set_render_mode mode=%s client_id=%s seq=%s", mode, client_id, client_seq)
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=s.current_step,
                            volume_mode=str(mode),
                        )
                    self._schedule_coro(
                        self._rebroadcast_meta(client_id),
                        'rebroadcast-volume-mode',
                    )
                handled = True
                return

            if t == 'volume.intent.set_clim':
                pair = self._normalize_clim(data.get('lo'), data.get('hi'))
                client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
                if pair is not None:
                    lo, hi = pair
                    self._volume_state['clim'] = [lo, hi]
                    self._log_volume_intent("intent: volume.set_clim lo=%.4f hi=%.4f client_id=%s seq=%s", lo, hi, client_id, client_seq)
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=s.current_step,
                            volume_clim=(float(lo), float(hi)),
                        )
                    self._schedule_coro(
                        self._rebroadcast_meta(client_id),
                        'rebroadcast-volume-clim',
                    )
                handled = True
                return

            if t == 'volume.intent.set_colormap':
                name = data.get('name')
                client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
                if isinstance(name, str) and name.strip():
                    self._volume_state['colormap'] = str(name)
                    self._log_volume_intent("intent: volume.set_colormap name=%s client_id=%s seq=%s", name, client_id, client_seq)
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=s.current_step,
                            volume_colormap=str(name),
                        )
                    self._schedule_coro(
                        self._rebroadcast_meta(client_id),
                        'rebroadcast-volume-colormap',
                    )
                handled = True
                return

            if t == 'volume.intent.set_opacity':
                a = self._clamp_opacity(data.get('alpha'))
                client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
                if a is not None:
                    self._volume_state['opacity'] = float(a)
                    self._log_volume_intent("intent: volume.set_opacity alpha=%.3f client_id=%s seq=%s", a, client_id, client_seq)
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=s.current_step,
                            volume_opacity=float(a),
                        )
                    self._schedule_coro(
                        self._rebroadcast_meta(client_id),
                        'rebroadcast-volume-opacity',
                    )
                handled = True
                return

            if t == 'volume.intent.set_sample_step':
                rr = self._clamp_sample_step(data.get('relative'))
                client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
                if rr is not None:
                    self._volume_state['sample_step'] = float(rr)
                    self._log_volume_intent("intent: volume.set_sample_step relative=%.3f client_id=%s seq=%s", rr, client_id, client_seq)
                    with self._state_lock:
                        s = self._latest_state
                        self._latest_state = ServerSceneState(
                            center=s.center,
                            zoom=s.zoom,
                            angles=s.angles,
                            current_step=s.current_step,
                            volume_sample_step=float(rr),
                        )
                    self._schedule_coro(
                        self._rebroadcast_meta(client_id),
                        'rebroadcast-volume-sample-step',
                    )
                handled = True
                return

            if t == 'multiscale.intent.set_policy':
                pol = str(data.get('policy') or '').lower()
                client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
                allowed = {'oversampling', 'thresholds', 'ratio'}
                if pol not in allowed:
                    self._log_volume_intent(
                        "intent: multiscale.set_policy rejected policy=%s client_id=%s seq=%s",
                        pol,
                        client_id,
                        client_seq,
                    )
                    handled = True
                    return
                self._ms_state['policy'] = pol
                self._log_volume_intent(
                    "intent: multiscale.set_policy policy=%s client_id=%s seq=%s",
                    pol, client_id, client_seq,
                )
                if self._worker is not None:
                    try:
                        (logger.info if self._log_state_traces else logger.debug)(
                            "state: set_policy -> worker.set_policy start"
                        )
                        self._worker.set_policy(pol)
                        (logger.info if self._log_state_traces else logger.debug)(
                            "state: set_policy -> worker.set_policy done"
                        )
                    except Exception:
                        logger.exception("worker set_policy failed for %s", pol)
                self._schedule_coro(
                    self._rebroadcast_meta(client_id),
                    'rebroadcast-policy',
                )
                handled = True
                return

            if t == 'multiscale.intent.set_level':
                lvl = self._clamp_level(data.get('level'))
                client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
                if lvl is None:
                    handled = True
                    return
                self._ms_state['current_level'] = int(lvl)
                # Keep ms_state policy unchanged
                self._log_volume_intent("intent: multiscale.set_level level=%d client_id=%s seq=%s", int(lvl), client_id, client_seq)
                if self._worker is not None:
                    try:
                        levels = self._ms_state.get('levels') or []
                        path = None
                        if isinstance(levels, list) and 0 <= int(lvl) < len(levels):
                            path = levels[int(lvl)].get('path')
                        (logger.info if self._log_state_traces else logger.debug)(
                            "state: set_level -> worker.request level=%s start", lvl
                        )
                        self._worker.request_multiscale_level(int(lvl), path)
                        (logger.info if self._log_state_traces else logger.debug)(
                            "state: set_level -> worker.request done"
                        )
                        (logger.info if self._log_state_traces else logger.debug)(
                            "state: set_level -> worker.force_idr start"
                        )
                        self._worker.force_idr()
                        (logger.info if self._log_state_traces else logger.debug)(
                            "state: set_level -> worker.force_idr done"
                        )
                        self._bypass_until_key = True
                    except Exception:
                        logger.exception("multiscale level switch request failed")
                self._schedule_coro(
                    self._rebroadcast_meta(client_id),
                    'rebroadcast-ms-level',
                )
                handled = True
                return

            if t == 'view.intent.set_ndisplay':
                try:
                    ndisp_raw = data.get('ndisplay')
                    ndisp = int(ndisp_raw) if ndisp_raw is not None else 2
                except Exception:
                    ndisp = 2
                client_seq = data.get('client_seq'); client_id = data.get('client_id') or None
                (logger.info if self._log_state_traces else logger.debug)(
                    "state: set_ndisplay start target=%s", ndisp
                )
                await self._handle_set_ndisplay(ndisp, client_id, client_seq)
                (logger.info if self._log_state_traces else logger.debug)(
                    "state: set_ndisplay done target=%s", ndisp
                )
                handled = True
                return

            if t == 'camera.zoom_at':
                try:
                    factor = float(data.get('factor') or 0.0)
                except Exception:
                    factor = 0.0
                anchor = data.get('anchor_px')
                anc_t = tuple(anchor) if isinstance(anchor, (list, tuple)) and len(anchor) >= 2 else None
                if factor > 0.0 and anc_t is not None:
                    self.metrics.inc('napari_cuda_state_camera_intents')
                    if self._log_cam_info:
                        logger.info("state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)", factor, float(anc_t[0]), float(anc_t[1]))
                    elif self._log_cam_debug:
                        logger.debug("state: camera.zoom_at factor=%.4f anchor=(%.1f,%.1f)", factor, float(anc_t[0]), float(anc_t[1]))
                    self._enqueue_camera_command(
                        CameraCommand(
                            kind='zoom',
                            factor=float(factor),
                            anchor_px=(float(anc_t[0]), float(anc_t[1])),
                        )
                    )
                handled = True
                return

            if t == 'camera.pan_px':
                try:
                    dx = float(data.get('dx_px') or 0.0)
                    dy = float(data.get('dy_px') or 0.0)
                except Exception:
                    dx = 0.0; dy = 0.0
                if dx != 0.0 or dy != 0.0:
                    if self._log_cam_info:
                        logger.info("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
                    elif self._log_cam_debug:
                        logger.debug("state: camera.pan_px dx=%.2f dy=%.2f", dx, dy)
                    self.metrics.inc('napari_cuda_state_camera_intents')
                    self._enqueue_camera_command(
                        CameraCommand(kind='pan', dx_px=float(dx), dy_px=float(dy))
                    )
                handled = True
                return

            if t == 'camera.orbit':
                try:
                    daz = float(data.get('d_az_deg') or 0.0)
                    delv = float(data.get('d_el_deg') or 0.0)
                except Exception:
                    daz = 0.0; delv = 0.0
                if daz != 0.0 or delv != 0.0:
                    if self._log_cam_info:
                        logger.info("state: camera.orbit daz=%.2f del=%.2f", daz, delv)
                    elif self._log_cam_debug:
                        logger.debug("state: camera.orbit daz=%.2f del=%.2f", daz, delv)
                    self.metrics.inc('napari_cuda_state_camera_intents')
                    self._enqueue_camera_command(
                        CameraCommand(kind='orbit', d_az_deg=float(daz), d_el_deg=float(delv))
                    )
                    self.metrics.inc('napari_cuda_orbit_events')
                handled = True
                return

            if t == 'camera.reset':
                if self._log_cam_info:
                    logger.info("state: camera.reset")
                elif self._log_cam_debug:
                    logger.debug("state: camera.reset")
                self.metrics.inc('napari_cuda_state_camera_intents')
                self._enqueue_camera_command(CameraCommand(kind='reset'))
                if self._idr_on_reset and self._worker is not None:
                    logger.info("state: camera.reset -> ensure_keyframe start")
                    await self._ensure_keyframe()
                    logger.info("state: camera.reset -> ensure_keyframe done")
                handled = True
                return

            if t == 'ping':
                await ws.send(json.dumps({'type': 'pong'}))
                handled = True
                return

            if t in ('request_keyframe', 'force_idr'):
                await self._ensure_keyframe()
                handled = True
                return

            logger.debug("state message ignored type=%s", t)
        finally:
            (logger.info if self._log_state_traces else logger.debug)(
                "state message end type=%s seq=%s handled=%s", t, seq, handled
            )
    async def _handle_pixel(self, ws: websockets.WebSocketServerProtocol):
        self._clients.add(ws)
        self._update_client_gauges()
        # Reduce latency: disable Nagle for binary pixel stream
        try:
            sock = ws.transport.get_extra_info('socket')  # type: ignore[attr-defined]
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        # Reset encoder on new client to guarantee an immediate keyframe
        try:
            if self._worker is not None:
                logger.info("Resetting encoder for new pixel client to force keyframe")
                self._worker.reset_encoder()
                # Allow immediate send of the first keyframe when pacing is enabled
                self._bypass_until_key = True
                # Record reset time for watchdog cooldown
                self._kf_last_reset_ts = time.time()
                # Request re-send of video configuration for new clients
                self._needs_config = True
                # If we already have a cached avcC, proactively re-broadcast on state channel
                if self._last_avcc is not None:
                    try:
                        msg = {
                            'type': 'video_config',
                            'codec': 'h264',
                            'format': 'avcc',
                            'data': base64.b64encode(self._last_avcc).decode('ascii'),
                            'width': self.width,
                            'height': self.height,
                            'fps': self.cfg.fps,
                        }
                        await self._broadcast_state_json(msg)
                    except Exception as e:
                        logger.debug("Proactive video_config broadcast failed: %s", e)
                try:
                    self.metrics.inc('napari_cuda_encoder_resets')
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Encoder reset on client connect failed: %s", e)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)
            self._update_client_gauges()

    async def _broadcast_loop(self) -> None:
        # Always use paced broadcasting with latest-wins coalescing for smooth delivery
        try:
            target_fps = float(os.getenv('NAPARI_CUDA_BROADCAST_FPS', str(self.cfg.fps)))
        except Exception:
            target_fps = float(self.cfg.fps)
        loop = asyncio.get_running_loop()
        tick = 1.0 / max(1.0, target_fps)
        next_t = loop.time()
        # Map loop.time() monotonic clock to wall time for stable header timestamps
        mono0 = loop.time()
        wall0 = time.time()
        latest: Optional[tuple[bytes, int, int, float]] = None

        async def _fill_until(deadline: float) -> None:
            nonlocal latest
            remaining = max(0.0, deadline - loop.time())
            try:
                item = await asyncio.wait_for(self._frame_q.get(), timeout=remaining if remaining > 0 else 1e-6)
                latest = item
                # Drain rest without waiting; keep the newest only
                while True:
                    try:
                        latest = self._frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            except asyncio.TimeoutError:
                return

        async def _fill_and_find_key(deadline: float) -> Optional[tuple[bytes, int, int, float]]:
            """Drain queue up to deadline and return the FIRST keyframe seen (if any).

            If no keyframe found, returns None and leaves `latest` pointing to the newest item.
            """
            nonlocal latest
            found: Optional[tuple[bytes, int, int, float]] = None
            remaining = max(0.0, deadline - loop.time())
            try:
                item = await asyncio.wait_for(self._frame_q.get(), timeout=remaining if remaining > 0 else 1e-6)
            except asyncio.TimeoutError:
                return None
            latest = item
            payload, flags, _seq, _stamp_ts = item
            if (flags & 0x01) != 0:
                found = item
            # Drain the rest without waiting, but capture the first keyframe encountered
            while True:
                try:
                    it2 = self._frame_q.get_nowait()
                    latest = it2
                    if found is None:
                        _payload2, flags2, _seq2, _stamp_ts2 = it2
                        if (flags2 & 0x01) != 0:
                            found = it2
                except asyncio.QueueEmpty:
                    break
            return found

        while True:
            # Immediate keyframe bypass for new clients or after keyframe request
            if self._bypass_until_key:
                key_item = await _fill_and_find_key(loop.time() + 0.050)
                if key_item is not None:
                    if self._clients:
                        # Build header timestamp from stamp minted post-pack (common-mode with encode/pack)
                        payload, flags, seq_cap, stamp_ts = key_item
                        send_ts_mono = loop.time()
                        send_ts_wall = wall0 + (send_ts_mono - mono0)
                        seq32 = int(seq_cap) & 0xFFFFFFFF
                        header = struct.pack('!IdIIBBH', seq32, float(stamp_ts), self.width, self.height, self.cfg.codec & 0xFF, flags & 0xFF, 0)
                        to_send = header + payload
                        await asyncio.gather(*(self._safe_send(c, to_send) for c in list(self._clients)), return_exceptions=True)
                        # Update counters/metrics and log
                        try:
                            self.metrics.inc('napari_cuda_frames_total')
                            self.metrics.inc('napari_cuda_bytes_total', len(payload))
                            if flags & 0x01:
                                self.metrics.inc('napari_cuda_keyframes_total')
                                self._last_key_seq = seq32
                                # Track last keyframe timestamp based on stamped header time
                                self._last_key_ts = float(stamp_ts)
                                try:
                                    self.metrics.set('napari_cuda_last_key_seq', float(self._last_key_seq))
                                    self.metrics.set('napari_cuda_last_key_ts', float(self._last_key_ts))
                                except Exception:
                                    pass
                                # Cancel any pending keyframe watchdog once a keyframe is observed
                                try:
                                    if self._kf_watchdog_task is not None and not self._kf_watchdog_task.done():
                                        self._kf_watchdog_task.cancel()
                                except Exception:
                                    pass
                            if self._log_sends:
                                # Keep broadcaster-to-stamp delta for observability
                                stamp_to_send_ms = (send_ts_wall - float(stamp_ts)) * 1000.0
                                logger.info("Send frame seq=%d send_ts=%.6f stamp_ts=%.6f delta=%.3f ms (bypass)", seq32, send_ts_mono, float(stamp_ts), stamp_to_send_ms)
                        except Exception:
                            pass
                    latest = None
                    self._bypass_until_key = False
                    next_t = loop.time() + tick
                    continue

            now = loop.time()
            if now < next_t:
                await _fill_until(next_t)
            else:
                missed = int((now - next_t) // tick) + 1
                next_t += missed * tick

            if latest is not None:
                if self._clients:
                    payload, flags, seq_cap, stamp_ts = latest
                    send_ts_mono = loop.time()
                    send_ts_wall = wall0 + (send_ts_mono - mono0)
                    seq32 = int(seq_cap) & 0xFFFFFFFF
                    # Use stamped post-pack timestamp in header for paced sends
                    header = struct.pack('!IdIIBBH', seq32, float(stamp_ts), self.width, self.height, self.cfg.codec & 0xFF, flags & 0xFF, 0)
                    to_send = header + payload
                    await asyncio.gather(*(self._safe_send(c, to_send) for c in list(self._clients)), return_exceptions=True)
                    # Update counters/metrics and log
                    try:
                        self.metrics.inc('napari_cuda_frames_total')
                        self.metrics.inc('napari_cuda_bytes_total', len(payload))
                        if flags & 0x01:
                            self.metrics.inc('napari_cuda_keyframes_total')
                            self._last_key_seq = seq32
                            # Track last keyframe timestamp based on stamped header time
                            self._last_key_ts = float(stamp_ts)
                            try:
                                self.metrics.set('napari_cuda_last_key_seq', float(self._last_key_seq))
                                self.metrics.set('napari_cuda_last_key_ts', float(self._last_key_ts))
                            except Exception:
                                pass
                            # Cancel any pending keyframe watchdog once a keyframe is observed
                            try:
                                if self._kf_watchdog_task is not None and not self._kf_watchdog_task.done():
                                    self._kf_watchdog_task.cancel()
                            except Exception:
                                pass
                        if self._log_sends:
                            # Keep broadcaster-to-stamp delta for observability
                            stamp_to_send_ms = (send_ts_wall - float(stamp_ts)) * 1000.0
                            logger.info("Send frame seq=%d send_ts=%.6f stamp_ts=%.6f delta=%.3f ms", seq32, send_ts_mono, float(stamp_ts), stamp_to_send_ms)
                    except Exception:
                        pass
                    # Simple send timing log for smoothing diagnostics
                    try:
                        now2 = loop.time()
                        if self._last_send_ts is not None:
                            dt = now2 - self._last_send_ts
                            self._send_count += 1
                            if self._log_sends and (self._send_count % int(max(1, self.cfg.fps))) == 0:
                                logger.debug("Pixel send dt=%.3f s (target=%.3f), drops=%d", dt, 1.0/max(1, self.cfg.fps), self._drops_total)
                        self._last_send_ts = now2
                    except Exception:
                        pass
                latest = None
            next_t += tick

    async def _safe_send(self, ws: websockets.WebSocketServerProtocol, data: bytes) -> None:
        try:
            await ws.send(data)
        except Exception as e:
            logger.debug("Pixel send error: %s", e)
            try:
                await ws.close()
            except Exception as e2:
                logger.debug("Pixel WS close error: %s", e2)
            self._clients.discard(ws)

    def _scene_spec_json(self) -> Optional[str]:
        try:
            # Refresh scene manager from live sources to avoid stale dims in scene.spec
            try:
                self._update_scene_manager()
            except Exception:
                logger.debug("scene manager update failed before scene.spec", exc_info=True)
            msg = self._scene_manager.scene_message(time.time())
            if self._log_dims_info:
                try:
                    dims = msg.scene.dims.to_dict() if msg.scene and msg.scene.dims else {}
                    ms = None
                    if msg.scene and msg.scene.layers:
                        layer0 = msg.scene.layers[0]
                        ms = layer0.multiscale.to_dict() if layer0.multiscale else None
                    logger.info(
                        "scene.spec dims: sizes=%s range=%s ms_level=%s",
                        dims.get('sizes'),
                        dims.get('range'),
                        (ms or {}).get('current_level') if isinstance(ms, dict) else None,
                    )
                except Exception:
                    logger.debug("scene.spec dims log failed", exc_info=True)
            return msg.to_json()
        except Exception:
            logger.debug("scene.spec build failed", exc_info=True)
            return None

    async def _send_scene_spec(self, ws: websockets.WebSocketServerProtocol, *, reason: str) -> None:
        payload = self._scene_spec_json()
        if payload is None:
            return
        await self._safe_state_send(ws, payload)
        try:
            if self._log_dims_info:
                logger.info("%s: scene.spec sent", reason)
            else:
                logger.debug("%s: scene.spec sent", reason)
        except Exception:
            pass

    async def _broadcast_scene_spec(self, *, reason: str) -> None:
        payload = self._scene_spec_json()
        if payload is None or not self._state_clients:
            return
        coros = []
        for c in list(self._state_clients):
            coros.append(self._safe_state_send(c, payload))
        try:
            await asyncio.gather(*coros, return_exceptions=True)
            if self._log_dims_info:
                logger.info("%s: scene.spec broadcast to %d clients", reason, len(coros))
            else:
                logger.debug("%s: scene.spec broadcast to %d clients", reason, len(coros))
        except Exception as e:
            logger.debug("scene.spec broadcast error: %s", e)

    async def _broadcast_state_json(self, obj: dict) -> None:
        data = json.dumps(obj)
        if not self._state_clients:
            return
        coros = []
        for c in list(self._state_clients):
            coros.append(self._safe_state_send(c, data))
        try:
            await asyncio.gather(*coros, return_exceptions=True)
        except Exception as e:
            logger.debug("State broadcast error: %s", e)

    

    async def _safe_state_send(self, ws: websockets.WebSocketServerProtocol, text: str) -> None:
        try:
            await ws.send(text)
        except Exception as e:
            logger.debug("State send error: %s", e)
            try:
                await ws.close()
            except Exception as e2:
                logger.debug("State WS close error: %s", e2)
            self._state_clients.discard(ws)

    def _update_client_gauges(self) -> None:
        try:
            self.metrics.set('napari_cuda_pixel_clients', float(len(self._clients)))
            # We could track state clients separately if desired; here we reuse pixel_clients for demo
        except Exception:
            pass
        self._publish_policy_metrics()

    def _publish_policy_metrics(self) -> None:
        worker = self._worker
        if worker is None:
            return
        try:
            snapshot = worker.policy_metrics_snapshot()
        except Exception:
            logger.debug('policy metrics snapshot failed', exc_info=True)
            return

        if not isinstance(snapshot, dict):
            return

        self._policy_metrics_snapshot = snapshot
        try:
            self._ms_state['prime_complete'] = bool(snapshot.get('prime_complete'))
            if 'active_level' in snapshot:
                self._ms_state['current_level'] = int(snapshot['active_level'])
            self._ms_state['downgraded'] = bool(snapshot.get('level_downgraded'))
        except Exception:
            logger.debug('ms_state prime update failed', exc_info=True)
        try:
            out = Path('tmp/policy_metrics_latest.json')
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(snapshot, indent=2))
        except Exception:
            logger.debug('policy metrics file write failed', exc_info=True)

        levels = snapshot.get('levels') if isinstance(snapshot.get('levels'), dict) else {}
        if isinstance(levels, dict):
            for level, stats in levels.items():
                try:
                    lvl = int(level)
                except Exception:
                    continue
                if not isinstance(stats, dict):
                    continue
                prefix = f'napari_cuda_policy_level_{lvl}'
                mean_time = float(stats.get('mean_time_ms', 0.0))
                last_time = float(stats.get('last_time_ms', 0.0))
                overs = float(stats.get('latest_oversampling', 0.0))
                mean_bytes = float(stats.get('mean_bytes', 0.0))
                samples = float(stats.get('samples', 0.0))
                try:
                    self.metrics.set(f'{prefix}_mean_time_ms', mean_time)
                    self.metrics.set(f'{prefix}_last_time_ms', last_time)
                    self.metrics.set(f'{prefix}_oversampling', overs)
                    self.metrics.set(f'{prefix}_mean_bytes', mean_bytes)
                    self.metrics.set(f'{prefix}_samples', samples)
                except Exception:
                    logger.debug('policy metrics gauge update failed for level %s', lvl, exc_info=True)

        try:
            self.metrics.set('napari_cuda_ms_prime_complete', 1.0 if self._ms_state.get('prime_complete') else 0.0)
            self.metrics.set('napari_cuda_ms_active_level', float(self._ms_state.get('current_level', 0)))
        except Exception:
            logger.debug('prime metrics update failed', exc_info=True)

        decision = snapshot.get('last_decision')
        if isinstance(decision, dict) and decision:
            try:
                intent = float(decision.get('intent_level', -1))
            except Exception:
                intent = -1.0
            try:
                desired = float(decision.get('desired_level', -1))
            except Exception:
                desired = -1.0
            try:
                applied = float(decision.get('applied_level', -1))
            except Exception:
                applied = -1.0
            try:
                idle_ms = float(decision.get('idle_ms', 0.0))
            except Exception:
                idle_ms = 0.0
            try:
                self.metrics.set('napari_cuda_policy_intent_level', intent)
                self.metrics.set('napari_cuda_policy_desired_level', desired)
                self.metrics.set('napari_cuda_policy_applied_level', applied)
                self.metrics.set('napari_cuda_policy_idle_ms', idle_ms)
                self.metrics.set(
                    'napari_cuda_policy_downgraded', 1.0 if decision.get('downgraded') else 0.0
                )
            except Exception:
                logger.debug('policy decision gauge update failed', exc_info=True)

    async def _start_metrics_server(self):
        # Start Dash/Plotly dashboard on a background thread with a Flask server.
        try:
            port = int(os.getenv('NAPARI_CUDA_METRICS_PORT', '8083'))
        except Exception:
            port = 8083
        try:
            refresh_ms = int(os.getenv('NAPARI_CUDA_METRICS_REFRESH_MS', '1000'))
        except Exception:
            refresh_ms = 1000
        try:
            # Import here to allow running without dash installed (graceful fallback)
            from .dash_dashboard import start_dash_dashboard  # type: ignore
            th = start_dash_dashboard(self.host, port, self.metrics, refresh_ms)
            return th
        except Exception as e:
            logger.error("Dashboard init failed; continuing without UI: %s", e)
            return None

    async def _stop_metrics_server(self, runner):
        # Dash thread is daemonized; nothing to stop cleanly at shutdown.
        return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='napari-cuda EGL headless server')
    parser.add_argument('--host', default=os.getenv('NAPARI_CUDA_HOST', '0.0.0.0'))
    parser.add_argument('--state-port', type=int, default=int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')))
    parser.add_argument('--pixel-port', type=int, default=int(os.getenv('NAPARI_CUDA_PIXEL_PORT', '8082')))
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--animate', action='store_true', help='Enable simple turntable camera animation')
    parser.add_argument('--animate-dps', type=float, default=float(os.getenv('NAPARI_CUDA_TURNTABLE_DPS', '30.0')),
                        help='Turntable speed in degrees per second (default 30)')
    parser.add_argument('--volume', action='store_true', help='Use 3D volume visual')
    parser.add_argument('--zarr', dest='zarr_path', default=os.getenv('NAPARI_CUDA_ZARR_PATH'), help='Path to OME-Zarr root (enables 2D slice from ZYX volume)')
    parser.add_argument('--zarr-level', dest='zarr_level', default=os.getenv('NAPARI_CUDA_ZARR_LEVEL'), help='Dataset level path inside OME-Zarr (e.g., level_02). If omitted, inferred from multiscales.')
    parser.add_argument('--zarr-axes', dest='zarr_axes', default=os.getenv('NAPARI_CUDA_ZARR_AXES', 'zyx'), help='Axes order of the dataset (default: zyx)')
    parser.add_argument('--zarr-z', dest='zarr_z', type=int, default=int(os.getenv('NAPARI_CUDA_ZARR_Z', '-1')), help='Initial Z index for 2D slice (default: mid-slice)')
    parser.add_argument('--log-sends', action='store_true', help='Log per-send timing (seq, send_ts, stamp_ts, delta)')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG for this server module only')
    parser.add_argument('--encoder-profile', choices=['latency', 'quality'], default='latency',
                        help='Apply predefined NVENC configuration (defaults to latency focus)')
    args = parser.parse_args()

    async def run():
        # Apply encoder profile before worker initialization so env vars are honored
        _apply_encoder_profile(args.encoder_profile)
        srv = EGLHeadlessServer(width=args.width, height=args.height, use_volume=args.volume,
                                host=args.host, state_port=args.state_port, pixel_port=args.pixel_port, fps=args.fps,
                                animate=args.animate, animate_dps=args.animate_dps, log_sends=bool(args.log_sends),
                                zarr_path=args.zarr_path, zarr_level=args.zarr_level,
                                zarr_axes=args.zarr_axes, zarr_z=(None if int(args.zarr_z) < 0 else int(args.zarr_z)),
                                debug=bool(args.debug))
        await srv.start()

    asyncio.run(run())


if __name__ == '__main__':
    main()
