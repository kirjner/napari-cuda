"""
EGLHeadlessServer - Async server harness for headless EGL rendering + NVENC streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Dict, Optional, Set, Mapping, TYPE_CHECKING

import websockets
import importlib.resources as ilr
import socket
from websockets.exceptions import ConnectionClosed

def _merge_encoder_config(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {k: v for k, v in base.items()}
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_encoder_config(current, value)
        else:
            merged[key] = value
    return merged


def _build_encoder_override_env(base_env: Mapping[str, str], patch: Mapping[str, object]) -> dict[str, str]:
    if not patch:
        return {}
    merged_base: dict[str, object] = {}
    raw = base_env.get('NAPARI_CUDA_ENCODER_CONFIG')
    if raw:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                merged_base = loaded
        except Exception:
            logger.debug("Failed to parse existing encoder config; starting fresh", exc_info=True)
    merged = _merge_encoder_config(merged_base, patch)
    return {'NAPARI_CUDA_ENCODER_CONFIG': json.dumps(merged)}


# Encoder profile presets for convenient NVENC tuning
def _apply_encoder_profile(profile: str) -> dict[str, object]:
    profiles: dict[str, dict[str, object]] = {
        'latency': {
            'runtime': {
                'rc_mode': 'cbr',
                'lookahead': 0,
                'aq': 0,
                'temporalaq': 0,
                'bframes': 0,
                'preset': 'P3',
            },
        },
        'quality': {
            'encode': {
                'bitrate': 35_000_000,
            },
            'runtime': {
                'rc_mode': 'vbr',
                'max_bitrate': 45_000_000,
                'lookahead': 10,
                'aq': 1,
                'temporalaq': 1,
                'bframes': 2,
                'preset': 'P5',
                'idr_period': 120,
            },
        },
    }
    settings = profiles.get((profile or '').lower())
    if not settings:
        return {}
    if logger.isEnabledFor(logging.INFO):
        logger.info("Encoder profile '%s' resolved overrides: %s", profile, settings)
    return settings

from .scene_state import ServerSceneState
from .server_scene import ServerSceneData, create_server_scene_data
from .server_scene_queue import (
    PendingServerSceneUpdate,
    ServerSceneCommand,
    ServerSceneQueue,
    WorkerSceneNotification,
    WorkerSceneNotificationQueue,
)
from .server_scene_spec import (
    build_scene_spec_json,
)
from .layer_manager import ViewerSceneManager
from .bitstream import ParamCache, configure_bitstream
from .metrics_core import Metrics
from napari_cuda.utils.env import env_bool
from .zarr_source import ZarrSceneSource, ZarrSceneSourceError
from napari_cuda.server.config import (
    ServerConfig,
    ServerCtx,
    load_server_ctx,
)
from . import pixel_broadcaster, pixel_channel, metrics_server
from .server_scene_control import (
    broadcast_dims_update,
    build_dims_update_message,
    handle_state,
    process_worker_notifications,
    rebroadcast_meta,
)
from .worker_lifecycle import (
    WorkerLifecycleState,
    start_worker as lifecycle_start_worker,
    stop_worker as lifecycle_stop_worker,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .egl_worker import EGLRendererWorker


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
                 debug: bool = False,
                 env_overrides: Optional[Mapping[str, str]] = None) -> None:
        # Build once: resolved runtime context (observe-only for now)
        base_env: dict[str, str] = dict(os.environ)
        if env_overrides:
            for key, value in env_overrides.items():
                base_env[str(key)] = str(value)
        self._ctx_env: dict[str, str] = base_env
        try:
            self._ctx: ServerCtx = load_server_ctx(self._ctx_env)
        except Exception:
            # Fallback to minimal defaults if ctx load fails for any reason
            self._ctx = load_server_ctx({})  # type: ignore[arg-type]
        try:
            configure_bitstream(self._ctx.bitstream)
        except Exception:
            logger.debug("Bitstream configuration failed", exc_info=True)

        self.width = width
        self.height = height
        self.use_volume = use_volume
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        encode_cfg = getattr(self._ctx.cfg, 'encode', None)
        codec_map = {'h264': 1, 'hevc': 2, 'av1': 3}
        if encode_cfg is None:
            self._codec_name = 'h264'
            self.cfg = EncodeConfig(fps=fps)
        else:
            codec_name = str(getattr(encode_cfg, 'codec', 'h264')).lower()
            self._codec_name = codec_name if codec_name in codec_map else 'h264'
            self.cfg = EncodeConfig(
                fps=int(getattr(encode_cfg, 'fps', fps)),
                codec=codec_map.get(codec_name, 1),
                bitrate=int(getattr(encode_cfg, 'bitrate', 10_000_000)),
                keyint=int(getattr(encode_cfg, 'keyint', 120)),
            )
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception:
            self._animate_dps = 30.0

        policy_logging = self._ctx.debug_policy.logging
        self.metrics = Metrics()
        self._metrics_runner = None
        self._state_clients: Set[websockets.WebSocketServerProtocol] = set()
        # Keep queue size at 1 for latest-wins, never-block behavior
        qsize = int(self._ctx.frame_queue)
        frame_queue: asyncio.Queue[tuple[bytes, int, int, float]] = asyncio.Queue(maxsize=max(1, qsize))
        log_sends_flag = bool(log_sends or policy_logging.log_sends_env)
        broadcast_state = pixel_broadcaster.PixelBroadcastState(
            frame_queue=frame_queue,
            clients=set(),
            log_sends=log_sends_flag,
        )
        self._pixel_config = pixel_channel.PixelChannelConfig(
            width=self.width,
            height=self.height,
            fps=float(self.cfg.fps),
            codec_id=self.cfg.codec,
            codec_name=self._codec_name,
            kf_watchdog_cooldown_s=float(self._ctx.kf_watchdog_cooldown_s),
        )
        self._pixel_channel = pixel_channel.PixelChannelState(
            broadcast=broadcast_state,
            needs_config=True,
            last_avcc=None,
            kf_watchdog_cooldown_s=self._pixel_config.kf_watchdog_cooldown_s,
        )
        self._seq = 0
        self._worker_lifecycle = WorkerLifecycleState()
        self._scene_manager = ViewerSceneManager((self.width, self.height))
        self._worker_notifications = WorkerSceneNotificationQueue()
        self._scene: ServerSceneData = create_server_scene_data(
            policy_event_path=self._ctx.policy_event_path
        )
        # Bitstream parameter cache and config tracking (server-side)
        self._param_cache = ParamCache()
        # Optional bitstream dump for validation
        self._dump_remaining = int(self._ctx.dump_bitstream)
        self._dump_dir = self._ctx.dump_dir
        self._dump_path: Optional[str] = None
        # State access synchronization for latest-wins camera op coalescing
        self._state_lock = threading.Lock()
        if logger.isEnabledFor(logging.INFO):
            logger.info("Server debug policy: %s", self._ctx.debug_policy)
        # Logging controls for camera ops
        self._log_cam_info = bool(policy_logging.log_camera_info)
        self._log_cam_debug = bool(policy_logging.log_camera_debug)
        logger.info(
            "camera logging flags info=%s debug=%s",
            self._log_cam_info,
            self._log_cam_debug,
        )
        # State trace toggles (per-message start/end logs)
        self._log_state_traces = bool(policy_logging.log_state_traces)
        # Logging controls for volume/multiscale intents
        self._log_volume_info = bool(policy_logging.log_volume_info)
        # Force IDR on reset (default True)
        self._idr_on_reset = bool(self._ctx.cfg.idr_on_reset)

        # Data configuration (optional OME-Zarr dataset for real data)
        self._zarr_path = zarr_path or self._ctx.cfg.zarr_path or None
        self._zarr_level = zarr_level or self._ctx.cfg.zarr_level or None
        self._zarr_axes = zarr_axes or self._ctx.cfg.zarr_axes or None
        _z_fallback = self._ctx.cfg.zarr_z
        _z_resolved = zarr_z if zarr_z is not None else _z_fallback
        self._zarr_z = (_z_resolved if (_z_resolved is not None and int(_z_resolved) >= 0) else None)
        # Verbose dims logging control: default debug, upgrade to info with flag
        self._log_dims_info = bool(policy_logging.log_dims_info)
        # Dedicated debug control for this module only (no dependency loggers)
        self._debug_only_this_logger = bool(debug) or bool(self._ctx.debug_policy.enabled)
        self._allowed_render_modes = {'mip', 'translucent', 'iso'}
        # Populate multiscale description from NGFF metadata if available
        try:
            self._populate_multiscale_state()
        except Exception:
            logger.debug("populate multiscale state failed", exc_info=True)

    @property
    def _worker(self) -> Optional["EGLRendererWorker"]:
        return self._worker_lifecycle.worker

    @_worker.setter
    def _worker(self, worker: Optional["EGLRendererWorker"]) -> None:
        self._worker_lifecycle.worker = worker

    # --- Logging + Broadcast helpers --------------------------------------------
    def _enqueue_camera_command(self, cmd: ServerSceneCommand) -> None:
        with self._state_lock:
            self._scene.camera_commands.append(cmd)
            queue_len = len(self._scene.camera_commands)
        if self._log_cam_info or self._log_cam_debug:
            logger.info(
                "enqueue camera command kind=%s queue_len=%d payload=%s",
                cmd.kind,
                queue_len,
                cmd,
            )

    def _process_worker_notifications(self) -> None:
        notifications = self._worker_notifications.drain()
        if not notifications:
            return
        process_worker_notifications(self, notifications)

    def _log_volume_intent(self, fmt: str, *args) -> None:
        try:
            if self._log_volume_info:
                logger.info(fmt, *args)
            else:
                logger.debug(fmt, *args)
        except Exception:
            logger.debug("volume intent log failed", exc_info=True)

    def _try_reset_encoder(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        worker.reset_encoder()
        return True

    def _try_force_idr(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        worker.force_idr()
        return True

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

    async def _ensure_keyframe(self) -> None:
        """Request a clean keyframe and arrange for watchdog + config resend."""

        await pixel_channel.ensure_keyframe(
            self._pixel_channel,
            config=self._pixel_config,
            metrics=self.metrics,
            try_force_idr=self._try_force_idr,
            reset_encoder=self._try_reset_encoder,
            send_state_json=self._broadcast_state_json,
        )

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
        if self._log_dims_info:
            logger.info("intent: view.set_ndisplay ndisplay=%d client_id=%s seq=%s", int(ndisp), client_id, client_seq)
        else:
            logger.debug("intent: view.set_ndisplay ndisplay=%d client_id=%s seq=%s", int(ndisp), client_id, client_seq)
        self.use_volume = bool(ndisp == 3)
        print(
            "_handle_set_ndisplay",
            {
                'requested': int(ndisp),
                'use_volume': self.use_volume,
                'latest_step': getattr(self._scene.latest_state, 'current_step', None),
            },
            flush=True,
        )
        # Ask worker to apply the mode switch on the render thread
        if self._worker is not None and hasattr(self._worker, 'request_ndisplay'):
            try:
                self._worker.request_ndisplay(int(ndisp))  # type: ignore[attr-defined]
                # Force a keyframe and bypass pacing so the switch is immediate
                try:
                    if self._try_force_idr():
                        self._pixel_channel.broadcast.bypass_until_key = True
                        pixel_channel.mark_config_dirty(self._pixel_channel)
                except Exception:
                    logger.debug("view.set_ndisplay: force_idr failed", exc_info=True)
            except Exception:
                logger.exception("view.set_ndisplay: worker request failed")
        # Let the worker-driven scene refresh broadcast updated dims once the toggle completes

    def _start_kf_watchdog(self) -> None:
        pixel_channel.start_watchdog(
            self._pixel_channel,
            reset_encoder=self._try_reset_encoder,
        )

    # --- Meta builders ------------------------------------------------------------
    def _populate_multiscale_state(self) -> None:
        """Populate self._scene.multiscale_state['levels'] from NGFF multiscales if available.

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
            self._scene.multiscale_state['levels'] = levels
            self._scene.multiscale_state['current_level'] = int(source.current_level)
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
            if self._scene.latest_state.current_step is not None:
                current_step = list(self._scene.latest_state.current_step)  # type: ignore[arg-type]
        extras = {
            'zarr_axes': (self._worker._zarr_axes if self._worker is not None else None),
            'zarr_level': self._zarr_level,
        }
        if self._scene.policy_metrics_snapshot:
            extras['policy_metrics'] = self._scene.policy_metrics_snapshot
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
                    self._scene.multiscale_state['policy'] = 'auto'
                    self._scene.multiscale_state['current_level'] = int(source.current_level)
                    # Refresh levels if descriptor count changed (defensive)
                    levels = self._scene.multiscale_state.get('levels') or []
                    if not isinstance(levels, list) or len(levels) != len(source.level_descriptors):
                        self._scene.multiscale_state['levels'] = [
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
            scene_state=self._scene.latest_state,
            multiscale_state=dict(self._scene.multiscale_state),
            volume_state=dict(self._scene.volume_state),
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
        levels = self._scene.multiscale_state.get('levels') or []
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
                    lowered = [str(x).lower() for x in order]
                    if ax in lowered:
                        pos = lowered.index(ax)
                        return pos if 0 <= pos < max(0, int(cur_len)) else None
                labels = meta.get('axis_labels') or []
                if isinstance(labels, (list, tuple)):
                    lowered = [str(x).lower() for x in labels]
                    if ax in lowered:
                        pos = lowered.index(ax)
                        return pos if 0 <= pos < max(0, int(cur_len)) else None
        except Exception:
            logger.debug("resolve axis failed", exc_info=True)
        return None

    def _apply_dims_intent(self, axis: object, step_delta: Optional[int], set_value: Optional[int]) -> Optional[list[int]]:
        """Apply a dims intent to server state and return the new step list.

        Axis may be label or index. Clamps result to meta range when available.
        """
        # Capture current step and infer length
        with self._state_lock:
            cur = self._scene.latest_state.current_step
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
            s = self._scene.latest_state
            self._scene.latest_state = ServerSceneState(
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
            _cfg: ServerConfig = self._ctx.cfg
            if self._debug_only_this_logger:
                logger.info("Resolved ServerConfig: %s", _cfg)
            else:
                logger.debug("Resolved ServerConfig: %s", _cfg)
        except Exception:
            logger.debug("ServerConfig log failed", exc_info=True)
        logger.info("Starting EGLHeadlessServer %dx%d @ %dfps", self.width, self.height, self.cfg.fps)
        loop = asyncio.get_running_loop()
        try:
            self._update_scene_manager()
        except Exception:
            logger.debug("Initial scene manager sync failed", exc_info=True)
        self._start_worker(loop)
        # Start websocket servers; disable permessage-deflate to avoid CPU and latency on large frames
        async def state_handler(ws: websockets.WebSocketServerProtocol) -> None:
            await handle_state(self, ws)

        state_server = await websockets.serve(
            state_handler, self.host, self.state_port, compression=None
        )
        pixel_server = await websockets.serve(
            self._handle_pixel, self.host, self.pixel_port, compression=None
        )
        self._metrics_runner = metrics_server.start_metrics_dashboard(
            self.host,
            int(self._ctx.metrics_port),
            self.metrics,
            int(self._ctx.metrics_refresh_ms),
        )
        logger.info(
            "WS listening on %s:%d (state), %s:%d (pixel) | Dashboard: http://%s:%s/dash/ JSON: http://%s:%s/metrics.json",
            self.host,
            self.state_port,
            self.host,
            self.pixel_port,
            self.host,
            str(self._ctx.metrics_port),
            self.host,
            str(self._ctx.metrics_port),
        )
        broadcaster = asyncio.create_task(self._broadcast_loop())
        try:
            await asyncio.Future()
        finally:
            broadcaster.cancel()
            state_server.close(); await state_server.wait_closed()
            pixel_server.close(); await pixel_server.wait_closed()
            metrics_server.stop_metrics_dashboard(self._metrics_runner)
            self._stop_worker()

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        lifecycle_start_worker(self, loop, self._worker_lifecycle)

    def _stop_worker(self) -> None:
        lifecycle_stop_worker(self._worker_lifecycle)

    async def _handle_pixel(self, ws: websockets.WebSocketServerProtocol):
        await pixel_channel.handle_client(
            self._pixel_channel,
            ws,
            config=self._pixel_config,
            metrics=self.metrics,
            reset_encoder=self._try_reset_encoder,
            send_state_json=self._broadcast_state_json,
            on_clients_change=self._update_client_gauges,
        )

    async def _broadcast_loop(self) -> None:
        await pixel_channel.run_channel_loop(
            self._pixel_channel,
            config=self._pixel_config,
            metrics=self.metrics,
        )

    def _scene_spec_json(self) -> Optional[str]:
        try:
            self._update_scene_manager()
        except Exception:
            logger.debug("scene manager update failed before scene.spec", exc_info=True)

        try:
            json_payload = build_scene_spec_json(
                self._scene,
                self._scene_manager,
                timestamp=time.time(),
            )
        except Exception:
            logger.debug("scene.spec build failed", exc_info=True)
            return None

        if self._log_dims_info:
            spec = self._scene_manager.scene_spec()
            dims = spec.dims.to_dict() if spec is not None and spec.dims is not None else {}
            ms = None
            if spec is not None and spec.layers:
                layer0 = spec.layers[0]
                if layer0.multiscale is not None:
                    ms = layer0.multiscale.to_dict()
            logger.info(
                "scene.spec dims: sizes=%s range=%s ms_level=%s",
                dims.get('sizes'),
                dims.get('range'),
                (ms or {}).get('current_level') if isinstance(ms, dict) else None,
            )

        return json_payload

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
        count = len(self._pixel_channel.broadcast.clients)
        self.metrics.set('napari_cuda_pixel_clients', float(count))
        # We could track state clients separately if desired; here we reuse pixel_clients for demo
        self._publish_policy_metrics()

    def _publish_policy_metrics(self) -> None:
        worker = self._worker
        if worker is None:
            return
        snapshot = worker.policy_metrics_snapshot()
        if not isinstance(snapshot, Mapping):
            raise TypeError("policy metrics snapshot must be a mapping")

        metrics_server.update_policy_metrics(
            self._scene,
            self.metrics,
            snapshot,
        )


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
        encoder_patch = _apply_encoder_profile(args.encoder_profile)
        encode_section = encoder_patch.setdefault('encode', {})
        if not isinstance(encode_section, dict):
            encode_section = {}
            encoder_patch['encode'] = encode_section
        encode_section['fps'] = int(args.fps)
        profile_overrides = _build_encoder_override_env(os.environ, encoder_patch)
        srv = EGLHeadlessServer(width=args.width, height=args.height, use_volume=args.volume,
                                host=args.host, state_port=args.state_port, pixel_port=args.pixel_port, fps=args.fps,
                                animate=args.animate, animate_dps=args.animate_dps, log_sends=bool(args.log_sends),
                                zarr_path=args.zarr_path, zarr_level=args.zarr_level,
                                zarr_axes=args.zarr_axes, zarr_z=(None if int(args.zarr_z) < 0 else int(args.zarr_z)),
                                debug=bool(args.debug), env_overrides=profile_overrides)
        await srv.start()

    asyncio.run(run())


if __name__ == '__main__':
    main()
