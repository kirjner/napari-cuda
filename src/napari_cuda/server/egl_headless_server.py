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
from typing import Awaitable, Dict, List, Optional, Set, Mapping

import websockets
from websockets.exceptions import ConnectionClosed
import importlib.resources as ilr
import socket

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

from .egl_worker import EGLRendererWorker
from .scene_state import ServerSceneState
from .server_scene import ServerSceneData, create_server_scene_data
from .server_scene_queue import (
    PendingServerSceneUpdate,
    ServerSceneCommand,
    ServerSceneQueue,
)
from .server_scene_spec import (
    build_dims_payload,
    build_scene_spec_json,
)
from .layer_manager import ViewerSceneManager
from .bitstream import ParamCache, pack_to_avcc, build_avcc_config, configure_bitstream
from .metrics import Metrics
from napari_cuda.utils.env import env_bool
from .zarr_source import ZarrSceneSource, ZarrSceneSourceError
from napari_cuda.server.config import (
    ServerConfig,
    ServerCtx,
    load_server_ctx,
)
from . import pixel_broadcaster

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
            self.cfg = EncodeConfig(fps=fps)
        else:
            codec_name = str(getattr(encode_cfg, 'codec', 'h264')).lower()
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
        self._state_clients: Set[websockets.WebSocketServerProtocol] = set()
        # Keep queue size at 1 for latest-wins, never-block behavior
        qsize = int(self._ctx.frame_queue)
        frame_queue: asyncio.Queue[tuple[bytes, int, int, float]] = asyncio.Queue(maxsize=max(1, qsize))
        log_sends_flag = bool(log_sends or policy_logging.log_sends_env)
        self._pixel = pixel_broadcaster.PixelBroadcastState(
            frame_queue=frame_queue,
            clients=set(),
            log_sends=log_sends_flag,
        )
        self._seq = 0
        self._stop = threading.Event()
        

        self._worker: Optional[EGLRendererWorker] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._scene: ServerSceneData = create_server_scene_data(
            policy_event_path=self._ctx.policy_event_path
        )
        # Bitstream parameter cache and config tracking (server-side)
        self._param_cache = ParamCache()
        self._needs_config = True
        self._last_avcc: Optional[bytes] = None
        # Optional bitstream dump for validation
        self._dump_remaining = int(self._ctx.dump_bitstream)
        self._dump_dir = self._ctx.dump_dir
        self._dump_path: Optional[str] = None
        # Keyframe watchdog cooldown to avoid rapid encoder resets
        self._kf_watchdog_cooldown_s = float(self._ctx.kf_watchdog_cooldown_s)
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

        self._scene_manager = ViewerSceneManager((self.width, self.height))

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
            # Gather steps from server state, worker viewer, and source
            state_step: list[int] | None = None
            with self._state_lock:
                cur = self._scene.latest_state.current_step
                state_step = list(cur) if isinstance(cur, (list, tuple)) else None

            w_step = None
            try:
                if self._worker is not None:
                    vm = self._worker.viewer_model()
                    if vm is not None:
                        w_step = tuple(int(x) for x in vm.dims.current_step)  # type: ignore[attr-defined]
            except Exception:
                w_step = None

            s_step = None
            try:
                src = getattr(self._worker, '_scene_source', None) if self._worker is not None else None
                if src is not None:
                    s_step = tuple(int(x) for x in (src.current_step or ()))
            except Exception:
                s_step = None

            worker_volume = False
            try:
                if self._worker is not None:
                    worker_volume = bool(getattr(self._worker, 'use_volume', False))
            except Exception:
                worker_volume = False

            # Choose authoritative step: favour worker viewer when volume mode is active
            chosen: list[int] = []
            source_of_truth = 'server'
            if worker_volume and w_step is not None and len(w_step) > 0:
                chosen = [int(x) for x in w_step]
                source_of_truth = 'viewer-volume'
            elif s_step is not None and len(s_step) > 0:
                chosen = [int(x) for x in s_step]
                source_of_truth = 'source'
            elif w_step is not None and len(w_step) > 0:
                chosen = [int(x) for x in w_step]
                source_of_truth = 'viewer'
            elif state_step is not None:
                chosen = [int(x) for x in state_step]
                source_of_truth = 'server'
            else:
                chosen = [0]
                source_of_truth = 'default'

            if worker_volume and chosen:
                while len(chosen) < 3:
                    chosen.append(0)
            print(
                "_rebroadcast_meta",
                {
                    'source': source_of_truth,
                    'chosen': chosen,
                    'state_step': state_step,
                    'worker_step': w_step,
                    'source_step': s_step,
                    'worker_volume': worker_volume,
                },
                flush=True,
            )

            # Synchronize server state with chosen step to keep intents consistent
            try:
                with self._state_lock:
                    s = self._scene.latest_state
                    self._scene.latest_state = ServerSceneState(
                        center=s.center,
                        zoom=s.zoom,
                        angles=s.angles,
                        current_step=tuple(chosen),
                    )
            except Exception:
                logger.debug('rebroadcast: failed to sync server state step', exc_info=True)

            # Diagnostics: compare and note source of truth
            if self._log_dims_info:
                logger.info(
                    "rebroadcast: source=%s step=%s server=%s viewer=%s source_step=%s",
                    source_of_truth, chosen, state_step, w_step, s_step,
                )
            else:
                logger.debug(
                    "rebroadcast: source=%s step=%s server=%s viewer=%s source_step=%s",
                    source_of_truth, chosen, state_step, w_step, s_step,
                )

            await self._broadcast_dims_update(chosen, last_client_id=client_id, ack=True)
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
            self._pixel.kf_last_reset_ts = time.time()
        # Bypass pacing once to deliver next keyframe immediately
        self._pixel.bypass_until_key = True
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
                    self._worker.force_idr()
                    self._pixel.bypass_until_key = True
                except Exception:
                    logger.debug("view.set_ndisplay: force_idr failed", exc_info=True)
            except Exception:
                logger.exception("view.set_ndisplay: worker request failed")
        # Let the worker-driven scene refresh broadcast updated dims once the toggle completes

    def _start_kf_watchdog(self) -> None:
        state = self._pixel

        async def _kf_watchdog(last_key_seq: Optional[int]):
            await asyncio.sleep(0.30)
            if state.last_key_seq == last_key_seq and self._worker is not None:
                now = time.time()
                if state.kf_last_reset_ts is not None and (now - state.kf_last_reset_ts) < self._kf_watchdog_cooldown_s:
                    rem = self._kf_watchdog_cooldown_s - (now - state.kf_last_reset_ts)
                    logger.debug("Keyframe watchdog cooldown active (%.2fs remaining); skip reset", rem)
                    return
                logger.warning("Keyframe watchdog fired; resetting encoder")
                try:
                    self._worker.reset_encoder()
                except Exception:
                    logger.exception("Encoder reset failed during keyframe watchdog")
                    return
                state.bypass_until_key = True
                state.kf_last_reset_ts = now
        try:
            task = state.kf_watchdog_task
            if task is not None and not task.done():
                task.cancel()
            state.kf_watchdog_task = asyncio.create_task(_kf_watchdog(state.last_key_seq))
        except Exception:
            logger.debug("start watchdog failed", exc_info=True)

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

    def _build_dims_update_message(
        self,
        step_list: list[int],
        last_client_id: Optional[str],
        *,
        ack: bool = False,
        intent_seq: Optional[int] = None,
    ) -> dict:
        meta = self._dims_metadata() or {}
        payload = build_dims_payload(
            self._scene,
            step_list=step_list,
            last_client_id=last_client_id,
            meta=meta,
            worker_scene_source=(getattr(self._worker, '_scene_source', None) if self._worker is not None else None),
            use_volume=bool(self._worker.use_volume if self._worker is not None else self.use_volume),
            ack=ack,
            intent_seq=intent_seq,
        )

        if self._log_dims_info:
            meta_out = payload.get('meta')
            if isinstance(meta_out, dict):
                logger.info(
                    "dims.update meta: level=%s level_shape=%s sizes=%s range=%s",
                    meta_out.get('level'),
                    meta_out.get('level_shape'),
                    meta_out.get('sizes'),
                    meta_out.get('range'),
                )

        return payload

    async def _broadcast_dims_update(self, step_list: list[int], last_client_id: Optional[str], *, ack: bool = False, intent_seq: Optional[int] = None) -> None:
        """Broadcast dims update in both new and legacy formats.

        Never raises; logs and continues on failure.
        """
        try:
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
            await self._stop_metrics_server(metrics_server)
            self._stop_worker()

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        def on_frame(payload_obj, _flags: int, capture_wall_ts: Optional[float] = None, seq: Optional[int] = None) -> None:
            # Hold pixel emission until worker orientation is ready to avoid a momentary inverted frame
            if self._worker is not None:
                if self._worker._orientation_ready is False:
                    return
            # Convert encoder AU (Annex B or AVCC) to AVCC and detect keyframe, and record pack time
            # Optional: raw NAL summary before packing for diagnostics
            try:
                if self._ctx.debug_policy.encoder.log_nals:
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
            avcc_pkt, is_key = pack_to_avcc(
                payload_obj,
                self._param_cache,
                encoder_logging=self._ctx.debug_policy.encoder,
            )
            t_p1 = time.perf_counter()
            try:
                self.metrics.observe_ms('napari_cuda_pack_ms', (t_p1 - t_p0) * 1000.0)
            except Exception:
                logger.debug('metrics observe napari_cuda_pack_ms failed', exc_info=True)
            try:
                if self._ctx.debug_policy.encoder.log_nals and avcc_pkt is not None:
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
                        logger.debug('metrics inc napari_cuda_video_config_sends failed', exc_info=True)
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
            queue = self._pixel.frame_queue

            def _enqueue() -> None:
                try:
                    queue.put_nowait((avcc_pkt, flags, seq_val, stamp_ts))
                    try:
                        self.metrics.set('napari_cuda_frame_queue_depth', float(queue.qsize()))
                    except Exception:
                        logger.debug('metrics set frame_queue_depth failed', exc_info=True)
                except asyncio.QueueFull:
                    try:
                        self._drain_and_put((avcc_pkt, flags, seq_val, stamp_ts))
                        try:
                            self.metrics.inc('napari_cuda_frames_dropped')
                            self._pixel.drops_total += 1
                            if (self._pixel.drops_total % 100) == 1:
                                logger.info(
                                    "Pixel queue full: dropped oldest (total drops=%d)",
                                    self._pixel.drops_total,
                                )
                        except Exception:
                            logger.debug('metrics inc frames_dropped failed', exc_info=True)
                    except Exception as e:
                        logger.debug("Failed to drain and enqueue frame: %s", e)
            loop.call_soon_threadsafe(_enqueue)

        def worker_loop() -> None:
            try:
                # Scene refresh callback: rebroadcast current dims/meta on worker-driven changes
                def _on_scene_refresh(step: object = None) -> None:
                    # If worker provided an explicit step, broadcast it directly; else rebroadcast meta
                    try:
                        if step is not None and isinstance(step, (list, tuple)):
                            chosen = [int(x) for x in step]
                            # Synchronize server state so future intents use this baseline
                            try:
                                with self._state_lock:
                                    s = self._scene.latest_state
                                    self._scene.latest_state = ServerSceneState(
                                        center=s.center,
                                        zoom=s.zoom,
                                        angles=s.angles,
                                        current_step=tuple(int(x) for x in chosen),
                                    )
                            except Exception:
                                logger.debug("scene.refresh: failed to sync server state step", exc_info=True)
                            loop.call_soon_threadsafe(
                                lambda c=chosen: asyncio.create_task(self._broadcast_dims_update(c, last_client_id=None, ack=True))
                            )
                        else:
                            loop.call_soon_threadsafe(
                                lambda: asyncio.create_task(self._rebroadcast_meta(client_id=None))
                            )
                    except Exception:
                        logger.debug("scene.refresh scheduling failed", exc_info=True)

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
                    policy_name=self._scene.multiscale_state.get('policy'),
                    scene_refresh_cb=_on_scene_refresh,
                    ctx=self._ctx,
                    env=self._ctx_env,
                )
                # After worker init, capture initial Z (if any) and broadcast a baseline dims_update.
                z0 = self._worker._z_index
                if z0 is not None:
                    with self._state_lock:
                        s = self._scene.latest_state
                        self._scene.latest_state = ServerSceneState(
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
                        queued = self._scene.latest_state
                        commands = list(self._scene.camera_commands)
                        self._scene.camera_commands.clear()
                        # Clear one-shot fields so subsequent intents accumulate deltas until next frame
                        self._scene.latest_state = ServerSceneState(
                            center=queued.center,
                            zoom=queued.zoom,
                            angles=queued.angles,
                            current_step=queued.current_step,
                        )
                    # Only log command snapshots when state trace logging is enabled
                    if commands and self._log_state_traces:
                        logger.info("frame commands snapshot count=%d", len(commands))
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
                        logger.debug('metrics observe render timings failed', exc_info=True)
                    # Track last keyframe timestamp/seq for metrics if needed
                    # Keyframe tracking is handled after AVCC packing in on_frame
                    # Forward the encoder output along with the capture wall timestamp
                    cap_ts = getattr(timings, 'capture_wall_ts', None)
                    on_frame(pkt, flags, cap_ts, seq)
                    # Refresh policy metrics + append event if new decision was recorded
                    try:
                        self._publish_policy_metrics()
                        snap = self._scene.policy_metrics_snapshot if isinstance(self._scene.policy_metrics_snapshot, dict) else {}
                        last = snap.get('last_decision') if isinstance(snap, dict) else None
                        if isinstance(last, dict):
                            seq_dec = int(last.get('seq') or 0)
                            if seq_dec > self._scene.last_written_decision_seq:
                                self._scene.policy_event_path.parent.mkdir(parents=True, exist_ok=True)
                                with self._scene.policy_event_path.open('a', encoding='utf-8') as f:
                                    f.write(json.dumps(last) + "\n")
                                self._scene.last_written_decision_seq = seq_dec
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
        queue = self._pixel.frame_queue
        try:
            while not queue.empty():
                queue.get_nowait()
        except Exception as e:
            logger.debug("Queue drain error: %s", e)
        try:
            queue.put_nowait(data)
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
                logger.debug('state ws: TCP_NODELAY toggle failed', exc_info=True)
            # Send current dims baseline FIRST so client can gate inputs and inflate UI deterministically
            try:
                await self._await_adapter_level_ready(0.5)
                cur = None
                with self._state_lock:
                    cur = self._scene.latest_state.current_step
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
        if self._log_state_traces:
            logger.info("state message start type=%s seq=%s", t, seq)
        handled = False
        try:
            if t == 'set_camera':
                center = data.get('center')
                zoom = data.get('zoom')
                angles = data.get('angles')
                if self._log_cam_info:
                    logger.info("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
                elif self._log_cam_debug:
                    logger.debug("state: set_camera center=%s zoom=%s angles=%s", center, zoom, angles)
                with self._state_lock:
                    self._scene.latest_state = ServerSceneState(
                        center=tuple(center) if center else None,
                        zoom=float(zoom) if zoom is not None else None,
                        angles=tuple(angles) if angles else None,
                        current_step=self._scene.latest_state.current_step,
                    )
                handled = True
                return

            if t == 'dims.set' or t == 'set_dims':
                logger.debug("state: dims.set ignored (use dims.intent.*)")
                handled = True
                return

            if t == 'dims.intent.step':
                axis = data.get('axis')
                delta = int(data.get('delta') or 0)
                client_seq = data.get('client_seq')
                client_id = data.get('client_id') or None
                if self._log_dims_info:
                    logger.info("intent: step axis=%r delta=%d client_id=%s seq=%s", axis, delta, client_id, client_seq)
                else:
                    logger.debug("intent: step axis=%r delta=%d client_id=%s seq=%s", axis, delta, client_id, client_seq)
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
                if self._log_dims_info:
                    logger.info("intent: set_index axis=%r value=%d client_id=%s seq=%s", axis, value, client_id, client_seq)
                else:
                    logger.debug("intent: set_index axis=%r value=%d client_id=%s seq=%s", axis, value, client_id, client_seq)
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
                    self._scene.volume_state['mode'] = mode
                    self._log_volume_intent("intent: volume.set_render_mode mode=%s client_id=%s seq=%s", mode, client_id, client_seq)
                    with self._state_lock:
                        s = self._scene.latest_state
                        self._scene.latest_state = ServerSceneState(
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
                    self._scene.volume_state['clim'] = [lo, hi]
                    self._log_volume_intent("intent: volume.set_clim lo=%.4f hi=%.4f client_id=%s seq=%s", lo, hi, client_id, client_seq)
                    with self._state_lock:
                        s = self._scene.latest_state
                        self._scene.latest_state = ServerSceneState(
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
                    self._scene.volume_state['colormap'] = str(name)
                    self._log_volume_intent("intent: volume.set_colormap name=%s client_id=%s seq=%s", name, client_id, client_seq)
                    with self._state_lock:
                        s = self._scene.latest_state
                        self._scene.latest_state = ServerSceneState(
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
                    self._scene.volume_state['opacity'] = float(a)
                    self._log_volume_intent("intent: volume.set_opacity alpha=%.3f client_id=%s seq=%s", a, client_id, client_seq)
                    with self._state_lock:
                        s = self._scene.latest_state
                        self._scene.latest_state = ServerSceneState(
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
                    self._scene.volume_state['sample_step'] = float(rr)
                    self._log_volume_intent("intent: volume.set_sample_step relative=%.3f client_id=%s seq=%s", rr, client_id, client_seq)
                    with self._state_lock:
                        s = self._scene.latest_state
                        self._scene.latest_state = ServerSceneState(
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
                self._scene.multiscale_state['policy'] = pol
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
                self._scene.multiscale_state['current_level'] = int(lvl)
                # Keep ms_state policy unchanged
                self._log_volume_intent("intent: multiscale.set_level level=%d client_id=%s seq=%s", int(lvl), client_id, client_seq)
                if self._worker is not None:
                    try:
                        levels = self._scene.multiscale_state.get('levels') or []
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
                        self._pixel.bypass_until_key = True
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
                        ServerSceneCommand(
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
                        ServerSceneCommand(kind='pan', dx_px=float(dx), dy_px=float(dy))
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
                        ServerSceneCommand(kind='orbit', d_az_deg=float(daz), d_el_deg=float(delv))
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
                self._enqueue_camera_command(ServerSceneCommand(kind='reset'))
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

            if self._log_state_traces:
                logger.info("state message ignored type=%s", t)
        finally:
            if self._log_state_traces:
                logger.info("state message end type=%s seq=%s handled=%s", t, seq, handled)
    async def _handle_pixel(self, ws: websockets.WebSocketServerProtocol):
        self._pixel.clients.add(ws)
        self._update_client_gauges()
        # Reduce latency: disable Nagle for binary pixel stream
        pixel_broadcaster.configure_socket(ws, label='pixel ws')
        # Reset encoder on new client to guarantee an immediate keyframe
        try:
            if self._worker is not None:
                logger.info("Resetting encoder for new pixel client to force keyframe")
                self._worker.reset_encoder()
                # Allow immediate send of the first keyframe when pacing is enabled
                self._pixel.bypass_until_key = True
                # Record reset time for watchdog cooldown
                self._pixel.kf_last_reset_ts = time.time()
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
                    logger.debug('metrics inc encoder_resets failed', exc_info=True)
        except Exception as e:
                logger.debug("Encoder reset on client connect failed: %s", e)
        try:
            await ws.wait_closed()
        finally:
            self._pixel.clients.discard(ws)
            self._update_client_gauges()

    async def _broadcast_loop(self) -> None:
        cfg = pixel_broadcaster.PixelBroadcastConfig(
            width=self.width,
            height=self.height,
            codec=self.cfg.codec,
            fps=float(getattr(self._ctx.cfg.encode, 'fps', getattr(self.cfg, 'fps', 60))),
        )
        await pixel_broadcaster.broadcast_loop(self._pixel, cfg, self.metrics)

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

    async def _send_scene_spec(self, ws: websockets.WebSocketServerProtocol, *, reason: str) -> None:
        payload = self._scene_spec_json()
        if payload is None:
            return
        await self._safe_state_send(ws, payload)
        if self._log_dims_info:
            logger.info("%s: scene.spec sent", reason)
        else:
            logger.debug("%s: scene.spec sent", reason)

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
            self.metrics.set('napari_cuda_pixel_clients', float(len(self._pixel.clients)))
            # We could track state clients separately if desired; here we reuse pixel_clients for demo
        except Exception:
            logger.debug('metrics set pixel_clients failed', exc_info=True)
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

        self._scene.policy_metrics_snapshot = snapshot
        try:
            self._scene.multiscale_state['prime_complete'] = bool(snapshot.get('prime_complete'))
            if 'active_level' in snapshot:
                self._scene.multiscale_state['current_level'] = int(snapshot['active_level'])
            self._scene.multiscale_state['downgraded'] = bool(snapshot.get('level_downgraded'))
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
            self.metrics.set('napari_cuda_ms_prime_complete', 1.0 if self._scene.multiscale_state.get('prime_complete') else 0.0)
            self.metrics.set('napari_cuda_ms_active_level', float(self._scene.multiscale_state.get('current_level', 0)))
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
        port = int(self._ctx.metrics_port)
        refresh_ms = int(self._ctx.metrics_refresh_ms)
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
