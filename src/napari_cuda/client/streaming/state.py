from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Callable, Optional
import threading

import websockets

from napari_cuda.protocol.messages import (
    LAYER_REMOVE_TYPE,
    LAYER_UPDATE_TYPE,
    SCENE_SPEC_TYPE,
    LayerRemoveMessage,
    LayerUpdateMessage,
    SceneSpecMessage,
)


def _normalize_meta(meta_raw: dict) -> dict:
    """Return a shallow copy with a few backwards-compatible aliases expanded."""
    meta = dict(meta_raw)
    axes = meta_raw.get('axes')
    if axes is not None and meta.get('axis_labels') is None:
        labels: list[str] = []
        order: list[int] = []
        try:
            if isinstance(axes, dict):
                label = axes.get('label') or axes.get('name')
                if label is not None:
                    labels.append(str(label))
                    order.append(int(axes.get('index', 0)))
            elif isinstance(axes, (list, tuple)):
                for idx, entry in enumerate(axes):
                    if isinstance(entry, dict):
                        label = entry.get('label') or entry.get('name') or entry.get('id')
                        labels.append(str(label) if label is not None else str(idx))
                        order.append(int(entry.get('index', idx)))
                    else:
                        labels.append(str(entry))
                        order.append(idx)
            if labels:
                meta['axis_labels'] = labels
            if order and meta.get('order') is None:
                meta['order'] = order
        except Exception:
            logger.debug("dims.update axes normalisation failed", exc_info=True)
    displayed = meta_raw.get('displayed_axes')
    if displayed is not None and meta.get('displayed') is None:
        meta['displayed'] = displayed
    return meta

logger = logging.getLogger(__name__)


def _maybe_enable_debug_logger() -> bool:
    flag = (os.getenv('NAPARI_CUDA_STATE_DEBUG') or os.getenv('NAPARI_CUDA_CLIENT_DEBUG') or '').lower()
    if flag not in ('1', 'true', 'yes', 'on', 'dbg', 'debug'):
        return False
    has_local = any(getattr(h, '_napari_cuda_local', False) for h in logger.handlers)
    if not has_local:
        handler = logging.StreamHandler()
        fmt = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(logging.DEBUG)
        setattr(handler, '_napari_cuda_local', True)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return True


_STATE_DEBUG = _maybe_enable_debug_logger()


def _normalize_current_step(cur: object, meta: dict) -> object:
    """Inflate/clamp current_step to match meta.ndim and meta.range when needed.

    Returns the original value if already consistent or normalization fails.
    """
    try:
        ndim = meta.get('ndim')
        if not isinstance(cur, (list, tuple)) or not isinstance(ndim, (int, float)):
            return cur
        nd = int(ndim)
        if nd <= 0 or len(cur) == nd:
            return cur
        rng = meta.get('range') or []
        order = meta.get('order') or []
        inflated = [0] * nd
        try:
            # If order provides axis indices, prefer mapping by index
            if isinstance(order, (list, tuple)) and len(order) == nd and all(
                isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()) for x in order
            ):
                ord_idx = [int(x) for x in order]
                for i, val in enumerate(list(cur)[:nd]):
                    ax = ord_idx[i] if i < len(ord_idx) else i
                    if 0 <= ax < nd:
                        inflated[ax] = int(val) if isinstance(val, (int, float)) else 0
            else:
                for i in range(min(nd, len(cur))):
                    val = cur[i]
                    inflated[i] = int(val) if isinstance(val, (int, float)) else 0
        except Exception:
            for i in range(min(nd, len(cur))):
                val = cur[i]
                try:
                    inflated[i] = int(val) if isinstance(val, (int, float)) else 0
                except Exception:
                    inflated[i] = 0
        # Clamp to range if available: range may be list of (low, high[, step])
        try:
            if isinstance(rng, (list, tuple)) and len(rng) >= nd:
                for i in range(nd):
                    r = rng[i]
                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                        lo = r[0]
                        hi = r[1]
                        try:
                            lo_i = int(lo) if isinstance(lo, (int, float)) else 0
                            hi_i = int(hi) if isinstance(hi, (int, float)) else inflated[i]
                            if lo_i > hi_i:
                                lo_i, hi_i = hi_i, lo_i
                            if inflated[i] < lo_i:
                                inflated[i] = lo_i
                            elif inflated[i] > hi_i:
                                inflated[i] = hi_i
                        except Exception:
                            pass
        except Exception:
            logger.debug("dims.update inflate/clamp failed", exc_info=True)
        return inflated
    except Exception:
        logger.debug("dims.update optional guard failed", exc_info=True)
        return cur


class StateChannel:
    """Maintains a WebSocket connection to the state channel.

    - Pings periodically to keep connection alive.
    - Forwards `video_config` messages via callback.
    - Can send best-effort keyframe requests (throttled).
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_video_config: Optional[Callable[[dict], None]] = None,
        on_dims_update: Optional[Callable[[dict], None]] = None,
        on_scene_spec: Optional[Callable[[SceneSpecMessage], None]] = None,
        on_layer_update: Optional[Callable[[LayerUpdateMessage], None]] = None,
        on_layer_remove: Optional[Callable[[LayerRemoveMessage], None]] = None,
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[Optional[Exception]], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.on_video_config = on_video_config
        self.on_dims_update = on_dims_update
        self.on_scene_spec = on_scene_spec
        self.on_layer_update = on_layer_update
        self.on_layer_remove = on_layer_remove
        self.on_connected = on_connected
        self.on_disconnect = on_disconnect
        self._last_key_req: Optional[float] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._out_q: Optional[asyncio.Queue[str]] = None
        self._stop = False

    def run(self) -> None:
        """Start the state channel event loop and keep reconnecting.

        Mirrors PixelReceiver's reconnection strategy so that when the server
        drops and comes back, we resume receiving dims/video_config and our
        send_json path becomes live again.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._run())
        finally:
            try:
                if self._out_q is not None:
                    # Drain without blocking
                    while not self._out_q.empty():
                        _ = self._out_q.get_nowait()
            except Exception:
                logger.debug("StateChannel.run: drain out_q failed", exc_info=True)
            self._loop = None

    async def _run(self) -> None:
        url = f"ws://{self.host}:{self.port}"
        logger.info("Connecting to state channel at %s", url)
        retry_delay = 5.0
        max_delay = 30.0
        while not self._stop:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Connected to state channel")
                    # Reset backoff on successful connect
                    retry_delay = 5.0
                    self._ws = ws
                    # Notify connected
                    if self.on_connected:
                        try:
                            self.on_connected()
                        except Exception:
                            logger.debug("on_connected callback failed (state)", exc_info=True)
                    # Fresh sender queue per-connection; we intentionally drop any
                    # queued messages from a previous disconnected session.
                    self._out_q = asyncio.Queue()

                    async def _pinger() -> None:
                        while True:
                            try:
                                await ws.send('{"type":"ping"}')
                            except Exception:
                                logger.debug("State ping failed; will stop pinger", exc_info=True)
                                break
                            await asyncio.sleep(2.0)

                    async def _sender() -> None:
                        # Relay queued outbound messages (e.g., keyframe requests)
                        while True:
                            try:
                                msg = await self._out_q.get()  # type: ignore[arg-type]
                                logger.info("StateChannel sender -> %s", msg)
                                await ws.send(msg)
                                logger.info("StateChannel sender delivered")
                            except Exception:
                                logger.debug("State sender failed; stopping", exc_info=True)
                                break

                    ping_task = asyncio.create_task(_pinger())
                    send_task = asyncio.create_task(_sender())
                    # Request a keyframe on connect
                    try:
                        await ws.send('{"type":"request_keyframe"}')
                    except Exception:
                        logger.debug("Initial keyframe request failed", exc_info=True)
                    import json as _json
                    async for msg in ws:
                        try:
                            data = _json.loads(msg)
                        except _json.JSONDecodeError:
                            continue
                        try:
                            self._handle_message(data)
                        except Exception:
                            logger.debug("StateChannel message dispatch failed", exc_info=True)
                    # Connection closed: cancel helpers and clear handles
                    try:
                        ping_task.cancel()
                        send_task.cancel()
                    finally:
                        self._ws = None
                        self._out_q = None
                    # Normal close -> disconnect callback with None
                    if self.on_disconnect:
                        try:
                            self.on_disconnect(None)
                        except Exception:
                            logger.debug("on_disconnect callback failed (state)", exc_info=True)
            except Exception as e:
                # Graceful handling similar to PixelReceiver
                msg = str(e) or e.__class__.__name__
                try:
                    import websockets as _ws
                except Exception:
                    _ws = None  # type: ignore[assignment]
                if isinstance(e, (EOFError, ConnectionRefusedError)):
                    logger.info("State channel unavailable (%s); retrying in %.0fs", msg, retry_delay)
                elif (_ws is not None) and isinstance(e, _ws.exceptions.InvalidMessage):  # type: ignore[attr-defined]
                    logger.info("State channel handshake failed (%s); retrying in %.0fs", msg, retry_delay)
                elif isinstance(e, OSError):
                    logger.info("State channel socket error (%s); retrying in %.0fs", msg, retry_delay)
                else:
                    logger.exception("State channel error")
                # Exceptional close -> disconnect callback with error
                if self.on_disconnect:
                    try:
                        self.on_disconnect(e)
                    except Exception:
                        logger.debug("on_disconnect callback failed (state-exc)", exc_info=True)
                if self._stop:
                    break
                await asyncio.sleep(retry_delay)
                retry_delay = min(max_delay, retry_delay * 1.5)
                logger.info("Reconnecting to state channel...")

    def stop(self) -> None:
        """Request the channel loop to shut down."""

        self._stop = True
        loop = self._loop
        if loop is None:
            return

        def _schedule_shutdown() -> None:
            async def _shutdown() -> None:
                try:
                    if self._ws is not None and not self._ws.closed:
                        await self._ws.close()
                except Exception:
                    logger.debug("StateChannel.stop: close failed", exc_info=True)
                if self._out_q is not None:
                    try:
                        self._out_q.put_nowait('{"type":"shutdown"}')
                    except Exception:
                        logger.debug("StateChannel.stop: queue notify failed", exc_info=True)

            try:
                loop.create_task(_shutdown())
            except Exception:
                logger.debug("StateChannel.stop: schedule failed", exc_info=True)

        try:
            loop.call_soon_threadsafe(_schedule_shutdown)
        except Exception:
            logger.debug("StateChannel.stop: loop notify failed", exc_info=True)

    def _handle_message(self, data: dict) -> None:
        """Dispatch a single decoded message to registered callbacks."""

        raw_type = data.get('type')
        msg_type = (raw_type or '').lower()

        if msg_type == 'video_config':
            if self.on_video_config:
                try:
                    self.on_video_config(data)
                except Exception:
                    logger.debug("on_video_config callback failed", exc_info=True)
            return

        if msg_type == 'dims.update':
            if self.on_dims_update:
                try:
                    meta_raw = data.get('meta') or {}
                    logger.info(
                        "dims.update raw meta: level=%s level_shape=%s range=%s",
                        meta_raw.get('level'),
                        meta_raw.get('level_shape') or meta_raw.get('sizes'),
                        meta_raw.get('range') or meta_raw.get('ranges'),
                    )
                    meta = _normalize_meta(data.get('meta') or {})
                    cur = _normalize_current_step(data.get('current_step'), meta)
                    ack_val = data.get('ack') if isinstance(data.get('ack'), bool) else None
                    fwd = {
                        'current_step': cur,
                        'ndim': meta.get('ndim'),
                        'order': meta.get('order'),
                        'axis_labels': meta.get('axis_labels'),
                        'sizes': meta.get('sizes'),
                        'range': meta.get('range'),
                        'ndisplay': meta.get('ndisplay'),
                        'displayed': meta.get('displayed'),
                        'volume': bool(meta.get('volume')) if 'volume' in meta else None,
                        'render': meta.get('render') or None,
                        'multiscale': meta.get('multiscale') or None,
                        'level': meta.get('level'),
                        'level_shape': meta.get('level_shape'),
                        'dtype': meta.get('dtype'),
                        'normalized': meta.get('normalized'),
                        'ack': ack_val,
                        'intent_seq': data.get('intent_seq'),
                        'seq': data.get('seq'),
                        'last_client_id': data.get('last_client_id'),
                    }
                    self.on_dims_update(fwd)
                except Exception:
                    logger.debug("on_dims_update callback failed", exc_info=True)
            return

        if msg_type == SCENE_SPEC_TYPE:
            if self.on_scene_spec:
                try:
                    spec = SceneSpecMessage.from_dict(data)
                    if _STATE_DEBUG:
                        logger.debug(
                            "received scene.spec: layers=%s capabilities=%s",
                            [layer.layer_id for layer in spec.scene.layers],
                            spec.scene.capabilities,
                        )
                    self.on_scene_spec(spec)
                except Exception:
                    logger.debug("on_scene_spec callback failed", exc_info=True)
            return

        if msg_type == LAYER_UPDATE_TYPE:
            if self.on_layer_update:
                try:
                    update = LayerUpdateMessage.from_dict(data)
                    if _STATE_DEBUG:
                        logger.debug(
                            "received layer.update: id=%s partial=%s",
                            update.layer.layer_id if update.layer else None,
                            update.partial,
                        )
                    self.on_layer_update(update)
                except Exception:
                    logger.debug("on_layer_update callback failed", exc_info=True)
            return

        if msg_type == LAYER_REMOVE_TYPE:
            if self.on_layer_remove:
                try:
                    removal = LayerRemoveMessage.from_dict(data)
                    if _STATE_DEBUG:
                        logger.debug("received layer.remove: id=%s reason=%s", removal.layer_id, removal.reason)
                    self.on_layer_remove(removal)
                except Exception:
                    logger.debug("on_layer_remove callback failed", exc_info=True)
            return

    def request_keyframe_once(self) -> None:
        """Best-effort request for a keyframe via state channel (throttled)."""
        now = time.time()
        if self._last_key_req is not None and (now - self._last_key_req) < 0.5:
            return
        self._last_key_req = now
        # Prefer sending on the persistent connection via the sender task
        try:
            loop = self._loop
            q = self._out_q
            if loop is not None and q is not None:
                loop.call_soon_threadsafe(q.put_nowait, '{"type":"request_keyframe"}')
                return
        except Exception:
            logger.debug("Keyframe request enqueue failed; falling back", exc_info=True)
        # Fallback: fire-and-forget short-lived connection
        async def _send_once(url: str) -> None:
            try:
                async with websockets.connect(url) as ws:
                    await ws.send('{"type":"request_keyframe"}')
            except Exception:
                logger.debug("StateChannel._send_once: send failed", exc_info=True)
        try:
            url = f"ws://{self.host}:{self.port}"
            # If we're already inside an event loop (e.g., called from async context),
            # run the coroutine on a short-lived background thread to avoid
            # 'coroutine was never awaited' or 'asyncio.run() in running loop' issues.
            try:
                asyncio.get_running_loop()
                def _runner(u: str) -> None:
                    try:
                        asyncio.run(_send_once(u))
                    except Exception:
                        logger.debug("StateChannel._send_once thread failed", exc_info=True)
                threading.Thread(target=_runner, args=(url,), daemon=True).start()
            except RuntimeError:
                # No running loop in this thread; safe to run directly
                asyncio.run(_send_once(url))
        except Exception:
            logger.debug("Keyframe request (fallback) failed", exc_info=True)

    # --- Generic outbound messaging -------------------------------------------------
    def send_json(self, obj: dict) -> bool:
        """Enqueue a JSON message for the state channel sender.

        Returns True if enqueued, False if the channel is not ready.

        Thread-safe: uses loop.call_soon_threadsafe to put_nowait on the
        sender queue when available.
        """
        try:
            import json as _json
            msg = _json.dumps(obj, separators=(",", ":"))
        except Exception:
            logger.debug("send_json: JSON encode failed", exc_info=True)
            return False
        try:
            loop = self._loop
            q = self._out_q
            if loop is None or q is None:
                return False
            loop.call_soon_threadsafe(q.put_nowait, msg)
            return True
        except Exception:
            logger.debug("send_json: enqueue failed", exc_info=True)
            return False
