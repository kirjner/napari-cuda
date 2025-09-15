from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Callable, Optional
import threading

import websockets

logger = logging.getLogger(__name__)


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
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[Optional[Exception]], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.on_video_config = on_video_config
        self.on_dims_update = on_dims_update
        self.on_connected = on_connected
        self.on_disconnect = on_disconnect
        self._last_key_req: Optional[float] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._out_q: Optional[asyncio.Queue[str]] = None

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
        while True:
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
                                await ws.send(msg)
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
                        t = (data.get('type') or '').lower()
                        if t == 'video_config':
                            if self.on_video_config:
                                try:
                                    self.on_video_config(data)
                                except Exception:
                                    logger.debug("on_video_config callback failed", exc_info=True)
                        elif t == 'dims.update':
                            if self.on_dims_update:
                                try:
                                    # Normalize new-style dims.update (nested meta) to flattened keys
                                    meta = data.get('meta') or {}
                                    fwd = {
                                        'current_step': data.get('current_step'),
                                        'ndim': meta.get('ndim'),
                                        'order': meta.get('order'),
                                        'axis_labels': meta.get('axis_labels'),
                                        'sizes': meta.get('sizes'),
                                        'range': meta.get('range'),
                                        'ndisplay': meta.get('ndisplay'),
                                        'seq': data.get('seq'),
                                        'last_client_id': data.get('last_client_id'),
                                    }
                                    self.on_dims_update(fwd)
                                except Exception:
                                    logger.debug("on_dims_update callback failed", exc_info=True)
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
                await asyncio.sleep(retry_delay)
                retry_delay = min(max_delay, retry_delay * 1.5)
                logger.info("Reconnecting to state channel...")

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
