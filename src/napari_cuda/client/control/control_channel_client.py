from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import websockets

try:
    from napari_cuda import __version__ as _NAPARI_CUDA_VERSION
except Exception:  # pragma: no cover - fallback for unusual import issues
    _NAPARI_CUDA_VERSION = "dev"

from napari_cuda.client.streaming.dims_payload import inflate_current_step, normalize_meta

from napari_cuda.protocol import (
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    EnvelopeParser,
    SESSION_REJECT_TYPE,
    SESSION_WELCOME_TYPE,
    HelloClientInfo,
    SessionReject,
    SessionWelcome,
    build_session_hello,
)
from napari_cuda.protocol.messages import (
    LAYER_REMOVE_TYPE,
    LAYER_UPDATE_TYPE,
    SCENE_SPEC_TYPE,
    STATE_UPDATE_TYPE,
    LayerRemoveMessage,
    LayerUpdateMessage,
    SceneSpecMessage,
    StateUpdateMessage,
)


logger = logging.getLogger(__name__)


@dataclass
class StateChannelLoop:
    loop: asyncio.AbstractEventLoop | None = None
    websocket: websockets.WebSocketClientProtocol | None = None
    outbox: asyncio.Queue[str] | None = None
    stop_requested: bool = False


def _maybe_enable_debug_logger() -> bool:
    flag = (os.getenv("NAPARI_CUDA_STATE_DEBUG") or os.getenv("NAPARI_CUDA_CLIENT_DEBUG") or "").lower()
    if flag not in ("1", "true", "yes", "on", "dbg", "debug"):
        return False
    has_local = any(getattr(h, "_napari_cuda_local", False) for h in logger.handlers)
    if not has_local:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(logging.DEBUG)
        setattr(handler, "_napari_cuda_local", True)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return True


_STATE_DEBUG = _maybe_enable_debug_logger()


_ENVELOPE_PARSER = EnvelopeParser()

_HANDSHAKE_TIMEOUT_S = 5.0
_RESUMABLE_TOPICS = ("notify.scene", "notify.layers", "notify.stream")
_REQUIRED_FEATURES = {
    "notify.scene": True,
    "notify.layers": True,
    "notify.stream": True,
    "notify.dims": True,
    "notify.camera": True,
}
_LEGACY_NOTIFY_STATE_TYPE = "notify.state"


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
        handle_video_config: Optional[Callable[[dict], None]] = None,
        handle_dims_update: Optional[Callable[[dict], None]] = None,
        handle_scene_spec: Optional[Callable[[SceneSpecMessage], None]] = None,
        handle_layer_update: Optional[Callable[[LayerUpdateMessage], None]] = None,
        handle_layer_remove: Optional[Callable[[LayerRemoveMessage], None]] = None,
        handle_state_update: Optional[Callable[[StateUpdateMessage], None]] = None,
        handle_connected: Optional[Callable[[], None]] = None,
        handle_disconnect: Optional[Callable[[Optional[Exception]], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.handle_video_config = handle_video_config
        self.handle_dims_update = handle_dims_update
        self.handle_scene_spec = handle_scene_spec
        self.handle_layer_update = handle_layer_update
        self.handle_layer_remove = handle_layer_remove
        self.handle_state_update = handle_state_update
        self.handle_connected = handle_connected
        self.handle_disconnect = handle_disconnect
        self._last_key_req: Optional[float] = None
        self._loop_state = StateChannelLoop()

    def run(self) -> None:
        """Start the state channel event loop and keep reconnecting."""

        loop_state = self._loop_state
        loop_state.stop_requested = False
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_state.loop = loop
        try:
            loop.run_until_complete(self._run(loop_state))
        finally:
            try:
                outbox = loop_state.outbox
                if outbox is not None:
                    while not outbox.empty():
                        _ = outbox.get_nowait()
            except Exception:
                logger.debug("StateChannel.run: drain outbox failed", exc_info=True)
            loop_state.loop = None
            loop_state.websocket = None
            loop_state.outbox = None

    async def _run(self, loop_state: StateChannelLoop) -> None:
        url = f"ws://{self.host}:{self.port}"
        logger.info("Connecting to state channel at %s", url)
        retry_delay = 5.0
        max_delay = 30.0
        while not loop_state.stop_requested:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Connected to state channel")
                    retry_delay = 5.0
                    loop_state.websocket = ws
                    if self.handle_connected:
                        try:
                            self.handle_connected()
                        except Exception:
                            logger.debug("handle_connected callback failed (state)", exc_info=True)
                    loop_state.outbox = asyncio.Queue()

                    await self._perform_handshake(ws)

                    async def _pinger() -> None:
                        while True:
                            try:
                                await ws.send('{"type":"ping"}')
                            except Exception:
                                logger.debug("State ping failed; will stop pinger", exc_info=True)
                                break
                            await asyncio.sleep(2.0)

                    async def _sender() -> None:
                        outbox = loop_state.outbox
                        if outbox is None:
                            return
                        while True:
                            try:
                                msg = await outbox.get()  # type: ignore[arg-type]
                                logger.debug("StateChannel sender -> %s", msg)
                                await ws.send(msg)
                                logger.debug("StateChannel sender delivered")
                            except Exception:
                                logger.debug("State sender failed; stopping", exc_info=True)
                                break

                    ping_task = asyncio.create_task(_pinger())
                    send_task = asyncio.create_task(_sender())
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
                    ping_task.cancel()
                    send_task.cancel()
                    try:
                        await asyncio.gather(ping_task, send_task, return_exceptions=True)
                    except Exception:
                        logger.debug("StateChannel helper cancel failed", exc_info=True)
                    finally:
                        loop_state.websocket = None
                        loop_state.outbox = None
                    if self.handle_disconnect:
                        try:
                            self.handle_disconnect(None)
                        except Exception:
                            logger.debug("handle_disconnect callback failed (state)", exc_info=True)
            except Exception as e:
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
                loop_state.websocket = None
                loop_state.outbox = None
                if self.handle_disconnect:
                    try:
                        self.handle_disconnect(e)
                    except Exception:
                        logger.debug("handle_disconnect callback failed (state-exc)", exc_info=True)
                if loop_state.stop_requested:
                    break
                await asyncio.sleep(retry_delay)
                retry_delay = min(max_delay, retry_delay * 1.5)
                logger.info("Reconnecting to state channel...")

    def stop(self) -> None:
        """Request the channel loop to shut down."""

        loop_state = self._loop_state
        loop_state.stop_requested = True
        loop = loop_state.loop
        if loop is None:
            return

        def _schedule_shutdown() -> None:
            async def _shutdown() -> None:
                try:
                    ws = loop_state.websocket
                    if ws is not None and not ws.closed:
                        await ws.close()
                except Exception:
                    logger.debug("StateChannel.stop: close failed", exc_info=True)
                outbox = loop_state.outbox
                if outbox is not None:
                    try:
                        outbox.put_nowait('{"type":"shutdown"}')
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

        if msg_type == _LEGACY_NOTIFY_STATE_TYPE:
            self._handle_legacy_notify_state(data)
            return

        if msg_type == STATE_UPDATE_TYPE:
            self._handle_state_update_frame(data)
            return

        if msg_type == NOTIFY_SCENE_TYPE:
            self._handle_notify_scene(data)
            return

        if msg_type == NOTIFY_STREAM_TYPE:
            self._handle_notify_stream(data)
            return

        if msg_type in (SESSION_WELCOME_TYPE, SESSION_REJECT_TYPE):
            logger.debug("Ignoring post-handshake control message: %s", msg_type)
            return

        if msg_type == 'video_config':
            if self.handle_video_config:
                try:
                    self.handle_video_config(data)
                except Exception:
                    logger.debug("handle_video_config callback failed", exc_info=True)
            return

        if msg_type == 'dims.update':
            if self.handle_dims_update:
                try:
                    meta_raw = data.get('meta') or {}
                    logger.debug(
                        "dims.update raw meta: level=%s level_shape=%s range=%s",
                        meta_raw.get('level'),
                        meta_raw.get('level_shape') or meta_raw.get('sizes'),
                        meta_raw.get('range') or meta_raw.get('ranges'),
                    )
                    meta = normalize_meta(data.get('meta') or {})
                    cur = inflate_current_step(data.get('current_step'), meta)
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
                    self.handle_dims_update(fwd)
                except Exception:
                    logger.debug("handle_dims_update callback failed", exc_info=True)
            return

        if msg_type == SCENE_SPEC_TYPE:
            if self.handle_scene_spec:
                try:
                    spec = SceneSpecMessage.from_dict(data)
                    if _STATE_DEBUG:
                        logger.debug(
                            "received scene.spec: layers=%s capabilities=%s",
                            [layer.layer_id for layer in spec.scene.layers],
                            spec.scene.capabilities,
                        )
                    self.handle_scene_spec(spec)
                except Exception:
                    logger.debug("handle_scene_spec callback failed", exc_info=True)
            return

        if msg_type == LAYER_UPDATE_TYPE:
            if self.handle_layer_update:
                try:
                    update = LayerUpdateMessage.from_dict(data)
                    if _STATE_DEBUG:
                        logger.debug(
                            "received layer.update: id=%s partial=%s",
                            update.layer.layer_id if update.layer else None,
                            update.partial,
                        )
                    self.handle_layer_update(update)
                except Exception:
                    logger.debug("handle_layer_update callback failed", exc_info=True)
            return

        if msg_type == LAYER_REMOVE_TYPE:
            if self.handle_layer_remove:
                try:
                    removal = LayerRemoveMessage.from_dict(data)
                    if _STATE_DEBUG:
                        logger.debug("received layer.remove: id=%s reason=%s", removal.layer_id, removal.reason)
                    self.handle_layer_remove(removal)
                except Exception:
                    logger.debug("handle_layer_remove callback failed", exc_info=True)
            return

        if msg_type == STATE_UPDATE_TYPE:
            if self.handle_state_update:
                try:
                    update = StateUpdateMessage.from_dict(data)
                    if _STATE_DEBUG:
                        logger.debug(
                            "received state.update: scope=%s target=%s key=%s phase=%s",
                            update.scope,
                            update.target,
                            update.key,
                            update.phase,
                        )
                    self.handle_state_update(update)
                except Exception:
                    logger.debug("handle_state_update callback failed", exc_info=True)
            return

    def _handle_legacy_notify_state(self, data: dict) -> None:
        if not self.handle_state_update:
            return
        try:
            payload_block = data.get('payload')
            if isinstance(payload_block, dict):
                update = StateUpdateMessage.from_dict(dict(payload_block))
            else:
                update = StateUpdateMessage.from_dict(dict(data))
            if _STATE_DEBUG:
                logger.debug(
                    "received notify.state: scope=%s target=%s key=%s phase=%s",
                    update.scope,
                    update.target,
                    update.key,
                    update.phase,
                )
            self.handle_state_update(update)
        except Exception:
            logger.debug("notify.state dispatch failed", exc_info=True)

    def _handle_state_update_frame(self, data: Mapping[str, Any]) -> None:
        if not self.handle_state_update:
            return
        try:
            frame = _ENVELOPE_PARSER.parse_state_update(data)
            payload = frame.payload
            update = StateUpdateMessage.from_dict(
                {
                    'scope': payload.scope,
                    'target': payload.target,
                    'key': payload.key,
                    'value': payload.value,
                }
            )
            if _STATE_DEBUG:
                logger.debug(
                    "received state.update frame: scope=%s target=%s key=%s",
                    update.scope,
                    update.target,
                    update.key,
                )
            self.handle_state_update(update)
        except Exception:
            logger.debug("state.update dispatch failed", exc_info=True)

    async def _perform_handshake(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Exchange the session handshake with the state server before use."""

        client_info = HelloClientInfo(
            name="napari-cuda-client",
            version=_NAPARI_CUDA_VERSION,
            platform=platform.platform(),
        )
        resume_tokens = {topic: None for topic in _RESUMABLE_TOPICS}
        hello = build_session_hello(
            client=client_info,
            features=_REQUIRED_FEATURES,
            resume_tokens=resume_tokens,
            timestamp=time.time(),
        )
        text = json.dumps(hello.to_dict(), separators=(",", ":"))
        if _STATE_DEBUG:
            logger.debug("StateChannel handshake -> %s", text)
        await ws.send(text)

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=_HANDSHAKE_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("state handshake timed out waiting for session.welcome") from exc
        except Exception as exc:  # pragma: no cover - transport errors bubble up
            raise RuntimeError(f"state handshake failed: {exc}") from exc

        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception as exc:  # pragma: no cover - unexpected encoding
                raise RuntimeError("state handshake payload was not UTF-8") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("state handshake payload was not valid JSON") from exc

        msg_type = str(data.get("type") or "").lower()
        if msg_type == SESSION_WELCOME_TYPE:
            welcome = SessionWelcome.from_dict(data)
            feature_toggles = welcome.payload.features
            enabled = sorted(
                name for name, toggle in feature_toggles.items() if getattr(toggle, "enabled", False)
            )
            logger.info("State handshake accepted; features=%s", enabled or "unknown")
            return

        if msg_type == SESSION_REJECT_TYPE:
            reject = SessionReject.from_dict(data)
            details = reject.payload.details or {}
            code = reject.payload.code
            message = reject.payload.message
            raise RuntimeError(
                f"state handshake rejected: {code}: {message} ({details})" if details else f"state handshake rejected: {code}: {message}"
            )

        raise RuntimeError(f"unexpected handshake response type: {msg_type or 'unknown'}")

    def _handle_notify_scene(self, data: dict) -> None:
        if not self.handle_scene_spec:
            return
        try:
            envelope = _ENVELOPE_PARSER.parse_notify_scene(data)
            state_block = envelope.payload.state or {}
            capabilities = None
            if isinstance(state_block, dict) and state_block.get('capabilities') is not None:
                try:
                    capabilities = [str(entry) for entry in state_block.get('capabilities', [])]
                except Exception:
                    capabilities = None
            spec = SceneSpecMessage(
                type=SCENE_SPEC_TYPE,
                version=envelope.payload.version,
                scene=envelope.payload.scene,
                capabilities=capabilities,
                timestamp=envelope.timestamp,
            )
            if _STATE_DEBUG:
                logger.debug(
                    "received notify.scene: layers=%s capabilities=%s",
                    [layer.layer_id for layer in spec.scene.layers],
                    spec.scene.capabilities,
                )
            self.handle_scene_spec(spec)
        except Exception:
            logger.debug("notify.scene dispatch failed", exc_info=True)

    def _handle_notify_stream(self, data: dict) -> None:
        if not self.handle_video_config:
            return
        try:
            envelope = _ENVELOPE_PARSER.parse_notify_stream(data)
            payload = envelope.payload
            config: dict[str, object] = {
                'type': 'video_config',
                'codec': payload.codec,
                'format': payload.format,
                'fps': payload.fps,
                'width': payload.frame_size[0],
                'height': payload.frame_size[1],
                'nal_length_size': payload.nal_length_size,
                'avcc': payload.avcc,
                'latency_policy': dict(payload.latency_policy),
            }
            if payload.vt_hint is not None:
                config['vt_hint'] = dict(payload.vt_hint)
            self.handle_video_config(config)  # type: ignore[arg-type]
        except Exception:
            logger.debug("notify.stream dispatch failed", exc_info=True)

    def request_keyframe_once(self) -> None:
        """Best-effort request for a keyframe via state channel (throttled)."""
        now = time.time()
        if self._last_key_req is not None and (now - self._last_key_req) < 0.5:
            return
        self._last_key_req = now
        # Prefer sending on the persistent connection via the sender task
        try:
            loop = self._loop_state.loop
            outbox = self._loop_state.outbox
            if loop is not None and outbox is not None:
                loop.call_soon_threadsafe(outbox.put_nowait, '{"type":"request_keyframe"}')
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
    def post(self, obj: dict) -> bool:
        """Enqueue a JSON message for the state channel sender.

        Returns True if enqueued, False if the channel is not ready.

        Thread-safe: uses loop.call_soon_threadsafe to put_nowait on the
        sender queue when available.
        """
        try:
            import json as _json
            msg = _json.dumps(obj, separators=(",", ":"))
        except Exception:
            logger.debug("post: JSON encode failed", exc_info=True)
            return False
        try:
            loop = self._loop_state.loop
            outbox = self._loop_state.outbox
            if loop is None or outbox is None:
                return False
            loop.call_soon_threadsafe(outbox.put_nowait, msg)
            return True
        except Exception:
            logger.debug("post: enqueue failed", exc_info=True)
            return False
