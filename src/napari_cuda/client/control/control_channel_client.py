from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import websockets

try:
    from napari_cuda import __version__ as _NAPARI_CUDA_VERSION
except Exception:  # pragma: no cover - fallback for unusual import issues
    _NAPARI_CUDA_VERSION = "dev"

from napari_cuda.protocol import (
    ACK_STATE_TYPE,
    ERROR_COMMAND_TYPE,
    NOTIFY_DIMS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    REPLY_COMMAND_TYPE,
    EnvelopeParser,
    SESSION_REJECT_TYPE,
    SESSION_WELCOME_TYPE,
    AckState,
    ErrorCommand,
    HelloClientInfo,
    ReplyCommand,
    SessionReject,
    SessionWelcome,
    build_session_hello,
)
from napari_cuda.protocol.messages import (
    LAYER_REMOVE_TYPE,
    LAYER_UPDATE_TYPE,
    SCENE_SPEC_TYPE,
    LayerRemoveMessage,
    LayerUpdateMessage,
    SceneSpecMessage,
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


@dataclass(frozen=True)
class ResumeCursor:
    seq: int
    delta_token: str


@dataclass(frozen=True)
class SessionMetadata:
    session_id: str
    heartbeat_s: float
    ack_timeout_ms: int | None
    resume_tokens: Dict[str, ResumeCursor | None]


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
        handle_ack_state: Optional[Callable[[AckState], None]] = None,
        handle_reply_command: Optional[Callable[[ReplyCommand], None]] = None,
        handle_error_command: Optional[Callable[[ErrorCommand], None]] = None,
        handle_session_ready: Optional[Callable[[SessionMetadata], None]] = None,
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
        self.handle_ack_state = handle_ack_state
        self.handle_reply_command = handle_reply_command
        self.handle_error_command = handle_error_command
        self.handle_session_ready = handle_session_ready
        self.handle_connected = handle_connected
        self.handle_disconnect = handle_disconnect
        self._last_key_req: Optional[float] = None
        self._loop_state = StateChannelLoop()
        self._session_metadata: SessionMetadata | None = None

    @property
    def session_metadata(self) -> SessionMetadata | None:
        return self._session_metadata

    @property
    def session_id(self) -> str | None:
        metadata = self._session_metadata
        return metadata.session_id if metadata is not None else None

    @property
    def ack_timeout_ms(self) -> int | None:
        metadata = self._session_metadata
        return metadata.ack_timeout_ms if metadata is not None else None

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
    def _handle_message(self, data: Mapping[str, object]) -> None:
        """Dispatch a single decoded message to registered callbacks."""

        raw_type = data.get("type")
        msg_type = (raw_type or "").lower()

        if msg_type == ACK_STATE_TYPE:
            self._handle_ack_state(data)
            return

        if msg_type == REPLY_COMMAND_TYPE:
            self._handle_reply_command(data)
            return

        if msg_type == ERROR_COMMAND_TYPE:
            self._handle_error_command(data)
            return

        if msg_type == NOTIFY_SCENE_TYPE:
            self._handle_notify_scene(data)
            return

        if msg_type == NOTIFY_STREAM_TYPE:
            self._handle_notify_stream(data)
            return

        if msg_type == NOTIFY_DIMS_TYPE:
            self._handle_notify_dims(data)
            return

        if msg_type in (SESSION_WELCOME_TYPE, SESSION_REJECT_TYPE):
            logger.debug("Ignoring post-handshake control message: %s", msg_type)
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

    async def _perform_handshake(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Exchange the session handshake with the state server before use."""

        self._session_metadata = None
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
            metadata = self._record_session_metadata(welcome)
            feature_toggles = welcome.payload.features
            enabled = sorted(name for name, toggle in feature_toggles.items() if getattr(toggle, "enabled", False))
            logger.info(
                "State handshake accepted; session=%s ack_timeout_ms=%s features=%s",
                metadata.session_id,
                metadata.ack_timeout_ms,
                enabled or "unknown",
            )
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

    def _handle_notify_dims(self, data: Mapping[str, object]) -> None:
        if not self.handle_dims_update:
            return
        try:
            frame = _ENVELOPE_PARSER.parse_notify_dims(data)
            envelope = frame.envelope
            payload = frame.payload
            message = {
                'frame_id': envelope.frame_id,
                'session': envelope.session,
                'timestamp': envelope.timestamp,
                'current_step': tuple(payload.current_step),
                'ndisplay': payload.ndisplay,
                'mode': payload.mode,
                'source': payload.source,
            }
            if _STATE_DEBUG:
                logger.debug(
                    "received notify.dims: frame=%s step=%s ndisplay=%s mode=%s",
                    envelope.frame_id,
                    payload.current_step,
                    payload.ndisplay,
                    payload.mode,
                )
            self.handle_dims_update(message)
        except Exception:
            logger.debug("notify.dims dispatch failed", exc_info=True)

    def _record_session_metadata(self, welcome: SessionWelcome) -> SessionMetadata:
        session_info = welcome.payload.session
        resume_tokens: Dict[str, ResumeCursor | None] = {}
        for name, toggle in welcome.payload.features.items():
            resume_state = getattr(toggle, "resume_state", None)
            if resume_state is not None:
                resume_tokens[name] = ResumeCursor(seq=int(resume_state.seq), delta_token=str(resume_state.delta_token))
            else:
                resume_tokens[name] = None
        metadata = SessionMetadata(
            session_id=session_info.id,
            heartbeat_s=float(session_info.heartbeat_s),
            ack_timeout_ms=int(session_info.ack_timeout_ms) if session_info.ack_timeout_ms is not None else None,
            resume_tokens=resume_tokens,
        )
        self._session_metadata = metadata
        if self.handle_session_ready:
            try:
                self.handle_session_ready(metadata)
            except Exception:
                logger.debug("handle_session_ready callback failed", exc_info=True)
        return metadata

    def _handle_ack_state(self, data: Mapping[str, object]) -> None:
        try:
            frame = _ENVELOPE_PARSER.parse_ack_state(data)
        except Exception:
            logger.debug("ack.state dispatch failed", exc_info=True)
            return

        payload = frame.payload
        logger.info(
            "ack.state received intent=%s in_reply_to=%s status=%s",
            payload.intent_id,
            payload.in_reply_to,
            payload.status,
        )
        if _STATE_DEBUG and payload.applied_value is not None:
            logger.debug(
                "ack.state applied_value intent=%s in_reply_to=%s value=%s",
                payload.intent_id,
                payload.in_reply_to,
                payload.applied_value,
            )
        if self.handle_ack_state:
            try:
                self.handle_ack_state(frame)
            except Exception:
                logger.debug("handle_ack_state callback failed", exc_info=True)

    def _handle_reply_command(self, data: Mapping[str, object]) -> None:
        try:
            frame = _ENVELOPE_PARSER.parse_reply_command(data)
        except Exception:
            logger.debug("reply.command dispatch failed", exc_info=True)
            return

        payload = frame.payload
        logger.info(
            "reply.command received intent=%s in_reply_to=%s",
            payload.intent_id,
            payload.in_reply_to,
        )
        if self.handle_reply_command:
            try:
                self.handle_reply_command(frame)
            except Exception:
                logger.debug("handle_reply_command callback failed", exc_info=True)

    def _handle_error_command(self, data: Mapping[str, object]) -> None:
        try:
            frame = _ENVELOPE_PARSER.parse_error_command(data)
        except Exception:
            logger.debug("error.command dispatch failed", exc_info=True)
            return

        payload = frame.payload
        error = payload.error
        logger.info(
            "error.command received intent=%s in_reply_to=%s code=%s message=%s",
            payload.intent_id,
            payload.in_reply_to,
            error.code,
            error.message,
        )
        if self.handle_error_command:
            try:
                self.handle_error_command(frame)
            except Exception:
                logger.debug("handle_error_command callback failed", exc_info=True)

    def request_keyframe_once(self) -> None:
        """Best-effort request for a keyframe via state channel (throttled)."""
        now = time.time()
        if self._last_key_req is not None and (now - self._last_key_req) < 0.5:
            return
        self._last_key_req = now
        # Prefer sending on the persistent connection via the sender task
        try:
            if self._enqueue_text('{"type":"request_keyframe"}'):
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
    def post(self, obj: Mapping[str, Any]) -> bool:
        """Enqueue a JSON message for the state channel sender.

        Returns True if enqueued, False if the channel is not ready.

        Thread-safe: uses loop.call_soon_threadsafe to put_nowait on the
        sender queue when available.
        """
        return self._enqueue_mapping(obj)

    def send_frame(self, frame: Any) -> bool:
        """Serialize a builder-backed frame and enqueue it for delivery."""

        try:
            payload = frame.to_dict()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("send_frame: to_dict failed", exc_info=True)
            return False
        if not isinstance(payload, Mapping):
            logger.debug("send_frame: payload must be a mapping")
            return False
        return self._enqueue_mapping(payload)

    def _enqueue_mapping(self, payload: Mapping[str, Any]) -> bool:
        try:
            msg = json.dumps(payload, separators=(",", ":"))
        except Exception:
            logger.debug("enqueue_mapping: JSON encode failed", exc_info=True)
            return False
        return self._enqueue_text(msg)

    def _enqueue_text(self, text: str) -> bool:
        try:
            loop = self._loop_state.loop
            outbox = self._loop_state.outbox
            if loop is None or outbox is None:
                return False
            loop.call_soon_threadsafe(outbox.put_nowait, text)
            return True
        except Exception:
            logger.debug("enqueue_text: enqueue failed", exc_info=True)
            return False
