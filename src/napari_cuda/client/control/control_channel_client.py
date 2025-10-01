from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import websockets

try:
    from napari_cuda import __version__ as _NAPARI_CUDA_VERSION
except Exception:  # pragma: no cover - fallback for unusual import issues
    _NAPARI_CUDA_VERSION = "dev"

from napari_cuda.protocol import (
    ACK_STATE_TYPE,
    ERROR_COMMAND_TYPE,
    NOTIFY_CAMERA_TYPE,
    NOTIFY_DIMS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_STREAM_TYPE,
    REPLY_COMMAND_TYPE,
    SESSION_HEARTBEAT_TYPE,
    EnvelopeParser,
    SESSION_REJECT_TYPE,
    SESSION_WELCOME_TYPE,
    AckState,
    ErrorCommand,
    FeatureToggle,
    HelloClientInfo,
    ReplyCommand,
    SessionReject,
    SessionWelcome,
    build_session_ack,
    build_session_hello,
)
from napari_cuda.protocol.messages import (
    LAYER_REMOVE_TYPE,
    LAYER_UPDATE_TYPE,
    SCENE_SPEC_TYPE,
    LayerRemoveMessage,
    LayerUpdateMessage,
    NotifyDimsFrame,
    NotifySceneLevelPayload,
    NotifyStreamFrame,
    SceneSpecMessage,
)


logger = logging.getLogger(__name__)


@dataclass
class StateChannelLoop:
    loop: asyncio.AbstractEventLoop | None = None
    websocket: websockets.WebSocketClientProtocol | None = None
    outbox: asyncio.Queue[str] | None = None
    stop_requested: bool = False


class HeartbeatAckError(RuntimeError):
    """Raised when a heartbeat acknowledgement cannot be enqueued."""


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
_RESUMABLE_TOPICS = ("notify.scene", "notify.scene.level", "notify.layers", "notify.stream")
_REQUIRED_FEATURES = {
    "notify.scene": True,
    "notify.scene.level": True,
    "notify.layers": True,
    "notify.stream": True,
    "notify.dims": True,
    "notify.camera": True,
}


def _normalize_level_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_level_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_level_value(v) for v in value]
    return value


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
    features: Dict[str, FeatureToggle]


class StateChannel:
    """Maintains a WebSocket connection to the state channel.

    Pings periodically to keep the connection alive and forwards parsed
    greenfield envelopes to the caller via callbacks.
    """

    def __init__(
        self,
        host: str,
        port: int,
        handle_notify_stream: Optional[Callable[[NotifyStreamFrame], None]] = None,
        handle_dims_update: Optional[Callable[[NotifyDimsFrame], None]] = None,
        handle_scene_spec: Optional[Callable[[SceneSpecMessage], None]] = None,
        handle_scene_level: Optional[Callable[[NotifySceneLevelPayload], None]] = None,
        handle_layer_update: Optional[Callable[[LayerUpdateMessage], None]] = None,
        handle_layer_remove: Optional[Callable[[LayerRemoveMessage], None]] = None,
        handle_notify_camera: Optional[Callable[[Any], None]] = None,
        handle_ack_state: Optional[Callable[[AckState], None]] = None,
        handle_reply_command: Optional[Callable[[ReplyCommand], None]] = None,
        handle_error_command: Optional[Callable[[ErrorCommand], None]] = None,
        handle_session_ready: Optional[Callable[[SessionMetadata], None]] = None,
        handle_connected: Optional[Callable[[], None]] = None,
        handle_disconnect: Optional[Callable[[Optional[Exception]], None]] = None,
        handle_scene_policies: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.handle_notify_stream = handle_notify_stream
        self.handle_dims_update = handle_dims_update
        self.handle_scene_spec = handle_scene_spec
        self.handle_scene_level = handle_scene_level
        self.handle_layer_update = handle_layer_update
        self.handle_layer_remove = handle_layer_remove
        self.handle_notify_camera = handle_notify_camera
        self.handle_ack_state = handle_ack_state
        self.handle_reply_command = handle_reply_command
        self.handle_error_command = handle_error_command
        self.handle_session_ready = handle_session_ready
        self.handle_connected = handle_connected
        self.handle_disconnect = handle_disconnect
        self.handle_scene_policies = handle_scene_policies
        self._loop_state = StateChannelLoop()
        self._session_metadata: SessionMetadata | None = None
        self._resume_tokens: Dict[str, ResumeCursor | None] = {topic: None for topic in _RESUMABLE_TOPICS}
        self._last_heartbeat_ts: float | None = None
        self._feature_toggles: Dict[str, FeatureToggle] = {}

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

    def feature_enabled(self, name: str) -> bool:
        toggle = self._feature_toggles.get(name)
        return bool(getattr(toggle, "enabled", False))

    def command_catalog(self) -> tuple[str, ...]:
        toggle = self._feature_toggles.get("call.command")
        if toggle and toggle.commands:
            return tuple(toggle.commands)
        return ()

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

                    heartbeat_task: asyncio.Task[None] | None = None
                    metadata = self._session_metadata
                    if metadata is not None and metadata.heartbeat_s > 0:
                        heartbeat_task = asyncio.create_task(
                            self._monitor_heartbeat(ws, metadata.heartbeat_s)
                        )

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
                    logger.debug("StateChannel: awaiting server keyframe; legacy request disabled")
                    import json as _json
                    heartbeat_exc: HeartbeatAckError | None = None
                    async for msg in ws:
                        try:
                            data = _json.loads(msg)
                        except _json.JSONDecodeError:
                            continue
                        try:
                            self._handle_message(data)
                        except HeartbeatAckError as exc:
                            heartbeat_exc = exc
                            with suppress(Exception):
                                await ws.close(code=1011, reason="heartbeat failure")
                            break
                        except Exception:
                            logger.debug("StateChannel message dispatch failed", exc_info=True)
                    ping_task.cancel()
                    send_task.cancel()
                    if heartbeat_task is not None:
                        heartbeat_task.cancel()
                    try:
                        tasks = [ping_task, send_task]
                        if heartbeat_task is not None:
                            tasks.append(heartbeat_task)
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result in results:
                            if isinstance(result, HeartbeatAckError):
                                heartbeat_exc = result
                                break
                    except Exception:
                        logger.debug("StateChannel helper cancel failed", exc_info=True)
                    finally:
                        loop_state.websocket = None
                        loop_state.outbox = None
                    if heartbeat_exc is not None:
                        raise heartbeat_exc
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
                elif isinstance(e, HeartbeatAckError):
                    logger.warning("State channel heartbeat failed (%s); reconnecting in %.0fs", msg, retry_delay)
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

        if msg_type == SESSION_HEARTBEAT_TYPE:
            self._handle_session_heartbeat(data)
            return

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

        if msg_type == NOTIFY_SCENE_LEVEL_TYPE:
            self._handle_scene_level(data)
            return

        if msg_type == NOTIFY_STREAM_TYPE:
            self._handle_notify_stream(data)
            return

        if msg_type == NOTIFY_DIMS_TYPE:
            self._handle_notify_dims(data)
            return

        if msg_type == NOTIFY_CAMERA_TYPE:
            self._handle_notify_camera(data)
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
        self._feature_toggles = {}
        client_info = HelloClientInfo(
            name="napari-cuda-client",
            version=_NAPARI_CUDA_VERSION,
            platform=platform.platform(),
        )
        resume_tokens = self._resume_token_payload()
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
            self._handle_handshake_reject(reject)

        raise RuntimeError(f"unexpected handshake response type: {msg_type or 'unknown'}")

    def _handle_notify_scene(self, data: Mapping[str, object]) -> None:
        try:
            envelope = _ENVELOPE_PARSER.parse_notify_scene(data)
            self._store_resume_cursor(NOTIFY_SCENE_TYPE, envelope.envelope)
        except Exception:
            logger.debug("notify.scene dispatch failed", exc_info=True)
            return

        if not self.handle_scene_spec:
            return

        payload = envelope.payload
        env_meta = envelope.envelope
        scene_payload: Dict[str, Any] = {
            "layers": [dict(layer) for layer in payload.layers],
        }

        viewer_block = payload.viewer
        dims_block = viewer_block.get("dims") if isinstance(viewer_block, Mapping) else None
        camera_block = viewer_block.get("camera") if isinstance(viewer_block, Mapping) else None
        if isinstance(dims_block, Mapping):
            scene_payload["dims"] = dict(dims_block)
        if isinstance(camera_block, Mapping):
            scene_payload["camera"] = dict(camera_block)

        ancillary = payload.ancillary if isinstance(payload.ancillary, Mapping) else {}
        metadata_block: Dict[str, Any] = {}
        if isinstance(ancillary, Mapping):
            meta_section = ancillary.get("metadata")
            if isinstance(meta_section, Mapping):
                metadata_block.update({str(k): v for k, v in meta_section.items()})
            for extra_key in ("volume_state", "policy_metrics"):
                extra_value = ancillary.get(extra_key)
                if extra_value is not None:
                    metadata_block[extra_key] = extra_value
        if metadata_block:
            scene_payload["metadata"] = metadata_block

        policies_mapping = payload.policies if isinstance(payload.policies, Mapping) else None
        if isinstance(policies_mapping, Mapping) and self.handle_scene_policies:
            policies_payload = {str(k): _normalize_level_value(v) for k, v in policies_mapping.items()}
            try:
                self.handle_scene_policies(policies_payload)
            except Exception:
                logger.debug("handle_scene_policies callback failed", exc_info=True)

        capabilities = None
        anc_caps = ancillary.get("capabilities") if isinstance(ancillary, Mapping) else None
        if isinstance(anc_caps, Sequence):
            capabilities = [str(entry) for entry in anc_caps if entry is not None]
            scene_payload["capabilities"] = capabilities

        scene_message_dict: Dict[str, Any] = {
            "type": SCENE_SPEC_TYPE,
            "version": 1,
            "timestamp": env_meta.timestamp,
            "scene": scene_payload,
        }
        if capabilities:
            scene_message_dict["capabilities"] = capabilities

        try:
            spec = SceneSpecMessage.from_dict(scene_message_dict)
        except Exception:
            logger.debug("notify.scene conversion failed", exc_info=True)
            return

        if _STATE_DEBUG:
            logger.debug(
                "received notify.scene: layers=%s capabilities=%s",
                [layer.layer_id for layer in spec.scene.layers],
                spec.scene.capabilities,
            )
        self.handle_scene_spec(spec)

    def _handle_scene_level(self, data: Mapping[str, object]) -> None:
        if not self.handle_scene_level:
            return
        try:
            frame = _ENVELOPE_PARSER.parse_notify_scene_level(data)
            self._store_resume_cursor(NOTIFY_SCENE_LEVEL_TYPE, frame.envelope)
        except Exception:
            logger.debug("notify.scene.level dispatch failed", exc_info=True)
            return

        try:
            self.handle_scene_level(frame.payload)
        except Exception:
            logger.debug("handle_scene_level callback failed", exc_info=True)

    def _handle_notify_stream(self, data: Mapping[str, object]) -> None:
        try:
            frame = _ENVELOPE_PARSER.parse_notify_stream(data)
            self._store_resume_cursor(NOTIFY_STREAM_TYPE, frame.envelope)
        except Exception:
            logger.debug("notify.stream dispatch failed", exc_info=True)
            return

        if not self.handle_notify_stream:
            return

        if _STATE_DEBUG:
            payload = frame.payload
            logger.debug(
                "received notify.stream: codec=%s fps=%.3f size=%sx%s",
                payload.codec,
                payload.fps,
                payload.frame_size[0],
                payload.frame_size[1],
            )
        self.handle_notify_stream(frame)

    def _handle_notify_dims(self, data: Mapping[str, object]) -> None:
        if not self.handle_dims_update:
            return
        try:
            frame = _ENVELOPE_PARSER.parse_notify_dims(data)
            if _STATE_DEBUG:
                payload = frame.payload
                logger.debug(
                    "received notify.dims: frame=%s step=%s ndisplay=%s mode=%s",
                    frame.envelope.frame_id,
                    payload.current_step,
                    payload.ndisplay,
                    payload.mode,
                )
            self.handle_dims_update(frame)
        except Exception:
            logger.debug("notify.dims dispatch failed", exc_info=True)

    def _handle_notify_camera(self, data: Mapping[str, object]) -> None:
        if not self.handle_notify_camera:
            return
        try:
            frame = _ENVELOPE_PARSER.parse_notify_camera(data)
            if _STATE_DEBUG:
                payload = frame.payload
                logger.debug(
                    "received notify.camera: mode=%s origin=%s intent=%s",
                    payload.mode,
                    payload.origin,
                    frame.envelope.intent_id,
                )
            self.handle_notify_camera(frame)
        except Exception:
            logger.debug("notify.camera dispatch failed", exc_info=True)

    def _record_session_metadata(self, welcome: SessionWelcome) -> SessionMetadata:
        session_info = welcome.payload.session
        feature_toggles = welcome.payload.features
        resume_tokens: Dict[str, ResumeCursor | None] = {}
        for name, toggle in feature_toggles.items():
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
            features=feature_toggles,
        )
        self._session_metadata = metadata
        self._feature_toggles = feature_toggles
        for topic in _RESUMABLE_TOPICS:
            self._resume_tokens[topic] = resume_tokens.get(topic)
        self._last_heartbeat_ts = time.time()
        if self.handle_session_ready:
            try:
                self.handle_session_ready(metadata)
            except Exception:
                logger.debug("handle_session_ready callback failed", exc_info=True)
        return metadata

    def _handle_session_heartbeat(self, data: Mapping[str, object]) -> None:
        frame = _ENVELOPE_PARSER.parse_heartbeat(data)
        session_id = frame.envelope.session or self.session_id
        if not session_id:
            raise ValueError("session.heartbeat missing session identifier")
        ack = build_session_ack(session_id=session_id, timestamp=time.time())
        if not self.send_frame(ack):
            raise HeartbeatAckError("unable to enqueue session.ack response")
        self._last_heartbeat_ts = time.time()
        if _STATE_DEBUG:
            logger.debug(
                "session.heartbeat received frame=%s",
                frame.envelope.frame_id,
            )

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

    def _store_resume_cursor(self, topic: str, envelope: Any) -> None:
        if topic not in _RESUMABLE_TOPICS:
            return
        delta_token = getattr(envelope, "delta_token", None)
        if delta_token is None:
            return
        seq_value = getattr(envelope, "seq", None)
        seq = 0 if seq_value is None else int(seq_value)
        cursor = ResumeCursor(seq=seq, delta_token=str(delta_token))
        self._resume_tokens[topic] = cursor
        metadata = self._session_metadata
        if metadata is not None:
            metadata.resume_tokens[topic] = cursor

    def _resume_token_payload(self) -> Dict[str, str | None]:
        payload: Dict[str, str | None] = {topic: None for topic in _RESUMABLE_TOPICS}
        for topic, cursor in self._resume_tokens.items():
            if cursor is not None:
                payload[topic] = cursor.delta_token
        return payload

    def _handle_handshake_reject(self, reject: SessionReject) -> None:
        details = reject.payload.details or {}
        code = reject.payload.code
        message = reject.payload.message
        if code == "invalid_resume_token":
            topic = details.get("topic") if isinstance(details, Mapping) else None
            if topic:
                self._resume_tokens[str(topic)] = None
            else:
                for name in _RESUMABLE_TOPICS:
                    self._resume_tokens[name] = None
        detail_text = f" ({details})" if details else ""
        raise RuntimeError(f"state handshake rejected: {code}: {message}{detail_text}")

    async def _monitor_heartbeat(
        self,
        ws: websockets.WebSocketClientProtocol,
        interval: float,
    ) -> None:
        grace = max(interval + 1.0, interval * 1.5)
        while True:
            await asyncio.sleep(interval)
            last = self._last_heartbeat_ts
            if last is None:
                continue
            elapsed = time.time() - last
            if elapsed > grace:
                logger.warning(
                    "State heartbeat timeout after %.2fs (expected %.2fs)",
                    elapsed,
                    interval,
                )
                with suppress(Exception):
                    await ws.close(code=1011, reason="heartbeat timeout")
                raise HeartbeatAckError("heartbeat timeout")

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
