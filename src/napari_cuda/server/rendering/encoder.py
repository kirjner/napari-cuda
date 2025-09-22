from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

import PyNvVideoCodec as pnvc  # type: ignore

from napari_cuda.server.config import ServerCtx


logger = logging.getLogger(__name__)


_ALLOWED_INPUT_FORMATS = {"YUV444", "NV12", "ARGB", "ABGR"}


def _env_int(env: Mapping[str, str], name: str, default: Optional[int] = None) -> Optional[int]:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_str(env: Mapping[str, str], name: str, default: Optional[str] = None) -> Optional[str]:
    raw = env.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    low = raw.strip().lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    try:
        return bool(int(raw))
    except Exception:
        return default


@dataclass(frozen=True)
class EncoderTimings:
    encode_ms: float
    pack_ms: float


class Encoder:
    """NVENC helper encapsulating creation, encode calls, and resets."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fps_hint: int = 60,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._env: Mapping[str, str] = dict(env) if env is not None else os.environ
        self._encoder: Optional[Any] = None
        self._input_format: str = "YUV444"
        self._log_keyframes = False
        self._log_settings = True
        self._force_next_idr = False
        self._frame_index = 0
        self._last_kwargs: dict[str, Any] = {}
        self._ctx: Optional[ServerCtx] = None
        self._fps_hint = int(fps_hint)

    @property
    def input_format(self) -> str:
        return self._input_format

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def is_ready(self) -> bool:
        return self._encoder is not None

    def set_fps_hint(self, fps_hint: int) -> None:
        self._fps_hint = int(fps_hint)

    def setup(self, ctx: Optional[ServerCtx]) -> None:
        self._ctx = ctx
        env = self._env

        fmt = (_env_str(env, "NAPARI_CUDA_ENCODER_INPUT_FMT", "YUV444") or "YUV444").upper()
        if fmt not in _ALLOWED_INPUT_FORMATS:
            fmt = "YUV444"
        self._input_format = fmt
        self._log_settings = bool(_env_int(env, "NAPARI_CUDA_LOG_ENCODER_SETTINGS", 1))
        self._log_keyframes = _env_bool(env, "NAPARI_CUDA_LOG_KEYFRAMES", False)

        rc_mode = (_env_str(env, "NAPARI_CUDA_RC", "cbr") or "cbr").lower()
        preset_env = _env_str(env, "NAPARI_CUDA_PRESET", "") or ""

        bitrate_default = 10_000_000
        if ctx is not None:
            try:
                bitrate_default = int(getattr(ctx.cfg.encode, "bitrate", bitrate_default))
            except Exception:
                bitrate_default = 10_000_000
        bitrate = _env_int(env, "NAPARI_CUDA_BITRATE", bitrate_default)
        max_bitrate = _env_int(env, "NAPARI_CUDA_MAXBITRATE")
        lookahead = _env_int(env, "NAPARI_CUDA_LOOKAHEAD", 0) or 0
        aq = _env_int(env, "NAPARI_CUDA_AQ", 0) or 0
        temporalaq = _env_int(env, "NAPARI_CUDA_TEMPORALAQ", 0) or 0
        nonrefp = _env_int(env, "NAPARI_CUDA_NONREFP", 0) or 0
        bf_override = _env_int(env, "NAPARI_CUDA_BFRAMES")
        try:
            idr_period = int(_env_str(env, "NAPARI_CUDA_IDR_PERIOD", "600") or "600")
        except Exception:
            idr_period = 600

        preset = preset_env.strip() if preset_env.strip() else "P3"
        kwargs: dict[str, Any] = {
            "codec": "h264",
            "tuning_info": "low_latency",
            "preset": preset,
            "bf": 0,
            "repeatspspps": 1,
            "idrperiod": int(idr_period),
            "rc": rc_mode,
        }
        if bitrate and bitrate > 0:
            kwargs["bitrate"] = int(bitrate)
        if max_bitrate and max_bitrate > 0:
            kwargs["maxbitrate"] = int(max_bitrate)
        elif rc_mode == "cbr" and bitrate and bitrate > 0:
            kwargs["maxbitrate"] = int(bitrate)
        bf = max(0, bf_override) if bf_override is not None else 0
        kwargs["bf"] = int(bf)

        fps_num = self._fps_hint if self._fps_hint > 0 else None
        if (fps_num is None or fps_num <= 0) and ctx is not None:
            try:
                fps_num = int(getattr(ctx.cfg.encode, "fps", 60))
            except Exception:
                fps_num = None
        if fps_num is None or fps_num <= 0:
            fps_env = _env_int(env, "NAPARI_CUDA_FPS")
            if fps_env is not None and fps_env > 0:
                fps_num = fps_env
        if fps_num is None or fps_num <= 0:
            fps_num = 60
        kwargs["frameRateNum"] = int(max(1, fps_num))
        kwargs["frameRateDen"] = 1

        if lookahead > 0:
            kwargs["lookahead"] = int(lookahead)
        if aq > 0:
            kwargs["aq"] = int(max(0, aq))
        if temporalaq > 0:
            kwargs["temporalaq"] = int(max(0, temporalaq))
        if nonrefp > 0:
            kwargs["nonrefp"] = 1

        logger.info(
            "Creating NVENC encoder: %dx%d fmt=%s preset=%s rc=%s",
            self._width,
            self._height,
            self._input_format,
            preset,
            rc_mode,
        )
        self._last_kwargs = kwargs

        try:
            self._encoder = pnvc.CreateEncoder(
                width=self._width,
                height=self._height,
                fmt=self._input_format,
                usecpuinputbuffer=False,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("Preset NVENC path failed (%s)", exc, exc_info=True)
            self._encoder = None
        else:
            if self._log_settings:
                self._log_encoder_settings("preset", kwargs)
            logger.info(
                "NVENC encoder created (preset): %dx%d fmt=%s preset=%s tuning=low_latency bf=%d rc=%s bitrate=%s max=%s idrperiod=%d lookahead=%d aq=%d temporalaq=%d",
                self._width,
                self._height,
                self._input_format,
                preset,
                int(bf),
                rc_mode.upper(),
                kwargs.get("bitrate"),
                kwargs.get("maxbitrate"),
                int(idr_period),
                kwargs.get("lookahead", 0),
                kwargs.get("aq", 0),
                kwargs.get("temporalaq", 0),
            )
            self._force_next_idr = True
            self._frame_index = 0
            return

        if self._encoder is None:
            logger.info("Creating NVENC fallback encoder (no preset kwargs)")
            try:
                self._encoder = pnvc.CreateEncoder(
                    width=self._width,
                    height=self._height,
                    fmt=self._input_format,
                    usecpuinputbuffer=False,
                )
            except Exception as exc:
                logger.exception("NVENC fallback encoder creation failed: %s", exc)
                self._encoder = None
            else:
                logger.warning("NVENC encoder created without preset kwargs; cadence may vary")
                self._force_next_idr = True
                self._frame_index = 0

    def encode(self, frame: Any) -> tuple[Optional[Any], EncoderTimings]:
        encoder = self._encoder
        if encoder is None:
            return None, EncoderTimings(encode_ms=0.0, pack_ms=0.0)

        t_e0 = time.perf_counter()
        pic_flags = 0
        if self._force_next_idr:
            try:
                pic_flags |= int(pnvc.NV_ENC_PIC_FLAGS.FORCEIDR)
                pic_flags |= int(pnvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)
            except Exception:
                logger.debug("NVENC pic flag constants unavailable", exc_info=True)
            self._force_next_idr = False
        packet = encoder.Encode(frame, pic_flags) if pic_flags else encoder.Encode(frame)
        t_e1 = time.perf_counter()
        encode_ms = (t_e1 - t_e0) * 1000.0
        pack_ms = 0.0
        self._frame_index += 1
        if self._log_keyframes:
            self._log_keyframe(packet)
        return packet, EncoderTimings(encode_ms=encode_ms, pack_ms=pack_ms)

    def request_idr(self) -> None:
        self._force_next_idr = True

    def force_idr(self) -> None:
        encoder = self._encoder
        if encoder is None:
            return
        if hasattr(encoder, "GetEncodeReconfigureParams") and hasattr(encoder, "Reconfigure"):
            params = encoder.GetEncodeReconfigureParams()
            if hasattr(params, "forceIDR"):
                setattr(params, "forceIDR", 1)
            encoder.Reconfigure(params)

    def reset(self, ctx: Optional[ServerCtx]) -> None:
        self.shutdown()
        self.setup(ctx)

    def shutdown(self) -> None:
        encoder = self._encoder
        if encoder is None:
            return
        try:
            encoder.EndEncode()
        except Exception:
            logger.debug("Encoder EndEncode failed", exc_info=True)
        self._encoder = None

    def _log_keyframe(self, packet: Any) -> None:
        if packet is None:
            return

        def _contains_idr(buf: Any) -> bool:
            try:
                data = bytes(buf)
            except Exception:
                return False
            n = len(data)
            i = 0
            seen = False
            while i + 3 < n:
                if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
                    if i + 3 < n:
                        nal_type = data[i + 3] & 0x1F
                        if nal_type == 5:
                            seen = True
                    i += 3
                elif i + 4 < n and data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 0 and data[i + 3] == 1:
                    if i + 4 < n:
                        nal_type = data[i + 4] & 0x1F
                        if nal_type == 5:
                            seen = True
                    i += 4
                else:
                    i += 1
            if seen:
                return True
            i = 0
            while i + 4 <= n:
                ln = int.from_bytes(data[i : i + 4], "big")
                i += 4
                if ln <= 0 or i + ln > n:
                    break
                nal_type = data[i] & 0x1F if ln >= 1 else 0
                if nal_type == 5:
                    return True
                i += ln
            return False

        keyframe = False
        if isinstance(packet, (list, tuple)):
            for part in packet:
                if _contains_idr(part):
                    keyframe = True
                    break
        else:
            keyframe = _contains_idr(packet)
        logger.debug("Encode frame %d: keyframe=%s", self._frame_index, keyframe)

    def _log_encoder_settings(self, path: str, init_kwargs: Mapping[str, Any]) -> None:
        encoder = self._encoder
        if encoder is None:
            return
        canon: MutableMapping[str, Any] = {}

        def take(src: str, dst: Optional[str] = None) -> None:
            if src in init_kwargs and init_kwargs[src] is not None:
                canon[dst or src] = init_kwargs[src]

        take("codec", "codec")
        take("preset", "preset")
        take("tuning_info", "tuning")
        take("bf", "bf")
        take("repeatspspps", "repeatSPSPPS")
        take("idrperiod", "idrPeriod")
        take("rc", "rcMode")
        take("bitrate", "bitrate")
        take("maxbitrate", "maxBitrate")
        take("frameIntervalP", "frameIntervalP")
        take("gopLength", "gopLength")
        take("lookahead", "enableLookahead")
        take("aq", "enableAQ")
        take("temporalaq", "enableTemporalAQ")
        take("nonrefp", "enableNonRefP")
        take("frameRateNum", "frameRateNum")
        take("frameRateDen", "frameRateDen")
        take("enableIntraRefresh", "enableIntraRefresh")
        take("maxNumRefFrames", "maxNumRefFrames")
        take("vbvBufferSize", "vbvBufferSize")
        take("vbvInitialDelay", "vbvInitialDelay")

        live: dict[str, Any] = {}
        if hasattr(encoder, "GetEncodeReconfigureParams"):
            try:
                params = encoder.GetEncodeReconfigureParams()
            except Exception:
                params = None
            if params is not None:
                for attr in (
                    "rateControlMode",
                    "averageBitrate",
                    "maxBitRate",
                    "vbvBufferSize",
                    "vbvInitialDelay",
                    "frameRateNum",
                    "frameRateDen",
                    "multiPass",
                ):
                    if hasattr(params, attr):
                        live[attr] = getattr(params, attr)

        for src, dst in (
            ("rateControlMode", "rcMode"),
            ("averageBitrate", "bitrate"),
            ("maxBitRate", "maxBitrate"),
            ("vbvBufferSize", "vbvBufferSize"),
            ("vbvInitialDelay", "vbvInitialDelay"),
            ("frameRateNum", "frameRateNum"),
            ("frameRateDen", "frameRateDen"),
        ):
            if dst not in canon and src in live:
                canon[dst] = live[src]

        order = [
            "codec",
            "preset",
            "tuning",
            "frameIntervalP",
            "bf",
            "maxNumRefFrames",
            "gopLength",
            "idrPeriod",
            "repeatSPSPPS",
            "rcMode",
            "bitrate",
            "maxBitrate",
            "vbvBufferSize",
            "vbvInitialDelay",
            "enableLookahead",
            "enableAQ",
            "enableTemporalAQ",
            "enableNonRefP",
            "frameRateNum",
            "frameRateDen",
            "enableIntraRefresh",
        ]
        parts = [f"{key}={canon[key]}" for key in order if key in canon]
        logger.info("Encoder settings (%s): %s", path, ", ".join(parts))


__all__ = ["Encoder", "EncoderTimings"]
