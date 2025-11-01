from __future__ import annotations

"""
PyAV H.264 encoder wrapper that yields AVCC access units and exposes avcC.

Goals:
- Normalize encoder quirks (AnnexB vs AVCC, side-data avcC) into a stable API
- Build SPS/PPS cache and provide avcC via server bitstream helpers
- Return typed AccessUnit objects suitable for VT submission or PyAV decode

This module avoids importing PyAV at import time; it loads PyAV lazily.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from napari_cuda.codec.avcc import AccessUnit, parse_avcc
from napari_cuda.server.engine.encoding.bitstream import (
    ParamCache,
    build_avcc_config,
    pack_to_avcc,
)
from napari_cuda.utils.env import env_bool


@dataclass
class EncoderConfig:
    name: str = 'h264_videotoolbox'  # or 'libx264', 'h264'
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    pix_fmt: Optional[str] = None  # defaults: nv12 for VT, yuv420p otherwise
    options: Optional[dict[str, str]] = None


class H264Encoder:
    """Thin H.264 encoder abstraction producing AVCC AUs and avcC.

    Usage:
        enc = H264Encoder(EncoderConfig(...))
        enc.open()
        au_list = enc.encode_rgb_frame(np.ndarray[h,w,3], 'rgb24')  # returns List[AccessUnit]
        avcc = enc.get_avcc_config()  # bytes or None until SPS/PPS observed
        enc.close()
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        self._cfg = cfg
        self._enc = None  # type: ignore[var-annotated]
        self._cache = ParamCache()
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened

    def open(self) -> None:
        import av  # lazy import

        c = self._cfg
        enc = av.CodecContext.create(c.name, 'w')
        enc.width = int(c.width)
        enc.height = int(c.height)
        enc.pix_fmt = c.pix_fmt or ('nv12' if c.name == 'h264_videotoolbox' else 'yuv420p')
        enc.time_base = Fraction(1, int(round(c.fps)))
        opts: dict[str, str] = {}
        if c.options:
            opts.update({str(k): str(v) for k, v in c.options.items()})
        else:
            try:
                if c.name == 'h264_videotoolbox':
                    opts.update({'realtime': '1'})
                    try:
                        enc.max_b_frames = 0  # type: ignore[attr-defined]
                    except Exception:
                        pass
                elif c.name in ('libx264', 'h264'):
                    opts.update({
                        'tune': 'zerolatency',
                        'preset': 'veryfast',
                        'bf': '0',
                        'x264-params': 'keyint=1:min-keyint=1:scenecut=0:repeat-headers=1',
                        'annexb': '0',  # prefer AVCC; packer will normalize either way
                    })
            except Exception:
                pass
        if opts:
            enc.options = opts
        enc.open()
        self._enc = enc
        self._opened = True

        # Respect server packer default: allow Python fallback only if env permits
        allow_fallback = env_bool('NAPARI_CUDA_ALLOW_PY_FALLBACK', False)
        if not allow_fallback:
            # No action needed here; pack_to_avcc will raise if Cython is absent
            pass

    def _pkt_bytes(self, pkt) -> bytes:
        try:
            return pkt.to_bytes()
        except Exception:
            try:
                return bytes(pkt)
            except Exception:
                return memoryview(pkt).tobytes()

    def _ingest_side_extradata(self, pkt) -> None:
        """Capture avcC/vps/sps/pps from packet side-data when present."""
        if pkt is None:
            return
        try:
            sd_list = getattr(pkt, 'side_data', None)
            if not sd_list:
                return
            for sd in sd_list:
                t = getattr(sd, 'type', None)
                tname = str(t).lower() if t is not None else ''
                if ('new_extradata' in tname) or ('parameter_sets' in tname):
                    try:
                        b = sd.to_bytes()  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            b = bytes(sd)  # type: ignore[arg-type]
                        except Exception:
                            b = memoryview(sd).tobytes()  # type: ignore[arg-type]
                    if len(b) >= 5 and b[0] == 1:
                        try:
                            sps_list, pps_list, _nsz = parse_avcc(b)
                            if sps_list:
                                self._cache.sps = sps_list[0]
                            if pps_list:
                                self._cache.pps = pps_list[0]
                            return
                        except Exception:
                            # Not valid avcC; ignore
                            pass
        except Exception:
            return

    def encode_rgb_frame(self, rgb: object, pixfmt: str = 'rgb24', pts: Optional[float] = None) -> list[AccessUnit]:
        """Encode a single RGB ndarray and return a list of AVCC AUs for the frame.

        The returned list usually contains 0-1 AUs; encoders can emit multiple packets
        per input frame, which are packed into at most one AU via the server packer.
        """
        if not self._opened or self._enc is None:
            raise RuntimeError('Encoder is not open')
        import av

        vf = av.VideoFrame.from_ndarray(rgb, format=pixfmt)
        try:
            vf = vf.reformat(self._cfg.width, self._cfg.height, format=str(self._enc.pix_fmt))
        except Exception:
            pass
        out = self._enc.encode(vf)
        payloads: list[bytes] = []
        for p in out:
            # Ingest side-data extradata early to seed SPS/PPS if needed
            if self._cache.sps is None or self._cache.pps is None:
                self._ingest_side_extradata(p)
            payloads.append(self._pkt_bytes(p))
        au_bytes, is_key = pack_to_avcc(payloads, self._cache)
        result: list[AccessUnit] = []
        if au_bytes is not None:
            # Attach provided PTS if any; caller computes based on frame index/fps
            result.append(AccessUnit(payload=au_bytes, is_keyframe=bool(is_key), pts=pts))
        return result

    def flush(self) -> list[AccessUnit]:
        if not self._opened or self._enc is None:
            return []
        out = []
        try:
            pkts = self._enc.encode(None)
        except Exception:
            pkts = []
        payloads: list[bytes] = []
        for p in pkts:
            if self._cache.sps is None or self._cache.pps is None:
                self._ingest_side_extradata(p)
            payloads.append(self._pkt_bytes(p))
        au_bytes, is_key = pack_to_avcc(payloads, self._cache)
        if au_bytes is not None:
            out.append(AccessUnit(payload=au_bytes, is_keyframe=bool(is_key), pts=None))
        return out

    def get_avcc_config(self) -> Optional[bytes]:
        return build_avcc_config(self._cache)

    def close(self) -> None:
        self._enc = None
        self._opened = False
