from __future__ import annotations

"""
VideoToolbox H.264 decoder (macOS) – AVCC input → BGRA output.

This module provides a thin wrapper around VideoToolbox using PyObjC.
It expects AVCC bitstream input (length‑prefixed NAL units) and uses
the avcC config (SPS/PPS) sent by the server to initialize the session.

If PyObjC and the macOS frameworks are not available, the decoder will
not be usable; callers should detect availability and fall back.
"""

from typing import List, Tuple, Optional
import logging
import sys

logger = logging.getLogger(__name__)


def parse_avcc_sps_pps(avcc: bytes) -> Tuple[List[bytes], List[bytes], int]:
    """Parse avcC (AVCDecoderConfigurationRecord) and return (sps_list, pps_list, nal_length_size).

    avcC format (ISO/IEC 14496-15):
      - configurationVersion: 1 byte
      - AVCProfileIndication: 1 byte
      - profile_compatibility: 1 byte
      - AVCLevelIndication: 1 byte
      - lengthSizeMinusOne: low 2 bits of next byte; nal_length_size = value + 1
      - numOfSequenceParameterSets: low 5 bits of next byte
      - for each SPS: 2-byte length + SPS bytes
      - numOfPictureParameterSets: 1 byte
      - for each PPS: 2-byte length + PPS bytes
    """
    if not avcc or len(avcc) < 7:
        raise ValueError("Invalid avcC: too short")
    i = 0
    configuration_version = avcc[i]; i += 1
    if configuration_version != 1:
        logger.debug("Unexpected avcC configurationVersion=%d", configuration_version)
    _profile = avcc[i]; i += 1
    _compat = avcc[i]; i += 1
    _level = avcc[i]; i += 1
    length_size_minus_one = avcc[i] & 0x03; i += 1
    nal_length_size = int(length_size_minus_one) + 1
    num_sps = avcc[i] & 0x1F; i += 1
    sps_list: List[bytes] = []
    for _ in range(num_sps):
        if i + 2 > len(avcc):
            raise ValueError("Invalid avcC: truncated SPS length")
        ln = int.from_bytes(avcc[i:i+2], 'big'); i += 2
        if i + ln > len(avcc):
            raise ValueError("Invalid avcC: truncated SPS data")
        sps_list.append(avcc[i:i+ln]); i += ln
    if i >= len(avcc):
        raise ValueError("Invalid avcC: missing PPS count")
    num_pps = avcc[i]; i += 1
    pps_list: List[bytes] = []
    for _ in range(num_pps):
        if i + 2 > len(avcc):
            raise ValueError("Invalid avcC: truncated PPS length")
        ln = int.from_bytes(avcc[i:i+2], 'big'); i += 2
        if i + ln > len(avcc):
            raise ValueError("Invalid avcC: truncated PPS data")
        pps_list.append(avcc[i:i+ln]); i += ln
    return sps_list, pps_list, nal_length_size


def is_vt_available() -> bool:
    if sys.platform != 'darwin':
        return False
    try:
        import objc  # noqa: F401
        import VideoToolbox  # type: ignore  # noqa: F401
        import CoreMedia  # type: ignore  # noqa: F401
        import CoreVideo  # type: ignore  # noqa: F401
        return True
    except Exception as e:
        logger.debug("VideoToolbox not available: %s", e)
        return False


class VideoToolboxDecoder:
    """Minimal VT wrapper. Initializes from avcC and decodes AVCC AUs to BGRA CVPixelBuffers.

    For the first milestone, we expose a decode(avcc_au: bytes) -> Optional[object]
    that returns a CVPixelBuffer object (PyObjC proxy) or None. The caller can then
    map it to QImage as needed.
    """

    def __init__(self, avcc: bytes, width: int, height: int) -> None:
        if not is_vt_available():
            raise RuntimeError(
                "VideoToolbox is not available. On macOS, install PyObjC frameworks:\n"
                "  pip install 'pyobjc-framework-VideoToolbox' 'pyobjc-framework-CoreMedia' 'pyobjc-framework-CoreVideo' 'pyobjc-framework-AVFoundation'"
            )
        # Lazy imports now that we know they’re present
        import objc  # type: ignore
        import CoreMedia  # type: ignore
        import VideoToolbox  # type: ignore
        import CoreVideo  # type: ignore
        from CoreFoundation import CFAllocatorGetDefault  # type: ignore

        self._objc = objc
        self._cm = CoreMedia
        self._vt = VideoToolbox
        self._cv = CoreVideo
        self._width = int(width)
        self._height = int(height)

        # Parse SPS/PPS from avcC and create CMFormatDescription
        sps_list, pps_list, nal_len_size = parse_avcc_sps_pps(avcc)
        self._nal_length_size = nal_len_size
        # CMVideoFormatDescriptionCreateFromH264ParameterSets expects a list of parameter sets
        ps = sps_list + pps_list
        num_sets = len(ps)
        # Build a tuple of bytes objects acceptable to PyObjC; lengths as a list
        parameter_set_pointers = ps
        parameter_set_sizes = [len(x) for x in ps]
        fmt_out = self._objc.nil
        status, fmt_desc = self._cm.CMVideoFormatDescriptionCreateFromH264ParameterSets(
            CFAllocatorGetDefault(),
            num_sets,
            parameter_set_pointers,
            parameter_set_sizes,
            nal_len_size,
            None,
        )
        if status != 0:
            raise RuntimeError(f"CMVideoFormatDescriptionCreateFromH264ParameterSets failed: {status}")
        self._fmt_desc = fmt_desc

        # Create VTDecompressionSession with BGRA output
        pix_attrs = {self._cv.kCVPixelBufferPixelFormatTypeKey: self._cv.kCVPixelFormatType_32BGRA}

        # Output callback stores the most recent CVPixelBuffer
        self._last_image_buffer = None

        def _output_callback(_refcon, _source_frame_refcon, _status, _info_flags, image_buffer, _pts, _duration):
            # image_buffer is a CVPixelBufferRef (PyObjC proxy)
            self._last_image_buffer = image_buffer

        cb = self._vt.VTDecompressionOutputCallbackRecord(
            decompressionOutputCallback=_output_callback, decompressionOutputRefCon=0
        )
        status, session = self._vt.VTDecompressionSessionCreate(
            CFAllocatorGetDefault(),
            self._fmt_desc,
            None,
            pix_attrs,
            cb,
        )
        if status != 0:
            raise RuntimeError(f"VTDecompressionSessionCreate failed: {status}")
        self._session = session

    def close(self) -> None:
        try:
            if self._session is not None:
                self._vt.VTDecompressionSessionInvalidate(self._session)
        except Exception:
            pass

    def decode(self, avcc_au: bytes) -> Optional[object]:
        """Decode one AVCC access unit. Returns a CVPixelBufferRef or None."""
        # Build CMBlockBuffer from raw bytes and then CMSampleBuffer associated with our format description
        from CoreFoundation import CFAllocatorGetDefault  # type: ignore
        status, bb = self._cm.CMBlockBufferCreateWithMemoryBlock(
            CFAllocatorGetDefault(),
            avcc_au,
            len(avcc_au),
            CFAllocatorGetDefault(),
            None,
            0,
            len(avcc_au),
            0,
            None,
        )
        if status != 0:
            logger.debug("CMBlockBufferCreateWithMemoryBlock failed: %s", status)
            return None
        sample_sizes = [len(avcc_au)]
        timing = None  # Let VT derive timing; server ts is used only for playout
        status, sbuf = self._cm.CMSampleBufferCreateReady(
            CFAllocatorGetDefault(), bb, self._fmt_desc, 1, 0, None, 1, sample_sizes, None
        )
        if status != 0:
            logger.debug("CMSampleBufferCreateReady failed: %s", status)
            return None
        flags_out = self._objc.nil
        status = self._vt.VTDecompressionSessionDecodeFrame(
            self._session, sbuf, 0, None, flags_out
        )
        if status != 0:
            logger.debug("VTDecompressionSessionDecodeFrame failed: %s", status)
            return None
        # Output callback sets _last_image_buffer
        return self._last_image_buffer

