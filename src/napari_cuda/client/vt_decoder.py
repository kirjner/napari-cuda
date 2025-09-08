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
import ctypes
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
        from Quartz import CoreVideo as _CV  # type: ignore  # noqa: F401
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
        logger.info("VT decoder init: width=%d height=%d avcc_len=%d", width, height, len(avcc))
        if not is_vt_available():
            raise RuntimeError(
                "VideoToolbox is not available. On macOS, install PyObjC frameworks:\n"
                "  pip install 'pyobjc-framework-VideoToolbox' 'pyobjc-framework-CoreMedia' 'pyobjc-framework-CoreVideo' 'pyobjc-framework-AVFoundation'"
            )
        # Lazy imports now that we know they’re present
        import objc  # type: ignore
        import CoreMedia  # type: ignore
        import VideoToolbox  # type: ignore
        from Quartz import CoreVideo as CV  # type: ignore
        from CoreFoundation import CFAllocatorGetDefault  # type: ignore

        self._objc = objc
        self._cm = CoreMedia
        self._vt = VideoToolbox
        self._cv = CV
        self._width = int(width)
        self._height = int(height)

        # Parse SPS/PPS from avcC and create CMFormatDescription
        try:
            sps_list, pps_list, nal_len_size = parse_avcc_sps_pps(avcc)
            logger.info("Parsed avcC: %d SPS, %d PPS, nal_len_size=%d", len(sps_list), len(pps_list), nal_len_size)
        except Exception as e:
            logger.error("Failed to parse avcC: %s", e)
            raise
        self._nal_length_size = nal_len_size
        # CMVideoFormatDescriptionCreateFromH264ParameterSets expects a tuple of buffers
        ps = tuple(sps_list + pps_list)
        num_sets = len(ps)
        parameter_set_pointers = tuple(ps)
        parameter_set_sizes = tuple(len(x) for x in ps)
        try:
            logger.debug(
                "VT avcC parse: sps=%d pps=%d nal_len_size=%d sizes=%s",
                len(sps_list), len(pps_list), nal_len_size, list(parameter_set_sizes),
            )
        except Exception:
            pass
        fmt_out = self._objc.nil
        try:
            status, fmt_desc = self._cm.CMVideoFormatDescriptionCreateFromH264ParameterSets(
                CFAllocatorGetDefault(),
                num_sets,
                parameter_set_pointers,
                parameter_set_sizes,
                nal_len_size,
                None,
            )
            logger.info("CMVideoFormatDescriptionCreate: status=%s", status)
            if status != 0:
                raise RuntimeError(f"CMVideoFormatDescriptionCreateFromH264ParameterSets failed: {status}")
            self._fmt_desc = fmt_desc
        except Exception as e:
            logger.error("Failed to create format description: %s", e)
            raise

        # Create VTDecompressionSession with BGRA output
        pix_attrs = {
            self._cv.kCVPixelBufferPixelFormatTypeKey: self._cv.kCVPixelFormatType_32BGRA,
        }
        # Prefer IOSurface-backed buffers and GL compatibility when available
        try:
            pix_attrs[self._cv.kCVPixelBufferIOSurfacePropertiesKey] = {}
        except Exception as e:
            logger.debug("IOSurfacePropertiesKey not set: %s", e)
        try:
            pix_attrs[self._cv.kCVPixelBufferOpenGLCompatibilityKey] = True
        except Exception as e:
            logger.debug("OpenGLCompatibilityKey not set: %s", e)

        # Output callback stores the most recent CVPixelBuffer
        self._last_image_buffer = None
        self._out_count = 0

        def _output_callback(_refcon, _source_frame_refcon, status_cb, info_flags, image_buffer, pts, duration):
            # image_buffer is a CVPixelBufferRef (PyObjC proxy)
            try:
                if status_cb != 0:
                    logger.error("VT output callback: decode error status=%s", status_cb)
                    return
                if image_buffer is None:
                    logger.warning("VT output callback: null image buffer")
                    return
                self._last_image_buffer = image_buffer
                self._out_count += 1
                logger.debug("VT output: status=%s flags=0x%x out_count=%d", status_cb, int(info_flags or 0), self._out_count)
            except Exception as e:
                logger.exception("VT output callback error: %s", e)

        # In PyObjC, pass (callback, refcon) tuple instead of a C record struct
        cb = (_output_callback, 0)
        try:
            status, session = self._vt.VTDecompressionSessionCreate(
                CFAllocatorGetDefault(),
                self._fmt_desc,
                None,
                pix_attrs,
                cb,
                None,
            )
            logger.info("VTDecompressionSessionCreate: status=%s", status)
            if status != 0:
                raise RuntimeError(f"VTDecompressionSessionCreate failed: {status}")
            self._session = session
            logger.info("VT decoder initialized successfully")
        except Exception as e:
            logger.error("Failed to create VT session: %s", e)
            raise

    def close(self) -> None:
        try:
            if self._session is not None:
                self._vt.VTDecompressionSessionInvalidate(self._session)
        except Exception as e:
            logger.warning("VT session invalidate failed: %s", e)

    def decode(self, avcc_au: bytes) -> Optional[object]:
        """Decode one AVCC access unit. Returns a CVPixelBufferRef or None."""
        logger.debug("VT decode: input len=%d, first bytes=%s", len(avcc_au), avcc_au[:8].hex() if avcc_au else "empty")
        # Build CMBlockBuffer from raw bytes and then CMSampleBuffer associated with our format description
        from CoreFoundation import CFAllocatorGetDefault  # type: ignore
        try:
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
                logger.error("CMBlockBufferCreateWithMemoryBlock failed: %s (len=%d)", status, len(avcc_au))
                return None
            logger.debug("CMBlockBuffer created successfully")
        except Exception as e:
            logger.exception("CMBlockBuffer creation exception: %s", e)
            return None
        # Provide a ctypes c_size_t array for sample sizes (PyObjC-friendly)
        sample_sizes = (ctypes.c_size_t * 1)(len(avcc_au))
        timing = None  # Let VT derive timing; server ts is used only for playout
        # Provide a single CMSampleTimingInfo entry; use kCMTimeInvalid for unknown timing
        try:
            timing = ((self._cm.kCMTimeInvalid, self._cm.kCMTimeInvalid, self._cm.kCMTimeInvalid),)
            sample_sizes = (len(avcc_au),)
            logger.debug("Creating CMSampleBuffer with timing=%s, sizes=%s", timing, sample_sizes)
            status, sbuf = self._cm.CMSampleBufferCreateReady(
                CFAllocatorGetDefault(), bb, self._fmt_desc, 1, 1, timing, 1, sample_sizes, None
            )
            if status != 0:
                logger.error("CMSampleBufferCreateReady failed: %s", status)
                return None
            logger.debug("CMSampleBuffer created successfully")
        except Exception as e:
            logger.exception("CMSampleBuffer creation exception: %s", e)
            return None
        # infoFlagsOut can be None; PyObjC returns status directly
        try:
            logger.debug("Calling VTDecompressionSessionDecodeFrame")
            status = self._vt.VTDecompressionSessionDecodeFrame(
                self._session, sbuf, 0, None, None
            )
            if status != 0:
                logger.error("VTDecompressionSessionDecodeFrame failed: %s", status)
                return None
            logger.debug("VTDecompressionSessionDecodeFrame succeeded")
        except Exception as e:
            logger.exception("VTDecompressionSessionDecodeFrame exception: %s", e)
            return None
        # Ensure asynchronous frames are delivered before returning
        try:
            logger.debug("Waiting for async frames")
            self._vt.VTDecompressionSessionWaitForAsynchronousFrames(self._session)
            logger.debug("Async frames wait completed")
        except Exception as e:
            logger.warning("VT wait for async frames error: %s", e)
        # Output callback sets _last_image_buffer
        result = self._last_image_buffer
        logger.debug("VT decode returning: %s (out_count=%d)", "image" if result else "None", self._out_count)
        return result
