from __future__ import annotations

"""
VideoToolbox H.264 decoder (macOS) – AVCC input → BGRA output.

This module provides a thin wrapper around VideoToolbox using PyObjC.
It expects AVCC bitstream input (length‑prefixed NAL units) and uses
the avcC config (SPS/PPS) sent by the server to initialize the session.

If PyObjC and the macOS frameworks are not available, the decoder will
not be usable; callers should detect availability and fall back.
"""

from typing import List, Tuple, Optional, Union
import ctypes
import logging
import sys
import queue
import threading

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
        from CoreFoundation import CFAllocatorGetDefault, CFRetain  # type: ignore

        self._objc = objc
        self._cm = CoreMedia
        self._vt = VideoToolbox
        self._cv = CV
        self._width = int(width)
        self._height = int(height)
        self._need_reset = True

        # Parse SPS/PPS from avcC and create CMFormatDescription
        try:
            sps_list, pps_list, nal_len_size = parse_avcc_sps_pps(avcc)
            logger.info("Parsed avcC: %d SPS, %d PPS, nal_len_size=%d", len(sps_list), len(pps_list), nal_len_size)
        except Exception as e:
            logger.error("Failed to parse avcC: %s", e)
            raise
        self._nal_length_size = nal_len_size
        # First try: create format description using avcC atom extension (preferred by some VT paths)
        fmt_created = False
        try:
            atoms = { 'avcC': avcc }
            exts = { self._cm.kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms: atoms }
            codec = getattr(self._cm, 'kCMVideoCodecType_H264', None)
            if codec is None:
                # Fallback fourCC for avc1
                codec = 0x61766331
            # PyObjC expects the out-parameter; pass None to receive (status, desc)
            status, fmt_desc = self._cm.CMVideoFormatDescriptionCreate(
                CFAllocatorGetDefault(), codec, self._width, self._height, exts, None
            )
            logger.info("CMVideoFormatDescriptionCreate(avcC): status=%s", status)
            if status == 0:
                self._fmt_desc = fmt_desc
                fmt_created = True
        except Exception as e:
            logger.debug("CMVideoFormatDescriptionCreate(avcC) failed: %s", e)
        if not fmt_created:
            # Fallback: create from parameter sets
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
            try:
                status, fmt_desc = self._cm.CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    CFAllocatorGetDefault(),
                    num_sets,
                    parameter_set_pointers,
                    parameter_set_sizes,
                    nal_len_size,
                    None,
                )
                logger.info("CMVideoFormatDescriptionCreateFromH264ParameterSets: status=%s", status)
                if status != 0:
                    raise RuntimeError(f"CMVideoFormatDescriptionCreateFromH264ParameterSets failed: {status}")
                self._fmt_desc = fmt_desc
            except Exception as e:
                logger.error("Failed to create format description: %s", e)
                raise

        # Create VTDecompressionSession; prefer NV12 (bi-planar 4:2:0) for zero-copy GPU friendly output
        import os as _os
        force_pf = (_os.getenv('NAPARI_CUDA_CLIENT_VT_PIXFMT', '') or '').upper()
        if force_pf == 'BGRA':
            preferred_pix = self._cv.kCVPixelFormatType_32BGRA
            logger.info("VT pixel format override: BGRA")
        elif force_pf in ('NV12', '420BIPLANAR'):
            preferred_pix = getattr(self._cv, 'kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange', None) or getattr(self._cv, 'kCVPixelFormatType_420YpCbCr8BiPlanarFullRange', None) or self._cv.kCVPixelFormatType_32BGRA
            logger.info("VT pixel format override: NV12")
        else:
            preferred_pix = getattr(self._cv, 'kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange', None)
            if preferred_pix is None:
                preferred_pix = self._cv.kCVPixelFormatType_32BGRA
        pix_attrs = {
            self._cv.kCVPixelBufferPixelFormatTypeKey: preferred_pix,
        }
        try:
            pf_name = 'NV12' if preferred_pix in (
                getattr(self._cv, 'kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange', -1),
                getattr(self._cv, 'kCVPixelFormatType_420YpCbCr8BiPlanarFullRange', -2),
            ) else 'BGRA'
            logger.info("VT requested pixel format: %s (0x%08x)", pf_name, int(preferred_pix))
        except Exception:
            pass
        # Prefer IOSurface-backed buffers and GL compatibility when available
        try:
            pix_attrs[self._cv.kCVPixelBufferIOSurfacePropertiesKey] = {}
        except Exception as e:
            logger.debug("IOSurfacePropertiesKey not set: %s", e)
        try:
            pix_attrs[self._cv.kCVPixelBufferOpenGLCompatibilityKey] = True
        except Exception as e:
            logger.debug("OpenGLCompatibilityKey not set: %s", e)
        # Hint expected dimensions (some paths benefit)
        try:
            pix_attrs[self._cv.kCVPixelBufferWidthKey] = self._width
            pix_attrs[self._cv.kCVPixelBufferHeightKey] = self._height
        except Exception:
            pass

        # Output callback stores the most recent CVPixelBuffer
        self._last_image_buffer = None
        self._out_count = 0
        self._submit_count = 0
        self._first_output_logged = False
        # Decoded output queue: items are tuples (CVPixelBufferRef, pts_seconds|None)
        self._out_q: "queue.Queue[Tuple[object, Optional[float]]]" = queue.Queue(maxsize=8)
        # Inflight frame context mapping: numeric handle -> context (e.g., pts)
        self._inflight: dict[int, dict] = {}
        self._inflight_lock = threading.Lock()
        self._next_frame_id = 1

        # Wrap the Python callback with a C function pointer using PyObjC's callbackFor
        def _output_callback(_refcon, _source_frame_refcon, status_cb, info_flags, image_buffer, pts, duration):
            # image_buffer is a CVPixelBufferRef (PyObjC proxy)
            try:
                if status_cb != 0:
                    logger.error("VT output callback: decode error status=%s", status_cb)
                    # Best-effort cleanup of inflight entry if we got a refcon
                    try:
                        fid = int(_source_frame_refcon) if _source_frame_refcon is not None else None
                        if fid is not None:
                            with self._inflight_lock:
                                self._inflight.pop(fid, None)
                    except Exception:
                        pass
                    return
                if image_buffer is None:
                    logger.warning("VT output callback: null image buffer")
                    # Best-effort cleanup
                    try:
                        fid = int(_source_frame_refcon) if _source_frame_refcon is not None else None
                        if fid is not None:
                            with self._inflight_lock:
                                self._inflight.pop(fid, None)
                    except Exception:
                        pass
                    return
                # Retain the pixel buffer for use outside the callback per VT contract
                try:
                    CFRetain(image_buffer)
                except Exception:
                    pass
                self._last_image_buffer = image_buffer
                self._out_count += 1
                # Derive PTS: prefer source_frame_refcon (server ts) if provided; else CMTime
                pts_seconds: Optional[float] = None
                # Look up by numeric handle if provided
                try:
                    fid = int(_source_frame_refcon) if _source_frame_refcon is not None else None
                except Exception:
                    fid = None
                ctx = None
                if fid is not None:
                    with self._inflight_lock:
                        ctx = self._inflight.pop(fid, None)
                if ctx is not None:
                    try:
                        v = ctx.get('pts', None)
                        if isinstance(v, (int, float)):
                            pts_seconds = float(v)
                    except Exception:
                        pass
                if pts_seconds is None:
                    try:
                        pts_seconds = float(self._cm.CMTimeGetSeconds(pts)) if pts is not None else None
                    except Exception:
                        pts_seconds = None
                # Enqueue; drop oldest on overflow (latest wins)
                try:
                    self._out_q.put_nowait((image_buffer, pts_seconds))
                except queue.Full:
                    try:
                        _ = self._out_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._out_q.put_nowait((image_buffer, pts_seconds))
                    except Exception:
                        pass
                # Log first few outputs at INFO with pixel format and dimensions
                try:
                    if not self._first_output_logged or self._out_count <= 3:
                        pf = None
                        w = h = bpr = None
                        try:
                            pf = self._cv.CVPixelBufferGetPixelFormatType(image_buffer)
                        except Exception:
                            pass
                        try:
                            w = self._cv.CVPixelBufferGetWidth(image_buffer)
                            h = self._cv.CVPixelBufferGetHeight(image_buffer)
                        except Exception:
                            pass
                        try:
                            bpr = self._cv.CVPixelBufferGetBytesPerRow(image_buffer)
                        except Exception:
                            pass
                        logger.info(
                            "VT output #%d: pts=%s pixfmt=0x%08x size=%sx%s bpr=%s cb_info=0x%x",
                            self._out_count,
                            f"{pts_seconds:.3f}" if isinstance(pts_seconds, (int, float)) else "None",
                            int(pf) if pf is not None else -1,
                            w,
                            h,
                            bpr,
                            int(info_flags or 0),
                        )
                        self._first_output_logged = True
                    else:
                        logger.debug("VT output: out_count=%d pts=%s", self._out_count, pts_seconds)
                except Exception:
                    logger.debug("VT output logging failed")
            except Exception as e:
                logger.exception("VT output callback error: %s", e)

        # Keep a strong reference to the callback to avoid GC; ensure proper C callback trampoline
        try:
            self._output_cb = self._objc.callbackFor(self._vt.VTDecompressionSessionCreate)(_output_callback)  # type: ignore
            logger.debug("VT output callback wrapped with PyObjC callbackFor")
        except Exception:
            # Fallback to raw function (may still work on some PyObjC versions)
            self._output_cb = _output_callback
            logger.debug("VT output callback using raw Python callable (no callbackFor wrapper)")
        # In PyObjC, pass a (callback, refcon) tuple
        cb = (self._output_cb, 0)
        try:
            # Signature with out parameter: VTDecompressionSessionCreate(allocator, formatDesc, decoderSpec,
            #                                                            imageBufferAttributes, outputCallback, decompressionSessionOut)
            # Passing None for the out parameter lets PyObjC return (status, session)
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
            # Mark session as real-time; helps with timely output and avoids deep reordering
            try:
                self._vt.VTSessionSetProperty(self._session, self._vt.kVTDecompressionPropertyKey_RealTime, True)
            except Exception as e:
                logger.debug("VT set RealTime property failed: %s", e)
            # (Optional) Hardware acceleration status query is version-sensitive; omit for stability
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

    def decode(self, avcc_au: bytes, src_pts: Optional[Union[float, int, Tuple[Union[float, int], ...]]] = None) -> bool:
        """Queue one AVCC access unit for decode. Returns True on submit success.

        Delivery is asynchronous via the output callback; poll get_frame_nowait()/get_frame()
        to retrieve decoded CVPixelBuffers.
        """
        logger.debug("VT decode: input len=%d, first bytes=%s", len(avcc_au), avcc_au[:8].hex() if avcc_au else "empty")
        
        # Validate AVCC structure
        if len(avcc_au) < 5:
            logger.error("AVCC AU too short: %d bytes", len(avcc_au))
            return False
            
        # Quick AVCC sanity scan: iterate length-prefixed NAL units
        off = 0
        total = len(avcc_au)
        first_type = None
        vcl_present = False
        idr_present = False
        # Count NAL types for diagnostics
        nal_counts = {1: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        while off + 4 <= total:
            ln = int.from_bytes(avcc_au[off:off+4], 'big'); off += 4
            if ln <= 0 or off + ln > total:
                logger.warning("AVCC AU parse error: ln=%d off=%d total=%d", ln, off, total)
                break
            ntype = avcc_au[off] & 0x1F
            if first_type is None:
                first_type = ntype
            if ntype in (1, 5):
                vcl_present = True
            if ntype == 5:
                idr_present = True
            if ntype in nal_counts:
                nal_counts[ntype] += 1
            off += ln
        if not vcl_present:
            logger.debug("AVCC AU has no VCL slice; skipping decode (first_type=%s)", first_type)
            return None
        # Log first few AUs' composition at INFO
        if self._submit_count < 3:
            try:
                logger.info(
                    "VT AU composition: AUD=%d SPS=%d PPS=%d SEI=%d IDR=%d NONIDR=%d",
                    nal_counts.get(9, 0), nal_counts.get(7, 0), nal_counts.get(8, 0), nal_counts.get(6, 0), nal_counts.get(5, 0), nal_counts.get(1, 0)
                )
            except Exception:
                pass
        
        # Build CMBlockBuffer from raw bytes and then CMSampleBuffer associated with our format description
        from CoreFoundation import CFAllocatorGetDefault  # type: ignore
        try:
            # Robust path: allocate CM-managed memory and then copy AU bytes in.
            # Avoids passing Python memory to CoreMedia entirely.
            status, bb = self._cm.CMBlockBufferCreateWithMemoryBlock(
                CFAllocatorGetDefault(),
                None,  # allocate internally
                len(avcc_au),
                CFAllocatorGetDefault(),
                None,
                0,
                len(avcc_au),
                0,  # no flags; we'll copy explicitly below
                None,
            )
            if status != 0:
                logger.error("CMBlockBufferCreateWithMemoryBlock failed: %s (len=%d)", status, len(avcc_au))
                return False
            # Copy data into the CM-owned block
            try:
                self._cm.CMBlockBufferReplaceDataBytes(avcc_au, bb, 0, len(avcc_au))
            except Exception as e:
                logger.exception("CMBlockBufferReplaceDataBytes failed: %s", e)
                return False
            logger.debug("CMBlockBuffer created and filled successfully")
        except Exception as e:
            logger.exception("CMBlockBuffer creation exception: %s", e)
            return False
        # Provide a ctypes c_size_t array for sample sizes (PyObjC-friendly)
        sample_sizes_c = (ctypes.c_size_t * 1)(len(avcc_au))
        timing = None  # Let VT derive timing; server ts is used only for playout
        # Provide a single CMSampleTimingInfo entry; if src_pts provided, set valid PTS
        try:
            if src_pts is not None:
                try:
                    pts_s = float(src_pts) if not isinstance(src_pts, (tuple, list)) else float(src_pts[0])
                except Exception:
                    pts_s = None
            else:
                pts_s = None
            if pts_s is not None:
                # Use 1e6 timescale to preserve microsecond precision
                pts_time = self._cm.CMTimeMakeWithSeconds(pts_s, 1_000_000)
                timing = ((pts_time, self._cm.kCMTimeInvalid, self._cm.kCMTimeInvalid),)
            else:
                timing = ((self._cm.kCMTimeInvalid, self._cm.kCMTimeInvalid, self._cm.kCMTimeInvalid),)
            logger.debug("Creating CMSampleBuffer with timing=%s, sizes=%s", timing, (len(avcc_au),))
            status, sbuf = self._cm.CMSampleBufferCreateReady(
                CFAllocatorGetDefault(), bb, self._fmt_desc, 1, 1, timing, 1, sample_sizes_c, None
            )
            if status != 0:
                logger.error("CMSampleBufferCreateReady failed: %s", status)
                return None
            logger.debug("CMSampleBuffer created successfully")
            # Mark sample attachments to help VT with sync frames
            atts = None
            try:
                atts = self._cm.CMSampleBufferGetSampleAttachmentsArray(sbuf, True)
                if atts and len(atts) > 0:
                    d = atts[0]
                    # Set core attachments first; ignore missing keys gracefully
                    try:
                        d[self._cm.kCMSampleAttachmentKey_NotSync] = bool(not idr_present)
                    except Exception:
                        pass
                    try:
                        d[self._cm.kCMSampleAttachmentKey_DependsOnOthers] = bool(not idr_present)
                    except Exception:
                        pass
                    try:
                        d[self._cm.kCMSampleAttachmentKey_DisplayImmediately] = True
                    except Exception:
                        pass
                    # Attempt reset-on-first only if key exists on this platform
                    if self._need_reset:
                        try:
                            key = getattr(self._cm, 'kCMSampleAttachmentKey_ResetDecoderBeforeDecoding', None)
                            if key is not None:
                                d[key] = True
                                self._need_reset = False
                        except Exception:
                            # If unsupported, continue without resetting
                            self._need_reset = False
                    try:
                        logger.debug(
                            "Set sample attachments (idr=%s): NotSync=%s DependsOnOthers=%s DisplayImmediately=%s",
                            idr_present,
                            d.get(getattr(self._cm, 'kCMSampleAttachmentKey_NotSync', None), None),
                            d.get(getattr(self._cm, 'kCMSampleAttachmentKey_DependsOnOthers', None), None),
                            d.get(getattr(self._cm, 'kCMSampleAttachmentKey_DisplayImmediately', None), None),
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Could not set sample attachments: %s", e)
        except Exception as e:
            logger.exception("CMSampleBuffer creation exception: %s", e)
            return False
        # infoFlagsOut can be None; PyObjC returns status directly
        try:
            logger.debug("Calling VTDecompressionSessionDecodeFrame")
            # Enable asynchronous decompression and then wait for frames
            # kVTDecodeFrame_EnableAsynchronousDecompression = 1<<0
            # kVTDecodeFrame_1xRealTimePlayback = 1<<2 (hint for timely output)
            decode_flags = (1 << 0) | (1 << 2)
            # Allocate a numeric handle and store context (e.g., PTS) without passing Python objects to C
            frame_id: Optional[int] = None
            try:
                with self._inflight_lock:
                    frame_id = self._next_frame_id
                    self._next_frame_id += 1
                    self._inflight[frame_id] = {'pts': pts_s}
            except Exception:
                frame_id = None
            ret = self._vt.VTDecompressionSessionDecodeFrame(
                self._session, sbuf, decode_flags, int(frame_id) if frame_id is not None else 0, None
            )
            info_flags = 0
            if isinstance(ret, tuple):
                status, info_flags = ret
            else:
                status = ret
            if status != 0:
                logger.error("VTDecompressionSessionDecodeFrame failed: status=%s info_flags=%s", status, info_flags)
                # Try to get more info about the error
                if status == -12909:
                    logger.error("kVTVideoDecoderBadDataErr - corrupt data or invalid parameter")
                elif status == -12911:
                    logger.error("kVTVideoDecoderUnsupportedDataFormatErr - unsupported data format")
                elif status == -8969:
                    logger.error("codecBadDataErr - data could not be decompressed")
                # On failure, drop inflight context if we created one
                if frame_id is not None:
                    try:
                        with self._inflight_lock:
                            self._inflight.pop(frame_id, None)
                    except Exception:
                        pass
                return False
            # Decode info flags diagnostic
            try:
                flags = int(info_flags or 0)
                names = []
                if flags & 0x1:
                    names.append('Asynchronous')
                if flags & 0x2:
                    names.append('FrameDropped')
                logger.debug("VTDecompressionSessionDecodeFrame ok (info_flags=0x%x %s)", flags, ','.join(names) if names else '')
            except Exception:
                logger.debug("VTDecompressionSessionDecodeFrame succeeded (info_flags=0x%x)", int(info_flags or 0))
        except Exception as e:
            logger.exception("VTDecompressionSessionDecodeFrame exception: %s", e)
            return False
        # Do not wait synchronously; frames arrive via callback
        self._submit_count += 1
        if self._submit_count <= 3:
            logger.info(
                "VT submit #%d: len=%d idr=%s",
                self._submit_count,
                len(avcc_au),
                bool(idr_present),
            )
        return True

    def get_frame_nowait(self) -> Optional[Tuple[object, Optional[float]]]:
        try:
            return self._out_q.get_nowait()
        except Exception:
            return None

    def get_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[object, Optional[float]]]:
        try:
            return self._out_q.get(timeout=timeout if timeout is not None else 0.0)
        except Exception:
            return None

    # Diagnostics
    def get_counts(self) -> Tuple[int, int]:
        """Return (submit_count, output_count) for diagnostics."""
        return self._submit_count, self._out_count

    def flush(self) -> None:
        """Wait for asynchronous frames to be delivered (diagnostic)."""
        try:
            self._vt.VTDecompressionSessionWaitForAsynchronousFrames(self._session)
        except Exception:
            pass
