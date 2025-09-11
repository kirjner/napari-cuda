from __future__ import annotations

"""
Tiny helpers for parsing server video_config and avcC details.
"""

from typing import Any, Dict, Optional, Tuple


def extract_video_config(data: Dict[str, Any]) -> Tuple[int, int, float, str, Optional[str]]:
    """Return (width, height, fps, stream_format, avcc_b64_or_None).

    - stream_format is 'avcc' or 'annexb'
    - Missing/invalid fields fall back to safe defaults
    """
    try:
        width = int(data.get('width') or 0)
    except Exception:
        width = 0
    try:
        height = int(data.get('height') or 0)
    except Exception:
        height = 0
    try:
        fps = float(data.get('fps') or 0.0)
    except Exception:
        fps = 0.0
    fmt = (str(data.get('format') or '')).lower() or 'avcc'
    stream_format = 'annexb' if fmt.startswith('annex') else 'avcc'
    avcc_b64 = data.get('data') if isinstance(data.get('data'), str) else None
    return width, height, fps, stream_format, avcc_b64


def nal_length_size_from_avcc(avcc: bytes) -> int:
    """Return NAL length size from avcC (1..4). Defaults to 4 on error."""
    try:
        if len(avcc) >= 5:
            n = int((avcc[4] & 0x03) + 1)
            if n in (1, 2, 3, 4):
                return n
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).debug("nal_length_size_from_avcc failed", exc_info=True)
    return 4
