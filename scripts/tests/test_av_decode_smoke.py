import os
import pathlib
import pytest

try:
    import av  # type: ignore
except Exception:  # pragma: no cover
    av = None


@pytest.mark.skipif(av is None, reason="PyAV not installed")
def test_decode_sample_if_present():
    # Optional: decode a small sample bitstream if available
    sample = pathlib.Path('testdata/sample_idr.h264')
    if not sample.exists():
        pytest.skip("sample_idr.h264 not present; skipping")
    with av.open(str(sample), 'r') as container:
        vstreams = [s for s in container.streams if s.type == 'video']
        assert vstreams, "no video streams in sample"
        frames = 0
        for packet in container.demux(vstreams[0]):
            for _ in packet.decode():
                frames += 1
                if frames > 0:
                    break
            if frames > 0:
                break
        assert frames > 0, "failed to decode any frames from sample"

