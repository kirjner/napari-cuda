import glob
import os
import pathlib

import pytest

try:
    import av  # type: ignore
except Exception:  # pragma: no cover
    av = None


@pytest.mark.skipif(av is None, reason="PyAV not installed")
def test_decode_dump_if_present():
    # Check env override first
    dump_path = os.getenv('DUMP_PATH')
    if dump_path:
        candidate = pathlib.Path(dump_path)
        if not candidate.exists():
            pytest.skip("DUMP_PATH set but file does not exist")
    else:
        files = sorted(glob.glob('benchmarks/bitstreams/*.h264'))
        if not files:
            pytest.skip(
                "no dumped bitstreams found; set NAPARI_CUDA_DUMP_BITSTREAM=<count> "
                "to generate a dump"
            )
        candidate = pathlib.Path(files[-1])
    # Try to decode a few packets
    with av.open(str(candidate), 'r') as container:
        vstreams = [s for s in container.streams if s.type == 'video']
        assert vstreams, "no video streams in dump"
        frames = 0
        for packet in container.demux(vstreams[0]):
            for _ in packet.decode():
                frames += 1
                if frames >= 1:
                    break
            if frames >= 1:
                break
        assert frames >= 1, f"failed to decode any frames from {candidate}"
