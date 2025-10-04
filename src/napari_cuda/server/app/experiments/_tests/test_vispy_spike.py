import json
from types import SimpleNamespace

import numpy as np

from napari_cuda.server.runtime.egl_worker import FrameTimings
from napari_cuda.server.app.experiments import baseline_capture as base
from napari_cuda.server.app.experiments import vispy_spike as spike


def _dummy_timings() -> FrameTimings:
    return FrameTimings(
        render_ms=1.0,
        blit_gpu_ns=None,
        blit_cpu_ms=0.1,
        map_ms=0.05,
        copy_ms=0.07,
        convert_ms=0.2,
        encode_ms=0.0,
        pack_ms=0.0,
        total_ms=1.42,
        packet_bytes=None,
        capture_wall_ts=0.0,
    )


def test_parse_args_defaults() -> None:
    cfg = spike._parse_args([])
    assert cfg.layer == "image"
    assert cfg.frames == 60
    assert cfg.width == 1280
    assert cfg.use_nvenc is False


def test_parse_args_overrides(tmp_path) -> None:
    out = tmp_path / "frames.json"
    cfg = spike._parse_args(
        [
            "--layer",
            "volume",
            "--frames",
            "12",
            "--width",
            "300",
            "--height",
            "200",
            "--output",
            str(out),
            "--nvenc",
        ]
    )
    assert cfg.layer == "volume"
    assert cfg.frames == 12
    assert cfg.width == 300
    assert cfg.output == out
    assert cfg.use_nvenc is True


class _DummyWorker:
    def __init__(self, width: int, height: int, fps: int, use_volume: bool, disable_nvenc: bool = True) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.use_volume = use_volume
        self.view = SimpleNamespace(scene=SimpleNamespace(), camera=None)
        self.canvas = SimpleNamespace(render=lambda: None)
        self._captures = 0
        self.cleaned = False
        self.disable_nvenc = disable_nvenc

    def capture_and_encode_packet(self):
        self._captures += 1
        return _dummy_timings(), None, 0, self._captures

    def latest_rgba(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def cleanup(self) -> None:
        self.cleaned = True


def test_run_spike_with_dummies(monkeypatch) -> None:
    zero_hash = spike._hash_rgba(np.zeros((2, 2, 4), dtype=np.uint8))

    def fake_build(kind: str, seed: int = 13):
        layer = SimpleNamespace(data=np.zeros((2, 2), dtype=np.float32))
        adapter = SimpleNamespace(node=SimpleNamespace(parent=None))
        return SimpleNamespace(), layer, adapter

    monkeypatch.setattr(spike, "_build_viewer_and_layer", fake_build)
    monkeypatch.setattr(spike, "_attach_to_worker", lambda worker, adapter, kind: None)
    monkeypatch.setattr(spike, "SpikeRendererWorker", _DummyWorker)

    cfg = spike.SpikeConfig(width=2, height=2, frames=3, fps=10)
    results = spike.run_spike(cfg)

    assert len(results) == 3
    assert all(r.hash_hex == zero_hash for r in results)
    assert all(r.packet_bytes is None for r in results)


def test_write_results(tmp_path) -> None:
    results = [
        spike.FrameResult(index=0, timings=_dummy_timings(), hash_hex="aa"),
        spike.FrameResult(index=1, timings=_dummy_timings(), hash_hex="bb"),
    ]
    cfg = spike.SpikeConfig(output=tmp_path / "frames.json", profile=tmp_path / "profile.json")

    spike._write_results(results, cfg)

    metrics = json.loads((tmp_path / "frames.json").read_text())
    profile = json.loads((tmp_path / "profile.json").read_text())

    assert metrics[0]["index"] == 0
    assert profile["frames"] == 2
    assert profile["packet_bytes"]["min"] is None


def test_baseline_parse_args_defaults() -> None:
    cfg = base._parse_args([])
    assert cfg.frames == 60
    assert cfg.width == 1280
    assert cfg.use_volume is False


def test_baseline_run_with_dummies(monkeypatch) -> None:
    hashes: list[str] = []

    class DummyBaselineWorker:
        def __init__(self, width, height, fps, use_volume):
            self.width = width
            self.height = height
            self.fps = fps
            self.use_volume = use_volume
            self._calls = 0

        def capture_and_encode_packet(self):
            self._calls += 1
            return _dummy_timings(), b"pkt", 0, self._calls

        def latest_rgba(self):
            return np.zeros((2, 2, 4), dtype=np.uint8)

        def cleanup(self):
            hashes.append("cleaned")

    monkeypatch.setattr(base, "BaselineRendererWorker", DummyBaselineWorker)

    cfg = base.CaptureConfig(width=2, height=2, frames=2, fps=5)
    results = base.run_capture(cfg)

    assert len(results) == 2
    assert hashes[-1] == "cleaned"
