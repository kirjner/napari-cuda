"""VisPy adapter spike harness for the headless EGL renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from vispy import scene

from napari._vispy.layers.image import VispyImageLayer
from napari.components import ViewerModel
from napari.layers import Image

from ..config import load_server_ctx
from ..egl_worker import EGLRendererWorker, FrameTimings

logger = logging.getLogger(__name__)


def _ensure_env() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")


class SpikeRendererWorker(EGLRendererWorker):
    """Renderer worker variant that overrides scene init and optional NVENC."""

    def __init__(
        self,
        *args,
        disable_nvenc: bool = True,
        **kwargs,
    ) -> None:
        self._disable_nvenc = disable_nvenc
        super().__init__(*args, **kwargs)

    def _init_vispy_scene(self) -> None:  # type: ignore[override]
        canvas = scene.SceneCanvas(
            size=(self.width, self.height),
            bgcolor="black",
            show=False,
            app="egl",
        )
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
        self.canvas = canvas
        self.view = view
        self._visual = None
        self._data_wh = (self.width, self.height)
        canvas.render()

    def _init_encoder(self) -> None:  # type: ignore[override]
        if getattr(self, "_disable_nvenc", True):
            self._encoder = None
            self._enc_input_fmt = "ARGB"
            logger.info("SpikeRendererWorker: NVENC disabled for experiment")
            return
        logger.info("SpikeRendererWorker: using NVENC path")
        super()._init_encoder()

    def latest_rgba(self) -> np.ndarray:
        if getattr(self, "_torch_frame", None) is None:
            raise RuntimeError("No frame captured yet")
        tensor = self._torch_frame.detach().cpu()
        return tensor.numpy().copy()


@dataclass
class SpikeConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    layer: str = "image"
    frames: int = 60
    mutate: bool = False
    mutation_interval: float = 0.25
    output: Optional[Path] = None
    profile: Optional[Path] = None
    use_nvenc: bool = False


@dataclass
class FrameResult:
    index: int
    timings: FrameTimings
    hash_hex: str
    packet_bytes: Optional[int] = None

    def to_dict(self) -> dict:
        t = self.timings
        return {
            "index": self.index,
            "render_ms": t.render_ms,
            "blit_gpu_ns": t.blit_gpu_ns,
            "blit_cpu_ms": t.blit_cpu_ms,
            "map_ms": t.map_ms,
            "copy_ms": t.copy_ms,
            "convert_ms": t.convert_ms,
            "encode_ms": t.encode_ms,
            "pack_ms": t.pack_ms,
            "total_ms": t.total_ms,
            "hash": self.hash_hex,
            "packet_bytes": self.packet_bytes,
        }


class LayerMutator(threading.Thread):
    """Background mutator exercising napari events while rendering."""

    def __init__(self, layer: Image, interval: float, stop_event: threading.Event) -> None:
        super().__init__(name="layer-mutator", daemon=True)
        self._layer = layer
        self._interval = max(0.05, float(interval))
        self._stop_event = stop_event
        self._tick = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            data = np.asarray(self._layer.data)
            if data.ndim == 2:
                rolled = np.roll(data, self._tick % data.shape[1], axis=1)
            elif data.ndim == 3:
                rolled = np.roll(data, self._tick % data.shape[0], axis=0)
            else:
                noise = np.random.default_rng(self._tick).normal(loc=0.0, scale=0.002, size=data.shape)
                rolled = data + noise
            self._layer.data = rolled
            self._tick += 1
            self._stop_event.wait(self._interval)


def _build_viewer_and_layer(kind: str, seed: int = 13) -> tuple[ViewerModel, Image, VispyImageLayer]:
    rng = np.random.default_rng(seed)
    viewer = ViewerModel()

    if kind == "image":
        data = rng.random((512, 512), dtype=np.float32)
        layer = viewer.add_image(data, name="spike-image")
        viewer.dims.ndisplay = 2
    elif kind == "volume":
        data = rng.random((128, 256, 256), dtype=np.float32)
        layer = viewer.add_image(data, name="spike-volume")
        viewer.dims.ndisplay = 3
    elif kind == "multiscale":
        base = rng.random((512, 512), dtype=np.float32)
        level1 = base[:, ::2]
        level2 = level1[:, ::2]
        pyramid = [base, level1, level2]
        layer = viewer.add_image(pyramid, name="spike-pyramid", multiscale=True)
        viewer.dims.ndisplay = 2
    else:
        raise ValueError(f"Unsupported layer kind: {kind}")

    adapter = VispyImageLayer(layer)
    return viewer, layer, adapter


def _attach_to_worker(worker: SpikeRendererWorker, adapter: VispyImageLayer, layer_kind: str) -> None:
    if worker.view is None:
        raise RuntimeError("Worker view not initialized")
    if layer_kind == "volume":
        worker.use_volume = True
        worker.view.camera = scene.cameras.TurntableCamera(fov=60.0, elevation=30.0, azimuth=30.0)
    else:
        worker.use_volume = False
        worker.view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
    node = adapter.node
    node.parent = worker.view.scene
    worker._visual = node  # reuse existing hooks for state application
    worker.canvas.render()


def _hash_rgba(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(array).view(np.uint8).tobytes())
    return h.hexdigest()


def _parse_args(argv: Optional[Iterable[str]] = None) -> SpikeConfig:
    parser = argparse.ArgumentParser(description="Run napari VisPy adapter spike under EGL")
    parser.add_argument("--layer", choices=["image", "volume", "multiscale"], default="image")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to render")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mutate", action="store_true", help="Enable background layer mutations")
    parser.add_argument("--mutation-interval", type=float, default=0.25)
    parser.add_argument("--output", type=Path, help="Optional JSON file to store per-frame metrics")
    parser.add_argument("--profile", type=Path, help="Optional JSON file to store summary stats")
    parser.add_argument("--nvenc", action="store_true", help="Enable NVENC and retain encoded packets")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return SpikeConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        layer=args.layer,
        frames=args.frames,
        mutate=args.mutate,
        mutation_interval=args.mutation_interval,
        output=args.output,
        profile=args.profile,
        use_nvenc=args.nvenc,
    )


def run_spike(cfg: SpikeConfig, stop_event: Optional[threading.Event] = None) -> list[FrameResult]:
    _ensure_env()
    viewer, layer, adapter = _build_viewer_and_layer(cfg.layer)
    worker = SpikeRendererWorker(
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        use_volume=(cfg.layer == "volume"),
        disable_nvenc=not cfg.use_nvenc,
        ctx=load_server_ctx(),
    )
    try:
        _attach_to_worker(worker, adapter, cfg.layer)
        stop_evt = threading.Event()
        mutator: Optional[LayerMutator] = None
        if cfg.mutate:
            mutator = LayerMutator(layer, cfg.mutation_interval, stop_evt)
            mutator.start()

        results: list[FrameResult] = []
        for index in range(cfg.frames):
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop event set; breaking render loop at frame %d", index)
                break
            timings, pkt, _flags, _seq = worker.capture_and_encode_packet()
            rgba = worker.latest_rgba()
            hash_hex = _hash_rgba(rgba)
            packet_len = len(pkt) if pkt is not None else None
            results.append(
                FrameResult(index=index, timings=timings, hash_hex=hash_hex, packet_bytes=packet_len)
            )
        stop_evt.set()
        if mutator is not None:
            mutator.join(timeout=2.0)
        return results
    finally:
        try:
            worker.cleanup()
        except Exception:
            logger.exception("Worker cleanup failed")


def _write_results(results: list[FrameResult], cfg: SpikeConfig) -> None:
    if cfg.output:
        rows = [entry.to_dict() for entry in results]
        if cfg.output.parent:
            cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.write_text(json.dumps(rows, indent=2))
    if cfg.profile:
        render = [r.timings.render_ms for r in results]
        total = [r.timings.total_ms for r in results]
        encode = [r.timings.encode_ms for r in results]
        packets = [r.packet_bytes for r in results if r.packet_bytes is not None]
        profile = {
            "frames": len(results),
            "render_ms": {
                "min": float(np.min(render)),
                "max": float(np.max(render)),
                "mean": float(np.mean(render)),
            },
            "total_ms": {
                "min": float(np.min(total)),
                "max": float(np.max(total)),
                "mean": float(np.mean(total)),
            },
            "encode_ms": {
                "min": float(np.min(encode)),
                "max": float(np.max(encode)),
                "mean": float(np.mean(encode)),
            },
            "packet_bytes": {
                "min": int(np.min(packets)) if packets else None,
                "max": int(np.max(packets)) if packets else None,
                "mean": float(np.mean(packets)) if packets else None,
            },
        }
        if cfg.profile.parent:
            cfg.profile.parent.mkdir(parents=True, exist_ok=True)
        cfg.profile.write_text(json.dumps(profile, indent=2))


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info(
        "Starting VisPy spike: layer=%s frames=%d size=%dx%d mutate=%s",
        cfg.layer,
        cfg.frames,
        cfg.width,
        cfg.height,
        cfg.mutate,
    )

    stop_on_signal = threading.Event()

    def _handle_signal(signum: int, _frame) -> None:
        logger.warning("Received signal %s; stopping after current frame", signum)
        stop_on_signal.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        results = run_spike(cfg, stop_event=stop_on_signal)
    except KeyboardInterrupt:
        logger.warning("Spike interrupted")
        return 130
    except Exception:
        logger.exception("Spike failed")
        return 1

    _write_results(results, cfg)
    logger.info("Spike completed: %d frames", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
