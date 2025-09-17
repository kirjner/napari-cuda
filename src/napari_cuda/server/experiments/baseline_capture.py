"""Baseline EGL renderer capture harness for comparison against VisPy spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..egl_worker import EGLRendererWorker, FrameTimings

logger = logging.getLogger(__name__)


def _ensure_env() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")


class BaselineRendererWorker(EGLRendererWorker):
    """Original worker with a convenience accessor for captured RGBA frames."""

    def latest_rgba(self) -> np.ndarray:
        tensor = getattr(self, "_torch_frame", None)
        if tensor is None:
            raise RuntimeError("No frame captured yet")
        return tensor.detach().cpu().numpy().copy()


@dataclass
class CaptureConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    frames: int = 60
    use_volume: bool = False
    output: Optional[Path] = None
    profile: Optional[Path] = None


@dataclass
class CaptureResult:
    index: int
    timings: FrameTimings
    hash_hex: str
    packet_bytes: Optional[int]

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
            "packet_bytes": self.packet_bytes,
            "hash": self.hash_hex,
        }


def _hash_rgba(rgba: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rgba).view(np.uint8).tobytes())
    return h.hexdigest()


def _parse_args(argv: Optional[Iterable[str]] = None) -> CaptureConfig:
    parser = argparse.ArgumentParser(description="Capture baseline EGL renderer frames")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--volume", action="store_true", help="Render the synthetic volume path")
    parser.add_argument("--output", type=Path, help="JSON file for per-frame metrics")
    parser.add_argument("--profile", type=Path, help="JSON file for aggregate timings")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return CaptureConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        use_volume=args.volume,
        output=args.output,
        profile=args.profile,
    )


def run_capture(cfg: CaptureConfig) -> list[CaptureResult]:
    _ensure_env()
    worker = BaselineRendererWorker(
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        use_volume=cfg.use_volume,
    )
    try:
        results: list[CaptureResult] = []
        for idx in range(cfg.frames):
            timings, packet, _flags, _seq = worker.capture_and_encode_packet()
            rgba = worker.latest_rgba()
            results.append(
                CaptureResult(
                    index=idx,
                    timings=timings,
                    hash_hex=_hash_rgba(rgba),
                    packet_bytes=len(packet) if packet is not None else None,
                )
            )
        return results
    finally:
        worker.cleanup()


def _write_results(results: list[CaptureResult], cfg: CaptureConfig) -> None:
    if cfg.output:
        if cfg.output.parent:
            cfg.output.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in results]
        cfg.output.write_text(json.dumps(rows, indent=2))
    if cfg.profile:
        render = [r.timings.render_ms for r in results]
        total = [r.timings.total_ms for r in results]
        encode = [r.timings.encode_ms for r in results]
        pkt_sizes = [r.packet_bytes for r in results if r.packet_bytes is not None]
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
                "min": int(np.min(pkt_sizes)) if pkt_sizes else None,
                "max": int(np.max(pkt_sizes)) if pkt_sizes else None,
                "mean": float(np.mean(pkt_sizes)) if pkt_sizes else None,
            },
        }
        if cfg.profile.parent:
            cfg.profile.parent.mkdir(parents=True, exist_ok=True)
        cfg.profile.write_text(json.dumps(profile, indent=2))


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info(
        "Running baseline capture: frames=%d size=%dx%d volume=%s",
        cfg.frames,
        cfg.width,
        cfg.height,
        cfg.use_volume,
    )
    try:
        results = run_capture(cfg)
    except Exception:
        logger.exception("Baseline capture failed")
        return 1
    _write_results(results, cfg)
    logger.info("Baseline capture completed: %d frames", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
