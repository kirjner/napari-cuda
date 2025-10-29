"""Capture and encode helpers for the EGL render worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from napari_cuda.server.rendering.capture import FrameTimings, encode_frame
from napari_cuda.server.runtime.camera.animator import animate_if_enabled
from napari_cuda.server.runtime.worker.loop import run_render_tick

from . import camera

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


def capture_blit_gpu_ns(worker: "EGLRendererWorker") -> Optional[int]:
    return worker._resources.capture.pipeline.capture_blit_gpu_ns()  # noqa: SLF001


def render_tick(worker: "EGLRendererWorker") -> float:
    canvas = worker.canvas
    assert canvas is not None, "render_tick requires an initialized canvas"

    return run_render_tick(
        animate_camera=lambda: animate_if_enabled(
            enabled=bool(worker._animate),  # noqa: SLF001
            view=getattr(worker, "view", None),
            width=worker.width,
            height=worker.height,
            animate_dps=float(worker._animate_dps),  # noqa: SLF001
            anim_start=float(worker._anim_start),  # noqa: SLF001
        ),
        drain_scene_updates=lambda: camera.drain(worker),
        render_canvas=canvas.render,
        evaluate_policy_if_needed=lambda: None,
        mark_tick_complete=worker._mark_render_tick_complete,  # noqa: SLF001
        mark_loop_started=worker._mark_render_loop_started,  # noqa: SLF001
    )


def capture_and_encode_packet(
    worker: "EGLRendererWorker",
) -> tuple[FrameTimings, Optional[bytes], int, int]:
    encoded = encode_frame(
        capture=worker._resources.capture,  # noqa: SLF001
        render_frame=lambda: render_tick(worker),
        obtain_encoder=lambda: worker._resources.encoder,  # noqa: SLF001
        encoder_lock=worker._resources.enc_lock,  # noqa: SLF001
        debug_dumper=worker._debug,  # noqa: SLF001
    )

    return encoded.timings, encoded.packet, encoded.flags, encoded.sequence


__all__ = [
    "capture_and_encode_packet",
    "capture_blit_gpu_ns",
    "render_tick",
]
