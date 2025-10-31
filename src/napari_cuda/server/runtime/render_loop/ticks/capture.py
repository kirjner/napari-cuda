"""Capture and encode helpers for the EGL render worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from napari_cuda.server.rendering.capture import FrameTimings, encode_frame
from napari_cuda.server.runtime.camera.animator import animate_if_enabled

from ..loop import run_render_tick
from ..tick_interface import RenderTickInterface
from . import camera

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


def capture_blit_gpu_ns(worker: EGLRendererWorker) -> Optional[int]:
    tick_iface = RenderTickInterface(worker)
    resources = tick_iface.resources
    assert resources is not None and resources.capture is not None, "render resources must be initialised"
    return resources.capture.pipeline.capture_blit_gpu_ns()


def render_tick(worker: EGLRendererWorker) -> float:
    tick_iface = RenderTickInterface(worker)
    canvas = tick_iface.canvas
    assert canvas is not None, "render_tick requires an initialized canvas"

    return run_render_tick(
        animate_camera=lambda: animate_if_enabled(
            enabled=tick_iface.animate,
            view=tick_iface.view,
            width=worker.width,
            height=worker.height,
            animate_dps=tick_iface.animate_dps,
            anim_start=tick_iface.anim_start,
        ),
        drain_scene_updates=lambda: camera.drain(worker),
        render_canvas=canvas.render,
        evaluate_policy_if_needed=lambda: None,
        mark_tick_complete=tick_iface.mark_render_tick_complete,
        mark_loop_started=tick_iface.mark_render_loop_started,
    )


def capture_and_encode_packet(
    worker: EGLRendererWorker,
) -> tuple[FrameTimings, Optional[bytes], int, int]:
    tick_iface = RenderTickInterface(worker)
    resources = tick_iface.resources
    assert resources is not None, "render resources must be initialised"
    encoded = encode_frame(
        capture=resources.capture,
        render_frame=lambda: render_tick(worker),
        obtain_encoder=lambda: resources.encoder,
        encoder_lock=resources.enc_lock,
        debug_dumper=getattr(worker, "_debug", None),
    )

    return encoded.timings, encoded.packet, encoded.flags, encoded.sequence


__all__ = [
    "capture_and_encode_packet",
    "capture_blit_gpu_ns",
    "render_tick",
]
