"""Capture and encode helpers for the EGL render worker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from napari_cuda.server.engine.api import FrameTimings, encode_frame
from napari_cuda.server.runtime.camera.animator import animate_if_enabled

from ...loop import run_render_tick
from ...render_interface import RenderInterface
from . import camera

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


def capture_blit_gpu_ns(worker: EGLRendererWorker) -> Optional[int]:
    render_iface = RenderInterface(worker)
    resources = render_iface.resources
    assert resources is not None and resources.capture is not None, "render resources must be initialised"
    return resources.capture.pipeline.capture_blit_gpu_ns()


def render_tick(worker: EGLRendererWorker) -> float:
    render_iface = RenderInterface(worker)
    canvas = render_iface.canvas
    assert canvas is not None, "render_tick requires an initialized canvas"

    return run_render_tick(
        animate_camera=lambda: animate_if_enabled(
            enabled=render_iface.animate,
            view=render_iface.view,
            width=worker.width,
            height=worker.height,
            animate_dps=render_iface.animate_dps,
            anim_start=render_iface.anim_start,
        ),
        drain_scene_updates=lambda: camera.drain(worker),
        render_canvas=canvas.render,
        evaluate_policy_if_needed=lambda: None,
        mark_tick_complete=render_iface.mark_render_tick_complete,
        mark_loop_started=render_iface.mark_render_loop_started,
    )


def capture_and_encode_packet(
    worker: EGLRendererWorker,
) -> tuple[FrameTimings, Optional[bytes], int, int]:
    render_iface = RenderInterface(worker)
    resources = render_iface.resources
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
