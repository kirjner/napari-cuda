"""Render loop driver helpers for :mod:`napari_cuda.server.egl_worker`."""

from __future__ import annotations

from typing import Callable
import time


def run_render_tick(
    *,
    animate_camera: Callable[[], None],
    drain_scene_updates: Callable[[], None],
    refresh_slice: Callable[[], None],
    render_canvas: Callable[[], None],
    evaluate_policy_if_needed: Callable[[], None],
    mark_tick_complete: Callable[[], None],
    mark_loop_started: Callable[[], None],
    perf_counter: Callable[[], float] = time.perf_counter,
) -> float:
    """Execute the render loop once and return the render duration in milliseconds."""

    t0 = perf_counter()
    animate_camera()
    drain_scene_updates()
    refresh_slice()
    render_canvas()
    evaluate_policy_if_needed()
    mark_tick_complete()
    mark_loop_started()
    return (perf_counter() - t0) * 1000.0


__all__ = ["run_render_tick"]
