"""Pipeline helpers for the client stream loop."""

from __future__ import annotations

import logging
from typing import Callable, TYPE_CHECKING

from napari_cuda.client.streaming.pipelines.pyav_pipeline import PyAVPipeline
from napari_cuda.client.streaming.pipelines.vt_pipeline import VTPipeline

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop


def build_vt_pipeline(
    loop: "ClientStreamLoop",
    *,
    schedule_next_wake: Callable[[], None],
    logger: logging.Logger,
) -> VTPipeline:
    """Create the VT pipeline wired to the loop's state."""

    def _is_vt_gated() -> bool:
        return bool(loop._loop_state.vt_wait_keyframe)

    def _on_vt_backlog_gate() -> None:
        loop._loop_state.vt_wait_keyframe = True

    def _on_cache_last(payload: object, persistent: bool) -> None:
        try:
            loop._loop_state.fallbacks.update_vt_cache(payload, persistent)
        except Exception:
            logger.debug("cache last VT payload callback failed", exc_info=True)

    return VTPipeline(
        presenter=loop._presenter,
        source_mux=loop._source_mux,
        scene_canvas=loop._scene_canvas,
        backlog_trigger=loop._vt_backlog_trigger,
        is_gated=_is_vt_gated,
        on_backlog_gate=_on_vt_backlog_gate,
        request_keyframe=loop._request_keyframe_command,  # noqa: SLF001
        on_cache_last=_on_cache_last,
        metrics=loop._loop_state.metrics,
        schedule_next_wake=schedule_next_wake,
    )


def build_pyav_pipeline(
    loop: "ClientStreamLoop",
    *,
    schedule_next_wake: Callable[[], None],
) -> PyAVPipeline:
    """Create the PyAV pipeline wired to the loop's state."""

    return PyAVPipeline(
        presenter=loop._presenter,
        source_mux=loop._source_mux,
        scene_canvas=loop._scene_canvas,
        backlog_trigger=loop._pyav_backlog_trigger,
        latency_s=loop._pyav_latency_s,
        metrics=loop._loop_state.metrics,
        schedule_next_wake=schedule_next_wake,
    )
