"""Utilities to bootstrap smoke/test mode for the client stream loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari_cuda.client.streaming.pipelines.smoke_pipeline import SmokeConfig, SmokePipeline


if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop


logger = logging.getLogger(__name__)


def start_smoke_mode(loop: "ClientStreamLoop") -> SmokePipeline:
    """Configure and start the smoke harness for the stream loop."""

    logger.info("ClientStreamLoop in smoke test mode (offline)")

    smoke_source = loop._env_cfg.smoke_source  # noqa: SLF001
    if smoke_source == 'pyav':
        loop._init_decoder()  # noqa: SLF001
        dec = loop.decoder.decode if loop.decoder else None  # noqa: SLF001
        loop._pyav_pipeline.set_decoder(dec)  # noqa: SLF001

    width = loop._env_cfg.smoke_width  # noqa: SLF001
    height = loop._env_cfg.smoke_height  # noqa: SLF001
    preset = loop._env_cfg.smoke_preset  # noqa: SLF001
    if preset == '4k60':
        width = 3840
        height = 2160
        fps = 60.0
    else:
        fps = loop._env_cfg.smoke_fps  # noqa: SLF001

    cfg = SmokeConfig(
        width=width,
        height=height,
        fps=fps,
        smoke_mode=loop._env_cfg.smoke_mode,  # noqa: SLF001
        preencode=loop._env_cfg.smoke_preencode,  # noqa: SLF001
        pre_frames=loop._env_cfg.smoke_pre_frames,  # noqa: SLF001
        backlog_trigger=loop._vt_backlog_trigger if smoke_source == 'vt' else loop._pyav_backlog_trigger,  # noqa: SLF001
        target='pyav' if smoke_source == 'pyav' else 'vt',
        vt_latency_s=loop._vt_latency_s,  # noqa: SLF001
        pyav_latency_s=loop._pyav_latency_s,  # noqa: SLF001
        mem_cap_mb=loop._env_cfg.smoke_pre_mb,  # noqa: SLF001
        pre_path=loop._env_cfg.smoke_pre_path,  # noqa: SLF001
    )

    def _init_and_clear(avcc_b64: str, width: int, height: int) -> None:
        loop._init_vt_from_avcc(avcc_b64, width, height)  # noqa: SLF001
        loop._vt_wait_keyframe = False  # noqa: SLF001

    pipeline = loop._pyav_pipeline if smoke_source == 'pyav' else loop._vt_pipeline  # noqa: SLF001
    smoke = SmokePipeline(
        config=cfg,
        presenter=loop._presenter,  # noqa: SLF001
        source_mux=loop._source_mux,  # noqa: SLF001
        pipeline=pipeline,
        init_vt_from_avcc=_init_and_clear,
        metrics=loop._metrics,  # noqa: SLF001
    )
    smoke.start()
    return smoke

