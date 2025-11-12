from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_cuda.server.runtime.worker import lifecycle
from napari_cuda.server.scene.models import RenderLedgerSnapshot


def test_pull_cached_render_snapshot_invokes_ledger_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def _fake_pull(_server: object) -> RenderLedgerSnapshot:
        calls.append(1)
        return RenderLedgerSnapshot(
            op_seq=7,
            block_snapshot=None,
        )

    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.lifecycle.pull_render_snapshot",
        _fake_pull,
    )
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.lifecycle.ENABLE_VIEW_AXES_INDEX_BLOCKS",
        True,
    )

    cache: dict[str, object] = {}
    snap_a, seq_a = lifecycle._pull_cached_render_snapshot(SimpleNamespace(), cache)
    snap_b, seq_b = lifecycle._pull_cached_render_snapshot(SimpleNamespace(), cache)

    assert snap_a is snap_b
    assert seq_a == seq_b == 7
    assert len(calls) == 1, "pull_render_snapshot should run once while cache is warm"

    cache.clear()
    lifecycle._pull_cached_render_snapshot(SimpleNamespace(), cache)
    assert len(calls) == 2, "cache must pull again after tick reset"
