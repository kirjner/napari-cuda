from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_cuda.server.runtime.worker.lifecycle import (
    _OpSeqWatcherState,
    _op_seq_watcher_apply_snapshot,
)
from napari_cuda.server.scene.blocks import (
    AxisBlock,
    AxisExtentBlock,
    AxesBlock,
    CameraBlock,
    IndexBlock,
    LodBlock,
    PlaneCameraBlock,
    PlaneRestoreCacheBlock,
    PlaneRestoreCachePose,
    ViewBlock,
    VolumeCameraBlock,
    VolumeRestoreCacheBlock,
    VolumeRestoreCachePose,
)
from napari_cuda.server.scene.models import RenderLedgerSnapshot, SceneBlockSnapshot
from napari_cuda.server.utils.signatures import SignatureToken


def _tokens(value: int = 0) -> dict[str, SignatureToken]:
    return {
        "view": SignatureToken(("view", value)),
        "axes": SignatureToken(("axes", value)),
        "index": SignatureToken(("index", value)),
        "lod": SignatureToken(("lod", value)),
        "camera": SignatureToken(("camera", value)),
        "layers": SignatureToken(("layers", value)),
    }


def test_op_seq_watcher_state_tracks_changes() -> None:
    watcher = _OpSeqWatcherState()
    first = watcher.observe(1, **_tokens(1))
    assert first

    # No change when op_seq and tokens are identical.
    assert watcher.observe(1, **_tokens(1)) is False

    # Token change triggers an update even if op_seq stays constant.
    updated = _tokens(1)
    updated["camera"] = SignatureToken(("camera", 2))
    assert watcher.observe(1, **updated) is True

    # Bumping op_seq also reports a change and updates cached seq.
    assert watcher.observe(2, **updated) is True


def _sample_scene_blocks() -> SceneBlockSnapshot:
    axes = AxesBlock(
        axes=(
            AxisBlock(
                axis_id=0,
                label="z",
                role="space",
                displayed=False,
                world_extent=AxisExtentBlock(start=0.0, stop=10.0, step=1.0),
                margin_left_world=0.0,
                margin_right_world=0.0,
            ),
        ),
    )
    return SceneBlockSnapshot(
        view=ViewBlock(mode="plane", displayed_axes=(1, 2), ndim=3),
        axes=axes,
        index=IndexBlock(value=(0, 0, 0)),
        lod=LodBlock(level=0, roi=None, policy="auto"),
        camera=CameraBlock(
            plane=PlaneCameraBlock(rect=(0.0, 0.0, 1.0, 1.0), center=(0.0, 0.0), zoom=1.0),
            volume=VolumeCameraBlock(
                center=(0.0, 0.0, 0.0),
                angles=(0.0, 0.0, 0.0),
                distance=10.0,
                fov=60.0,
            ),
        ),
        plane_restore=PlaneRestoreCacheBlock(
            level=0,
            index=(0, 0, 0),
            pose=PlaneRestoreCachePose(
                rect=(0.0, 0.0, 1.0, 1.0),
                center=(0.0, 0.0),
                zoom=1.0,
            ),
        ),
        volume_restore=VolumeRestoreCacheBlock(
            level=0,
            index=(0, 0, 0),
            pose=VolumeRestoreCachePose(
                center=(0.0, 0.0, 0.0),
                angles=(0.0, 0.0, 0.0),
                distance=10.0,
                fov=60.0,
            ),
        ),
    )


def test_op_seq_watcher_uses_existing_block_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _sample_scene_blocks()
    snapshot = RenderLedgerSnapshot(op_seq=1, block_snapshot=blocks)
    worker = SimpleNamespace(_ledger=object(), _op_seq_watcher_state=None)
    consumed: list[RenderLedgerSnapshot] = []
    render_iface = SimpleNamespace(
        apply_scene_blocks=lambda snap: consumed.append(snap),
    )

    def _fail_fetch(_ledger) -> SceneBlockSnapshot:
        raise AssertionError("fetch_scene_blocks should not be called when block snapshot is provided")

    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.lifecycle.fetch_scene_blocks",
        _fail_fetch,
    )
    result = _op_seq_watcher_apply_snapshot(
        server=SimpleNamespace(),
        worker=worker,  # type: ignore[arg-type]
        snapshot=snapshot,
        current_op_seq=1,
        blocks=blocks,
        render_iface=render_iface,  # type: ignore[arg-type]
    )

    assert result is snapshot
    assert consumed == [snapshot]


def test_op_seq_watcher_accepts_cached_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _sample_scene_blocks()
    snapshot = RenderLedgerSnapshot(op_seq=1, block_snapshot=None)
    worker = SimpleNamespace(_ledger=object(), _op_seq_watcher_state=None)
    render_iface = SimpleNamespace(apply_scene_blocks=lambda snap: None)

    def _fail_fetch(_ledger) -> SceneBlockSnapshot:
        raise AssertionError("fetch_scene_blocks must not run when cached blocks are provided")

    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.lifecycle.fetch_scene_blocks",
        _fail_fetch,
    )
    result = _op_seq_watcher_apply_snapshot(
        server=SimpleNamespace(),
        worker=worker,  # type: ignore[arg-type]
        snapshot=snapshot,
        current_op_seq=2,
        blocks=blocks,
        render_iface=render_iface,  # type: ignore[arg-type]
    )

    assert result is snapshot


def test_op_seq_watcher_requires_blocks_when_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = RenderLedgerSnapshot(op_seq=3, block_snapshot=None)
    worker = SimpleNamespace(_ledger=object(), _op_seq_watcher_state=None)

    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.lifecycle.ENABLE_VIEW_AXES_INDEX_BLOCKS",
        True,
    )

    render_iface = SimpleNamespace(apply_scene_blocks=lambda snap: None)

    with pytest.raises(AssertionError):
        _op_seq_watcher_apply_snapshot(
            server=SimpleNamespace(),
            worker=worker,  # type: ignore[arg-type]
            snapshot=snapshot,
            current_op_seq=3,
            render_iface=render_iface,  # type: ignore[arg-type]
        )
