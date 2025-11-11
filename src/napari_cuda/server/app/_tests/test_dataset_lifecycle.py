from __future__ import annotations

from types import SimpleNamespace

from napari_cuda.server.app.dataset_lifecycle import (
    ServerLifecycleHooks,
    apply_dataset_bootstrap,
    enter_idle_state,
)
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.ledger import ServerStateLedger


class StubBroadcast:
    def __init__(self) -> None:
        self.last_key_seq = 17
        self.last_key_ts = 42.0
        self.waiting_for_keyframe = False


class StubPixelChannel:
    def __init__(self) -> None:
        self.broadcast = StubBroadcast()
        self.last_avcc = b"seed"
        self.needs_stream_config = False


def _make_hooks(pixel_channel: StubPixelChannel, snapshot: RenderLedgerSnapshot):
    calls: dict[str, int] = {
        "stop": 0,
        "clear": 0,
        "reset_mirrors": 0,
        "reset_thumbnail": 0,
        "start_mirrors": 0,
    }
    refreshed: list[RenderLedgerSnapshot | None] = []
    dataset_meta: list[tuple[str | None, str | None, str | None, int | None]] = []
    stored_snapshot: list[RenderLedgerSnapshot] = []

    def incr(key: str) -> None:
        calls[key] += 1

    hooks = ServerLifecycleHooks(
        stop_worker=lambda: incr("stop"),
        clear_frame_queue=lambda: incr("clear"),
        reset_mirrors=lambda: incr("reset_mirrors"),
        reset_thumbnail_state=lambda: incr("reset_thumbnail"),
        refresh_scene_snapshot=lambda render_state: refreshed.append(render_state),
        start_mirrors_if_needed=lambda: incr("start_mirrors"),
        pull_render_snapshot=lambda: snapshot,
        set_dataset_metadata=lambda path, level, axes, z: dataset_meta.append(
            (path, level, axes, z)
        ),
        set_bootstrap_snapshot=lambda snap: stored_snapshot.append(snap),
        pixel_channel=pixel_channel,
    )
    return hooks, calls, refreshed, dataset_meta, stored_snapshot


def test_enter_idle_state_resets_runtime() -> None:
    ledger = ServerStateLedger()
    pixel_channel = StubPixelChannel()
    snapshot = RenderLedgerSnapshot(op_seq=1, current_level=0, current_step=(0, 0))
    hooks, calls, refreshed, dataset_meta, stored_snapshot = _make_hooks(pixel_channel, snapshot)

    enter_idle_state(hooks, ledger)

    assert calls == {
        "stop": 1,
        "clear": 1,
        "reset_mirrors": 1,
        "reset_thumbnail": 1,
        "start_mirrors": 0,
    }
    assert pixel_channel.needs_stream_config is True
    assert pixel_channel.last_avcc is None
    broadcast = pixel_channel.broadcast
    assert broadcast.last_key_seq is None
    assert broadcast.last_key_ts is None
    assert broadcast.waiting_for_keyframe is True

    assert dataset_meta[-1] == (None, None, None, None)
    assert stored_snapshot[-1] is snapshot
    assert refreshed[-1] is snapshot


def test_apply_dataset_bootstrap_applies_metadata_and_snapshot() -> None:
    ledger = ServerStateLedger()
    pixel_channel = StubPixelChannel()
    snapshot = RenderLedgerSnapshot(op_seq=2, current_level=1, current_step=(4, 5, 6))
    hooks, calls, refreshed, dataset_meta, stored_snapshot = _make_hooks(pixel_channel, snapshot)

    bootstrap_meta = SimpleNamespace(
        step=(1, 2, 3),
        axis_labels=("z", "y", "x"),
        order=(0, 1, 2),
        level_shapes=((8, 8, 8),),
        levels=(
            {"index": 0, "shape": [8, 8, 8], "downsample": [1.0, 1.0, 1.0], "path": "level_0"},
        ),
        current_level=0,
        ndisplay=3,
    )

    result = apply_dataset_bootstrap(
        hooks,
        ledger,
        bootstrap_meta,
        resolved_path="/data/sample.zarr",
        preferred_level="level_0",
        z_override=5,
    )

    assert result is snapshot
    assert calls["stop"] == 1
    assert calls["clear"] == 1
    assert calls["reset_mirrors"] == 1
    assert calls["reset_thumbnail"] == 1
    assert calls["start_mirrors"] == 1

    assert dataset_meta[-1] == ("/data/sample.zarr", "level_0", "zyx", 5)
    assert stored_snapshot[-1] is snapshot
    assert refreshed[-1] is snapshot
    assert pixel_channel.needs_stream_config is True
    assert pixel_channel.last_avcc is None
    assert pixel_channel.broadcast.waiting_for_keyframe is True
