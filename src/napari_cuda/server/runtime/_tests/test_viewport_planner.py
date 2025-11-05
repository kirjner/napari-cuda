from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from napari_cuda.server.data import SliceROI
from napari_cuda.server.runtime.render_loop.planning.viewport_planner import (
    ViewportOps,
    ViewportPlanner,
)
from napari_cuda.server.scene.viewport import PoseEvent
from napari_cuda.server.scene import (
    RenderLedgerSnapshot,
)


@dataclass
class _StubSource:
    chunks: tuple[int, ...] = (1, 8, 8)
    axes: tuple[str, ...] = ("z", "y", "x")

    def get_level(self, level: int):
        return type("_Level", (), {"chunks": self.chunks})()


def _make_snapshot(
    *,
    level: int = 0,
    ndisplay: int = 2,
    rect: Optional[tuple[float, float, float, float]] = (0.0, 0.0, 100.0, 60.0),
) -> RenderLedgerSnapshot:
    return RenderLedgerSnapshot(
        ndisplay=ndisplay,
        current_level=level,
        current_step=(0, 0, 0),
        plane_rect=rect,
        plane_center=(0.0, 0.0),
        plane_zoom=1.0,
        layer_values=None,
    )


def _roi_resolver(level: int, rect: tuple[float, float, float, float]) -> SliceROI:
    x0, y0, w, h = rect
    x1 = x0 + w
    y1 = y0 + h
    return SliceROI(int(y0 // 4) * 4, int(y1 // 4) * 4, int(x0 // 4) * 4, int(x1 // 4) * 4)


def _apply_ops(runner: ViewportPlanner, ops: ViewportOps) -> None:
    if ops.level_change:
        target_level = (
            ops.slice_task.level if ops.slice_task is not None else runner.state.request.level
        )
        runner.mark_level_applied(target_level)
    if ops.slice_task is not None:
        runner.mark_slice_applied(ops.slice_task)


def test_level_switch_sets_level_change_flag() -> None:
    runner = ViewportPlanner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(level=2))

    ops = runner.plan_tick(source=source, roi_resolver=_roi_resolver)

    assert isinstance(ops, ViewportOps)
    assert ops.level_change is True
    _apply_ops(runner, ops)
    follow_up = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    assert follow_up.level_change is False


def test_repeated_roi_within_chunk_skips_reload() -> None:
    runner = ViewportPlanner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(rect=(0.0, 0.0, 64.0, 64.0)))
    first = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    _apply_ops(runner, first)

    runner.ingest_snapshot(_make_snapshot(rect=(2.0, 2.0, 62.0, 62.0)))
    ops = runner.plan_tick(source=source, roi_resolver=_roi_resolver)

    assert ops.slice_task is None


def test_roi_reload_triggers_on_chunk_change() -> None:
    runner = ViewportPlanner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(rect=(0.0, 0.0, 32.0, 32.0)))
    first = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    _apply_ops(runner, first)

    runner.ingest_snapshot(_make_snapshot(rect=(64.0, 0.0, 96.0, 32.0)))
    ops = runner.plan_tick(source=source, roi_resolver=_roi_resolver)

    assert ops.slice_task is not None


def test_volume_mode_disables_roi_reload() -> None:
    runner = ViewportPlanner()
    runner.ingest_snapshot(_make_snapshot(level=1, ndisplay=3))

    ops = runner.plan_tick(source=_StubSource(), roi_resolver=_roi_resolver)

    assert ops.slice_task is None


def test_volume_level_request_does_not_wait_for_confirm() -> None:
    runner = ViewportPlanner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(level=1, ndisplay=3))

    requested = runner.request_level(2)
    assert requested is True

    ops = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    assert ops.level_change is True
    _apply_ops(runner, ops)
    assert runner.state.applied.level == 2


def test_volume_pose_emits_only_on_dirty_events() -> None:
    runner = ViewportPlanner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(level=1, ndisplay=3))
    runner.request_level(2)
    ops = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    _apply_ops(runner, ops)
    assert ops.pose_event is PoseEvent.LEVEL_RELOAD

    second = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    assert second.pose_event is None

    zoom_cmd = type("Cmd", (), {"kind": "zoom", "factor": 1.1})()
    runner.ingest_camera_deltas([zoom_cmd])
    third = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    assert third.pose_event is PoseEvent.CAMERA_DELTA


def test_zoom_hint_consumed_once() -> None:
    runner = ViewportPlanner()
    commands: Sequence[object] = [
        type("Cmd", (), {"kind": "zoom", "factor": 1.2})(),
        type("Cmd", (), {"kind": "zoom", "factor": 0.85})(),
    ]
    runner.ingest_camera_deltas(commands)
    assert runner.state.zoom_hint == 0.85

    ops = runner.plan_tick(source=_StubSource(), roi_resolver=_roi_resolver)
    assert ops.zoom_hint == 0.85
    assert runner.state.zoom_hint is None
