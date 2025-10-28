from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from napari_cuda.server.runtime.viewport import ViewportIntent, ViewportRunner
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.scene_types import SliceROI
from napari_cuda.server.runtime.roi_math import chunk_shape_for_level


@dataclass
class _StubSource:
    chunks: Tuple[int, ...] = (1, 8, 8)
    axes: Tuple[str, ...] = ("z", "y", "x")

    def get_level(self, level: int):
        return type("_Level", (), {"chunks": self.chunks})()


def _make_snapshot(
    *,
    level: int = 0,
    ndisplay: int = 2,
    rect: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 100.0, 60.0),
) -> RenderLedgerSnapshot:
    return RenderLedgerSnapshot(
        ndisplay=ndisplay,
        current_level=level,
        current_step=(0, 0, 0),
        plane_rect=rect,
        plane_center=(0.0, 0.0),
        plane_zoom=1.0,
        layer_values=None,
        layer_versions=None,
    )


def _roi_resolver(level: int, rect: Tuple[float, float, float, float]) -> SliceROI:
    x0, y0, w, h = rect
    x1 = x0 + w
    y1 = y0 + h
    return SliceROI(int(y0 // 4) * 4, int(y1 // 4) * 4, int(x0 // 4) * 4, int(x1 // 4) * 4)


def _apply_intent(runner: ViewportRunner, source: _StubSource, intent: ViewportIntent) -> None:
    if intent.level_change:
        runner.mark_level_applied(runner.state.target_level)
    if intent.roi_change:
        pending = runner.state.pending_roi
        assert pending is not None
        runner.mark_roi_applied(
            pending,
            chunk_shape=chunk_shape_for_level(source, runner.state.target_level),
        )


def test_level_switch_sets_level_change_flag() -> None:
    runner = ViewportRunner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(level=2))

    intent = runner.plan_tick(source=source, roi_resolver=_roi_resolver)

    assert isinstance(intent, ViewportIntent)
    assert intent.level_change is True
    _apply_intent(runner, source, intent)
    assert runner.state.level_reload_required is False


def test_repeated_roi_within_chunk_skips_reload() -> None:
    runner = ViewportRunner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(rect=(0.0, 0.0, 64.0, 64.0)))
    first = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    _apply_intent(runner, source, first)

    runner.ingest_snapshot(_make_snapshot(rect=(2.0, 2.0, 62.0, 62.0)))
    intent = runner.plan_tick(source=source, roi_resolver=_roi_resolver)

    assert intent.roi_change is False


def test_roi_reload_triggers_on_chunk_change() -> None:
    runner = ViewportRunner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(rect=(0.0, 0.0, 32.0, 32.0)))
    first = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    _apply_intent(runner, source, first)

    runner.ingest_snapshot(_make_snapshot(rect=(64.0, 0.0, 96.0, 32.0)))
    intent = runner.plan_tick(source=source, roi_resolver=_roi_resolver)

    assert intent.roi_change is True


def test_volume_mode_disables_roi_reload() -> None:
    runner = ViewportRunner()
    runner.ingest_snapshot(_make_snapshot(level=1, ndisplay=3))

    intent = runner.plan_tick(source=_StubSource(), roi_resolver=_roi_resolver)

    assert intent.roi_change is False


def test_volume_level_request_does_not_wait_for_confirm() -> None:
    runner = ViewportRunner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(level=1, ndisplay=3))

    requested = runner.request_level(2)
    assert requested is True

    intent = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    assert intent.level_change is True
    _apply_intent(runner, source, intent)
    assert runner.state.applied_level == 2


def test_volume_pose_emits_only_on_dirty_events() -> None:
    runner = ViewportRunner()
    source = _StubSource()
    runner.ingest_snapshot(_make_snapshot(level=1, ndisplay=3))
    runner.request_level(2)
    intent = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    _apply_intent(runner, source, intent)
    assert intent.pose_reason == "level-reload"

    second = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    assert second.pose_reason is None

    zoom_cmd = type("Cmd", (), {"kind": "zoom", "factor": 1.1})()
    runner.ingest_camera_deltas([zoom_cmd])
    third = runner.plan_tick(source=source, roi_resolver=_roi_resolver)
    assert third.pose_reason == "camera-delta"


def test_zoom_hint_consumed_once() -> None:
    runner = ViewportRunner()
    commands: Sequence[object] = [
        type("Cmd", (), {"kind": "zoom", "factor": 1.2})(),
        type("Cmd", (), {"kind": "zoom", "factor": 0.85})(),
    ]
    runner.ingest_camera_deltas(commands)
    assert runner.state.zoom_hint == 0.85

    intent = runner.plan_tick(source=_StubSource(), roi_resolver=_roi_resolver)
    assert intent.zoom_hint == 0.85
    assert runner.state.zoom_hint is None
