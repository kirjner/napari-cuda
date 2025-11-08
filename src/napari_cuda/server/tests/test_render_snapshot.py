from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_cuda.server.data import SliceROI
from napari_cuda.server.runtime.render_loop.applying import (
    apply as snapshot_mod,
    plane as plane_mod,
)
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.server.scene.viewport import RenderMode, ViewportState
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.shared.dims_spec import AxisExtent, DimsSpec, DimsSpecAxis, dims_spec_to_payload


def _make_axes_spec(
    *,
    ndisplay: int,
    current_level: int,
    current_step: tuple[int, ...],
    level_shapes: tuple[tuple[int, ...], ...],
    order: tuple[int, ...] = (0, 1, 2),
    axis_labels: tuple[str, ...] = ("z", "y", "x"),
) -> DimsSpec:
    displayed = order[-ndisplay:]
    axes: list[DimsSpecAxis] = []
    for idx, label in enumerate(axis_labels):
        per_level_steps = tuple(shape[idx] for shape in level_shapes)
        per_level_world = tuple(
            AxisExtent(start=0.0, stop=float(max(0, count - 1)), step=1.0)
            for count in per_level_steps
        )
        axes.append(
            DimsSpecAxis(
                index=idx,
                label=label,
                role=label,
                displayed=idx in displayed,
                order_position=order.index(idx),
                current_step=current_step[idx],
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=per_level_steps,
                per_level_world=per_level_world,
            )
        )
    return DimsSpec(
        version=1,
        ndim=len(order),
        ndisplay=ndisplay,
        order=order,
        displayed=displayed,
        current_level=current_level,
        current_step=current_step,
        level_shapes=level_shapes,
        plane_mode=ndisplay < 3,
        axes=tuple(axes),
        levels=tuple({"index": idx, "shape": list(shape)} for idx, shape in enumerate(level_shapes)),
        labels=None,
    )


def _seed_ledger(worker: _StubWorker, spec: DimsSpec) -> None:
    ledger = worker._ledger
    ledger.record_confirmed("dims", "main", "current_step", spec.current_step, origin="test")
    ledger.record_confirmed("dims", "main", "dims_spec", dims_spec_to_payload(spec), origin="test")


class _NoopEvent:
    def connect(self, *_args, **_kwargs) -> None:
        return None

    def disconnect(self, *_args, **_kwargs) -> None:
        return None


class _FakeDescriptor:
    def __init__(self, shape: tuple[int, int, int], path: str | None = None) -> None:
        self.shape = tuple(int(dim) for dim in shape)
        self.path = path


class _FakeSource:
    def __init__(self) -> None:
        self.level_descriptors = [
            _FakeDescriptor((10, 10, 10)),
            _FakeDescriptor((5, 5, 5)),
            _FakeDescriptor((2, 2, 2)),
        ]
        self.dtype = "float32"
        self.axes = ("z", "y", "x")

    def ensure_contrast(self, level: int, **_kwargs) -> tuple[float, float]:
        _ = level
        return (0.0, 1.0)

    def level_scale(self, level: int) -> tuple[float, float, float]:
        _ = level
        return (1.0, 1.0, 1.0)


class _StubTurntableCamera:
    def __init__(self) -> None:
        self.center = (0.0, 0.0, 0.0)
        self.azimuth = 0.0
        self.elevation = 0.0
        self.roll = 0.0
        self.distance = 0.0
        self.fov = 45.0


class _StubPanZoomCamera:
    def __init__(self) -> None:
        self.rect = None
        self.center = (0.0, 0.0)
        self.zoom = 1.0


class _StubWorker:
    def __init__(self, *, use_volume: bool, level: int) -> None:
        mode = RenderMode.VOLUME if use_volume else RenderMode.PLANE
        self.viewport_state = ViewportState(mode=mode)
        self.viewport_state.plane.applied_level = level
        self.viewport_state.plane.target_level = level
        self.viewport_state.volume.level = level
        self.viewport_state.volume.scale = (1.0, 1.0, 1.0)
        self.viewport_state.volume.update_pose(
            center=(0.0, 0.0, 0.0),
            angles=(0.0, 0.0, 0.0),
            distance=0.0,
            fov=45.0,
        )
        self.configure_calls = 0
        self._scene_source = _FakeSource()
        self._volume_max_bytes = None
        self._volume_max_voxels = None
        self._hw_limits = SimpleNamespace(volume_max_bytes=None, volume_max_voxels=None)
        self._budget_error_cls = RuntimeError
        camera = _StubTurntableCamera() if use_volume else _StubPanZoomCamera()
        self.view = type("V", (), {"camera": camera})()
        self._viewport_runner = None
        self._level_metadata = None
        event = _NoopEvent()
        dims_events = SimpleNamespace(ndisplay=event, order=event)

        class _TestDims:
            def __init__(self) -> None:
                self.events = dims_events
                self._ndim = 0
                self._ndisplay = 2
                self._order: tuple[int, ...] = ()
                self._displayed: tuple[int, ...] = ()
                self.axis_labels: tuple[str, ...] = ()
                self.current_step: tuple[int, ...] = ()

            @property
            def ndim(self) -> int:
                return self._ndim

            @ndim.setter
            def ndim(self, value: int) -> None:
                self._ndim = int(value)

            @property
            def ndisplay(self) -> int:
                return self._ndisplay

            @ndisplay.setter
            def ndisplay(self, value: int) -> None:
                self._ndisplay = max(1, int(value))
                if self._order:
                    self._displayed = self._order[-self._ndisplay :]

            @property
            def order(self) -> tuple[int, ...]:
                return self._order

            @order.setter
            def order(self, value: tuple[int, ...]) -> None:
                self._order = tuple(int(v) for v in value)
                if self._order:
                    self._displayed = self._order[-self._ndisplay :]

            @property
            def displayed(self) -> tuple[int, ...]:
                return self._displayed

            @displayed.setter
            def displayed(self, value: tuple[int, ...]) -> None:
                self._displayed = tuple(int(v) for v in value)

        class _Viewer:
            def __init__(self) -> None:
                self.dims = _TestDims()

            def fit_to_view(self) -> None:
                return None

        self._viewer = _Viewer()
        class _Layer:
            def __init__(self) -> None:
                self.depiction = "plane"
                self.rendering = "mip"
                self.scale = (1.0, 1.0)
                self.visible = True

            def _set_view_slice(self) -> None:
                return None

        self._napari_layer = _Layer()
        self.pose_events: list[str] = []
        self._apply_dims_calls = 0
        self._roi_align_chunks = False
        self._roi_ensure_contains_viewport = False
        self._roi_edge_threshold = 0
        self._roi_pad_chunks = 0
        self._data_wh = (10, 10)
        self._data_d = 10
        self.width = 640
        self.height = 480
        self._log_layer_debug = False
        self._ledger = ServerStateLedger()
        self._plane_visual_handle = SimpleNamespace(visible=True)
        self._volume_visual_handle = SimpleNamespace(visible=False)

    def _ensure_scene_source(self) -> object:
        return self._scene_source

    def _configure_camera_for_mode(self) -> None:
        self.configure_calls += 1
        camera = (
            _StubTurntableCamera()
            if self.viewport_state.mode is RenderMode.VOLUME
            else _StubPanZoomCamera()
        )
        self.view.camera = camera

    def _ensure_plane_visual(self) -> Any:
        self._plane_visual_handle.visible = True
        self._volume_visual_handle.visible = False
        return self._plane_visual_handle

    def _ensure_volume_visual(self) -> Any:
        self._plane_visual_handle.visible = False
        self._volume_visual_handle.visible = True
        return self._volume_visual_handle

    def _estimate_level_bytes(self, source, level: int) -> tuple[int, int]:
        descriptor = source.level_descriptors[level]
        voxels = 1
        for dim in descriptor.shape:
            voxels *= int(dim)
        dtype_size = 4  # float32
        return voxels, voxels * dtype_size

    def _resolve_volume_intent_level(self, source, requested_level: int) -> int:
        descriptors = getattr(source, "level_descriptors", ())
        if not descriptors:
            return int(requested_level)
        coarsest = max(0, len(descriptors) - 1)
        return int(coarsest)

    def _update_level_metadata(self, descriptor, context) -> None:
        self._level_metadata = (descriptor, context)

    def _ledger_step(self):
        return self.viewport_state.plane.applied.step

    def _current_level_index(self) -> int:
        if self.viewport_state.mode is RenderMode.VOLUME:
            return int(self.viewport_state.volume.level)
        return int(self.viewport_state.plane.applied_level)

    def _set_current_level_index(self, value: int) -> None:
        level = int(value)
        self.viewport_state.plane.applied_level = level
        self.viewport_state.plane.target_level = level
        self.viewport_state.volume.level = level

    def _set_plane_level_index(self, value: int) -> None:
        level = int(value)
        self.viewport_state.plane.applied_level = level
        self.viewport_state.plane.target_level = level

    def _set_volume_level_index(self, value: int) -> None:
        level = int(value)
        self.viewport_state.volume.level = level

    @property
    def _volume_scale(self) -> tuple[float, float, float]:
        scale = self.viewport_state.volume.scale
        assert scale is not None
        return scale

    @_volume_scale.setter
    def _volume_scale(self, value: tuple[float, float, float]) -> None:
        self.viewport_state.volume.scale = tuple(float(component) for component in value)

    def _current_panzoom_rect(self):
        return self.viewport_state.plane.pose.rect

    def _emit_current_camera_pose(self, reason: str) -> None:
        self.pose_events.append(reason)


def _level_context(level: int, step: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        level=int(level),
        step=tuple(int(v) for v in step),
        scale_yx=(1.0, 1.0),
        contrast=(0.0, 1.0),
    )


def test_apply_viewport_plan_enters_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=0)
    snapshot_iface = RenderApplyInterface(worker)
    snapshot = RenderLedgerSnapshot(ndisplay=3)
    calls: list[str] = []

    monkeypatch.setattr(snapshot_mod, "reduce_level_update", lambda *_, **__: None)
    captured_plan: list[snapshot_mod.ViewportOps] = []
    captured_plan: list[snapshot_mod.ViewportOps] = []
    monkeypatch.setattr(
        snapshot_mod,
        "apply_volume_metadata",
        lambda *_args, **_kwargs: calls.append("metadata"),
    )
    monkeypatch.setattr(
        snapshot_mod,
        "apply_volume_level",
        lambda *_args, **_kwargs: calls.append("level"),
    )

    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.VOLUME,
        level_change=True,
        slice_task=None,
        slice_signature=None,
        level_context=_level_context(2, (0, 0, 0)),
        pose_event=None,
        zoom_hint=None,
        metadata_replay=True,
    )

    snapshot_mod.apply_viewport_plan(snapshot_iface, snapshot, plan, source=worker._scene_source)

    assert worker.viewport_state.mode is RenderMode.VOLUME
    assert calls == ["metadata", "level"]
    assert worker.configure_calls == 1


def test_apply_viewport_plan_skips_volume_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=True, level=2)
    snapshot_iface = RenderApplyInterface(worker)
    snapshot = RenderLedgerSnapshot(ndisplay=3)
    monkeypatch.setattr(snapshot_mod, "reduce_level_update", lambda *_, **__: None)
    monkeypatch.setattr(
        snapshot_mod,
        "apply_volume_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("metadata should not run")),
    )
    monkeypatch.setattr(
        snapshot_mod,
        "apply_volume_level",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("level should not run")),
    )

    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.VOLUME,
        level_change=False,
        slice_task=None,
        slice_signature=None,
        level_context=None,
        pose_event=None,
        zoom_hint=None,
        metadata_replay=False,
    )

    snapshot_mod.apply_viewport_plan(snapshot_iface, snapshot, plan, source=worker._scene_source)

    assert worker.viewport_state.mode is RenderMode.VOLUME
    assert worker.configure_calls == 0


def test_apply_viewport_plan_switches_to_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=True, level=2)
    snapshot_iface = RenderApplyInterface(worker)
    snapshot = RenderLedgerSnapshot(ndisplay=2)
    calls: list[str] = []

    monkeypatch.setattr(snapshot_mod, "reduce_level_update", lambda *_, **__: None)
    monkeypatch.setattr(
        snapshot_mod,
        "apply_plane_metadata",
        lambda *_args, **_kwargs: calls.append("plane-metadata"),
    )
    monkeypatch.setattr(
        snapshot_mod,
        "apply_slice_level",
        lambda *_args, **_kwargs: calls.append("slice-level"),
    )

    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.PLANE,
        level_change=True,
        slice_task=None,
        slice_signature=None,
        level_context=_level_context(1, (0, 0, 0)),
        pose_event=None,
        zoom_hint=None,
        metadata_replay=True,
    )

    snapshot_mod.apply_viewport_plan(snapshot_iface, snapshot, plan, source=worker._scene_source)

    assert worker.viewport_state.mode is RenderMode.PLANE
    assert worker.configure_calls == 1
    assert calls == ["plane-metadata", "slice-level"]


def test_apply_viewport_plan_plane_slice_task(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    snapshot_iface = RenderApplyInterface(worker)
    snapshot = RenderLedgerSnapshot(ndisplay=2)
    recorded: list[tuple[int, SliceROI]] = []

    monkeypatch.setattr(snapshot_mod, "reduce_level_update", lambda *_, **__: None)
    def _fake_slice_roi(snapshot_iface_obj, source, level, roi, *, update_contrast, step):
        recorded.append((level, roi))

    monkeypatch.setattr(snapshot_mod, "apply_slice_roi", _fake_slice_roi)

    slice_task = SliceTask(
        level=1,
        step=(0, 0, 0),
        roi=SliceROI(0, 4, 0, 4),
        chunk_shape=(4, 4),
        signature=(0, 0, 0, 0),
    )
    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.PLANE,
        level_change=False,
        slice_task=slice_task,
        slice_signature=None,
        level_context=None,
        pose_event=None,
        zoom_hint=None,
        metadata_replay=False,
    )

    snapshot_mod.apply_viewport_plan(snapshot_iface, snapshot, plan, source=worker._scene_source)

    assert recorded == [(1, SliceROI(0, 4, 0, 4))]


def test_apply_viewport_plan_updates_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    snapshot_iface = RenderApplyInterface(worker)
    snapshot = RenderLedgerSnapshot(ndisplay=2)
    recorded: list[tuple[int, RenderMode]] = []

    monkeypatch.setattr(
        snapshot_mod,
        "reduce_level_update",
        lambda ledger, *, level, mode, **_kwargs: recorded.append((level, mode)),
    )
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        RenderApplyInterface,
        "viewport_roi_for_level",
        lambda self, *_args, **_kwargs: SliceROI(0, 4, 0, 4),
    )

    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.PLANE,
        level_change=True,
        slice_task=None,
        slice_signature=None,
        level_context=_level_context(2, (1, 0, 0)),
        pose_event=None,
        zoom_hint=None,
        metadata_replay=True,
    )

    snapshot_mod.apply_viewport_plan(snapshot_iface, snapshot, plan, source=worker._scene_source)

    assert recorded == [(2, RenderMode.PLANE)]


def test_apply_render_snapshot_uses_planner_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    spec = _make_axes_spec(
        ndisplay=2,
        current_level=1,
        current_step=(0, 0, 0),
        level_shapes=((10, 10, 10), (5, 5, 5), (2, 2, 2)),
        order=(0, 1, 2),
        axis_labels=("z", "y", "x"),
    )
    _seed_ledger(worker, spec)
    snapshot = RenderLedgerSnapshot(ndisplay=2, dims_spec=spec)
    apply_calls: list[str] = []

    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.PLANE,
        level_change=False,
        slice_task=None,
        slice_signature=None,
        level_context=None,
        pose_event=None,
        zoom_hint=None,
        metadata_replay=False,
    )

    class _Runner:
        def __init__(self) -> None:
            self.requests: list[RenderLedgerSnapshot] = []

        def plan_from_snapshot(self, snapshot_obj, **_kwargs):
            self.requests.append(snapshot_obj)
            return plan

    worker._viewport_runner = _Runner()  # type: ignore[attr-defined]

    monkeypatch.setattr(snapshot_mod, "reduce_level_update", lambda *_, **__: None)

    def _fake_apply(snapshot_iface_obj, snapshot_obj, plan_obj, *, source):
        apply_calls.append("apply")
        assert snapshot_obj is snapshot
        assert plan_obj is plan

    monkeypatch.setattr(snapshot_mod, "apply_viewport_plan", _fake_apply)

    snapshot_mod.apply_render_snapshot(RenderApplyInterface(worker), snapshot)

    assert apply_calls == ["apply"]
    assert worker._viewport_runner.requests == [snapshot]  # type: ignore[attr-defined]


def test_apply_render_snapshot_replays_plane_mode_on_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=True, level=2)
    worker.viewport_state.mode = RenderMode.VOLUME
    plane_state = worker.viewport_state.plane
    plane_state.applied_level = 2
    plane_state.applied_step = (0, 0, 0)
    plane_state.applied_roi = SliceROI(0, 4, 0, 4)
    plane_state.applied_roi_signature = (0, 0, 0, 0)

    spec = _make_axes_spec(
        ndisplay=2,
        current_level=2,
        current_step=(0, 0, 0),
        level_shapes=((10, 10, 10), (5, 5, 5), (2, 2, 2)),
    )
    _seed_ledger(worker, spec)
    snapshot = RenderLedgerSnapshot(ndisplay=2, dims_spec=spec)

    calls: list[str] = []

    monkeypatch.setattr(snapshot_mod, "reduce_level_update", lambda *_, **__: None)

    captured_plan: list[snapshot_mod.ViewportOps] = []

    def _fake_apply_viewport_plan(snapshot_iface, snapshot_obj, plan_obj, *, source):
        captured_plan.append(plan_obj)

    monkeypatch.setattr(snapshot_mod, "apply_viewport_plan", _fake_apply_viewport_plan)

    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.PLANE,
        level_change=False,
        slice_task=None,
        slice_signature=None,
        level_context=None,
        pose_event=None,
        zoom_hint=None,
        metadata_replay=False,
    )

    class _Runner:
        def __init__(self):
            self.plans: list[RenderLedgerSnapshot] = []
            self.metadata_marks = 0

        def plan_from_snapshot(self, snapshot_obj, **_kwargs):
            self.plans.append(snapshot_obj)
            return plan

        def mark_plane_metadata_applied(self):
            self.metadata_marks += 1

        def mark_slice_applied(self, *_args, **_kwargs):
            return None

        def mark_level_applied(self, *_args, **_kwargs):
            return None

        def reset_for_volume(self):
            return None

    worker._viewport_runner = _Runner()  # type: ignore[attr-defined]

    snapshot_mod.apply_render_snapshot(RenderApplyInterface(worker), snapshot)

    assert isinstance(captured_plan[0], snapshot_mod.ViewportOps)
    assert captured_plan[0].mode is RenderMode.PLANE
    assert captured_plan[0].level_context is not None
    assert captured_plan[0].metadata_replay is True
    assert captured_plan[0].level_change is True
    # Mode switch and metadata marks occur during apply_viewport_plan; we stubbed it above.


def test_apply_render_snapshot_replays_volume_mode_on_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    worker.viewport_state.mode = RenderMode.PLANE
    worker.viewport_state.volume.level = 1
    worker.viewport_state.plane.applied_step = (0, 0, 0)

    spec = _make_axes_spec(
        ndisplay=3,
        current_level=1,
        current_step=(0, 0, 0),
        level_shapes=((10, 10, 10), (5, 5, 5), (2, 2, 2)),
    )
    _seed_ledger(worker, spec)
    snapshot = RenderLedgerSnapshot(ndisplay=3, dims_spec=spec)

    monkeypatch.setattr(snapshot_mod, "reduce_level_update", lambda *_, **__: None)
    captured_plan: list[snapshot_mod.ViewportOps] = []

    def _fake_apply_viewport_plan(snapshot_iface, snapshot_obj, plan_obj, *, source):
        captured_plan.append(plan_obj)

    monkeypatch.setattr(snapshot_mod, "apply_viewport_plan", _fake_apply_viewport_plan)

    plan = snapshot_mod.ViewportOps(
        mode=RenderMode.VOLUME,
        level_change=False,
        slice_task=None,
        slice_signature=None,
        level_context=None,
        pose_event=None,
        zoom_hint=None,
        metadata_replay=False,
    )

    class _Runner:
        def __init__(self):
            self.plans: list[RenderLedgerSnapshot] = []

        def plan_from_snapshot(self, snapshot_obj, **_kwargs):
            self.plans.append(snapshot_obj)
            return plan

        def mark_level_applied(self, *_args, **_kwargs):
            return None

        def mark_slice_applied(self, *_args, **_kwargs):
            return None

        def mark_plane_metadata_applied(self):
            return None

        def reset_for_volume(self):
            return None

    worker._viewport_runner = _Runner()  # type: ignore[attr-defined]

    snapshot_mod.apply_render_snapshot(RenderApplyInterface(worker), snapshot)

    assert isinstance(captured_plan[0], snapshot_mod.ViewportOps)
    assert captured_plan[0].mode is RenderMode.VOLUME
    assert captured_plan[0].level_context is not None
    assert captured_plan[0].metadata_replay is True
    assert captured_plan[0].level_change is True
    # Mode switch happens in apply_viewport_plan, which we replaced above.
