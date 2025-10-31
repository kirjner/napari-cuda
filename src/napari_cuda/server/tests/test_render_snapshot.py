from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_cuda.server.runtime.core.snapshot_build import RenderLedgerSnapshot
from napari_cuda.server.runtime.snapshots import apply as snapshot_mod
from napari_cuda.server.runtime.snapshots import plane as plane_mod
from napari_cuda.server.runtime.snapshots.interface import SnapshotInterface
from napari_cuda.server.runtime.data import SliceROI
from napari_cuda.server.runtime.viewport import RenderMode, ViewportState


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
        self._last_dims_signature: tuple[int, ...] | None = None
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
        self._viewer = SimpleNamespace(dims=SimpleNamespace(events=SimpleNamespace(ndisplay=event, order=event)))
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

    def _estimate_level_bytes(self, source, level: int) -> tuple[int, int]:
        descriptor = source.level_descriptors[level]
        voxels = 1
        for dim in descriptor.shape:
            voxels *= int(dim)
        dtype_size = 4  # float32
        return voxels, voxels * dtype_size

    def _resolve_volume_intent_level(self, source, requested_level: int) -> tuple[int, bool]:
        descriptors = getattr(source, "level_descriptors", ())
        if not descriptors:
            return int(requested_level), False
        coarsest = max(0, len(descriptors) - 1)
        downgraded = coarsest != int(requested_level)
        return int(coarsest), downgraded

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


def test_apply_snapshot_multiscale_enters_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    snapshot = RenderLedgerSnapshot(ndisplay=3, current_level=0, current_step=(5, 0, 0))
    calls: dict[str, object] = {}
    call_order: list[str] = []

    def _fake_build(decision, *, source, prev_level, last_step):
        calls["build"] = (decision.selected_level, prev_level, last_step)
        return SimpleNamespace(
            level=decision.selected_level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=last_step,
        )

    def _fake_volume(_snapshot_iface, _source, context, *, downgraded: bool):
        calls["volume"] = context.level
        call_order.append("volume")
        calls["downgraded"] = downgraded

    monkeypatch.setattr(snapshot_mod.lod, "build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", _fake_volume)
    monkeypatch.setattr(
        snapshot_mod,
        "apply_slice_level",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("slice apply should not run")),
    )  # type: ignore[arg-type]
    monkeypatch.setattr(
        SnapshotInterface,
        "viewport_roi_for_level",
        lambda self, *_args, **_kwargs: SliceROI(0, 10, 0, 10),
    )

    original_configure = worker._configure_camera_for_mode

    def _wrapped_configure() -> None:
        call_order.append("configure")
        original_configure()

    worker._configure_camera_for_mode = _wrapped_configure  # type: ignore[assignment]

    snapshot_iface = SnapshotInterface(worker)
    ops = snapshot_mod._resolve_snapshot_ops(snapshot_iface, snapshot)
    snapshot_mod._apply_snapshot_ops(snapshot_iface, snapshot, ops)

    assert worker.viewport_state.mode is RenderMode.VOLUME
    assert worker.configure_calls == 1
    assert calls["build"] == (2, 1, (5, 0, 0))
    assert calls["volume"] == 2
    assert call_order == ["volume", "configure"]
    assert worker.viewport_state.mode is RenderMode.VOLUME


def test_apply_snapshot_multiscale_stays_volume_skips_volume_load(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=True, level=2)
    snapshot = RenderLedgerSnapshot(ndisplay=3, current_level=2, current_step=(9, 0, 0))
    prepare_called = False
    volume_called = False

    def _fake_build(decision, *, source, prev_level, last_step):
        nonlocal prepare_called
        prepare_called = True
        return SimpleNamespace(
            level=decision.selected_level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=last_step,
        )

    def _fake_volume(_worker, _source, context, *, downgraded: bool):
        nonlocal volume_called
        volume_called = True

    monkeypatch.setattr(snapshot_mod.lod, "build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", _fake_volume)
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("slice apply should not run")))  # type: ignore[arg-type]

    snapshot_iface = SnapshotInterface(worker)
    ops = snapshot_mod._resolve_snapshot_ops(snapshot_iface, snapshot)
    snapshot_mod._apply_snapshot_ops(snapshot_iface, snapshot, ops)

    assert worker.viewport_state.mode is RenderMode.VOLUME
    assert worker.configure_calls == 0
    assert prepare_called is False  # no load performed
    assert volume_called is False
    assert worker.viewport_state.mode is RenderMode.VOLUME


def test_apply_snapshot_multiscale_exit_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=True, level=2)
    snapshot = RenderLedgerSnapshot(ndisplay=2, current_level=1, current_step=(3, 0, 0))
    calls: dict[str, object] = {}

    def _fake_build(decision, *, source, prev_level, last_step):
        calls["build"] = (decision.selected_level, prev_level, last_step)
        return SimpleNamespace(
            level=decision.selected_level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=last_step,
        )

    def _fake_slice(_snapshot_iface, _source, context):
        calls["slice"] = (context.level, context.step)

    monkeypatch.setattr(snapshot_mod.lod, "build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", _fake_slice)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("volume apply should not run")))  # type: ignore[arg-type]
    monkeypatch.setattr(
        SnapshotInterface,
        "viewport_roi_for_level",
        lambda self, *_args, **_kwargs: SliceROI(0, 10, 0, 10),
    )

    snapshot_iface = SnapshotInterface(worker)
    ops = snapshot_mod._resolve_snapshot_ops(snapshot_iface, snapshot)
    snapshot_mod._apply_snapshot_ops(snapshot_iface, snapshot, ops)

    assert worker.viewport_state.mode is RenderMode.PLANE
    assert worker.configure_calls == 1
    assert calls["build"] == (1, 1, (3, 0, 0))
    assert calls["slice"] == (1, (3, 0, 0))
    assert worker.viewport_state.mode is RenderMode.PLANE


def test_apply_snapshot_multiscale_falls_back_to_budget_level(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    worker._volume_max_voxels = 20  # force fallback from level 0
    snapshot = RenderLedgerSnapshot(ndisplay=3, current_level=0, current_step=(7, 0, 0))
    calls: dict[str, object] = {}

    def _fake_build(decision, *, source, prev_level, last_step):
        calls.setdefault("build", []).append((decision.selected_level, prev_level, last_step))
        return SimpleNamespace(
            level=decision.selected_level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=last_step,
        )

    def _fake_volume(_snapshot_iface, _source, context, *, downgraded: bool):
        calls.setdefault("volume", []).append((context.level, downgraded))

    monkeypatch.setattr(snapshot_mod.lod, "build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", _fake_volume)
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("slice apply should not run")))  # type: ignore[arg-type]
    monkeypatch.setattr(
        SnapshotInterface,
        "viewport_roi_for_level",
        lambda self, *_args, **_kwargs: SliceROI(0, 10, 0, 10),
    )

    snapshot_iface = SnapshotInterface(worker)
    ops = snapshot_mod._resolve_snapshot_ops(snapshot_iface, snapshot)
    snapshot_mod._apply_snapshot_ops(snapshot_iface, snapshot, ops)

    assert worker.viewport_state.mode is RenderMode.VOLUME
    assert worker.configure_calls == 1
    assert calls["build"] == [(2, 1, (7, 0, 0))]
    assert calls["volume"] == [(2, True)]


def test_apply_render_snapshot_short_circuits_on_matching_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    snapshot = RenderLedgerSnapshot(ndisplay=2, current_level=1, current_step=(5, 0, 0))

    original_apply_dims = plane_mod.apply_dims_from_snapshot

    def _track_apply_dims(snapshot_iface_obj, snapshot_obj, *, signature):
        snapshot_iface_obj._worker._apply_dims_calls += 1  # type: ignore[attr-defined]
        original_apply_dims(snapshot_iface_obj, snapshot_obj, signature=signature)

    monkeypatch.setattr(plane_mod, "apply_dims_from_snapshot", _track_apply_dims)
    monkeypatch.setattr(
        SnapshotInterface,
        "viewport_roi_for_level",
        lambda self, *_args, **_kwargs: SliceROI(0, 10, 0, 10),
    )

    snapshot_iface = SnapshotInterface(worker)
    ops = snapshot_mod._resolve_snapshot_ops(snapshot_iface, snapshot)
    snapshot_iface.set_last_snapshot_signature(ops["signature"])
    snapshot_iface.set_last_dims_signature(plane_mod.dims_signature(snapshot))

    def _fail_apply(*_args, **_kwargs) -> None:
        raise AssertionError("apply_snapshot_ops should not be invoked when signature matches")

    monkeypatch.setattr(snapshot_mod, "_apply_snapshot_ops", _fail_apply)

    snapshot_mod.apply_render_snapshot(snapshot_iface, snapshot)
    assert worker._apply_dims_calls == 0
