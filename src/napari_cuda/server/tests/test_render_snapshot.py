from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime import render_snapshot as snapshot_mod


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
        self.use_volume = use_volume
        self._active_ms_level = level
        self.configure_calls = 0
        self._scene_source = _FakeSource()
        self._volume_max_bytes = None
        self._volume_max_voxels = None
        self._hw_limits = SimpleNamespace(volume_max_bytes=None, volume_max_voxels=None)
        self._budget_error_cls = RuntimeError
        camera = _StubTurntableCamera() if use_volume else _StubPanZoomCamera()
        self.view = type("V", (), {"camera": camera})()

    def _ensure_scene_source(self) -> object:
        return self._scene_source

    def _configure_camera_for_mode(self) -> None:
        self.configure_calls += 1
        camera = _StubTurntableCamera() if self.use_volume else _StubPanZoomCamera()
        self.view.camera = camera

    def _estimate_level_bytes(self, source, level: int) -> tuple[int, int]:
        descriptor = source.level_descriptors[level]
        voxels = 1
        for dim in descriptor.shape:
            voxels *= int(dim)
        dtype_size = 4  # float32
        return voxels, voxels * dtype_size


def test_apply_snapshot_multiscale_enters_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    snapshot = RenderLedgerSnapshot(ndisplay=3, current_level=0, current_step=(5, 0, 0))
    calls: dict[str, object] = {}
    call_order: list[str] = []

    def _fake_build(_worker, _source, level, *, prev_level, ledger_step):
        calls["build"] = (level, prev_level, ledger_step)
        return SimpleNamespace(
            level=level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=ledger_step,
        )

    def _fake_volume(_worker, _source, context):
        calls["volume"] = context.level
        call_order.append("volume")

    monkeypatch.setattr(snapshot_mod, "_build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", _fake_volume)
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("slice apply should not run")))  # type: ignore[arg-type]

    original_configure = worker._configure_camera_for_mode

    def _wrapped_configure() -> None:
        call_order.append("configure")
        original_configure()

    worker._configure_camera_for_mode = _wrapped_configure  # type: ignore[assignment]

    snapshot_mod._apply_snapshot_multiscale(worker, snapshot)

    assert worker.use_volume is True
    assert worker.configure_calls == 1
    assert calls["build"] == (2, 1, (5, 0, 0))
    assert calls["volume"] == 2
    assert call_order == ["volume", "configure"]


def test_apply_snapshot_multiscale_stays_volume_skips_volume_load(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=True, level=2)
    snapshot = RenderLedgerSnapshot(ndisplay=3, current_level=2, current_step=(9, 0, 0))
    prepare_called = False
    volume_called = False

    def _fake_build(_worker, _source, level, *, prev_level, ledger_step):
        nonlocal prepare_called
        prepare_called = True
        return SimpleNamespace(
            level=level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=ledger_step,
        )

    def _fake_volume(_worker, _source, context):
        nonlocal volume_called
        volume_called = True

    monkeypatch.setattr(snapshot_mod, "_build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", _fake_volume)
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("slice apply should not run")))  # type: ignore[arg-type]

    snapshot_mod._apply_snapshot_multiscale(worker, snapshot)

    assert worker.use_volume is True
    assert worker.configure_calls == 0
    assert prepare_called is False  # no load performed
    assert volume_called is False


def test_apply_snapshot_multiscale_exit_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=True, level=2)
    snapshot = RenderLedgerSnapshot(ndisplay=2, current_level=1, current_step=(3, 0, 0))
    calls: dict[str, object] = {}

    def _fake_build(_worker, _source, level, *, prev_level, ledger_step):
        calls["build"] = (level, prev_level, ledger_step)
        return SimpleNamespace(
            level=level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=ledger_step,
        )

    def _fake_slice(_worker, _source, context):
        calls["slice"] = (context.level, context.step)

    monkeypatch.setattr(snapshot_mod, "_build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", _fake_slice)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("volume apply should not run")))  # type: ignore[arg-type]

    snapshot_mod._apply_snapshot_multiscale(worker, snapshot)

    assert worker.use_volume is False
    assert worker.configure_calls == 1
    assert calls["build"] == (1, 1, (3, 0, 0))
    assert calls["slice"] == (1, (3, 0, 0))


def test_apply_snapshot_multiscale_falls_back_to_budget_level(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _StubWorker(use_volume=False, level=1)
    worker._volume_max_voxels = 20  # force fallback from level 0
    snapshot = RenderLedgerSnapshot(ndisplay=3, current_level=0, current_step=(7, 0, 0))
    calls: dict[str, object] = {}

    def _fake_build(_worker, _source, level, *, prev_level, ledger_step):
        calls.setdefault("build", []).append((level, prev_level, ledger_step))
        return SimpleNamespace(
            level=level,
            scale_yx=(1.0, 1.0),
            contrast=(0.0, 1.0),
            step=ledger_step,
        )

    def _fake_volume(_worker, _source, context):
        calls.setdefault("volume", []).append(context.level)

    monkeypatch.setattr(snapshot_mod, "_build_level_context", _fake_build)
    monkeypatch.setattr(snapshot_mod, "apply_volume_level", _fake_volume)
    monkeypatch.setattr(snapshot_mod, "apply_slice_level", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("slice apply should not run")))  # type: ignore[arg-type]

    snapshot_mod._apply_snapshot_multiscale(worker, snapshot)

    assert worker.use_volume is True
    assert worker.configure_calls == 1
    assert calls["build"] == [(2, 1, (7, 0, 0))]
    assert calls["volume"] == [2]
