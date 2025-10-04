from __future__ import annotations

import math
from types import MethodType, SimpleNamespace
from typing import Mapping

import numpy as np
import pytest

from napari_cuda.server.app.config import LevelPolicySettings, ServerConfig, ServerCtx
from napari_cuda.server.state.scene_state import ServerSceneState
from napari_cuda.server.rendering.display_mode import apply_ndisplay_switch
from napari_cuda.server.state.server_scene import ServerSceneCommand


class _IdentityTransform:
    """Minimal transform stub mirroring VisPy API used in tests."""

    matrix = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    def __mul__(self, other: object) -> "_IdentityTransform":
        return self

    def imap(self, coords):  # type: ignore[no-untyped-def]
        return coords


class _FakeCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.size = (int(width), int(height))

    def render(self) -> None:  # pragma: no cover - render is a noop stub
        return


class _FakeCamera2D:
    def __init__(self, aspect: float | None = None) -> None:
        self.center = (0.0, 0.0)
        self.angles = (0.0, 0.0, 0.0)
        self.zoom_factor = 1.0
        self.zoom_calls: list[tuple[float, tuple[float, float] | None]] = []
        self.set_range_calls: list[dict[str, tuple[float, float] | None]] = []
        self.azimuth = 0.0
        self.elevation = 0.0
        self.aspect = aspect

    def zoom(self, factor: float, center: tuple[float, float] | None = None) -> None:
        self.zoom_factor *= float(factor)
        self.zoom_calls.append((float(factor), center))
        if center is not None:
            self.center = (float(center[0]), float(center[1]))

    def set_range(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.set_range_calls.append({str(k): v for k, v in kwargs.items()})

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (0, 0, 1, 1)

    @property
    def scale(self) -> tuple[float, float]:
        return (1.0, 1.0)

    @property
    def _viewbox(self) -> SimpleNamespace:
        return SimpleNamespace(size=(320.0, 180.0))

    def get_state(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
        return {
            "center": tuple(self.center),
            "angles": tuple(self.angles),
            "zoom_factor": float(self.zoom_factor),
            "azimuth": float(self.azimuth),
            "elevation": float(self.elevation),
        }

    def set_state(self, state: Mapping[str, object]) -> None:  # type: ignore[no-untyped-def]
        if "center" in state:
            center = state["center"]
            self.center = tuple(float(v) for v in center)  # type: ignore[assignment]
        if "angles" in state:
            angles = state["angles"]
            self.angles = tuple(float(v) for v in angles)
        if "zoom_factor" in state:
            self.zoom_factor = float(state["zoom_factor"])
        if "azimuth" in state:
            self.azimuth = float(state["azimuth"])
        if "elevation" in state:
            self.elevation = float(state["elevation"])


class _FakeCamera3D(_FakeCamera2D):
    def __init__(self, elevation: float = 30.0, azimuth: float = 30.0, fov: float = 60.0) -> None:
        super().__init__()
        self.elevation = float(elevation)
        self.azimuth = float(azimuth)
        self.fov = float(fov)
        self.center = (0.0, 0.0, 0.0)
        self.distance = 1.0
        self.set_range_calls_3d: list[dict[str, tuple[float, float]]] = []

    def set_range(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.set_range_calls_3d.append({str(k): v for k, v in kwargs.items()})


class _FakeScene:
    transform = _IdentityTransform()


class _FakeView:
    def __init__(self, camera: _FakeCamera2D) -> None:
        self.camera = camera
        self.scene = _FakeScene()
        self.transform = _IdentityTransform()


class _FakeDims:
    def __init__(self) -> None:
        self.current_step: tuple[int, ...] = (0, 0, 0)
        self.range: tuple = ()
        self.ndisplay: int = 2
        self.ndim: int = 3
        self.order: tuple[int, ...] = (0, 1, 2)
        self.axis_labels: tuple[str, ...] = ("z", "y", "x")
        self.nsteps: tuple[int, ...] = (256, 256, 256)


class _FakeViewer:
    def __init__(self) -> None:
        self.dims = _FakeDims()


class _FakeLayer:
    def __init__(self) -> None:
        self.visible = False
        self.opacity = 0.0
        self.blending = "opaque"
        self.contrast_limits = [0.0, 1.0]
        self.applied: list[tuple[np.ndarray, object, bool]] = []
        self.ndim = 3
        self.data = SimpleNamespace(ndim=3)
        self.translate = (0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)


class _FakeVisual:
    def __init__(self) -> None:
        self.data = None
        self.cmap = None
        self.clim = None
        self.opacity = 1.0
        self.visible = True

    def set_data(self, data: np.ndarray) -> None:
        self.data = data


class _FakeSceneSource:
    def __init__(self) -> None:
        self.axes = ("z", "y", "x")
        self.current_step: tuple[int, ...] = (0, 0, 0)
        self.current_level = 1
        self.dtype = "float32"
        self.level_descriptors = [
            SimpleNamespace(shape=(1, 4, 4), path=None, index=i) for i in range(3)
        ]

    def level_shape(self, level: int) -> tuple[int, int, int]:
        return (1, 4, 4)

    def level_volume(self, level: int, *, compute: bool = True):
        return np.zeros((1, 4, 4), dtype=np.float32)

    def set_current_slice(self, step: tuple[int, ...], level: int) -> tuple[int, ...]:
        self.current_level = int(level)
        self.current_step = tuple(int(s) for s in step)
        return self.current_step

    def level_index_for_path(self, path: str | None) -> int:
        return 0

    def level_scale(self, level: int) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)

    def get_level(self, level: int) -> SimpleNamespace:
        return SimpleNamespace(
            chunks=(1, 4, 4),
            dtype=self.dtype,
            shape=(1, 4, 4),
        )

    def ensure_contrast(self, *, level: int) -> tuple[float, float]:
        return (0.0, 1.0)


@pytest.fixture()
def render_worker_fixture(monkeypatch) -> "napari_cuda.server.runtime.egl_worker.EGLRendererWorker":
    from napari_cuda.server.runtime import egl_worker as rw
    from napari_cuda.server.state import scene_state_applier as ssa

    class _DummyEglContext:
        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height

        def ensure(self) -> None:
            return

        def cleanup(self) -> None:
            return

    class _DummyCaptureFacade:
        def __init__(self, *, width: int, height: int) -> None:
            self.width = width
            self.height = height
            self.texture_id = 1

            class _DummyPipeline:
                def __init__(self) -> None:
                    self._debug = None
                    self._enc_fmt = "NV12"
                    self._raw_budget = 0

                def set_debug(self, debug):  # type: ignore[no-untyped-def]
                    self._debug = debug

                def set_raw_dump_budget(self, budget: int) -> None:  # type: ignore[no-untyped-def]
                    self._raw_budget = int(budget)

                @property
                def enc_input_format(self) -> str:
                    return self._enc_fmt

                def set_enc_input_format(self, fmt: str) -> None:
                    self._enc_fmt = str(fmt)

                def capture_blit_gpu_ns(self):  # type: ignore[no-untyped-def]
                    return 0

                def map_and_copy_to_torch(self, debug_cb=None):  # type: ignore[no-untyped-def]
                    return 0.0, 0.0

                def convert_for_encoder(self):  # type: ignore[no-untyped-def]
                    return SimpleNamespace(device="cpu"), 0.0

            class _DummyCuda:
                def set_force_tight_pitch(self, enabled: bool) -> None:  # type: ignore[no-untyped-def]
                    return

            self.pipeline = _DummyPipeline()
            self.cuda = _DummyCuda()

        def ensure(self) -> None:
            return

        def initialize_cuda_interop(self) -> None:
            return

        def cleanup(self) -> None:
            return

    class _DummyViewerBuilder:
        def __init__(self, worker):  # type: ignore[no-untyped-def]
            self._worker = worker

        def build(self, source):  # type: ignore[no-untyped-def]
            canvas = _FakeCanvas(self._worker.width, self._worker.height)
            view = _FakeView(_FakeCamera2D())
            viewer = _FakeViewer()
            return canvas, view, viewer

    class _DummyEncoder:
        def __init__(self, width: int, height: int, fps_hint: int) -> None:
            self.width = width
            self.height = height
            self.fps_hint = fps_hint
            self.input_format = "NV12"
            self.is_ready = True
            self.frame_index = 0

        def set_fps_hint(self, fps: int) -> None:
            self.fps_hint = fps

        def setup(self, ctx) -> None:  # type: ignore[no-untyped-def]
            self.is_ready = True

        def reset(self, ctx) -> None:  # type: ignore[no-untyped-def]
            self.is_ready = True

        def shutdown(self) -> None:
            self.is_ready = False

        def force_idr(self) -> None:
            return

        def request_idr(self) -> None:
            return

        def encode(self, frame):  # type: ignore[no-untyped-def]
            self.frame_index += 1
            timings = SimpleNamespace(encode_ms=0.0, pack_ms=0.0)
            return SimpleNamespace(), timings

    def _fake_plane_scale_for_level(source, level):  # type: ignore[no-untyped-def]
        return 1.0, 1.0

    def _fake_apply_slice_to_layer(ctx, *, source, slab, roi, update_contrast):  # type: ignore[no-untyped-def]
        layer = ctx.layer
        layer.applied.append((slab, roi, bool(update_contrast)))
        return 1.0, 1.0

    monkeypatch.setattr(rw.scene.cameras, "PanZoomCamera", _FakeCamera2D)
    monkeypatch.setattr(rw.scene.cameras, "TurntableCamera", _FakeCamera3D)
    monkeypatch.setattr(rw, "EglContext", _DummyEglContext)
    monkeypatch.setattr(rw, "CaptureFacade", _DummyCaptureFacade)
    monkeypatch.setattr(rw, "ViewerBuilder", _DummyViewerBuilder)
    monkeypatch.setattr(rw, "Encoder", _DummyEncoder)
    monkeypatch.setattr(rw, "plane_scale_for_level", _fake_plane_scale_for_level)
    monkeypatch.setattr(ssa.SceneStateApplier, "apply_slice_to_layer", staticmethod(_fake_apply_slice_to_layer))

    monkeypatch.setattr(rw.EGLRendererWorker, "_init_cuda", lambda self: None)
    monkeypatch.setattr(rw.EGLRendererWorker, "_init_vispy_scene", lambda self: None)
    monkeypatch.setattr(rw.EGLRendererWorker, "_init_egl", lambda self: None)
    monkeypatch.setattr(rw.EGLRendererWorker, "_init_capture", lambda self: None)
    monkeypatch.setattr(rw.EGLRendererWorker, "_init_cuda_interop", lambda self: None)
    monkeypatch.setattr(rw.EGLRendererWorker, "_init_encoder", lambda self: None)

    ctx = ServerCtx(
        cfg=ServerConfig(width=320, height=180),
        policy=LevelPolicySettings(
            threshold_in=1.05,
            threshold_out=1.35,
            hysteresis=0.0,
            fine_threshold=1.05,
            cooldown_ms=50.0,
            log_policy_eval=False,
            preserve_view_on_switch=True,
            sticky_contrast=True,
        ),
    )

    worker = rw.EGLRendererWorker(width=320, height=180, ctx=ctx)
    worker._init_cuda()
    worker._init_vispy_scene()
    worker._init_egl()
    worker._init_capture()
    worker._init_cuda_interop()
    worker._init_encoder()
    worker._is_ready = True
    worker.canvas = _FakeCanvas(worker.width, worker.height)
    camera = _FakeCamera2D()
    worker.view = _FakeView(camera)
    worker._viewer = _FakeViewer()
    worker._napari_layer = _FakeLayer()
    worker._visual = _FakeVisual()
    worker._scene_source = _FakeSceneSource()
    worker._zarr_path = "memory://dataset"
    worker._scene_refresh_cb = None
    worker._last_roi = None
    worker._lock_level = None
    worker._active_ms_level = 1
    worker._last_level_switch_ts = 0.0

    def _fake_ensure_scene_source(self):  # type: ignore[no-untyped-def]
        return self._scene_source

    def _fake_oversampling(self, source, level):  # type: ignore[no-untyped-def]
        return {0: 1.4, 1: 1.0, 2: 0.6}.get(int(level), 1.0)

    def _fake_load_slice(self, source, level, z_index):  # type: ignore[no-untyped-def]
        return np.full((4, 4), float(z_index), dtype=np.float32)

    worker._ensure_scene_source = MethodType(_fake_ensure_scene_source, worker)
    worker._oversampling_for_level = MethodType(_fake_oversampling, worker)
    worker._load_slice = MethodType(_fake_load_slice, worker)

    return worker


def test_zoom_hint_reaches_lod_selector(render_worker_fixture, monkeypatch):
    from napari_cuda.server.runtime import egl_worker as rw

    worker = render_worker_fixture
    ratios: list[float | None] = []

    def _fake_select_level(config, inputs):  # type: ignore[no-untyped-def]
        ratios.append(inputs.zoom_ratio)
        return SimpleNamespace(
            should_switch=False,
            blocked_reason=None,
            selected_level=inputs.current_level,
            desired_level=inputs.current_level,
            action="hold",
            cooldown_remaining_ms=0.0,
        )

    monkeypatch.setattr(rw, "select_level", _fake_select_level)

    worker.process_camera_commands([ServerSceneCommand(kind="zoom", factor=2.0)])
    assert worker._level_policy_refresh_needed is True

    worker._evaluate_level_policy()

    assert ratios, "select_level was not invoked"
    assert math.isclose(float(ratios[-1]), 0.5, rel_tol=1e-6)
    assert worker._render_mailbox.consume_zoom_hint(max_age=0.5) is None


def test_preserve_view_switch_keeps_camera_range(render_worker_fixture):
    worker = render_worker_fixture
    camera: _FakeCamera2D = worker.view.camera  # type: ignore[assignment]
    camera.set_range_calls.clear()
    worker._render_tick_required = False

    worker._render_mailbox.enqueue_scene_state(ServerSceneState(current_step=(2, 0, 0)))
    worker.drain_scene_updates()

    assert camera.set_range_calls == []
    assert worker._z_index == 2
    assert worker._last_step == (2, 0, 0)
    assert worker._render_tick_required is True
    assert tuple(worker._data_wh) == (4, 4)


def test_preserve_view_disabled_resets_camera(render_worker_fixture):
    worker = render_worker_fixture
    worker._preserve_view_on_switch = False
    camera: _FakeCamera2D = worker.view.camera  # type: ignore[assignment]
    camera.set_range_calls.clear()
    worker._render_tick_required = False

    worker._render_mailbox.enqueue_scene_state(ServerSceneState(current_step=(1, 0, 0)))
    worker.drain_scene_updates()

    assert camera.set_range_calls, "camera.set_range should be invoked when preserve-view is disabled"


def test_layer_updates_drive_napari_layer(render_worker_fixture):
    worker = render_worker_fixture
    worker._render_tick_required = False

    worker.apply_state(
        ServerSceneState(
            layer_updates={
                "layer-0": {
                    "colormap": "red",
                    "gamma": 0.5,
                }
            }
        )
    )

    worker.drain_scene_updates()

    layer = worker._napari_layer
    cmap = getattr(layer, "colormap", None)
    assert getattr(cmap, "name", cmap) == "red"
    if worker._visual is not None:
        vis_cmap = getattr(worker._visual, "cmap", None)
        from napari._vispy.layers.image import _napari_cmap_to_vispy  # type: ignore
        from napari.utils.colormaps.colormap_utils import ensure_colormap

        expected = _napari_cmap_to_vispy(ensure_colormap("red"))
        assert vis_cmap is not None
        assert np.allclose(expected.colors, vis_cmap.colors)
    assert math.isclose(float(getattr(layer, "gamma", 0.0)), 0.5, rel_tol=1e-6)
    assert worker._render_tick_required is True


def test_zoom_hint_triggers_level_switch_end_to_end(render_worker_fixture):
    worker = render_worker_fixture

    def _oversampling(self, source, level):  # type: ignore[no-untyped-def]
        return {0: 0.8, 1: 1.0, 2: 1.6}.get(int(level), 1.0)

    worker._oversampling_for_level = MethodType(_oversampling, worker)
    worker._napari_layer.applied.clear()

    worker.process_camera_commands([ServerSceneCommand(kind="zoom", factor=2.0)])
    worker._evaluate_level_policy()

    assert worker._active_ms_level == 0
    assert worker._napari_layer.applied, "slice should be applied when level changes"
    assert worker._render_mailbox.consume_zoom_hint(max_age=0.5) is None
    assert worker._last_level_switch_ts > 0.0


def test_render_tick_preserve_view_smoke(render_worker_fixture):
    worker = render_worker_fixture
    camera: _FakeCamera2D = worker.view.camera  # type: ignore[assignment]
    camera.set_range_calls.clear()
    worker._napari_layer.applied.clear()
    worker._last_roi = None

    elapsed_ms = worker.render_tick()

    assert elapsed_ms >= 0.0
    assert worker._napari_layer.applied, "render_tick should apply the latest slab"
    assert worker._last_roi is not None
    assert camera.set_range_calls == []


def test_ndisplay_switch_to_volume_pins_coarsest_level(render_worker_fixture, monkeypatch):
    worker = render_worker_fixture
    worker._last_step = (0, 0, 0)

    recorded: dict[str, object] = {}

    def _capture_switch(worker_param, *, target_level, reason, requested_level, selected_level, source=None, budget_error=None, restoring_plane_state=False):  # type: ignore[no-untyped-def]
        recorded.update(
            target_level=int(target_level),
            reason=reason,
            requested_level=int(requested_level) if requested_level is not None else None,
            selected_level=int(selected_level) if selected_level is not None else None,
            restoring_plane_state=bool(restoring_plane_state),
        )
        worker_param._active_ms_level = int(target_level)  # type: ignore[attr-defined]

    monkeypatch.setattr("napari_cuda.server.rendering.display_mode.perform_level_switch", _capture_switch)

    worker._render_tick_required = False
    worker._level_policy_refresh_needed = True
    apply_ndisplay_switch(worker, 3)

    assert worker.use_volume is True
    assert recorded.get("target_level") == 2
    assert recorded.get("reason") == "ndisplay-3d"
    assert recorded.get("restoring_plane_state") is True
    assert worker._level_policy_refresh_needed is False
    assert worker._render_tick_required is True
    assert worker._viewer is not None and worker._viewer.dims.ndisplay == 3
    assert tuple(worker._viewer.dims.current_step)[:3] == (0, 0, 0)
    assert worker._z_index == 0


def test_ndisplay_switch_back_to_plane_resumes_policy(render_worker_fixture, monkeypatch):
    worker = render_worker_fixture
    worker.use_volume = True
    worker._viewer.dims.ndisplay = 3

    calls: list[None] = []

    monkeypatch.setattr(worker, "_evaluate_level_policy", lambda: calls.append(None))

    worker._render_tick_required = False
    apply_ndisplay_switch(worker, 2)

    assert worker.use_volume is False
    assert worker._viewer.dims.ndisplay == 2
    assert calls, "Expected policy evaluation after returning to 2D"
    assert worker._level_policy_refresh_needed is False
    assert worker._render_tick_required is True
    assert tuple(worker._viewer.dims.current_step)[:2] == (0, 0)


def test_ndisplay_switch_notifies_scene_refresh(render_worker_fixture):
    worker = render_worker_fixture
    captured_steps: list[tuple[int, ...]] = []

    worker._scene_refresh_cb = captured_steps.append  # type: ignore[assignment]
    worker._last_step = (1, 2, 3)
    worker._render_tick_required = False

    apply_ndisplay_switch(worker, 3)

    assert worker._render_tick_required is True
    assert captured_steps, "Scene refresh callback was not invoked"
    from napari_cuda.server.state.plane_restore_state import PlaneRestoreState

    resolved_steps: list[tuple[int, ...]] = []
    for entry in captured_steps:
        if isinstance(entry, tuple):
            resolved_steps.append(entry)
        elif isinstance(entry, PlaneRestoreState):
            resolved_steps.append(entry.step)
    if resolved_steps:
        assert resolved_steps[-1] == worker._last_step
    else:
        assert worker._plane_restore_state is not None


def test_ndisplay_toggle_restores_plane_step(render_worker_fixture):
    worker = render_worker_fixture
    worker._viewer.dims.current_step = (4, 0, 0)
    worker._last_step = (4, 0, 0)
    worker._active_ms_level = 2
    worker._plane_restore_state = None
    worker._pending_plane_restore = None

    apply_ndisplay_switch(worker, 3)
    assert worker.use_volume is True
    assert worker._plane_restore_state is not None
    assert worker._plane_restore_state.step == (4, 0, 0)
    assert worker._plane_restore_state.level == 2

    worker._render_tick_required = False
    worker._last_step = (0, 0, 0)
    apply_ndisplay_switch(worker, 2)

    assert worker.use_volume is False
    restored = tuple(worker._viewer.dims.current_step)
    assert restored[:3] == (4, 0, 0)
    assert worker._last_step == restored
    assert worker._z_index == 4
