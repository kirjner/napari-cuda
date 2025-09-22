from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from napari_cuda.server.config import LevelPolicySettings, ServerConfig, ServerCtx
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.state_machine import CameraCommand


class _IdentityTransform:
    """Minimal transform stub mirroring VisPy API used in tests."""

    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
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
    def __init__(self) -> None:
        self.center = (0.0, 0.0)
        self.angles = (0.0, 0.0, 0.0)
        self.zoom_factor = 1.0
        self.zoom_calls: list[tuple[float, tuple[float, float] | None]] = []
        self.set_range_calls: list[dict[str, tuple[float, float] | None]] = []
        self.azimuth = 0.0
        self.elevation = 0.0

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


class _FakeVisual:
    def __init__(self) -> None:
        self.data = None

    def set_data(self, data: np.ndarray) -> None:
        self.data = data


class _FakeSceneSource:
    def __init__(self) -> None:
        self.axes = ("z", "y", "x")
        self.current_step: tuple[int, ...] = (0, 0, 0)
        self.current_level = 1
        self.dtype = "float32"
        self.level_descriptors = [
            SimpleNamespace(shape=(1, 4, 4), path=None) for _ in range(3)
        ]

    def level_shape(self, level: int) -> tuple[int, int, int]:
        return (1, 4, 4)

    def set_current_level(self, level: int, step: tuple[int, ...]):
        self.current_level = int(level)
        self.current_step = tuple(int(s) for s in step)
        return SimpleNamespace(level=int(level), step=self.current_step)

    def level_index_for_path(self, path: str | None) -> int:
        return 0


@pytest.fixture()
def egl_worker_fixture(monkeypatch) -> "napari_cuda.server.egl_worker.EGLRendererWorker":
    from napari_cuda.server import egl_worker as ew
    from napari_cuda.server import scene_state_applier as ssa

    class _DummyEglContext:
        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height

        def ensure(self) -> None:
            return

        def cleanup(self) -> None:
            return

    class _DummyGLCapture:
        def __init__(self, width: int, height: int) -> None:
            self.texture_id = 1

        def ensure(self) -> None:
            return

        def cleanup(self) -> None:
            return

        def blit_with_timing(self) -> int:
            return 0

    class _DummyCudaInterop:
        def __init__(self, width: int, height: int) -> None:
            self.torch_frame = SimpleNamespace(device="cpu")

        def map_and_copy(self, debug_cb=None):  # type: ignore[no-untyped-def]
            return 0.0, 0.0

        def cleanup(self) -> None:
            return

        def initialize(self, texture_id: int) -> None:
            return

    class _DummyFramePipeline:
        def __init__(self, *, gl_capture, cuda, width, height, debug=None):  # type: ignore[no-untyped-def]
            self._debug = debug
            self.orientation_ready = True
            self.black_reset_done = False

        def set_debug(self, debug):  # type: ignore[no-untyped-def]
            self._debug = debug

        def capture_blit_gpu_ns(self):  # type: ignore[no-untyped-def]
            return 0

        def map_and_copy_to_torch(self, debug_cb=None):  # type: ignore[no-untyped-def]
            return 0.0, 0.0

        def convert_for_encoder(self, *, reset_camera=None):  # type: ignore[no-untyped-def]
            return SimpleNamespace(device="cpu"), 0.0

        def set_dimensions(self, width: int, height: int) -> None:
            return

        def configure_auto_reset(self, enabled: bool) -> None:
            return

        def set_enc_input_format(self, fmt: str) -> None:
            return

        @property
        def enc_input_format(self) -> str:
            return "YUV444"

    class _DummyAdapterScene:
        def __init__(self, worker):  # type: ignore[no-untyped-def]
            self._worker = worker

        def init(self, source):  # type: ignore[no-untyped-def]
            canvas = _FakeCanvas(self._worker.width, self._worker.height)
            view = _FakeView(_FakeCamera2D())
            viewer = _FakeViewer()
            return canvas, view, viewer

    class _DummyEncoder:
        def __init__(self, width: int, height: int, fps_hint: int) -> None:
            self.width = width
            self.height = height
            self.fps_hint = fps_hint
            self.input_format = "YUV444"
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

    monkeypatch.setattr(ew, "EglContext", _DummyEglContext)
    monkeypatch.setattr(ew, "GLCapture", _DummyGLCapture)
    monkeypatch.setattr(ew, "CudaInterop", _DummyCudaInterop)
    monkeypatch.setattr(ew, "FramePipeline", _DummyFramePipeline)
    monkeypatch.setattr(ew, "AdapterScene", _DummyAdapterScene)
    monkeypatch.setattr(ew, "Encoder", _DummyEncoder)
    monkeypatch.setattr(ew, "plane_scale_for_level", _fake_plane_scale_for_level)
    monkeypatch.setattr(ssa.SceneStateApplier, "apply_slice_to_layer", staticmethod(_fake_apply_slice_to_layer))

    monkeypatch.setattr(ew.EGLRendererWorker, "_init_cuda", lambda self: None)
    monkeypatch.setattr(ew.EGLRendererWorker, "_init_vispy_scene", lambda self: None)
    monkeypatch.setattr(ew.EGLRendererWorker, "_init_egl", lambda self: None)
    monkeypatch.setattr(ew.EGLRendererWorker, "_init_capture", lambda self: None)
    monkeypatch.setattr(ew.EGLRendererWorker, "_init_encoder", lambda self: None)

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

    worker = ew.EGLRendererWorker(width=320, height=180, ctx=ctx)
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


def test_zoom_intent_reaches_lod_selector(egl_worker_fixture, monkeypatch):
    from napari_cuda.server import egl_worker as ew

    worker = egl_worker_fixture
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

    monkeypatch.setattr(ew, "select_level", _fake_select_level)

    worker.process_camera_commands([CameraCommand(kind="zoom", factor=2.0)])
    assert worker._level_policy_refresh_needed is True

    worker._evaluate_level_policy()

    assert ratios, "select_level was not invoked"
    assert math.isclose(float(ratios[-1]), 0.5, rel_tol=1e-6)
    assert worker._scene_state_queue.consume_zoom_intent(max_age=0.5) is None


def test_preserve_view_switch_keeps_camera_range(egl_worker_fixture):
    worker = egl_worker_fixture
    camera: _FakeCamera2D = worker.view.camera  # type: ignore[assignment]
    camera.set_range_calls.clear()
    worker._render_tick_required = False

    worker._scene_state_queue.queue_scene_state(ServerSceneState(current_step=(2, 0, 0)))
    worker.drain_scene_updates()

    assert camera.set_range_calls == []
    assert worker._z_index == 2
    assert worker._last_step == (2, 0, 0)
    assert worker._render_tick_required is True
    assert tuple(worker._data_wh) == (4, 4)


def test_preserve_view_disabled_resets_camera(egl_worker_fixture):
    worker = egl_worker_fixture
    worker._preserve_view_on_switch = False
    camera: _FakeCamera2D = worker.view.camera  # type: ignore[assignment]
    camera.set_range_calls.clear()
    worker._render_tick_required = False

    worker._scene_state_queue.queue_scene_state(ServerSceneState(current_step=(1, 0, 0)))
    worker.drain_scene_updates()

    assert camera.set_range_calls, "camera.set_range should be invoked when preserve-view is disabled"
