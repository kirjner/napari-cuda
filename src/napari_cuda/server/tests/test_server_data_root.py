from __future__ import annotations

from pathlib import Path

from napari_cuda.server.app.egl_headless_server import EGLHeadlessServer


def test_data_root_defaults_to_launch_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NAPARI_CUDA_DATA_ROOT", raising=False)

    server = EGLHeadlessServer()

    resolved_root = Path(server._data_root)
    assert resolved_root == tmp_path.resolve()
    assert server._require_data_root() == tmp_path.resolve()
