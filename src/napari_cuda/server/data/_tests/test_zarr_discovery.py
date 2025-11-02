from __future__ import annotations

import json
from pathlib import Path
import os
import stat

import pytest

from napari_cuda.server.data.zarr_discovery import (
    ZarrDatasetDisambiguationError,
    discover_dataset_root,
    inspect_zarr_directory,
    is_dataset_root,
)


def _write_multiscale(path: Path, *, name: str = "0") -> None:
    datasets = [{"path": name, "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]
    multiscales = [{"datasets": datasets}]
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zattrs").write_text(json.dumps({"multiscales": multiscales}))
    (path / name).mkdir(parents=True, exist_ok=True)
    (path / name / ".zarray").write_text(json.dumps({"shape": [1, 1, 1], "chunks": [1, 1, 1], "dtype": "<f4"}))


def test_is_dataset_root_true_for_multiscale(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.zarr"
    _write_multiscale(dataset)
    assert is_dataset_root(dataset) is True


def test_discover_dataset_root_returns_root(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.zarr"
    _write_multiscale(dataset)
    resolved = discover_dataset_root(dataset)
    assert resolved == dataset


def test_discover_dataset_root_descends_into_unique_child(tmp_path: Path) -> None:
    container = tmp_path / "container.zarr"
    child = container / "all"
    child_2 = container / "annotations"
    child_2.mkdir(parents=True)
    _write_multiscale(child)
    resolved = discover_dataset_root(container)
    assert resolved == child


def test_discover_dataset_root_raises_when_multiple_children(tmp_path: Path) -> None:
    container = tmp_path / "multi.zarr"
    first = container / "a"
    second = container / "b"
    _write_multiscale(first)
    _write_multiscale(second)
    with pytest.raises(ZarrDatasetDisambiguationError) as excinfo:
        discover_dataset_root(container)
    exc = excinfo.value
    assert exc.root == container
    assert set(exc.option_relatives()) == {"a", "b"}


def test_inspect_reports_children(tmp_path: Path) -> None:
    container = tmp_path / "nested.zarr"
    first = container / "primary"
    _write_multiscale(first)
    summary = inspect_zarr_directory(container)
    assert summary is not None
    assert summary.is_dataset is False
    assert "primary" in summary.dataset_children


def test_inspect_handles_permission_error(tmp_path: Path) -> None:
    protected = tmp_path / "protected.zarr"
    protected.mkdir()
    (protected / ".zattrs").write_text("{}")
    os.chmod(protected, 0)
    try:
        summary = inspect_zarr_directory(protected)
        assert summary is None
    finally:
        os.chmod(protected, stat.S_IRWXU)
