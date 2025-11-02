"""Helpers for resolving OME-Zarr datasets within directory hierarchies."""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ZarrDirectorySummary:
    """Summary metadata describing a potential Zarr directory."""

    path: Path
    is_dataset: bool
    has_zattrs: bool
    has_zarray: bool
    has_multiscales: bool
    dataset_children: Tuple[str, ...]


class ZarrDatasetDisambiguationError(ValueError):
    """Raised when multiple dataset candidates exist beneath a requested path."""

    def __init__(self, *, root: Path, options: Sequence[Path]) -> None:
        opts_tuple = tuple(options)
        message = (
            f"Multiple OME-Zarr datasets found under {root}; selection required."
        )
        super().__init__(message)
        self.root = root
        self.options = opts_tuple

    def option_relatives(self) -> Tuple[str, ...]:
        """Return option paths relative to the ambiguous root."""

        relatives: list[str] = []
        for option in self.options:
            try:
                relatives.append(str(option.relative_to(self.root)))
            except ValueError:
                relatives.append(option.name)
        return tuple(relatives)


def _read_zattrs(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def is_dataset_root(path: Path) -> bool:
    """Return True if ``path`` looks like a concrete OME-Zarr dataset root."""

    try:
        if not path.is_dir():
            return False
    except OSError:
        return False

    try:
        if (path / ".zarray").is_file():
            return True
    except OSError:
        return False

    meta = _read_zattrs(path / ".zattrs")
    multiscales = meta.get("multiscales") if isinstance(meta, dict) else None
    if isinstance(multiscales, Sequence) and multiscales:
        return True
    return False


def inspect_zarr_directory(path: Path, *, child_limit: int = 16) -> Optional[ZarrDirectorySummary]:
    """Inspect ``path`` and return a summary if it is a Zarr directory."""

    # Guard against unreadable directories; fall back to treating as non-dataset.
    try:
        if not path.is_dir():
            return None
    except OSError:
        return None

    zattrs_path = path / ".zattrs"
    # Permission issues are treated as absent metadata rather than hard failures.
    try:
        has_zattrs = zattrs_path.is_file()
    except OSError:
        return None
    try:
        has_zarray = (path / ".zarray").is_file()
    except OSError:
        return None
    has_multiscales = False
    if has_zattrs:
        meta = _read_zattrs(zattrs_path)
        multiscales = meta.get("multiscales") if isinstance(meta, dict) else None
        has_multiscales = isinstance(multiscales, Sequence) and bool(multiscales)

    is_dataset = bool(has_zarray or has_multiscales)
    dataset_children: list[str] = []
    if not is_dataset:
        try:
            for child in path.iterdir():
                if len(dataset_children) >= child_limit:
                    break
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    continue
                if is_dataset_root(child):
                    dataset_children.append(child.name)
        except OSError:
            dataset_children = []

    return ZarrDirectorySummary(
        path=path,
        is_dataset=is_dataset,
        has_zattrs=has_zattrs,
        has_zarray=has_zarray,
        has_multiscales=has_multiscales,
        dataset_children=tuple(dataset_children),
    )


def discover_dataset_root(
    path: Path,
    *,
    max_depth: int = 2,
    option_limit: int = 32,
) -> Path:
    """Locate a concrete dataset beneath ``path`` or raise if ambiguous."""

    dataset, options = _discover_dataset(path, max_depth=max_depth, option_limit=option_limit)
    if dataset is not None:
        return dataset
    if options:
        raise ZarrDatasetDisambiguationError(root=path, options=options)
    raise ValueError(f"No OME-Zarr dataset found under {path}")


def _discover_dataset(
    root: Path,
    *,
    max_depth: int,
    option_limit: int,
) -> tuple[Optional[Path], tuple[Path, ...]]:
    try:
        if not root.is_dir():
            return None, ()
    except OSError:
        return None, ()
    if is_dataset_root(root):
        return root, ()

    options: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    visited: set[Path] = {root}

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            if child in visited:
                continue
            visited.add(child)
            if is_dataset_root(child):
                options.append(child)
                if 0 < option_limit <= len(options):
                    break
            elif depth + 1 < max_depth:
                queue.append((child, depth + 1))
        if 0 < option_limit <= len(options):
            break

    if len(options) == 1:
        return options[0], ()
    return None, tuple(options)


__all__ = [
    "ZarrDatasetDisambiguationError",
    "ZarrDirectorySummary",
    "discover_dataset_root",
    "inspect_zarr_directory",
    "is_dataset_root",
]
