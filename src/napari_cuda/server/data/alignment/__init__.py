"""Optional Allen CCF alignment helpers for napari-cuda.

This package is imported lazily so deployments that do not install the
``napari-cuda[alignment]`` extra do not pay for the additional brainreg/
brainglobe stack.  Functions here provide light-weight sanity checks around
alignment profiles before the heavy lifting is wired into the server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

__all__ = [
    "AlignmentDependencyError",
    "AlignmentProfile",
    "ensure_alignment_dependencies",
    "load_alignment_profile",
]


class AlignmentDependencyError(RuntimeError):
    """Raised when optional alignment dependencies are missing."""


@dataclass(slots=True)
class AlignmentProfile:
    """Container for Allen CCF alignment configuration."""

    source: Path
    data: dict[str, Any]

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self.data.get(key, default)


def ensure_alignment_dependencies() -> None:
    """Verify brainreg and related packages are importable.

    Raises
    ------
    AlignmentDependencyError
        When at least one of the optional packages is missing.
    """

    required = {
        "brainreg": "brainreg",
        "brainreg-segment": "brainreg_segment",
        "brainglobe-atlasapi": "brainglobe_atlasapi",
        "ome-zarr": "ome_zarr",
    }
    missing: list[str] = []
    for dist_name, module_name in required.items():
        try:
            import_module(module_name)
        except ModuleNotFoundError:
            missing.append(dist_name)
        except Exception as exc:  # defensive guard
            raise AlignmentDependencyError(
                f"Unable to import optional alignment dependency '{module_name}': {exc}"
            ) from exc
    if missing:
        raise AlignmentDependencyError(
            "Optional Allen CCF workflow dependencies missing: "
            + ", ".join(sorted(missing))
            + ". Install via `uv sync --extra alignment` or `pip install napari-cuda[alignment]`."
        )


def load_alignment_profile(config_path: str | Path) -> AlignmentProfile:
    """Load an alignment profile JSON produced by the importer tooling.

    Parameters
    ----------
    config_path:
        Path to a JSON file describing alignment assets.

    Returns
    -------
    AlignmentProfile
        Thin wrapper containing parsed configuration.
    """

    ensure_alignment_dependencies()
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Alignment profile not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Alignment profile must be a JSON object: {path}")
    return AlignmentProfile(source=path, data=data)

