from __future__ import annotations

import os
from typing import Dict, List

# Central preset definitions. Values are strings for direct env use.
PRESETS: Dict[str, Dict[str, str]] = {
    # Disabled
    'off': {
        'NAPARI_CUDA_JIT_ENABLE': '0',
    },
    # Mild Wi‑Fi-like
    'mild': {
        'NAPARI_CUDA_JIT_ENABLE': '1',
        'NAPARI_CUDA_JIT_MODE': 'uniform',
        'NAPARI_CUDA_JIT_BASE_MS': '10',
        'NAPARI_CUDA_JIT_JITTER_MS': '15',
        'NAPARI_CUDA_JIT_LOSS_P': '0',
        'NAPARI_CUDA_JIT_REORDER_P': '0',
    },
    # Heavy tail + occasional burst delay
    'heavy': {
        'NAPARI_CUDA_JIT_ENABLE': '1',
        'NAPARI_CUDA_JIT_MODE': 'pareto',
        'NAPARI_CUDA_JIT_PARETO_ALPHA': '2',
        'NAPARI_CUDA_JIT_PARETO_SCALE': '8',
        'NAPARI_CUDA_JIT_BURST_P': '0.03',
        'NAPARI_CUDA_JIT_BURST_MS': '100',
        'NAPARI_CUDA_JIT_LOSS_P': '0',
        'NAPARI_CUDA_JIT_REORDER_P': '0',
    },
    # Bandwidth cap 4 Mbps with modest jitter
    'cap4mbps': {
        'NAPARI_CUDA_JIT_ENABLE': '1',
        'NAPARI_CUDA_JIT_BW_KBPS': '4000',
        'NAPARI_CUDA_JIT_BURST_BYTES': '65536',
        'NAPARI_CUDA_JIT_BASE_MS': '0',
        'NAPARI_CUDA_JIT_JITTER_MS': '10',
    },
    # Higher base RTT
    'wifi30': {
        'NAPARI_CUDA_JIT_ENABLE': '1',
        'NAPARI_CUDA_JIT_MODE': 'uniform',
        'NAPARI_CUDA_JIT_BASE_MS': '30',
        'NAPARI_CUDA_JIT_JITTER_MS': '20',
    },
}


def preset_names() -> List[str]:
    return sorted(PRESETS.keys())


def apply_preset(name: str, env: os._Environ[str] | None = None, override: bool = False) -> List[str]:
    """Apply a jitter preset to environment variables.

    - Uses setdefault semantics by default (explicit user envs win).
    - If override=True, existing env vars are overwritten.
    - Returns the list of keys that were set/overwritten.
    """
    if env is None:
        env = os.environ
    key = (name or '').strip().lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown jitter preset: {name}")
    applied: List[str] = []
    for k, v in PRESETS[key].items():
        if override:
            env[k] = v
            applied.append(k)
        else:
            if not env.get(k):
                env[k] = v
                applied.append(k)
    return applied

