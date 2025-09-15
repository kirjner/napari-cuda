from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HWLimits:
    gpu_name: str
    vram_mb: int
    # Budgets
    volume_max_bytes: int
    volume_max_voxels: int
    loader_parallel: int
    reserve_mb: int


def _detect_gpu() -> tuple[str, int]:
    name = os.getenv('NAPARI_CUDA_GPU_NAME')
    mem_env = os.getenv('NAPARI_CUDA_VRAM_MB')
    if name and mem_env:
        try:
            return str(name), int(mem_env)
        except Exception as e:
            logger.debug("env GPU parse failed", exc_info=True)
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'
        ], stderr=subprocess.STDOUT, text=True).strip()
        parts = out.split(',')
        if len(parts) >= 2:
            nm = parts[0].strip()
            mb = int(parts[1].strip())
            return nm, mb
    except Exception as e:
        logger.debug("nvidia-smi probe failed", exc_info=True)
    # Fallback
    return 'Unknown GPU', int(os.getenv('NAPARI_CUDA_VRAM_MB', '8192'))


def get_hw_limits() -> HWLimits:
    name, vram_mb = _detect_gpu()
    # Reserve ~25% of VRAM for GL/encoder/runtime; budget ~50% of remainder to volume
    reserve_mb = int(max(1024, vram_mb * 0.25))
    budget_mb = max(0, vram_mb - reserve_mb)
    volume_budget_mb = int(budget_mb * 0.5)
    volume_max_bytes = volume_budget_mb * 1024 * 1024
    # Conservative voxel cap for float32
    volume_max_voxels = volume_max_bytes // 4
    # Parallelism for loader (tuneable)
    loader_parallel = 4 if vram_mb <= 16_384 else 6 if vram_mb <= 24_576 else 8
    try:
        logger.info("HW: %s, VRAM=%d MB, vol_budget=%d MB", name, vram_mb, volume_budget_mb)
    except Exception:
        pass
    return HWLimits(
        gpu_name=name,
        vram_mb=vram_mb,
        volume_max_bytes=volume_max_bytes,
        volume_max_voxels=int(volume_max_voxels),
        loader_parallel=int(loader_parallel),
        reserve_mb=reserve_mb,
    )

