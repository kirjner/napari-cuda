"""Typed ledger block models for the new view/axes/index/lod/camera schema."""

from __future__ import annotations

import os

from .axes import AxisBlock, AxisExtentBlock, AxesBlock, axes_from_payload, axes_to_payload
from .camera import (
    CameraBlock,
    PlaneCameraBlock,
    VolumeCameraBlock,
    camera_block_from_payload,
    camera_block_to_payload,
)
from .index import IndexBlock, index_block_from_payload, index_block_to_payload
from .lod import LodBlock, lod_block_from_payload, lod_block_to_payload
from .view import ViewBlock, view_block_from_payload, view_block_to_payload
from .restore import (
    PlaneRestoreCachePose,
    PlaneRestoreCacheBlock,
    VolumeRestoreCachePose,
    VolumeRestoreCacheBlock,
    plane_restore_cache_block_from_payload,
    plane_restore_cache_block_to_payload,
    volume_restore_cache_block_from_payload,
    volume_restore_cache_block_to_payload,
)

ENABLE_VIEW_AXES_INDEX_BLOCKS = os.environ.get("NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX", "0") == "1"

__all__ = [
    "AxisExtentBlock",
    "AxisBlock",
    "AxesBlock",
    "axes_from_payload",
    "axes_to_payload",
    "CameraBlock",
    "PlaneCameraBlock",
    "VolumeCameraBlock",
    "camera_block_from_payload",
    "camera_block_to_payload",
    "IndexBlock",
    "index_block_from_payload",
    "index_block_to_payload",
    "LodBlock",
    "lod_block_from_payload",
    "lod_block_to_payload",
    "ViewBlock",
    "view_block_from_payload",
    "view_block_to_payload",
    "ENABLE_VIEW_AXES_INDEX_BLOCKS",
    # restore caches
    "PlaneRestoreCachePose",
    "PlaneRestoreCacheBlock",
    "VolumeRestoreCachePose",
    "VolumeRestoreCacheBlock",
    "plane_restore_cache_block_from_payload",
    "plane_restore_cache_block_to_payload",
    "volume_restore_cache_block_from_payload",
    "volume_restore_cache_block_to_payload",
]
