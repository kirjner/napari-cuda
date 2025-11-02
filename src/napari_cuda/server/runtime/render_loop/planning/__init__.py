"""Snapshot planning utilities for the render loop."""

from .interface import RenderPlanInterface
from .staging import (
    AppliedVersions,
    consume_render_snapshot,
    drain_scene_updates,
    extract_layer_changes,
    normalize_scene_state,
    record_snapshot_versions,
)
from .viewport_planner import ViewportPlanner, SliceTask, ViewportOps

__all__ = [
    "AppliedVersions",
    "RenderPlanInterface",
    "ViewportPlanner",
    "SliceTask",
    "ViewportOps",
    "consume_render_snapshot",
    "drain_scene_updates",
    "extract_layer_changes",
    "normalize_scene_state",
    "record_snapshot_versions",
]
