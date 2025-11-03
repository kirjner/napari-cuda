"""Procedural helpers for dataset lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from napari_cuda.server.control.state_reducers import reduce_bootstrap_state
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.server.engine.api import PixelChannelState, mark_stream_config_dirty


@dataclass(slots=True)
class ServerLifecycleHooks:
    stop_worker: Callable[[], None]
    clear_frame_queue: Callable[[], None]
    reset_mirrors: Callable[[], None]
    reset_thumbnail_state: Callable[[], None]
    refresh_scene_snapshot: Callable[[Optional[RenderLedgerSnapshot]], None]
    start_mirrors_if_needed: Callable[[], None]
    pull_render_snapshot: Callable[[], RenderLedgerSnapshot]
    set_dataset_metadata: Callable[[Optional[str], Optional[str], Optional[str], Optional[int]], None]
    set_bootstrap_snapshot: Callable[[RenderLedgerSnapshot], None]
    pixel_channel: PixelChannelState


def _reset_runtime_state(hooks: ServerLifecycleHooks, ledger: ServerStateLedger) -> None:
    hooks.stop_worker()
    hooks.clear_frame_queue()
    hooks.reset_mirrors()
    hooks.reset_thumbnail_state()
    ledger.clear_scope("layer")
    ledger.clear_scope("multiscale")
    ledger.clear_scope("volume")

    pixel_channel = hooks.pixel_channel
    mark_stream_config_dirty(pixel_channel)
    pixel_channel.last_avcc = None
    broadcast = pixel_channel.broadcast
    broadcast.last_key_seq = None
    broadcast.last_key_ts = None
    broadcast.waiting_for_keyframe = True


def enter_idle_state(hooks: ServerLifecycleHooks, ledger: ServerStateLedger) -> None:
    """Transition the server into idle mode with a trivial bootstrap state."""

    _reset_runtime_state(hooks, ledger)
    hooks.set_dataset_metadata(None, None, None, None)

    reduce_bootstrap_state(
        ledger,
        step=(0, 0),
        axis_labels=("y", "x"),
        order=(0, 1),
        level_shapes=((1, 1),),
        levels=(
            {"index": 0, "shape": [1, 1], "downsample": [1.0, 1.0], "path": ""},
        ),
        current_level=0,
        ndisplay=2,
        origin="server.idle-bootstrap",
    )

    snapshot = hooks.pull_render_snapshot()
    hooks.set_bootstrap_snapshot(snapshot)
    hooks.refresh_scene_snapshot(snapshot)


def apply_dataset_bootstrap(
    hooks: ServerLifecycleHooks,
    ledger: ServerStateLedger,
    bootstrap,
    *,
    resolved_path: str,
    preferred_level: Optional[str],
    z_override: Optional[int],
) -> RenderLedgerSnapshot:
    """Apply dataset bootstrap metadata and refresh cached snapshots."""

    _reset_runtime_state(hooks, ledger)
    reduce_bootstrap_state(
        ledger,
        step=bootstrap.step,
        axis_labels=bootstrap.axis_labels,
        order=bootstrap.order,
        level_shapes=bootstrap.level_shapes,
        levels=bootstrap.levels,
        current_level=bootstrap.current_level,
        ndisplay=bootstrap.ndisplay,
        origin="server.bootstrap",
    )

    hooks.set_dataset_metadata(
        resolved_path,
        preferred_level,
        "".join(str(a) for a in bootstrap.axis_labels),
        z_override,
    )

    snapshot = hooks.pull_render_snapshot()
    hooks.set_bootstrap_snapshot(snapshot)
    hooks.start_mirrors_if_needed()
    hooks.refresh_scene_snapshot(snapshot)
    return snapshot


__all__ = ["ServerLifecycleHooks", "apply_dataset_bootstrap", "enter_idle_state"]
