"""Worker-facing glue around the multiscale level policy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import napari_cuda.server.data.lod as lod
from napari_cuda.server.data.level_budget import LevelBudgetError
from napari_cuda.server.data.roi import plane_wh_for_level
from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.runtime.render_loop.plan.ledger_access import (
    dims_spec as ledger_dims_spec,
)
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.shared.dims_spec import dims_spec_remap_step_for_level

from .context import build_level_context
from .roi import viewport_roi_for_level

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


logger = logging.getLogger(__name__)


class BudgetGuardError(RuntimeError):
    """Marker error used to convert to the public LevelBudgetError."""


@dataclass(frozen=True)
class PolicyEvaluation:
    """Result of evaluating policy inputs for a worker."""

    source: ZarrSceneSource
    decision: lod.LevelDecision
    context: lod.LevelContext
    zoom_ratio: float | None


def _oversampling_for_level(
    worker: EGLRendererWorker,
    source: ZarrSceneSource,
    level: int,
) -> float:
    """Estimate oversampling ratio for ``level`` relative to the worker viewport."""

    try:
        roi = viewport_roi_for_level(worker, source, level, quiet=True, for_policy=True)
        if roi.is_empty():
            height, width = plane_wh_for_level(source, level)
        else:
            height, width = roi.height, roi.width
    except Exception:  # pragma: no cover - guarded by policy logging
        height = int(getattr(worker, "height", 0))
        width = int(getattr(worker, "width", 0))
    viewport_h = max(1, int(getattr(worker, "height", 0)))
    viewport_w = max(1, int(getattr(worker, "width", 0)))
    return float(max(height / viewport_h, width / viewport_w))


def _estimate_level_bytes(source: ZarrSceneSource, level: int) -> tuple[int, int]:
    descriptor = source.level_descriptors[level]
    voxels = 1
    for dim in descriptor.shape:
        voxels *= max(1, int(dim))
    dtype_size = int(np.dtype(source.dtype).itemsize)
    return int(voxels), int(voxels * dtype_size)


def volume_budget_allows(
    worker: EGLRendererWorker,
    source: ZarrSceneSource,
    level: int,
) -> None:
    """Raise when the estimated volume at ``level`` exceeds configured limits."""

    voxels, bytes_est = _estimate_level_bytes(source, level)
    limit_bytes = worker._volume_max_bytes or worker._hw_limits.volume_max_bytes
    limit_voxels = worker._volume_max_voxels or worker._hw_limits.volume_max_voxels
    if limit_voxels and voxels > limit_voxels:
        msg = f"voxels={voxels} exceeds cap={limit_voxels}"
        if worker._log_layer_debug:
            logger.info(
                "budget check (volume): level=%d voxels=%d bytes=%d -> REJECT: %s",
                level,
                voxels,
                bytes_est,
                msg,
            )
        raise BudgetGuardError(msg)
    if limit_bytes and bytes_est > limit_bytes:
        msg = f"bytes={bytes_est} exceeds cap={limit_bytes}"
        if worker._log_layer_debug:
            logger.info(
                "budget check (volume): level=%d voxels=%d bytes=%d -> REJECT: %s",
                level,
                voxels,
                bytes_est,
                msg,
            )
        raise BudgetGuardError(msg)
    if worker._log_layer_debug:
        logger.info(
            "budget check (volume): level=%d voxels=%d bytes=%d -> OK",
            level,
            voxels,
            bytes_est,
        )


def slice_budget_allows(
    worker: EGLRendererWorker,
    source: ZarrSceneSource,
    level: int,
) -> None:
    """Raise when the estimated slice bytes exceed the configured limit."""

    limit_bytes = int(worker._slice_max_bytes or 0)
    if limit_bytes <= 0:
        return
    height, width = plane_wh_for_level(source, level)
    dtype_size = int(np.dtype(source.dtype).itemsize)
    bytes_est = int(height) * int(width) * dtype_size
    if bytes_est > limit_bytes:
        msg = f"slice bytes={bytes_est} exceeds cap={limit_bytes} at level={level}"
        if worker._log_layer_debug:
            logger.info(
                "budget check (slice): level=%d h=%d w=%d dtype=%s bytes=%d -> REJECT: %s",
                level,
                height,
                width,
                str(source.dtype),
                bytes_est,
                msg,
            )
        raise BudgetGuardError(msg)
    if worker._log_layer_debug:
        logger.info(
            "budget check (slice): level=%d h=%d w=%d dtype=%s bytes=%d -> OK",
            level,
            height,
            width,
            str(source.dtype),
            bytes_est,
        )


def resolve_volume_intent_level(
    worker: EGLRendererWorker,
    source: ZarrSceneSource,
    requested_level: int,
) -> int:
    """Return the coarsest available descriptor, ignoring finer requests."""

    descriptors = source.level_descriptors
    if not descriptors:
        return int(requested_level)
    return max(0, len(descriptors) - 1)


def load_volume(
    worker: EGLRendererWorker,
    source: ZarrSceneSource,
    level: int,
):
    """Load volume data for ``level`` after enforcing budgets."""

    volume_budget_allows(worker, source, level)
    return worker._load_volume(source, level)


def evaluate(worker: EGLRendererWorker) -> PolicyEvaluation | None:
    """Evaluate policy inputs and return the selected level context."""

    if not worker._zarr_path:
        return None
    try:
        source = worker._ensure_scene_source()
    except Exception:
        logger.debug("ensure_scene_source failed in selection", exc_info=True)
        return None

    current = int(worker._current_level_index())
    zoom_hint = worker._render_mailbox.consume_zoom_hint(max_age=0.5)
    zoom_ratio = float(zoom_hint.ratio) if zoom_hint is not None else None

    config = lod.LevelPolicyConfig(
        threshold_in=float(worker._level_threshold_in),
        threshold_out=float(worker._level_threshold_out),
        fine_threshold=float(worker._level_fine_threshold),
        hysteresis=float(worker._level_hysteresis),
        cooldown_ms=float(worker._level_switch_cooldown_ms),
    )

    selector = worker._policy_func  # type: ignore[attr-defined]

    outcome = lod.evaluate_policy(
        source=source,
        current_level=current,
        oversampling_for_level=lambda scene, lvl: _oversampling_for_level(worker, scene, lvl),
        zoom_ratio=zoom_ratio,
        lock_level=worker._lock_level,
        last_switch_ts=float(worker._last_level_switch_ts),
        config=config,
        log_policy_eval=worker._log_policy_eval,
        select_level_fn=selector,
        logger_ref=logger,
    )

    if outcome is None:
        return None

    def _budget_check(scene: ZarrSceneSource, level: int) -> None:
        try:
            if worker._viewport_state.mode is RenderMode.VOLUME:
                volume_budget_allows(worker, scene, level)
            else:
                slice_budget_allows(worker, scene, level)
        except BudgetGuardError as exc:
            raise LevelBudgetError(str(exc)) from exc

    decision = lod.enforce_budgets(
        outcome,
        source=source,
        use_volume=worker._viewport_state.mode is RenderMode.VOLUME,
        budget_check=_budget_check,
        log_layer_debug=worker._log_layer_debug,
        logger_ref=logger,
    )

    spec = ledger_dims_spec(worker._ledger)
    assert spec is not None, "dims spec missing during level evaluation"
    base_step = tuple(int(v) for v in spec.current_step)
    step_hint = dims_spec_remap_step_for_level(
        spec,
        step=base_step,
        prev_level=current,
        next_level=int(decision.selected_level),
    )
    context = build_level_context(
        source=source,
        level=int(decision.selected_level),
        step=step_hint,
    )

    return PolicyEvaluation(
        source=source,
        decision=decision,
        context=context,
        zoom_ratio=zoom_ratio,
    )


__all__ = [
    "BudgetGuardError",
    "PolicyEvaluation",
    "evaluate",
    "load_volume",
    "resolve_volume_intent_level",
    "slice_budget_allows",
    "volume_budget_allows",
]
