"""Level-of-detail (multiscale) helpers.

Pure, testable functions that decide and apply multiscale level changes. This
module centralizes logic that was previously embedded in the worker so napari
remains the authority for level choice while keeping our stabilizers and
budgets.

Phase A: helpers only; wiring happens in the worker in a later step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Optional, Sequence, Tuple
import logging
import time

import numpy as np

from napari.components.viewer_model import ViewerModel
from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.data.roi import plane_scale_for_level
from napari_cuda.server.data.level_budget import LevelBudgetError


logger = logging.getLogger(__name__)


# ---- Types -------------------------------------------------------------------


@dataclass(frozen=True)
class AppliedLevel:
    level: int
    step: Tuple[int, ...]
    z_index: Optional[int]
    shape: Tuple[int, ...]
    scale_yx: Tuple[float, float]
    contrast: Tuple[float, float]
    axes: str
    dtype: str


@dataclass(frozen=True)
class LevelPolicyConfig:
    threshold_in: float
    threshold_out: float
    fine_threshold: float
    hysteresis: float
    cooldown_ms: float


@dataclass(frozen=True)
class LevelPolicyInputs:
    current_level: int
    oversampling: Mapping[int, float]
    zoom_ratio: Optional[float]
    lock_level: Optional[int]
    last_switch_ts: float
    now_ts: float


@dataclass(frozen=True)
class LevelPolicyDecision:
    desired_level: int
    selected_level: int
    action: str
    should_switch: bool
    blocked_reason: Optional[str]
    cooldown_remaining_ms: float = 0.0


@dataclass(frozen=True)
class PolicySwitchOutcome:
    timestamp: float
    decision: LevelPolicyDecision
    oversampling: Mapping[int, float]


# ---- Stabilizer --------------------------------------------------------------


def stabilize_level(napari_level: int, prev_level: int, *, hysteresis: float = 0.0) -> int:
    """Return a stabilized level around ``prev_level`` honouring hysteresis."""
    cur = int(prev_level)
    nxt = int(napari_level)
    if hysteresis <= 0.0:
        return nxt
    # Simple clamp: require two-step intent to move away from prev_level
    if abs(nxt - cur) == 1:
        return cur
    return nxt


# ---- ROI computation ---------------------------------------------------------


def _clamp_level(value: int, *, max_level: int) -> int:
    return max(0, min(int(value), int(max_level)))


def _cooldown_remaining(config: LevelPolicyConfig, inputs: LevelPolicyInputs) -> float:
    if inputs.last_switch_ts <= 0.0 or config.cooldown_ms <= 0.0:
        return 0.0
    elapsed_ms = (inputs.now_ts - inputs.last_switch_ts) * 1000.0
    return max(0.0, float(config.cooldown_ms) - elapsed_ms)


def select_level(
    config: LevelPolicyConfig,
    inputs: LevelPolicyInputs,
) -> LevelPolicyDecision:
    overs_map = {int(k): float(v) for k, v in inputs.oversampling.items() if v is not None}
    if not overs_map:
        return LevelPolicyDecision(
            desired_level=int(inputs.current_level),
            selected_level=int(inputs.current_level),
            action="no-oversampling",
            should_switch=False,
            blocked_reason="no-oversampling",
        )

    current = int(inputs.current_level)
    max_level = max(overs_map.keys())

    desired: Optional[int] = None
    action = "heuristic"
    zoom_ratio = inputs.zoom_ratio
    if zoom_ratio is not None:
        eps = 1e-3
        if zoom_ratio < 1.0 - eps and current > 0:
            desired = current - 1
            action = "zoom-in"
        elif zoom_ratio > 1.0 + eps and current < max_level:
            desired = current + 1
            action = "zoom-out"

    if desired is None:
        desired = current
        if current > 0 and overs_map.get(current - 1, float("inf")) <= config.fine_threshold:
            desired = current - 1
        elif current < max_level and overs_map.get(current, 0.0) > config.threshold_out:
            desired = current + 1
        action = "heuristic"
    else:
        desired = _clamp_level(desired, max_level=max_level)
        desired = max(desired, current - 1) if desired < current else min(desired, current + 1)

    selected = stabilize_level(int(desired), current, hysteresis=config.hysteresis)
    sel = int(selected)

    # Respect zoom hints by tightening the outgoing threshold when the user is
    # explicitly zooming in or out. This mirrors the previous inline policy in
    # ``EGLRendererWorker`` and preserves the softer 1.20 band for hint-driven
    # zoom-outs so they may trigger sooner than heuristic switching.
    thr_in = float(config.threshold_in)
    thr_out = float(config.threshold_out)
    if action == "zoom-in":
        thr_in = min(1.20, thr_in)
    elif action == "zoom-out":
        thr_out = min(1.20, thr_out)

    if sel < current:
        if overs_map.get(sel, float("inf")) > thr_in:
            sel = current
    elif sel > current:
        if overs_map.get(current, 0.0) <= thr_out:
            sel = current
    selected = sel

    reason = action
    if inputs.lock_level is not None:
        selected = int(inputs.lock_level)
        reason = "locked"

    if selected == current:
        return LevelPolicyDecision(
            desired_level=int(desired),
            selected_level=int(selected),
            action=reason,
            should_switch=False,
            blocked_reason=None,
        )

    remaining = _cooldown_remaining(config, inputs)
    if selected != current and remaining > 0.0:
        return LevelPolicyDecision(
            desired_level=int(desired),
            selected_level=int(selected),
            action=reason,
            should_switch=False,
            blocked_reason="cooldown",
            cooldown_remaining_ms=remaining,
        )

    return LevelPolicyDecision(
        desired_level=int(desired),
        selected_level=int(selected),
        action=reason,
        should_switch=True,
        blocked_reason=None,
    )


# ---- Budget checks -----------------------------------------------------------


def assert_volume_within_budget(
    source: ZarrSceneSource,
    level: int,
    *,
    max_bytes: int | None,
    max_voxels: int | None,
) -> None:
    """Raise when level volume exceeds inclusive memory/voxel budgets."""
    if not max_bytes and not max_voxels:
        return
    arr = source.get_level(level)
    dtype_sz = int(np.dtype(getattr(arr, "dtype", source.dtype)).itemsize)
    shape = tuple(int(s) for s in getattr(arr, "shape", ()))
    vox = 1
    for dim in shape:
        vox *= max(1, dim)
    bytes_est = int(vox) * int(dtype_sz)
    if max_voxels and vox > int(max_voxels):
        raise LevelBudgetError(f"volume voxels over budget: {vox} > {int(max_voxels)}")
    if max_bytes and bytes_est > int(max_bytes):
        raise LevelBudgetError(f"volume bytes over budget: {bytes_est} > {int(max_bytes)}")


def assert_slice_within_budget(
    source: ZarrSceneSource,
    level: int,
    *,
    dtype_size: int | None,
    max_bytes: int | None,
) -> None:
    """Raise when a full-frame slice at this level would exceed budget."""
    if not max_bytes:
        return
    desc = source.level_descriptors[level]
    axes = source.axes
    lower = [str(a).lower() for a in axes]
    try:
        y_pos = lower.index("y")
    except ValueError:
        y_pos = max(0, len(desc.shape) - 2)
    try:
        x_pos = lower.index("x")
    except ValueError:
        x_pos = max(0, len(desc.shape) - 1)
    h = int(desc.shape[y_pos]) if 0 <= y_pos < len(desc.shape) else int(desc.shape[-2])
    w = int(desc.shape[x_pos]) if 0 <= x_pos < len(desc.shape) else int(desc.shape[-1])
    if dtype_size is None:
        dtype_size = int(np.dtype(source.dtype).itemsize)
    bytes_est = int(h) * int(w) * int(dtype_size)
    if bytes_est > int(max_bytes):
        raise LevelBudgetError(f"slice bytes over budget: {bytes_est} > {int(max_bytes)}")


# ---- Application -------------------------------------------------------------


def _proportional_z_remap(
    *,
    prev_level: Optional[int],
    prev_shape: Optional[Sequence[int]],
    new_shape: Sequence[int],
    axes: Sequence[str],
    current_step: Optional[Sequence[int]],
    current_z: Optional[int],
) -> Optional[int]:
    lower = [str(a).lower() for a in axes]
    if "z" not in lower:
        return current_z
    try:
        zi = lower.index("z")
    except Exception:
        return current_z
    try:
        z_src = int(prev_shape[zi]) if prev_shape is not None else None
    except Exception:
        z_src = None
    try:
        z_tgt = int(new_shape[zi])
    except Exception:
        return current_z
    if z_src is None or z_src <= 1 or z_tgt <= 1:
        return 0 if z_tgt > 0 and current_z is None else current_z
    if current_z is None:
        if current_step is not None and len(current_step) > zi:
            current_z = int(current_step[zi])
        else:
            current_z = 0
    new_z = int(round(float(current_z) * float(max(0, z_tgt - 1)) / float(max(1, z_src - 1))))
    return max(0, min(int(new_z), int(z_tgt - 1)))


def set_dims_range_for_level(viewer: Optional[ViewerModel], source: ZarrSceneSource, level: int) -> None:
    if viewer is None:
        return
    desc = source.level_descriptors[level]
    shape = tuple(int(s) for s in desc.shape)
    ranges = tuple((0, max(0, s - 1), 1) for s in shape)
    try:
        viewer.dims.range = ranges
    except Exception:
        logger.debug("set_dims_range_for_level failed", exc_info=True)


def apply_level(
    *,
    source: ZarrSceneSource,
    target_level: int,
    prev_level: Optional[int],
    last_step: Optional[Sequence[int]],
    viewer: Optional[ViewerModel],
    restoring_plane_state: bool = False,
) -> AppliedLevel:
    """Apply target level, update dims, and return the applied snapshot."""
    axes = source.axes
    lower = [str(a).lower() for a in axes]
    desc = source.level_descriptors[target_level]

    prev_shape: Optional[Sequence[int]] = None
    if prev_level is not None and 0 <= int(prev_level) < len(source.level_descriptors):
        prev_shape = source.level_descriptors[int(prev_level)].shape

    # Current Z guess (None -> derive from last_step)
    cur_z = None
    if last_step is not None and "z" in lower:
        zi = lower.index("z")
        if len(last_step) > zi:
            try:
                cur_z = int(last_step[zi])
            except Exception:
                cur_z = None

    # Proportional remap if needed
    if restoring_plane_state:
        new_z = cur_z
    else:
        new_z = _proportional_z_remap(
            prev_level=prev_level,
            prev_shape=prev_shape,
            new_shape=desc.shape,
            axes=axes,
            current_step=last_step,
            current_z=cur_z,
        )

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "lod.apply_level: prev_level=%s target=%s prev_shape=%s new_shape=%s last_step=%s cur_z=%s new_z=%s",
            prev_level,
            target_level,
            prev_shape,
            desc.shape,
            last_step,
            cur_z,
            new_z,
        )

    # Build step hint and set level
    step_hint: list[int] = [0] * len(desc.shape)
    if new_z is not None and "z" in lower:
        zi = lower.index("z")
        if 0 <= zi < len(step_hint):
            step_hint[zi] = int(new_z)

    step = source.set_current_slice(tuple(step_hint), int(target_level))

    # Sync viewer: set range first, then current_step to avoid clamping to old range
    if viewer is not None:
        try:
            set_dims_range_for_level(viewer, source, int(target_level))
            viewer.dims.current_step = tuple(int(x) for x in step)
        except Exception:
            logger.debug("apply_level: syncing viewer dims failed", exc_info=True)

    # Contrast and scale
    contrast = source.ensure_contrast(level=int(target_level))
    sy, sx = plane_scale_for_level(source, int(target_level))
    axes_str = "".join(str(a) for a in axes)
    dtype_str = str(source.dtype)

    # z-index from applied step
    z_index: Optional[int] = None
    if "z" in lower and len(step) > lower.index("z"):
        z_index = int(step[lower.index("z")])
    elif step:
        z_index = int(step[0])

    return AppliedLevel(
        level=int(target_level),
        step=tuple(int(x) for x in step),
        z_index=z_index,
        shape=tuple(int(s) for s in desc.shape),
        scale_yx=(float(sy), float(sx)),
        contrast=(float(contrast[0]), float(contrast[1])),
        axes=axes_str,
        dtype=dtype_str,
    )


def format_oversampling(overs_map: Mapping[int, float]) -> str:
    return '{' + ', '.join(f"{k}:{overs_map[k]:.2f}" for k in sorted(overs_map)) + '}'


def evaluate_policy_decision(
    *,
    source: ZarrSceneSource,
    current_level: int,
    oversampling_for_level: Callable[[ZarrSceneSource, int], float],
    zoom_ratio: Optional[float],
    lock_level: Optional[int],
    last_switch_ts: float,
    now_ts: float,
    config: LevelPolicyConfig,
    log_policy_eval: bool,
    select_level_fn: Callable[[LevelPolicyConfig, LevelPolicyInputs], LevelPolicyDecision] = select_level,
    logger_ref: logging.Logger = logger,
) -> Optional[tuple[LevelPolicyDecision, Mapping[int, float]]]:
    level_indices = list(range(len(source.level_descriptors)))
    if not level_indices:
        return None

    overs_map: dict[int, float] = {}
    for lvl in level_indices:
        try:
            overs_map[int(lvl)] = float(oversampling_for_level(source, int(lvl)))
        except Exception:
            continue
    if not overs_map:
        return None

    inputs = LevelPolicyInputs(
        current_level=current_level,
        oversampling=overs_map,
        zoom_ratio=zoom_ratio,
        lock_level=lock_level,
        last_switch_ts=last_switch_ts,
        now_ts=now_ts,
    )
    decision = select_level_fn(config, inputs)

    if not decision.should_switch:
        if decision.blocked_reason == "cooldown":
            if log_policy_eval and logger_ref.isEnabledFor(logging.DEBUG):
                logger_ref.debug(
                    "lod.cooldown: level=%d target=%d remaining=%.1fms",
                    int(current_level),
                    int(decision.selected_level),
                    decision.cooldown_remaining_ms,
                )
        elif logger_ref.isEnabledFor(logging.DEBUG):
            logger_ref.debug(
                "lod.hold: level=%d desired=%d selected=%d overs=%s",
                int(current_level),
                int(decision.desired_level),
                int(decision.selected_level),
                format_oversampling(overs_map),
            )
        return None

    if logger_ref.isEnabledFor(logging.INFO):
        if int(decision.selected_level) < int(current_level):
            logger_ref.info(
                "lod.zoom_in: current=%d -> selected=%d overs=%.3f reason=%s",
                int(current_level),
                int(decision.selected_level),
                overs_map.get(int(decision.selected_level), float('nan')),
                decision.action,
            )
        elif int(decision.selected_level) > int(current_level):
            logger_ref.info(
                "lod.zoom_out: current=%d -> selected=%d overs=%.3f reason=%s",
                int(current_level),
                int(decision.selected_level),
                overs_map.get(int(decision.selected_level), float('nan')),
                decision.action,
            )

    return decision, overs_map


def apply_level_with_budget(
    *,
    desired_level: int,
    use_volume: bool,
    source: ZarrSceneSource,
    current_level: int,
    log_layer_debug: bool,
    budget_check: Callable[[ZarrSceneSource, int], None],
    apply_level_cb: Callable[[ZarrSceneSource, int, Optional[int]], None],
    on_switch: Callable[[int, int, float], None],
    logger_ref: logging.Logger = logger,
) -> tuple[int, bool]:
    """Select a level under budgets, apply it, and report the downgrade flag."""

    level_count = len(source.level_descriptors)
    if level_count == 0:
        return current_level, False

    desired_level = max(0, min(int(desired_level), level_count - 1))
    cand = range(desired_level, level_count)
    total = len(cand)

    for idx, level in enumerate(cand):
        try:
            budget_check(source, level)
            start = time.perf_counter()
            apply_level_cb(source, level, current_level)
            downgraded = level != desired_level
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            on_switch(current_level, level, elapsed_ms)
            if downgraded and log_layer_debug:
                logger_ref.info(
                    "level downgrade: requested=%d active=%d", desired_level, level
                )
            return level, downgraded
        except LevelBudgetError as exc:
            if idx == total - 1:
                raise
            if log_layer_debug:
                logger_ref.info(
                    "budget reject: mode=%s level=%d reason=%s",
                    'volume' if use_volume else 'slice',
                    int(level),
                    str(exc),
                )
            continue

    raise RuntimeError("Unable to select multiscale level within budget")


def apply_level_with_context(
    *,
    desired_level: int,
    use_volume: bool,
    source: ZarrSceneSource,
    current_level: int,
    log_layer_debug: bool,
    budget_check: Callable[[ZarrSceneSource, int], None],
    apply_level_fn: Callable[[ZarrSceneSource, int, Optional[int]], AppliedLevel],
    on_switch: Callable[[int, int, float], None],
    roi_cache: Optional[MutableMapping[int, object]] = None,
    roi_log_state: Optional[MutableMapping[int, object]] = None,
    logger_ref: logging.Logger = logger,
) -> tuple[AppliedLevel, bool]:
    """Apply a level within budgets and refresh ROI caches."""

    applied_snapshot: Optional[AppliedLevel] = None

    def _apply(scene: ZarrSceneSource, level: int, prev_level: Optional[int]) -> None:
        nonlocal applied_snapshot
        level_idx = int(level)
        if roi_cache is not None:
            roi_cache.pop(level_idx, None)
        if roi_log_state is not None:
            roi_log_state.pop(level_idx, None)
        applied_snapshot = apply_level_fn(scene, level, prev_level)

    def _on_switch(prev_level: int, applied_level: int, elapsed_ms: float) -> None:
        on_switch(prev_level, applied_level, elapsed_ms)

    applied_level_idx, downgraded = apply_level_with_budget(
        desired_level=desired_level,
        use_volume=use_volume,
        source=source,
        current_level=current_level,
        log_layer_debug=log_layer_debug,
        budget_check=budget_check,
        apply_level_cb=_apply,
        on_switch=_on_switch,
        logger_ref=logger_ref,
    )

    if applied_snapshot is None:
        raise RuntimeError("apply_level_with_context: apply_level_fn did not run")

    if int(applied_snapshot.level) != int(applied_level_idx) and logger_ref.isEnabledFor(logging.DEBUG):
        logger_ref.debug(
            "apply_level_with_context: snapshot level=%d idx=%d",  # pragma: no cover - debug aid
            int(applied_snapshot.level),
            int(applied_level_idx),
        )

    return applied_snapshot, downgraded


def run_policy_switch(
    *,
    source: ZarrSceneSource,
    current_level: int,
    oversampling_for_level: Callable[[ZarrSceneSource, int], float],
    zoom_ratio: Optional[float],
    lock_level: Optional[int],
    last_switch_ts: float,
    config: LevelPolicyConfig,
    log_policy_eval: bool,
    apply_level: Callable[[int, str], None],
    budget_error: type[Exception],
    select_level_fn: Callable[[LevelPolicyConfig, LevelPolicyInputs], LevelPolicyDecision] = select_level,
    logger_ref: logging.Logger = logger,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> Optional[PolicySwitchOutcome]:
    """Evaluate and apply a level policy decision, returning metadata on success."""

    now_ts = perf_counter()
    maybe = evaluate_policy_decision(
        source=source,
        current_level=current_level,
        oversampling_for_level=oversampling_for_level,
        zoom_ratio=zoom_ratio,
        lock_level=lock_level,
        last_switch_ts=last_switch_ts,
        now_ts=now_ts,
        config=config,
        log_policy_eval=log_policy_eval,
        select_level_fn=select_level_fn,
        logger_ref=logger_ref,
    )

    if maybe is None:
        return None

    decision, overs_map = maybe

    try:
        apply_level(int(decision.selected_level), decision.action)
    except budget_error as exc:  # type: ignore[arg-type]
        if logger_ref.isEnabledFor(logging.INFO):
            logger_ref.info(
                "ms.switch: hold=%d (budget reject %s)",
                int(current_level),
                str(exc),
            )
        return None
    except Exception as exc:
        setattr(exc, "policy_decision", decision)
        raise

    return PolicySwitchOutcome(
        timestamp=now_ts,
        decision=decision,
        oversampling=overs_map,
    )


__all__ = [
    "AppliedLevel",
    "LevelPolicyConfig",
    "LevelPolicyDecision",
    "LevelPolicyInputs",
    "stabilize_level",
    "assert_volume_within_budget",
    "assert_slice_within_budget",
    "apply_level",
    "apply_level_with_context",
    "select_level",
    "set_dims_range_for_level",
    "format_oversampling",
    "evaluate_policy_decision",
    "PolicySwitchOutcome",
    "run_policy_switch",
]
