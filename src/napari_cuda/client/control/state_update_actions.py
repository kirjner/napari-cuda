"""State update helpers for the streaming client loop."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TYPE_CHECKING
from numbers import Integral, Real

from napari_cuda.protocol import NotifyCamera

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.rendering.presenter_facade import PresenterFacade

    from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.client_state_ledger import (
    ClientStateLedger,
    IntentRecord,
    AckReconciliation,
    MirrorEvent,
)
from napari_cuda.client.control.mirrors import napari_dims_mirror


logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


def _default_dims_meta() -> dict[str, object | None]:
    return {
        'ndim': None,
        'order': None,
        'axis_labels': None,
        'range': None,
        'sizes': None,
        'ndisplay': None,
        'mode': None,
        'volume': None,
        'render': None,
        'multiscale': None,
    }


def _normalize_policy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_policy_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_policy_value(v) for v in value]
    return value


def _normalize_camera_delta_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_camera_delta_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_camera_delta_value(v) for v in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


@dataclass
class ControlStateContext:
    """Mutable control state hoisted out of the loop object."""

    dims_ready: bool = False
    dims_meta: dict[str, object | None] = field(default_factory=_default_dims_meta)
    primary_axis_index: int | None = None
    session_id: Optional[str] = None
    ack_timeout_ms: Optional[int] = None
    intent_counter: int = 0
    dims_min_dt: float = 0.0
    last_dims_send: float = 0.0
    wheel_px_accum: float = 0.0
    wheel_step: float = 1.0
    settings_min_dt: float = 0.0
    last_settings_send: float = 0.0
    dims_z: float | None = None
    dims_z_min: float | None = None
    dims_z_max: float | None = None
    control_runtimes: Dict[str, "ControlRuntime"] = field(default_factory=dict)
    dims_state: Dict[tuple[str, str], Any] = field(default_factory=dict)
    view_state: Dict[str, Any] = field(default_factory=dict)
    volume_state: Dict[str, Any] = field(default_factory=dict)
    multiscale_state: Dict[str, Any] = field(default_factory=dict)
    scene_policies: Dict[str, Any] = field(default_factory=dict)
    camera_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_cfg: Any) -> "ControlStateContext":
        state = cls()
        dims_rate = getattr(env_cfg, 'dims_rate_hz', 1.0) or 1.0
        state.dims_min_dt = 1.0 / max(1.0, float(dims_rate))
        state.wheel_step = float(getattr(env_cfg, 'wheel_step', 1.0) or 1.0)
        settings_rate = getattr(env_cfg, 'settings_rate_hz', 1.0) or 1.0
        state.settings_min_dt = 1.0 / max(1.0, float(settings_rate))
        state.dims_z = getattr(env_cfg, 'dims_z', None)
        state.dims_z_min = getattr(env_cfg, 'dims_z_min', None)
        state.dims_z_max = getattr(env_cfg, 'dims_z_max', None)
        if state.dims_z is not None:
            if state.dims_z_min is not None and state.dims_z < state.dims_z_min:
                state.dims_z = state.dims_z_min
            if state.dims_z_max is not None and state.dims_z > state.dims_z_max:
                state.dims_z = state.dims_z_max
        return state

    def next_intent_ids(self) -> tuple[str, str]:
        self.intent_counter = (int(self.intent_counter) + 1) & 0xFFFFFFFF
        base = f"{self.intent_counter:08x}"
        intent_id = f"intent-{base}"
        frame_id = f"state-{base}"
        return intent_id, frame_id


def apply_scene_policies(state: ControlStateContext, policies: Mapping[str, Any]) -> None:
    normalized = {str(k): _normalize_policy_value(v) for k, v in policies.items()}
    state.scene_policies = normalized

    multiscale = normalized.get('multiscale') if isinstance(normalized.get('multiscale'), Mapping) else None
    if isinstance(multiscale, Mapping):
        meta_obj = state.dims_meta.get('multiscale')
        if not isinstance(meta_obj, dict):
            meta_obj = {}
            state.dims_meta['multiscale'] = meta_obj
        meta = meta_obj

        policy_name = multiscale.get('policy')
        if policy_name is not None:
            meta['policy'] = str(policy_name)
            state.multiscale_state['policy'] = str(policy_name)

        level_value = multiscale.get('active_level')
        if level_value is None:
            level_value = multiscale.get('current_level')
        if level_value is not None:
            try:
                level_int = int(level_value)
            except Exception:
                level_int = level_value
            meta['current_level'] = level_int
            meta['level'] = level_int
            state.multiscale_state['level'] = level_int

        if 'downgraded' in multiscale:
            meta['downgraded'] = bool(multiscale.get('downgraded'))

        if 'index_space' in multiscale:
            meta['index_space'] = str(multiscale.get('index_space'))

        sizes: list[int] | None = None
        ranges: list[list[int]] | None = None

        levels_obj = multiscale.get('levels')
        if isinstance(levels_obj, Sequence) and not isinstance(levels_obj, (str, bytes, bytearray)):
            level_entries: list[dict[str, Any]] = []
            for entry in levels_obj:
                if isinstance(entry, Mapping):
                    level_entries.append({str(k): _normalize_policy_value(v) for k, v in entry.items()})
            if level_entries:
                meta['levels'] = level_entries
                if isinstance(level_int, int) and 0 <= level_int < len(level_entries):
                    active_entry = level_entries[level_int]
                    shape_obj = active_entry.get('shape')
                    if isinstance(shape_obj, Sequence) and not isinstance(shape_obj, (str, bytes, bytearray)):
                        try:
                            sizes = [max(1, int(x)) for x in shape_obj]
                        except Exception:
                            sizes = None
                    if sizes:
                        ranges = [[0, max(0, s - 1)] for s in sizes]

        if sizes:
            meta_root = state.dims_meta
            meta_root['sizes'] = sizes
            meta_root['ndim'] = len(sizes)
            if ranges:
                meta_root['range'] = ranges

            prev_sizes = meta_root.get('sizes_prev') if False else None
        if ranges is None and 'sizes' not in state.dims_meta:
            meta_root = state.dims_meta
            meta_root.pop('range', None)

        if logger.isEnabledFor(logging.DEBUG):
            levels_data = meta.get('levels') if isinstance(meta.get('levels'), list) else []
            logger.debug(
                "apply_scene_policies: multiscale policy=%s level=%s levels=%s",
                meta.get('policy'),
                meta.get('current_level'),
                len(levels_data),
            )
    elif logger.isEnabledFor(logging.DEBUG):
        logger.debug("apply_scene_policies: no multiscale section present")


@dataclass
class ControlRuntime:
    active: bool = False
    last_phase: Optional[str] = None
    last_send_ts: float = 0.0
    active_intent_id: Optional[str] = None
    active_frame_id: Optional[str] = None


def on_state_connected(state: ControlStateContext) -> None:
    state.dims_ready = False
    state.primary_axis_index = None


def on_state_disconnected(loop_state: "ClientLoopState", state: ControlStateContext) -> None:
    state.dims_ready = False
    state.primary_axis_index = None
    loop_state.pending_intents.clear()
    loop_state.last_dims_payload = None
    state.control_runtimes.clear()
    state.camera_state.clear()
def handle_notify_camera(
    state: ControlStateContext,
    state_ledger: "ClientStateLedger",
    frame: NotifyCamera,
    *,
    log_debug: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    payload = frame.payload
    mode = str(payload.mode or "")
    normalized_delta = _normalize_camera_delta_value(payload.delta)
    mode_key = mode or 'main'
    state.camera_state[mode_key] = normalized_delta

    timestamp = frame.envelope.timestamp
    state_ledger.record_confirmed(
        'camera',
        'main',
        mode_key,
        normalized_delta,
        timestamp=timestamp,
    )

    if log_debug:
        logger.debug(
            "notify.camera mode=%s origin=%s intent=%s delta=%s",
            mode,
            payload.origin,
            frame.envelope.intent_id,
            normalized_delta,
        )

    return mode_key, normalized_delta


def _axis_target_label(state: ControlStateContext, axis_idx: int) -> str:
    labels = state.dims_meta.get('axis_labels')
    if isinstance(labels, Sequence) and 0 <= axis_idx < len(labels):
        label = labels[axis_idx]
        if isinstance(label, str) and label.strip():
            return label
    return str(axis_idx)


def _axis_index_from_target(state: ControlStateContext, target: str) -> Optional[int]:
    target_lower = target.lower()
    labels = state.dims_meta.get('axis_labels')
    if isinstance(labels, Sequence):
        for idx, label in enumerate(labels):
            text = str(label)
            if text == target or text.lower() == target_lower:
                return int(idx)
    if target.startswith('axis-'):
        target = target.split('-', 1)[1]
    try:
        return int(target)
    except Exception:
        return None


def _runtime_key(scope: str, target: str, key: str) -> str:
    return f"{scope}:{target}:{key}"


def _emit_state_update(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    *,
    scope: str,
    target: str,
    key: str,
    value: Any,
    origin: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, Optional[Any]]:
    runtime_key = _runtime_key(scope, target, key)
    runtime = state.control_runtimes.setdefault(runtime_key, ControlRuntime())
    phase = "start" if not runtime.active else "update"
    intent_id, frame_id = state.next_intent_ids()
    pending = state_ledger.apply_local(
        scope,
        target,
        key,
        value,
        phase,
        intent_id=intent_id,
        frame_id=frame_id,
        metadata=metadata,
    )

    if pending is None:
        projection_value = state_ledger.confirmed_value(scope, target, key)
        if projection_value is None:
            projection_value = value
        logger.debug(
            "state.update suppressed (duplicate): scope=%s target=%s key=%s value=%s runtime.active=%s",
            scope,
            target,
            key,
            value,
            runtime.active,
        )
        return False, projection_value

    if not dispatch_state_update(pending, origin):
        state_ledger.discard_pending(frame_id)
        return False, None

    runtime.active = True
    runtime.last_phase = phase
    runtime.last_send_ts = time.perf_counter()
    runtime.active_intent_id = intent_id
    runtime.active_frame_id = frame_id
    return True, pending.projection_value

def _update_runtime_from_ack_outcome(state: ControlStateContext, outcome: "AckReconciliation") -> None:
    if outcome.scope is None or outcome.target is None or outcome.key is None:
        return
    runtime_key = _runtime_key(outcome.scope, outcome.target, outcome.key)
    runtime = state.control_runtimes.setdefault(runtime_key, ControlRuntime())
    runtime.last_phase = outcome.update_phase or runtime.last_phase
    runtime.last_send_ts = time.perf_counter()
    matched = outcome.in_reply_to == runtime.active_frame_id
    if matched:
        runtime.active_frame_id = None
        runtime.active_intent_id = None

    if matched and outcome.pending_len == 0:
        runtime.active = False
        runtime.last_phase = None
    else:
        runtime.active = outcome.pending_len > 0
        if not runtime.active:
            runtime.last_phase = None


def _sync_dims_payload_from_meta(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
) -> dict[str, Any]:
    meta = state.dims_meta

    current_step = meta.get('current_step')
    if current_step is not None:
        assert isinstance(current_step, Sequence)
        payload_current_step = list(current_step)
    else:
        payload_current_step = None

    dims_range = meta.get('range')
    if dims_range is not None:
        assert isinstance(dims_range, Sequence)
        payload_range = list(dims_range)
    else:
        payload_range = None

    order = meta.get('order')
    if order is not None:
        assert isinstance(order, Sequence)
        payload_order = list(order)
    else:
        payload_order = None

    axis_labels = meta.get('axis_labels')
    if axis_labels is not None:
        assert isinstance(axis_labels, Sequence)
        payload_axis_labels = list(axis_labels)
    else:
        payload_axis_labels = None

    sizes = meta.get('sizes')
    if sizes is not None:
        assert isinstance(sizes, Sequence)
        payload_sizes = list(sizes)
    else:
        payload_sizes = None

    displayed = meta.get('displayed')
    if displayed is not None:
        assert isinstance(displayed, Sequence)
        payload_displayed = list(displayed)
    else:
        payload_displayed = None

    payload = {
        'current_step': payload_current_step,
        'ndisplay': meta.get('ndisplay'),
        'ndim': meta.get('ndim'),
        'dims_range': payload_range,
        'order': payload_order,
        'axis_labels': payload_axis_labels,
        'sizes': payload_sizes,
        'displayed': payload_displayed,
        'mode': meta.get('mode'),
        'volume': meta.get('volume'),
        'source': meta.get('source'),
    }

    loop_state.last_dims_payload = payload
    return payload


def _seed_dims_baseline(
    state: ControlStateContext,
    state_ledger: "ClientStateLedger",
    payload: dict[str, Any],
) -> None:
    current_step = payload.get('current_step') or []
    if isinstance(current_step, list):
        for idx, value in enumerate(current_step):
            if value is None:
                continue
            target_label = _axis_target_label(state, idx)
            value_int = _coerce_step_value(value)
            if value_int is None:
                continue
            state_ledger.record_confirmed(
                'dims',
                target_label,
                'index',
                value_int,
                metadata={
                    'axis_index': idx,
                    'axis_target': target_label,
                    'update_kind': 'baseline',
                },
            )

    _seed_dims_indices(state, state_ledger, payload, update_kind='baseline')

    ndisplay = payload.get('ndisplay')
    if ndisplay is not None:
        state_ledger.record_confirmed('view', 'main', 'ndisplay', int(ndisplay))

    multiscale_meta = state.dims_meta.get('multiscale')
    if isinstance(multiscale_meta, dict):
        level_val = multiscale_meta.get('level')
        if level_val is not None:
            state_ledger.record_confirmed('multiscale', 'main', 'level', int(level_val))
        policy_val = multiscale_meta.get('policy')
        if policy_val is not None:
            state_ledger.record_confirmed('multiscale', 'main', 'policy', str(policy_val))

    render_meta = state.dims_meta.get('render')
    if isinstance(render_meta, dict):
        mode_val = render_meta.get('mode') or render_meta.get('render_mode')
        if mode_val is not None:
            state_ledger.record_confirmed('volume', 'main', 'render_mode', str(mode_val))
        clim_val = render_meta.get('contrast_limits')
        if isinstance(clim_val, (list, tuple)) and len(clim_val) >= 2:
            lo = float(clim_val[0])
            hi = float(clim_val[1])
            state_ledger.record_confirmed('volume', 'main', 'contrast_limits', (lo, hi))
        cmap_val = render_meta.get('colormap')
        if cmap_val is not None:
            state_ledger.record_confirmed('volume', 'main', 'colormap', str(cmap_val))
        opacity_val = render_meta.get('opacity')
        if opacity_val is not None:
            state_ledger.record_confirmed('volume', 'main', 'opacity', float(opacity_val))
        sample_val = render_meta.get('sample_step')
        if sample_val is not None:
            state_ledger.record_confirmed('volume', 'main', 'sample_step', float(sample_val))


def _seed_dims_indices(
    state: ControlStateContext,
    state_ledger: "ClientStateLedger",
    payload: Mapping[str, Any],
    *,
    update_kind: str,
) -> None:
    current_step = payload.get('current_step')
    if not isinstance(current_step, (list, tuple)):
        return
    for idx, value in enumerate(current_step):
        if value is None:
            continue
        value_int = _coerce_step_value(value)
        if value_int is None:
            continue
        target_label = _axis_target_label(state, idx)
        metadata = {
            'axis_index': idx,
            'axis_target': target_label,
            'update_kind': update_kind,
        }
        state_ledger.record_confirmed('dims', target_label, 'index', value_int, metadata=metadata)


def _coerce_step_value(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value_int = int(value)
        if abs(float(value) - float(value_int)) < 1e-6:
            return value_int
        return None
    return None

def _apply_dims_meta_snapshot(
    meta: dict[str, object | None],
    snapshot: Mapping[str, Any],
) -> None:
    assert isinstance(snapshot, Mapping)

    if 'ndim' in snapshot:
        value = snapshot['ndim']
        assert isinstance(value, int)
        meta['ndim'] = value

    if 'ndisplay' in snapshot:
        value = snapshot['ndisplay']
        assert isinstance(value, int)
        meta['ndisplay'] = value

    for key in ('order', 'axis_labels', 'range', 'sizes', 'displayed', 'current_step'):
        if key in snapshot:
            value = snapshot[key]
            assert isinstance(value, Sequence)
            meta[key] = list(value)

    if 'volume' in snapshot:
        meta['volume'] = bool(snapshot['volume'])

    if 'render' in snapshot:
        render = snapshot['render']
        assert isinstance(render, Mapping)
        meta['render'] = dict(render)

    if 'multiscale' in snapshot:
        multiscale = snapshot['multiscale']
        assert isinstance(multiscale, Mapping)
        meta['multiscale'] = dict(multiscale)

    if 'controls' in snapshot:
        controls = snapshot['controls']
        assert isinstance(controls, Mapping)
        meta['controls'] = dict(controls)
def mirror_dims_to_viewer(
    viewer_obj,
    ui_call,
    *,
    current_step,
    ndisplay,
    ndim,
    dims_range,
    order,
    axis_labels,
    sizes,
    displayed,
) -> None:
    if viewer_obj is None or not hasattr(viewer_obj, '_apply_remote_dims_update'):
        return
    apply_remote = viewer_obj._apply_remote_dims_update  # type: ignore[attr-defined]

    def _apply() -> None:
        apply_remote(
            current_step=current_step,
            ndisplay=ndisplay,
            ndim=ndim,
            dims_range=dims_range,
            order=order,
            axis_labels=axis_labels,
            sizes=sizes,
            displayed=displayed,
        )

    if ui_call is not None:
        ui_call.call.emit(_apply)
        return
    _apply()


def handle_dims_ack(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    outcome: "AckReconciliation",
    *,
    presenter: Optional["PresenterFacade"] = None,
    viewer_ref=None,
    ui_call=None,
    log_dims_info: bool = False,
) -> None:
    if outcome.scope != "dims":
        return

    _ = viewer_ref, ui_call
    _update_runtime_from_ack_outcome(state, outcome)

    axis_idx: Optional[int] = None
    metadata = outcome.metadata or {}
    if "axis_index" in metadata:
        try:
            axis_idx = int(metadata["axis_index"])
        except Exception:
            axis_idx = None
    if axis_idx is None and outcome.target is not None:
        axis_idx = _axis_index_from_target(state, str(outcome.target))

    if outcome.status == "accepted":
        payload = _sync_dims_payload_from_meta(state, loop_state)
        if presenter is not None:
            try:
                presenter.apply_dims_update(dict(payload))
            except Exception:
                logger.debug("presenter dims update failed", exc_info=True)
        if log_dims_info and axis_idx is not None and outcome.applied_value is not None:
            logger.info(
                "dims ack accepted: axis=%s value=%s",
                axis_idx,
                outcome.applied_value,
            )
        logger.debug(
            "ack.state dims accepted: target=%s key=%s pending=%d",
            outcome.target,
            outcome.key,
            outcome.pending_len,
        )
        return

    error = outcome.error or {}
    axis_label = metadata.get("axis_target") if isinstance(metadata, dict) else None
    logger.warning(
        "ack.state dims rejected: axis=%s label=%s target=%s key=%s code=%s message=%s details=%s",
        axis_idx,
        axis_label,
        outcome.target,
        outcome.key,
        error.get("code"),
        error.get("message"),
        error.get("details"),
    )

    payload = _sync_dims_payload_from_meta(state, loop_state)
    if presenter is not None:
        try:
            presenter.apply_dims_update(dict(payload))
        except Exception:
            logger.debug("presenter dims update failed", exc_info=True)

    if log_dims_info and axis_idx is not None:
        confirmed_value = outcome.confirmed_value
        if confirmed_value is None:
            confirmed_value = state.dims_state.get((str(outcome.target or ""), str(outcome.key or "")))
        logger.info(
            "dims intent reverted: axis=%s target=%s value=%s",
            axis_idx,
            outcome.target,
            confirmed_value,
        )


def handle_generic_ack(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    outcome: "AckReconciliation",
    *,
    presenter: Optional["PresenterFacade"] = None,
) -> None:
    if outcome.scope is None:
        return

    _update_runtime_from_ack_outcome(state, outcome)

    if outcome.status == "accepted" and outcome.scope == "camera" and outcome.key is not None:
        applied = outcome.applied_value
        if applied is None:
            applied = outcome.confirmed_value or outcome.pending_value
        if applied is not None:
            state.camera_state[str(outcome.key)] = applied

    if outcome.status != "accepted":
        error = outcome.error or {}
        logger.warning(
            "ack.state rejected: scope=%s target=%s key=%s code=%s message=%s details=%s",
            outcome.scope,
            outcome.target,
            outcome.key,
            error.get("code"),
            error.get("message"),
            error.get("details"),
        )

        confirmed_value = outcome.confirmed_value
        scope = outcome.scope
        key = outcome.key or ""

        if scope == 'view':
            if confirmed_value is None:
                confirmed_value = state.view_state.get(key)
        if scope == 'dims':
            axis_idx: Optional[int] = None
            metadata = outcome.metadata or {}
            if "axis_index" in metadata:
                try:
                    axis_idx = int(metadata["axis_index"])
                except Exception:
                    axis_idx = None
            if axis_idx is None and outcome.target is not None:
                axis_idx = _axis_index_from_target(state, str(outcome.target))

        if scope == 'dims':
            axis_idx: Optional[int] = None
            metadata = outcome.metadata or {}
            if "axis_index" in metadata:
                try:
                    axis_idx = int(metadata["axis_index"])
                except Exception:
                    axis_idx = None
            if axis_idx is None and outcome.target is not None:
                axis_idx = _axis_index_from_target(state, str(outcome.target))

            if axis_idx is not None and confirmed_value is not None:
                axis_label = metadata.get("axis_target") if isinstance(metadata, dict) else None
                logger.warning(
                    "ack.state dims rejected: axis=%s label=%s target=%s key=%s code=%s message=%s details=%s",
                    axis_idx,
                    axis_label,
                    outcome.target,
                    outcome.key,
                    error.get("code"),
                    error.get("message"),
                    error.get("details"),
                )

                meta = state.dims_meta
                current = list(meta.get('current_step') or [])
                while len(current) <= axis_idx:
                    current.append(0)
                try:
                    current[axis_idx] = int(confirmed_value)
                except Exception:
                    current[axis_idx] = 0
                meta['current_step'] = current
                state.dims_state[(str(outcome.target or axis_idx), str(outcome.key or "index"))] = confirmed_value

                payload = _sync_dims_payload_from_meta(state, loop_state)
                state.dims_ready = True
                state.primary_axis_index = _compute_primary_axis_index(meta)

                if presenter is not None:
                    try:
                        presenter.apply_dims_update(dict(payload))
                    except Exception:
                        logger.debug("presenter dims update failed", exc_info=True)

                viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
                if viewer_obj is not None:
                    mirror_dims_to_viewer(
                        viewer_obj,
                        ui_call,
                        current_step=payload.get('current_step'),
                        ndisplay=payload.get('ndisplay'),
                        ndim=payload.get('ndim'),
                        dims_range=payload.get('dims_range'),
                        order=payload.get('order'),
                        axis_labels=payload.get('axis_labels'),
                        sizes=payload.get('sizes'),
                        displayed=payload.get('displayed'),
                    )
            return

        logger.warning(
            "ack.state rejected: scope=%s target=%s key=%s code=%s message=%s details=%s",
            outcome.scope,
            outcome.target,
            outcome.key,
            error.get("code"),
            error.get("message"),
            error.get("details"),
        )
        return

    logger.debug(
        "ack.state accepted: scope=%s target=%s key=%s pending=%d",
        outcome.scope,
        outcome.target,
        outcome.key,
        outcome.pending_len,
    )

    confirmed_value = outcome.confirmed_value
    scope = outcome.scope
    key = outcome.key or ""

    if scope == 'view':
        if confirmed_value is not None:
            state.view_state[key] = confirmed_value
            if key == 'ndisplay':
                try:
                    state.dims_meta['ndisplay'] = int(confirmed_value)
                except Exception:
                    state.dims_meta['ndisplay'] = None
                payload = _sync_dims_payload_from_meta(state, loop_state)
                if presenter is not None:
                    try:
                        presenter.apply_dims_update(dict(payload))
                    except Exception:
                        logger.debug("presenter dims update failed", exc_info=True)
    elif scope == 'volume' and confirmed_value is not None:
        state.volume_state[key] = confirmed_value
        render_meta = state.dims_meta.setdefault('render', {})
        if isinstance(render_meta, dict):
            render_meta[key] = confirmed_value
    elif scope == 'multiscale' and confirmed_value is not None:
        state.multiscale_state[key] = confirmed_value
        ms_meta = state.dims_meta.setdefault('multiscale', {})
        if isinstance(ms_meta, dict):
            ms_meta[key] = confirmed_value


def current_ndisplay(state: ControlStateContext) -> Optional[int]:
    return _int_or_none(state.dims_meta.get('ndisplay'))


def handle_key_event(
    data: dict,
    *,
    reset_camera: Callable[[], None],
    step_primary: Callable[[int], None],
) -> bool:
    try:
        from qtpy import QtCore  # type: ignore
    except Exception:
        logger.debug("key handler skipped: QtCore unavailable", exc_info=True)
        return False

    key_raw = data.get('key')
    key = int(key_raw) if key_raw is not None else -1
    mods = int(data.get('mods') or 0)
    txt = str(data.get('text') or '')

    keypad_mask = int(QtCore.Qt.KeypadModifier)
    keypad_only = (mods & ~keypad_mask) == 0 and (mods & keypad_mask) != 0

    if txt == '0' and mods == 0:
        logger.info("keycb: '0' -> camera.reset")
        reset_camera()
        return True
    if key == int(QtCore.Qt.Key_0) and (mods == 0 or keypad_only):
        logger.info("keycb: Key_0 -> camera.reset")
        reset_camera()
        return True

    k_left = int(QtCore.Qt.Key_Left)
    k_right = int(QtCore.Qt.Key_Right)
    k_up = int(QtCore.Qt.Key_Up)
    k_down = int(QtCore.Qt.Key_Down)
    k_pgup = int(QtCore.Qt.Key_PageUp)
    k_pgdn = int(QtCore.Qt.Key_PageDown)

    if key in (k_left, k_right, k_up, k_down, k_pgup, k_pgdn):
        coarse = 10
        if key in (k_left, k_down):
            step_primary(-1)
        elif key in (k_right, k_up):
            step_primary(+1)
        elif key == k_pgup:
            step_primary(+coarse)
        elif key == k_pgdn:
            step_primary(-coarse)
        return True

    return False


def camera_zoom(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    *,
    factor: float,
    anchor_px: tuple[float, float],
    origin: str,
) -> bool:
    sanitized = {
        "factor": float(factor),
        "anchor_px": [float(anchor_px[0]), float(anchor_px[1])],
    }
    metadata = {
        "mode": "zoom",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="zoom",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['zoom'] = sanitized
    return ok


def camera_pan(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    *,
    dx_px: float,
    dy_px: float,
    origin: str,
) -> bool:
    sanitized = {"dx_px": float(dx_px), "dy_px": float(dy_px)}
    metadata = {
        "mode": "pan",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="pan",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['pan'] = sanitized
    return ok


def camera_orbit(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    *,
    d_az_deg: float,
    d_el_deg: float,
    origin: str,
) -> bool:
    sanitized = {"d_az_deg": float(d_az_deg), "d_el_deg": float(d_el_deg)}
    metadata = {
        "mode": "orbit",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="orbit",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['orbit'] = sanitized
    return ok


def camera_reset(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    *,
    reason: str,
    origin: str,
) -> bool:
    sanitized = {"reason": str(reason)}
    metadata = {
        "mode": "reset",
        "origin": origin,
        "delta": dict(sanitized),
        "update_kind": "delta",
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="reset",
        value=sanitized,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['reset'] = sanitized
    return ok


def camera_set(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    *,
    center: Optional[Sequence[float]] = None,
    zoom: Optional[float] = None,
    angles: Optional[Sequence[float]] = None,
    origin: str,
) -> bool:
    payload: Dict[str, Any] = {}
    if center is not None:
        payload['center'] = [float(c) for c in center]
    if zoom is not None:
        payload['zoom'] = float(zoom)
    if angles is not None:
        payload['angles'] = [float(a) for a in angles]
    if not payload:
        return False
    metadata = {
        "mode": "set",
        "origin": origin,
        "delta": dict(payload),
    }
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="camera",
        target="main",
        key="set",
        value=payload,
        origin=origin,
        metadata=metadata,
    )
    if ok:
        state.camera_state['set'] = payload
    return ok


def volume_set_render_mode(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    mode: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    mode_value = str(mode)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="render_mode",
        value=mode_value,
        origin=origin,
    )
    return ok


def volume_set_clim(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    lo: float,
    hi: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    lo_f, hi_f = _ensure_lo_hi(lo, hi)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="contrast_limits",
        value=(float(lo_f), float(hi_f)),
        origin=origin,
    )
    return ok


def volume_set_colormap(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    name: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="colormap",
        value=str(name),
        origin=origin,
    )
    return ok


def volume_set_opacity(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    alpha: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    a = _clamp01(alpha)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="opacity",
        value=float(a),
        origin=origin,
    )
    return ok


def volume_set_sample_step(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    relative: float,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready or not _is_volume_mode(state):
        return False
    if _rate_gate_settings(state, origin):
        return False
    r = _clamp_sample_step(relative)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="volume",
        target="main",
        key="sample_step",
        value=float(r),
        origin=origin,
    )
    return ok


def multiscale_set_policy(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    policy: str,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready:
        return False
    if _rate_gate_settings(state, origin):
        return False
    pol = str(policy).lower().strip()
    if pol not in {'oversampling', 'thresholds', 'ratio'}:
        logger.debug("multiscale_set_policy rejected: policy=%s", pol)
        return False
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="multiscale",
        target="main",
        key="policy",
        value=pol,
        origin=origin,
    )
    return ok


def multiscale_set_level(
    state: ControlStateContext,
    loop_state: "ClientLoopState",
    state_ledger: "ClientStateLedger",
    dispatch_state_update: Callable[["IntentRecord", str], bool],
    level: int,
    *,
    origin: str,
) -> bool:
    if not state.dims_ready:
        return False
    if _rate_gate_settings(state, origin):
        return False
    lv = _clamp_level(state, level)
    ok, _ = _emit_state_update(
        state,
        loop_state,
        state_ledger,
        dispatch_state_update,
        scope="multiscale",
        target="main",
        key="level",
        value=int(lv),
        origin=origin,
    )
    return ok


def hud_snapshot(state: ControlStateContext, *, video_size: tuple[Optional[int], Optional[int]], zoom_state: dict[str, object]) -> dict[str, object]:
    meta = state.dims_meta
    snap: dict[str, object] = {}
    snap['ndisplay'] = _int_or_none(meta.get('ndisplay'))
    snap['volume'] = _bool_or_none(meta.get('volume'))
    snap['vol_mode'] = bool(_is_volume_mode(state))

    controls = meta.get('controls') if isinstance(meta.get('controls'), dict) else None
    if isinstance(controls, dict):
        snap['render_mode'] = controls.get('rendering')
        clim = controls.get('contrast_limits')
        if isinstance(clim, Sequence):
            snap['clim_lo'] = _float_or_none(clim[0] if len(clim) > 0 else None)
            snap['clim_hi'] = _float_or_none(clim[1] if len(clim) > 1 else None)
        else:
            snap['clim_lo'] = None
            snap['clim_hi'] = None
        snap['colormap'] = controls.get('colormap')
        snap['opacity'] = _float_or_none(controls.get('opacity'))
    render_meta = meta.get('render') if isinstance(meta.get('render'), dict) else None
    if isinstance(render_meta, dict):
        snap['sample_step'] = _float_or_none(render_meta.get('sample_step'))
    elif 'sample_step' not in snap:
        snap['sample_step'] = None

    multiscale = meta.get('multiscale') if isinstance(meta.get('multiscale'), dict) else None
    if isinstance(multiscale, dict):
        snap['ms_policy'] = multiscale.get('policy')
        level_value = _int_or_none(multiscale.get('current_level'))
        snap['ms_level'] = level_value
        levels_obj = multiscale.get('levels')
        if isinstance(levels_obj, Sequence):
            snap['ms_levels'] = len(levels_obj)
            if level_value is not None and 0 <= level_value < len(levels_obj):
                entry = levels_obj[level_value]
                snap['ms_path'] = entry.get('path') if isinstance(entry, dict) else None
            else:
                snap['ms_path'] = None
        else:
            snap['ms_levels'] = None
            snap['ms_path'] = None

    snap['primary_axis'] = _int_or_none(state.primary_axis_index)
    snap.update(zoom_state)
    video_w, video_h = video_size
    snap['video_w'] = _int_or_none(video_w)
    snap['video_h'] = _int_or_none(video_h)
    return snap


def _axis_to_index(state: ControlStateContext, axis: int | str) -> Optional[int]:
    if axis == 'primary':
        return int(state.primary_axis_index) if state.primary_axis_index is not None else 0
    if isinstance(axis, (int, float)) or (isinstance(axis, str) and str(axis).isdigit()):
        return int(axis)
    labels = state.dims_meta.get('axis_labels')
    if isinstance(labels, Sequence):
        label_map = {str(lbl): i for i, lbl in enumerate(labels)}
        match = label_map.get(str(axis))
        return int(match) if match is not None else None
    return None


def _compute_primary_axis_index(meta: dict[str, object | None]) -> Optional[int]:
    order = meta.get('order')
    ndisplay = meta.get('ndisplay')
    labels = meta.get('axis_labels')
    nd = int(ndisplay) if ndisplay is not None else 2
    idx_order: list[int] | None = None
    if isinstance(order, Sequence) and len(order) > 0:
        if all(isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()) for x in order):
            idx_order = [int(x) for x in order]
        elif isinstance(labels, Sequence) and all(isinstance(x, str) for x in order):
            label_to_index = {str(lbl): i for i, lbl in enumerate(labels)}
            idx_order = [int(label_to_index.get(str(lbl), i)) for i, lbl in enumerate(order)]
    if idx_order and len(idx_order) > nd:
        return int(idx_order[0])
    return 0


def _rate_gate_settings(state: ControlStateContext, origin: str) -> bool:
    now = time.perf_counter()
    if (now - float(state.last_settings_send or 0.0)) < state.settings_min_dt:
        logger.debug("settings intent gated by rate limiter (%s)", origin)
        return True
    state.last_settings_send = now
    return False


def _is_volume_mode(state: ControlStateContext) -> bool:
    mode = str(state.dims_meta.get('mode') or '').lower()
    nd = int(state.dims_meta.get('ndisplay') or 2)
    if mode:
        return mode == 'volume' and nd == 3
    vol = bool(state.dims_meta.get('volume'))
    return vol and nd == 3


def _is_axis_playing(viewer_obj, axis_index: int) -> bool:
    if viewer_obj is None:
        return False
    is_playing = bool(getattr(viewer_obj, '_is_playing', False))
    play_axis = getattr(viewer_obj, '_play_axis', None)
    return bool(is_playing) and play_axis is not None and int(play_axis) == int(axis_index)


def _clamp01(a: float) -> float:
    a = float(a)
    if a < 0.0:
        return 0.0
    if a > 1.0:
        return 1.0
    return a


def _clamp_sample_step(r: float) -> float:
    r = float(r)
    if r < 0.1:
        return 0.1
    if r > 4.0:
        return 4.0
    return r


def _ensure_lo_hi(lo: float, hi: float) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f <= lo_f:
        lo_f, hi_f = hi_f, lo_f
    return lo_f, hi_f


def _clamp_level(state: ControlStateContext, level: int) -> int:
    multiscale = state.dims_meta.get('multiscale')
    if isinstance(multiscale, dict):
        levels = multiscale.get('levels')
        if isinstance(levels, Sequence) and levels:
            lo, hi = 0, len(levels) - 1
            lv = int(level)
            if lv < lo:
                return lo
            if lv > hi:
                return hi
            return lv
    return int(level)


def _int_or_none(value: object) -> Optional[int]:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _float_or_none(value: object) -> Optional[float]:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _bool_or_none(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)
