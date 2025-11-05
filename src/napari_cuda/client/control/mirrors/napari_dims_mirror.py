"""Mirror consumer dims (and multiscale) state from the client ledger into napari."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Optional

from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import (
    ClientStateLedger,
    MirrorEvent,
)
from napari_cuda.protocol.messages import NotifyDimsFrame
from napari_cuda.shared.axis_spec import (
    AxisSpec,
    axis_by_index,
    axis_spec_to_payload,
)

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.control.emitters.napari_dims_intent_emitter import (
        NapariDimsIntentEmitter,
    )
    from napari_cuda.client.control.state_update_actions import (
        ControlStateContext,
    )
    from napari_cuda.client.rendering.presenter_facade import PresenterFacade
    from napari_cuda.client.runtime.client_loop.loop_state import (
        ClientLoopState,
    )

logger = logging.getLogger(__name__)



class NapariDimsMirror:
    """Subscribe to dims-related ledger events and mirror them into napari."""

    def __init__(
        self,
        *,
        ledger: ClientStateLedger,
        state: ControlStateContext,
        loop_state: ClientLoopState,
        viewer_ref,
        ui_call,
        presenter: Optional[PresenterFacade],
        log_dims_info: bool,
        notify_first_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        app = QtCore.QCoreApplication.instance()
        assert app is not None, "Qt application instance must exist"
        self._ledger = ledger
        self._state = state
        self._loop_state = loop_state
        self._viewer_ref = viewer_ref
        self._ui_call = ui_call
        self._presenter = presenter
        self._log_dims_info = bool(log_dims_info)
        self._notify_first_ready = notify_first_ready
        self._emitter: Optional[NapariDimsIntentEmitter] = None
        self._ui_thread = app.thread()
        self._last_multiscale_snapshot: Mapping[str, Any] | None = None
        ledger.subscribe_all(self._handle_ledger_update)

    def set_logging(self, enabled: bool) -> None:
        self._log_dims_info = bool(enabled)

    def attach_emitter(self, emitter: NapariDimsIntentEmitter) -> None:
        self._emitter = emitter

    def ingest_dims_notify(self, frame: NotifyDimsFrame) -> None:
        """Record a ``notify.dims`` snapshot and mirror it into napari."""

        self._assert_gui_thread()
        state = self._state
        meta = state.dims_meta
        was_ready = bool(state.dims_ready)
        payload = frame.payload

        axis_spec = payload.axes_spec
        state.axis_spec = axis_spec
        meta['axis_spec'] = axis_spec

        storage_step = [int(value) for value in payload.current_step]
        meta['current_step'] = storage_step
        meta['ndisplay'] = axis_spec.ndisplay
        mode_text = 'volume' if not axis_spec.plane_mode else 'plane'
        meta['mode'] = mode_text
        meta['volume'] = not axis_spec.plane_mode

        level_shapes = [[int(dim) for dim in shape] for shape in axis_spec.level_shapes]
        meta['level_shapes'] = level_shapes

        active_level = int(axis_spec.current_level)
        assert 0 <= active_level < len(level_shapes), 'active level out of bounds for level_shapes'
        active_shape = [int(dim) for dim in level_shapes[active_level]]
        meta['active_level_shape'] = active_shape
        meta['range'] = [[0, max(0, dim - 1)] for dim in active_shape]
        meta['ndim'] = len(active_shape)

        meta['order'] = [int(idx) for idx in axis_spec.order]
        meta['axis_labels'] = [axis.label for axis in axis_spec.axes]
        meta['displayed'] = [int(idx) for idx in axis_spec.displayed]

        if 'sizes' in meta:
            meta.pop('sizes', None)

        multiscale_snapshot = _build_multiscale_snapshot(payload)
        if multiscale_snapshot is not None:
            meta['multiscale'] = multiscale_snapshot
            state.multiscale_state = {
                'level': multiscale_snapshot['level'],
                'current_level': multiscale_snapshot['current_level'],
                'levels': [dict(entry) for entry in multiscale_snapshot['levels']],
            }
            if 'downgraded' in multiscale_snapshot:
                state.multiscale_state['downgraded'] = multiscale_snapshot['downgraded']
        else:
            meta['multiscale'] = None
            state.multiscale_state = {}
        self._apply_multiscale_to_presenter(multiscale_snapshot)

        snapshot_payload = _build_consumer_dims_payload(state, self._loop_state)

        if not was_ready:
            _record_dims_snapshot(state, self._ledger, snapshot_payload)
        else:
            _record_dims_delta(state, self._ledger, snapshot_payload, update_kind='notify')

        if not state.dims_ready:
            state.dims_ready = True
            logger.info('notify.dims: metadata received; client intents enabled')
            if self._notify_first_ready is not None:
                self._notify_first_ready()

        state.primary_axis_index = _compute_primary_axis_index(meta)

        if logger.isEnabledFor(logging.DEBUG):
            level_val = multiscale_snapshot['current_level'] if multiscale_snapshot else None
            logger.debug(
                'ingest_dims_notify: frame=%s step=%s level=%s shape=%s',
                frame.envelope.frame_id,
                storage_step,
                level_val,
                active_shape,
            )
        if storage_step and self._log_dims_info:
            logger.info(
                'notify.dims: step=%s ndisplay=%s order=%s labels=%s',
                storage_step,
                meta.get('ndisplay'),
                meta.get('order'),
                meta.get('axis_labels'),
            )

        self._mirror_confirmed_dims(reason='notify', payload=snapshot_payload)


    def replay_last_payload(self) -> None:
        """Replay the last consumer dims payload into the viewer."""

        payload = self._loop_state.last_dims_payload
        if not payload:
            return
        self._mirror_confirmed_dims(reason="replay", payload=payload)

    def refresh_from_state(self, *, reason: str) -> None:
        """Mirror the current consumer dims payload into the viewer."""

        self._mirror_confirmed_dims(reason=reason)

    # ------------------------------------------------------------------
    def _handle_ledger_update(self, update: MirrorEvent) -> None:
        scope = update.scope
        if scope == 'dims' and update.key in {'index', 'step'}:
            self._handle_axis_update(update)
            return
        if scope == 'view' and update.target == 'main' and update.key == 'ndisplay':
            self._handle_ndisplay_update(update)

    def _handle_axis_update(self, update: MirrorEvent) -> None:
        metadata = update.metadata or {}
        axis_idx: Optional[int] = None
        if 'axis_index' in metadata:
            try:
                axis_idx = int(metadata['axis_index'])
            except Exception:
                axis_idx = None
        if axis_idx is None:
            axis_idx = _axis_index_from_target(self._state, str(update.target))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dims ledger update: axis=%s target=%s key=%s value=%s metadata=%s",
                axis_idx,
                update.target,
                update.key,
                update.value,
                metadata,
            )

        try:
            value_int = int(update.value)
        except Exception as exc:  # pragma: no cover - intentional crash path
            raise AssertionError(f"dims value must be int-like: {update.value!r}") from exc

        meta = self._state.dims_meta
        current = list(meta.get('current_step') or [])
        while len(current) <= axis_idx:
            current.append(0)
        current[axis_idx] = value_int
        meta['current_step'] = current

        if self._state.axis_spec is not None:
            meta['axis_spec'] = self._state.axis_spec

        self._state.dims_state[(str(update.target), str(update.key))] = value_int
        self._state.primary_axis_index = _compute_primary_axis_index(meta)
        self._mirror_confirmed_dims(reason=f"axis:{axis_idx}")

    def _handle_ndisplay_update(self, update: MirrorEvent) -> None:
        try:
            ndisplay = int(update.value)
        except Exception as exc:  # pragma: no cover - intentional crash path
            raise AssertionError(f"ndisplay value must be int-like: {update.value!r}") from exc
        self._state.dims_meta['ndisplay'] = ndisplay
        self._mirror_confirmed_dims(reason="ndisplay")

    def _mirror_confirmed_dims(
        self,
        *,
        reason: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        confirmed_payload = (
            dict(payload)
            if payload is not None
            else _build_consumer_dims_payload(self._state, self._loop_state)
        )

        # Multiscale snapshot already applied during ingest_dims_notify.

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dims mirror (%s): payload step=%s range=%s",
                reason,
                confirmed_payload.get('current_step'),
                confirmed_payload.get('dims_range'),
            )

        if self._log_dims_info and confirmed_payload.get('current_step') is not None:
            logger.info(
                "dims mirror (%s): step=%s ndisplay=%s",
                reason,
                confirmed_payload.get('current_step'),
                confirmed_payload.get('ndisplay'),
            )

        emitter = self._emitter
        assert emitter is not None, "mirror requires a dims emitter"
        with emitter.suppressing():
            if self._presenter is not None:
                self._presenter.apply_dims_update(dict(confirmed_payload))

        viewer_obj = self._viewer_ref() if callable(self._viewer_ref) else None  # type: ignore[misc]
        mirror_dims_to_viewer(
            viewer_obj,
            self._ui_call,
            current_step=confirmed_payload.get('current_step'),
            ndisplay=confirmed_payload.get('ndisplay'),
            ndim=confirmed_payload.get('ndim'),
            dims_range=confirmed_payload.get('dims_range'),
            order=confirmed_payload.get('order'),
            axis_labels=confirmed_payload.get('axis_labels'),
            displayed=confirmed_payload.get('displayed'),
        )

    def _apply_multiscale_to_presenter(self, multiscale: Mapping[str, Any] | None) -> None:
        self._last_multiscale_snapshot = dict(multiscale) if multiscale is not None else None
        presenter = self._presenter
        if presenter is None or multiscale is None:
            return
        presenter.apply_multiscale_policy(dict(multiscale))

    def _assert_gui_thread(self) -> None:
        current = QtCore.QThread.currentThread()
        assert current is self._ui_thread, "NapariDimsMirror methods must run on the Qt GUI thread"


# Shared helpers (lifted from the legacy state_update_actions module)


def _record_dims_snapshot(
    state: ControlStateContext,
    ledger: ClientStateLedger,
    payload: Mapping[str, Any],
) -> None:
    _record_dims_delta(state, ledger, payload, update_kind='snapshot')

    ndisplay = payload.get('ndisplay')
    entries: list[tuple[Any, ...]] = []
    if ndisplay is not None:
        entries.append(('view', 'main', 'ndisplay', int(ndisplay)))

    order = payload.get('order')
    if isinstance(order, (list, tuple)):
        entries.append(('view', 'main', 'order', tuple(int(v) for v in order)))

    displayed = payload.get('displayed')
    if isinstance(displayed, (list, tuple)):
        entries.append(('view', 'main', 'displayed', tuple(int(v) for v in displayed)))

    if state.axis_spec is None:
        raise AssertionError('axis spec missing in control state during snapshot record')
    entries.append(('dims', 'main', 'axes', axis_spec_to_payload(state.axis_spec)))

    if entries:
        ledger.batch_record_confirmed(entries)

    _record_multiscale_metadata(state, ledger)


def _build_multiscale_snapshot(payload: Any) -> dict[str, Any] | None:
    levels_source = getattr(payload, 'levels', None)
    if not levels_source:
        return None
    levels = [dict(entry) for entry in levels_source]
    level_value = int(payload.current_level)
    snapshot: dict[str, Any] = {
        'levels': levels,
        'current_level': level_value,
        'level': level_value,
    }
    downgraded = getattr(payload, 'downgraded', None)
    if downgraded is not None:
        snapshot['downgraded'] = bool(downgraded)
    return snapshot


def _record_dims_delta(
    state: ControlStateContext,
    ledger: ClientStateLedger,
    payload: Mapping[str, Any],
    *,
    update_kind: str,
) -> None:
    current_step = payload.get('current_step')
    if state.axis_spec is None:
        raise AssertionError('axis spec missing in control state during delta record')
    spec_payload = axis_spec_to_payload(state.axis_spec)
    if not isinstance(current_step, (list, tuple)):
        ledger.batch_record_confirmed([('dims', 'main', 'axes', spec_payload)])
        return
    entries: list[tuple[Any, ...]] = []
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
        entries.append(('dims', target_label, 'index', value_int, metadata))

    entries.append(('dims', 'main', 'axes', spec_payload))

    if entries:
        ledger.batch_record_confirmed(entries)


def _build_consumer_dims_payload(state: ControlStateContext, loop_state: ClientLoopState) -> dict[str, Any]:
    meta = state.dims_meta
    payload: dict[str, Any] = {}

    current_step = meta.get('current_step')
    if isinstance(current_step, Sequence):
        payload['current_step'] = [int(value) for value in current_step]
    else:
        payload['current_step'] = None

    spec = state.axis_spec

    if spec is None:
        raise AssertionError('axis spec missing from control state')

    payload['ndisplay'] = int(spec.ndisplay)
    payload['ndim'] = int(spec.ndim)

    dims_range = meta.get('range')
    if isinstance(dims_range, Sequence):
        payload['dims_range'] = [list(pair) for pair in dims_range]  # ensure JSON-friendly
    else:
        payload['dims_range'] = None

    payload['order'] = [int(idx) for idx in spec.order]
    payload['axis_labels'] = [axis.label for axis in spec.axes]
    payload['displayed'] = [int(idx) for idx in spec.displayed]
    payload['axes_spec'] = axis_spec_to_payload(spec)

    mode = meta.get('mode')
    payload['mode'] = str(mode) if mode is not None else None

    volume_flag = meta.get('volume')
    payload['volume'] = bool(volume_flag) if volume_flag is not None else None

    payload['source'] = meta.get('source')

    loop_state.last_dims_payload = dict(payload)
    return payload


def _record_multiscale_metadata(state: ControlStateContext, ledger: ClientStateLedger) -> None:
    multiscale_meta = state.dims_meta.get('multiscale')
    if not isinstance(multiscale_meta, Mapping):
        return

    entries: list[tuple[Any, ...]] = []

    level_val = multiscale_meta.get('level')
    if level_val is None:
        level_val = multiscale_meta.get('current_level')
    if level_val is not None:
        entries.append(('multiscale', 'main', 'level', int(level_val)))

    policy_val = multiscale_meta.get('policy')
    if policy_val is not None:
        entries.append(('multiscale', 'main', 'policy', str(policy_val)))

    downgraded = multiscale_meta.get('downgraded')
    if downgraded is not None:
        entries.append(('multiscale', 'main', 'downgraded', bool(downgraded)))

    if entries:
        ledger.batch_record_confirmed(entries)

def _record_volume_metadata(state: ControlStateContext, ledger: ClientStateLedger) -> None:  # pragma: no cover - removed
    return


def _axis_index_from_target(state: ControlStateContext, target: str) -> int:
    if state.axis_spec is None:
        raise AssertionError('axis spec unavailable for target lookup')
    text = target.strip()
    if text == '':
        raise ValueError('axis target must not be empty')
    if text.lstrip('-').isdigit():
        return state.axis_spec.axis_by_index(int(text)).index
    lowered = text.lower()
    if lowered.startswith('axis-') and lowered[5:].isdigit():
        return state.axis_spec.axis_by_index(int(lowered[5:])).index
    return state.axis_spec.axis_by_label(text).index


def _compute_primary_axis_index(meta: dict[str, object | None]) -> int:
    axis_spec_obj = meta.get('axis_spec')
    if not isinstance(axis_spec_obj, AxisSpec):
        raise AssertionError('axis spec missing for primary axis compute')
    order = axis_spec_obj.order
    assert order, "axis spec missing order sequence"
    nd = max(1, axis_spec_obj.ndisplay)
    return int(order[0] if len(order) > nd else order[0])


def _axis_target_label(state: ControlStateContext, axis_idx: int) -> str:
    if state.axis_spec is None:
        raise AssertionError('axis spec unavailable for target label lookup')
    extent = axis_by_index(state.axis_spec, axis_idx)
    return extent.label


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
            displayed=displayed,
        )

    if ui_call is not None:
        ui_call.call.emit(_apply)
        return
    _apply()



__all__ = [
    "NapariDimsMirror",
    "mirror_dims_to_viewer",
]
