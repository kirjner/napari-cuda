"""Mirror consumer dims (and multiscale) state from the client ledger into napari."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Optional

from qtpy import QtCore

from napari_cuda.client.control import state_update_actions as control_actions
from napari_cuda.client.control.client_state_ledger import (
    ClientStateLedger,
    MirrorEvent,
)
from napari_cuda.protocol.messages import NotifyDimsFrame
from napari_cuda.shared.dims_spec import (
    DimsSpec,
    dims_spec_axis_index_for_target,
    dims_spec_axis_labels,
    dims_spec_primary_axis,
    dims_spec_to_payload,
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
        was_ready = bool(state.dims_ready)
        payload = frame.payload

        dims_spec = payload.dims_spec
        assert isinstance(dims_spec, DimsSpec), 'notify.dims requires dims_spec'

        state.dims_spec = dims_spec
        state.dims_step_override = tuple(int(value) for value in dims_spec.current_step)
        state.dims_ndisplay_override = int(dims_spec.ndisplay)
        state.primary_axis_index = dims_spec_primary_axis(dims_spec)

        multiscale_snapshot = _build_multiscale_snapshot(payload)
        if multiscale_snapshot is not None:
            state.multiscale_state = {
                'level': multiscale_snapshot['level'],
                'current_level': multiscale_snapshot['current_level'],
                'levels': [dict(entry) for entry in multiscale_snapshot['levels']],
            }
            if 'downgraded' in multiscale_snapshot:
                state.multiscale_state['downgraded'] = multiscale_snapshot['downgraded']
        else:
            state.multiscale_state = {}
        self._apply_multiscale_to_presenter(multiscale_snapshot)

        snapshot_payload = control_actions.build_dims_payload(state)
        self._loop_state.last_dims_payload = dict(snapshot_payload)
        self._loop_state.last_dims_spec = state.last_dims_spec

        axis_labels = snapshot_payload.get('axis_labels') or []
        current_step = snapshot_payload.get('current_step') or []
        entries: list[tuple[Any, ...]] = []
        for axis_idx, raw_value in enumerate(current_step):
            label = axis_labels[axis_idx] if axis_idx < len(axis_labels) else str(axis_idx)
            entries.append(('dims', str(label), 'index', int(raw_value)))
        entries.append(("dims", "main", "dims_spec", dims_spec_to_payload(dims_spec)))

        ndisplay = snapshot_payload.get('ndisplay')
        if ndisplay is not None:
            entries.append(('view', 'main', 'ndisplay', int(ndisplay)))

        order = snapshot_payload.get('order')
        if isinstance(order, (list, tuple)):
            entries.append(('view', 'main', 'order', tuple(int(v) for v in order)))

        displayed = snapshot_payload.get('displayed')
        if isinstance(displayed, (list, tuple)):
            entries.append(('view', 'main', 'displayed', tuple(int(v) for v in displayed)))

        level_snapshot = state.multiscale_state
        if isinstance(level_snapshot, Mapping):
            if 'level' in level_snapshot:
                entries.append(('multiscale', 'main', 'level', int(level_snapshot['level'])))
            if 'policy' in level_snapshot:
                entries.append(('multiscale', 'main', 'policy', level_snapshot['policy']))
            if 'downgraded' in level_snapshot:
                entries.append(('multiscale', 'main', 'downgraded', bool(level_snapshot['downgraded'])))
        if entries:
            self._ledger.batch_record_confirmed(entries)

        if not state.dims_ready:
            state.dims_ready = True
            logger.info('notify.dims: metadata received; client intents enabled')
            if self._notify_first_ready is not None:
                self._notify_first_ready()

        if logger.isEnabledFor(logging.DEBUG):
            level_val = multiscale_snapshot['current_level'] if multiscale_snapshot else None
            logger.debug(
                'ingest_dims_notify: frame=%s step=%s level=%s shape=%s',
                frame.envelope.frame_id,
                list(snapshot_payload.get('current_step') or ()),
                level_val,
                snapshot_payload.get('dims_range'),
            )
        if snapshot_payload.get('current_step') and self._log_dims_info:
            logger.info(
                'notify.dims: step=%s ndisplay=%s order=%s labels=%s',
                snapshot_payload.get('current_step'),
                snapshot_payload.get('ndisplay'),
                snapshot_payload.get('order'),
                snapshot_payload.get('axis_labels'),
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
        spec = control_actions.ensure_dims_spec(self._state)
        target = str(update.target)
        axis_idx = dims_spec_axis_index_for_target(spec, target)
        assert axis_idx is not None, f"unknown dims axis target={update.target!r}"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dims ledger update: axis=%s target=%s key=%s value=%s metadata=%s",
                axis_idx,
                update.target,
                update.key,
                update.value,
                update.metadata,
            )

        value_int = int(update.value)

        state = self._state
        current = list(control_actions.resolve_current_step(state) or spec.current_step)
        while len(current) <= axis_idx:
            current.append(0)
        current[axis_idx] = value_int
        state.dims_step_override = tuple(int(v) for v in current)
        state.primary_axis_index = dims_spec_primary_axis(spec)
        self._mirror_confirmed_dims(reason=f"axis:{axis_idx}")

    def _handle_ndisplay_update(self, update: MirrorEvent) -> None:
        self._state.dims_ndisplay_override = int(update.value)
        self._mirror_confirmed_dims(reason="ndisplay")

    def _mirror_confirmed_dims(
        self,
        *,
        reason: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if payload is not None:
            confirmed_payload = dict(payload)
        else:
            snapshot = control_actions.build_dims_payload(self._state)
            confirmed_payload = dict(snapshot)
            self._loop_state.last_dims_payload = dict(confirmed_payload)
        if self._state.last_dims_spec is not None:
            self._loop_state.last_dims_spec = self._state.last_dims_spec

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

        viewer = self._viewer_ref() if callable(self._viewer_ref) else self._viewer_ref  # type: ignore[misc]
        assert viewer is not None, "viewer reference must be alive"
        mirror_dims_to_viewer(
            viewer,
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
