"""Mirror consumer dims (and multiscale) state from the client ledger into napari."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Optional

from qtpy import QtCore

from napari_cuda.client.control.control_state import ControlStateContext, _mirror_viewer_dims
from napari_cuda.client.control.dims_projection import project_dims, viewer_update_from_spec
from napari_cuda.client.control.client_state_ledger import (
    AckReconciliation,
    ClientStateLedger,
    MirrorEvent,
)
from napari_cuda.protocol.messages import NotifyDimsFrame
from napari_cuda.shared.dims_spec import DimsSpec, dims_spec_axis_index_for_target, dims_spec_to_payload

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.control.emitters.napari_dims_intent_emitter import (
        NapariDimsIntentEmitter,
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
        ledger.subscribe_all(self._handle_ledger_update)

    def set_logging(self, enabled: bool) -> None:
        self._log_dims_info = bool(enabled)

    def attach_emitter(self, emitter: NapariDimsIntentEmitter) -> None:
        self._emitter = emitter

    def ingest_dims_notify(self, frame: NotifyDimsFrame) -> None:
        """Record a ``notify.dims`` snapshot and mirror it into napari."""

        self._assert_gui_thread()
        state = self._state
        payload = frame.payload

        dims_spec = payload.dims_spec
        assert isinstance(dims_spec, DimsSpec), 'notify.dims requires dims_spec'

        state.dims_spec = dims_spec
        projection = project_dims(dims_spec)
        state.primary_axis_index = projection.primary_axis

        axis_index_entries = [
            ('dims', axis.label, 'index', int(axis.current_step))
            for axis in dims_spec.axes
        ]
        self._ledger.batch_record_confirmed(
            axis_index_entries,
            timestamp=frame.envelope.timestamp,
        )

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

        viewer_update = viewer_update_from_spec(dims_spec, projection)
        self._loop_state.last_dims_spec = dims_spec

        entries: list[tuple[Any, ...]] = [
            ("dims", "main", "dims_spec", dims_spec_to_payload(dims_spec)),
            ('view', 'main', 'ndisplay', int(viewer_update['ndisplay'])),
        ]

        order = viewer_update['order']
        if order:
            entries.append(('view', 'main', 'order', order))

        displayed = viewer_update['displayed']
        if displayed:
            entries.append(('view', 'main', 'displayed', displayed))

        level_snapshot = state.multiscale_state
        if isinstance(level_snapshot, Mapping):
            if 'level' in level_snapshot:
                entries.append(('multiscale', 'main', 'level', int(level_snapshot['level'])))
            if 'policy' in level_snapshot:
                entries.append(('multiscale', 'main', 'policy', level_snapshot['policy']))
            if 'downgraded' in level_snapshot:
                entries.append(('multiscale', 'main', 'downgraded', bool(level_snapshot['downgraded'])))
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
                list(viewer_update['current_step']),
                level_val,
                viewer_update['dims_range'],
            )
        if viewer_update['current_step'] and self._log_dims_info:
            logger.info(
                'notify.dims: step=%s ndisplay=%s order=%s labels=%s',
                viewer_update['current_step'],
                viewer_update['ndisplay'],
                viewer_update['order'],
                viewer_update['axis_labels'],
            )

        self._mirror_confirmed_dims(reason='notify', spec=dims_spec)


    def replay_last_spec(self) -> None:
        """Replay the last confirmed dims spec into the viewer."""

        spec = self._loop_state.last_dims_spec
        if spec is None:
            return
        self._mirror_confirmed_dims(reason="replay", spec=spec)

    def handle_ack(self, outcome: AckReconciliation) -> None:
        scope = outcome.scope
        if scope == 'dims':
            spec = self._state.dims_spec
            assert spec is not None, "dims_spec must be available"
            axis_idx: Optional[int] = None
            if outcome.target is not None:
                axis_idx = dims_spec_axis_index_for_target(spec, str(outcome.target))
            if outcome.status == 'accepted':
                if self._log_dims_info and axis_idx is not None and outcome.applied_value is not None:
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
            else:
                error = outcome.error or {}
                logger.warning(
                    "ack.state dims rejected: axis=%s target=%s key=%s code=%s message=%s details=%s",
                    axis_idx,
                    outcome.target,
                    outcome.key,
                    error.get("code"),
                    error.get("message"),
                    error.get("details"),
                )
                if self._log_dims_info and axis_idx is not None:
                    logger.info(
                        "dims intent reverted: axis=%s target=%s value=%s",
                        axis_idx,
                        outcome.target,
                        outcome.confirmed_value,
                    )
            self._mirror_confirmed_dims(reason="ack:dims", spec=spec)
            return

        if scope == 'view' and outcome.target == 'main' and outcome.key == 'ndisplay':
            if outcome.status != 'accepted':
                error = outcome.error or {}
                logger.warning(
                    "ack.state view rejected: key=ndisplay code=%s message=%s details=%s",
                    error.get("code"),
                    error.get("message"),
                    error.get("details"),
                )
            self._mirror_confirmed_dims(reason="ack:ndisplay")
            return

    def refresh_from_state(self, *, reason: str) -> None:
        """Mirror the current consumer dims state into the viewer."""

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
        spec = self._state.dims_spec
        assert spec is not None, "dims_spec must be available"
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

        self._mirror_confirmed_dims(reason=f"axis:{axis_idx}", spec=spec)

    def _handle_ndisplay_update(self, update: MirrorEvent) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dims ledger update: target=%s key=ndisplay value=%s metadata=%s",
                update.target,
                update.value,
                update.metadata,
            )
        self._mirror_confirmed_dims(reason="ndisplay")

    def _mirror_confirmed_dims(
        self,
        *,
        reason: str,
        spec: Optional[DimsSpec] = None,
    ) -> None:
        if spec is None:
            spec = self._state.dims_spec
            assert spec is not None, "dims_spec must be available"

        projection = project_dims(spec, self._ledger)
        viewer_update_local = viewer_update_from_spec(spec, projection)

        self._loop_state.last_dims_spec = spec
        self._state.primary_axis_index = projection.primary_axis

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dims mirror (%s): step=%s range=%s",
                reason,
                viewer_update_local['current_step'],
                viewer_update_local['dims_range'],
            )

        if self._log_dims_info and viewer_update_local['current_step']:
            logger.info(
                "dims mirror (%s): step=%s ndisplay=%s",
                reason,
                viewer_update_local['current_step'],
                viewer_update_local['ndisplay'],
            )

        emitter = self._emitter
        assert emitter is not None, "mirror requires a dims emitter"
        with emitter.suppressing():
            if self._presenter is not None:
                self._presenter.apply_dims_update(spec=spec, viewer_update=viewer_update_local)

        viewer = self._viewer_ref() if callable(self._viewer_ref) else self._viewer_ref  # type: ignore[misc]
        assert viewer is not None, "viewer reference must be alive"
        _mirror_viewer_dims(viewer, self._ui_call, viewer_update_local)

    def _apply_multiscale_to_presenter(self, multiscale: Mapping[str, Any] | None) -> None:
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


__all__ = [
    "NapariDimsMirror",
]
