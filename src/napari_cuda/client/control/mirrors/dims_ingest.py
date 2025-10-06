"""Ingest notify.dims frames and seed the client state ledger."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence, TYPE_CHECKING

from napari_cuda.protocol.messages import NotifyDimsFrame

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.mirrors.napari_dims_mirror import (
    _sync_dims_payload_from_meta,
    _seed_dims_baseline,
    _seed_dims_indices,
    _compute_primary_axis_index,
)

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.control.state_update_actions import ControlStateContext
    from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
    from napari_cuda.client.rendering.presenter_facade import PresenterFacade

logger = logging.getLogger(__name__)


def ingest_notify_dims(
    state: "ControlStateContext",
    loop_state: "ClientLoopState",
    frame: NotifyDimsFrame,
    *,
    presenter: "PresenterFacade",
    viewer_ref,
    ui_call,
    notify_first_dims_ready: Callable[[], None],
    log_dims_info: bool,
    state_ledger: Optional[ClientStateLedger] = None,
) -> None:
    """Process a notify.dims frame and seed the ledger before mirroring."""

    meta = state.dims_meta
    was_ready = bool(state.dims_ready)

    if logger.isEnabledFor(logging.DEBUG):
        payload = frame.payload
        logger.debug(
            "notify.dims ingest: frame=%s current_step=%s ndisplay=%s mode=%s",
            frame.envelope.frame_id,
            tuple(int(x) for x in payload.current_step),
            payload.ndisplay,
            payload.mode,
        )

    payload = frame.payload
    current_step = tuple(int(value) for value in payload.current_step)
    meta['current_step'] = list(current_step)
    meta['ndisplay'] = int(payload.ndisplay)
    mode_text = str(payload.mode)
    meta['mode'] = mode_text
    meta['volume'] = bool(mode_text.lower() == 'volume')
    meta['source'] = payload.source

    payload_dict = _sync_dims_payload_from_meta(state, loop_state)

    if state_ledger is not None:
        if not was_ready:
            _seed_dims_baseline(state, state_ledger, payload_dict)
        else:
            _seed_dims_indices(state, state_ledger, payload_dict, update_kind='notify')

    if not state.dims_ready:
        state.dims_ready = True
        logger.info("notify.dims: metadata received; client intents enabled")
        notify_first_dims_ready()

    state.primary_axis_index = _compute_primary_axis_index(meta)

    if current_step and log_dims_info:
        logger.info(
            "notify.dims: step=%s ndisplay=%s order=%s labels=%s",
            list(current_step),
            meta.get('ndisplay'),
            meta.get('order'),
            meta.get('axis_labels'),
        )

    presenter.apply_dims_update(dict(payload_dict))

    viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
    if viewer_obj is None:
        return
    from napari_cuda.client.control.mirrors.napari_dims_mirror import mirror_dims_to_viewer

    mirror_dims_to_viewer(
        viewer_obj,
        ui_call,
        current_step=payload_dict.get('current_step'),
        ndisplay=payload_dict.get('ndisplay'),
        ndim=payload_dict.get('ndim'),
        dims_range=payload_dict.get('dims_range'),
        order=payload_dict.get('order'),
        axis_labels=payload_dict.get('axis_labels'),
        sizes=payload_dict.get('sizes'),
        displayed=payload_dict.get('displayed'),
    )


__all__ = ["ingest_notify_dims"]
