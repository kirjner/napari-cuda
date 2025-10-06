"""Mirror confirmed dims state from the client ledger into napari."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

from napari_cuda.client.control.client_state_ledger import ClientStateLedger, MirrorEvent

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.client.control.state_update_actions import ControlStateContext
    from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
    from napari_cuda.client.rendering.presenter_facade import PresenterFacade

logger = logging.getLogger(__name__)


class NapariDimsMirror:
    """Subscribe to dims-related ledger events and mirror them into napari."""

    def __init__(
        self,
        *,
        ledger: ClientStateLedger,
        state: "ControlStateContext",
        loop_state: "ClientLoopState",
        viewer_ref,
        ui_call,
        presenter: Optional["PresenterFacade"],
        log_dims_info: bool,
    ) -> None:
        self._ledger = ledger
        self._state = state
        self._loop_state = loop_state
        self._viewer_ref = viewer_ref
        self._ui_call = ui_call
        self._presenter = presenter
        self._log_dims_info = bool(log_dims_info)
        ledger.subscribe_all(self._handle_ledger_update)

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
        assert axis_idx is not None, f"unknown dims axis target={update.target!r}"

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

        self._state.dims_state[(str(update.target), str(update.key))] = value_int
        self._state.primary_axis_index = _compute_primary_axis_index(meta)
        self._flush_dims_payload(reason=f"axis:{axis_idx}")

    def _handle_ndisplay_update(self, update: MirrorEvent) -> None:
        try:
            ndisplay = int(update.value)
        except Exception as exc:  # pragma: no cover - intentional crash path
            raise AssertionError(f"ndisplay value must be int-like: {update.value!r}") from exc
        self._state.dims_meta['ndisplay'] = ndisplay
        self._flush_dims_payload(reason="ndisplay")

    def _flush_dims_payload(self, *, reason: str) -> None:
        payload = _sync_dims_payload_from_meta(self._state, self._loop_state)
        if self._log_dims_info and payload.get('current_step') is not None:
            logger.info(
                "dims mirror (%s): step=%s ndisplay=%s",
                reason,
                payload.get('current_step'),
                payload.get('ndisplay'),
            )

        if self._presenter is not None:
            try:
                self._presenter.apply_dims_update(dict(payload))
            except Exception:
                logger.debug("presenter dims update failed", exc_info=True)

        viewer_obj = self._viewer_ref() if callable(self._viewer_ref) else None  # type: ignore[misc]
        mirror_dims_to_viewer(
            viewer_obj,
            self._ui_call,
            current_step=payload.get('current_step'),
            ndisplay=payload.get('ndisplay'),
            ndim=payload.get('ndim'),
            dims_range=payload.get('dims_range'),
            order=payload.get('order'),
            axis_labels=payload.get('axis_labels'),
            sizes=payload.get('sizes'),
            displayed=payload.get('displayed'),
        )


# ---------------------------------------------------------------------------
# Shared helpers (lifted from the legacy state_update_actions module)


def _sync_dims_payload_from_meta(state: "ControlStateContext", loop_state: "ClientLoopState") -> dict[str, Any]:
    meta = state.dims_meta
    payload: dict[str, Any] = {}

    current_step = meta.get('current_step')
    if isinstance(current_step, Sequence):
        payload['current_step'] = list(int(value) for value in current_step)
        loop_state.last_dims_payload = dict(payload)
    else:
        payload['current_step'] = None

    ndisplay = meta.get('ndisplay')
    payload['ndisplay'] = int(ndisplay) if ndisplay is not None else None

    ndim = meta.get('ndim')
    payload['ndim'] = int(ndim) if ndim is not None else None

    dims_range = meta.get('range')
    if isinstance(dims_range, Sequence):
        payload['dims_range'] = [list(pair) for pair in dims_range]  # ensure JSON-friendly
    else:
        payload['dims_range'] = None

    order = meta.get('order')
    if isinstance(order, Sequence):
        payload['order'] = list(order)
    else:
        payload['order'] = None

    labels = meta.get('axis_labels')
    if isinstance(labels, Sequence):
        payload['axis_labels'] = [str(label) for label in labels]
    else:
        payload['axis_labels'] = None

    sizes = meta.get('sizes')
    if isinstance(sizes, Sequence):
        payload['sizes'] = [int(size) for size in sizes]
    else:
        payload['sizes'] = None

    displayed = meta.get('displayed')
    if isinstance(displayed, Sequence):
        payload['displayed'] = [int(val) for val in displayed]
    else:
        payload['displayed'] = None

    return payload


def _axis_index_from_target(state: "ControlStateContext", target: str) -> Optional[int]:
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


def replay_last_dims_payload(state: "ControlStateContext", loop_state: "ClientLoopState", viewer_ref, ui_call) -> None:
    payload = loop_state.last_dims_payload
    if not payload:
        return
    viewer_obj = viewer_ref() if callable(viewer_ref) else None  # type: ignore[misc]
    if viewer_obj is None:
        return
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


__all__ = [
    "NapariDimsMirror",
    "mirror_dims_to_viewer",
    "replay_last_dims_payload",
]
