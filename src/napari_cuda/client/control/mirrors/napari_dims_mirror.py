"""Mirror confirmed dims state from the client ledger into napari."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence, TYPE_CHECKING
from numbers import Integral, Real

from napari_cuda.client.control.client_state_ledger import ClientStateLedger, MirrorEvent
from napari_cuda.protocol.messages import NotifyDimsFrame

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

    def ingest_notify(
        self,
        frame: NotifyDimsFrame,
        *,
        notify_first_ready: Callable[[], None],
    ) -> None:
        """Record a ``notify.dims`` snapshot and mirror it into napari."""

        state = self._state
        meta = state.dims_meta
        was_ready = bool(state.dims_ready)

        payload = frame.payload
        current_step = tuple(int(value) for value in payload.current_step)
        meta['current_step'] = list(current_step)
        meta['ndisplay'] = int(payload.ndisplay)
        mode_text = str(payload.mode)
        meta['mode'] = mode_text
        meta['volume'] = bool(mode_text.lower() == 'volume')
        meta['source'] = payload.source

        snapshot_payload = _build_confirmed_dims_payload(state, self._loop_state)

        if not was_ready:
            _record_dims_snapshot(state, self._ledger, snapshot_payload)
        else:
            _record_dims_delta(state, self._ledger, snapshot_payload, update_kind='notify')

        if not state.dims_ready:
            state.dims_ready = True
            logger.info("notify.dims: metadata received; client intents enabled")
            notify_first_ready()

        state.primary_axis_index = _compute_primary_axis_index(meta)

        if current_step and self._log_dims_info:
            logger.info(
                "notify.dims: step=%s ndisplay=%s order=%s labels=%s",
                list(current_step),
                meta.get('ndisplay'),
                meta.get('order'),
                meta.get('axis_labels'),
            )

        self._mirror_confirmed_dims(reason="notify", payload=snapshot_payload)

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
            else _build_confirmed_dims_payload(self._state, self._loop_state)
        )

        if self._log_dims_info and confirmed_payload.get('current_step') is not None:
            logger.info(
                "dims mirror (%s): step=%s ndisplay=%s",
                reason,
                confirmed_payload.get('current_step'),
                confirmed_payload.get('ndisplay'),
            )

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
            sizes=confirmed_payload.get('sizes'),
            displayed=confirmed_payload.get('displayed'),
        )


# Shared helpers (lifted from the legacy state_update_actions module)


def _record_dims_snapshot(
    state: "ControlStateContext",
    ledger: ClientStateLedger,
    payload: Mapping[str, Any],
) -> None:
    _record_dims_delta(state, ledger, payload, update_kind='snapshot')

    ndisplay = payload.get('ndisplay')
    if ndisplay is not None:
        ledger.record_confirmed('view', 'main', 'ndisplay', int(ndisplay))

    _record_multiscale_metadata(state, ledger)
    _record_volume_metadata(state, ledger)


def _record_dims_delta(
    state: "ControlStateContext",
    ledger: ClientStateLedger,
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
        ledger.record_confirmed('dims', target_label, 'index', value_int, metadata=metadata)


def _build_confirmed_dims_payload(state: "ControlStateContext", loop_state: "ClientLoopState") -> dict[str, Any]:
    meta = state.dims_meta
    payload: dict[str, Any] = {}

    current_step = meta.get('current_step')
    if isinstance(current_step, Sequence):
        payload['current_step'] = [int(value) for value in current_step]
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

    mode = meta.get('mode')
    payload['mode'] = str(mode) if mode is not None else None

    volume_flag = meta.get('volume')
    payload['volume'] = bool(volume_flag) if volume_flag is not None else None

    payload['source'] = meta.get('source')

    loop_state.last_dims_payload = dict(payload)
    return payload


def _record_multiscale_metadata(state: "ControlStateContext", ledger: ClientStateLedger) -> None:
    multiscale_meta = state.dims_meta.get('multiscale')
    if not isinstance(multiscale_meta, Mapping):
        return

    level_val = multiscale_meta.get('level')
    if level_val is None:
        level_val = multiscale_meta.get('current_level')
    if level_val is not None:
        ledger.record_confirmed('multiscale', 'main', 'level', int(level_val))

    policy_val = multiscale_meta.get('policy')
    if policy_val is not None:
        ledger.record_confirmed('multiscale', 'main', 'policy', str(policy_val))


def _record_volume_metadata(state: "ControlStateContext", ledger: ClientStateLedger) -> None:
    render_meta = state.dims_meta.get('render')
    if not isinstance(render_meta, Mapping):
        return

    mode_val = render_meta.get('mode') or render_meta.get('render_mode')
    if mode_val is not None:
        ledger.record_confirmed('volume', 'main', 'render_mode', str(mode_val))

    clim_val = render_meta.get('contrast_limits')
    if isinstance(clim_val, (list, tuple)) and len(clim_val) >= 2:
        lo = float(clim_val[0])
        hi = float(clim_val[1])
        ledger.record_confirmed('volume', 'main', 'contrast_limits', (lo, hi))

    cmap_val = render_meta.get('colormap')
    if cmap_val is not None:
        ledger.record_confirmed('volume', 'main', 'colormap', str(cmap_val))

    opacity_val = render_meta.get('opacity')
    if opacity_val is not None:
        ledger.record_confirmed('volume', 'main', 'opacity', float(opacity_val))

    sample_val = render_meta.get('sample_step')
    if sample_val is not None:
        ledger.record_confirmed('volume', 'main', 'sample_step', float(sample_val))


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


def _axis_target_label(state: "ControlStateContext", axis_idx: int) -> str:
    labels = state.dims_meta.get('axis_labels')
    if isinstance(labels, Sequence) and 0 <= axis_idx < len(labels):
        label = labels[axis_idx]
        if isinstance(label, str) and label.strip():
            return label
    return str(axis_idx)


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
