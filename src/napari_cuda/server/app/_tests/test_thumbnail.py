from __future__ import annotations

import numpy as np
import pytest

from napari_cuda.server.app.thumbnail import (
    ThumbnailState,
    build_render_signature,
    handle_render_tick,
    ingest_thumbnail_payload,
    queue_thumbnail_refresh,
    record_thumbnail_failure,
    record_thumbnail_success,
    request_thumbnail_refresh,
    reset_thumbnail_state,
)
from napari_cuda.server.runtime.ipc.mailboxes.worker_intent import ThumbnailIntent
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.state_ledger import ServerStateLedger


def _make_snapshot(**kwargs) -> RenderLedgerSnapshot:
    defaults = {
        "op_seq": 3,
        "current_level": 2,
        "current_step": (4, 5),
        "displayed": (0, 1),
        "camera_versions": {"plane.zoom": 7},
    }
    defaults.update(kwargs)
    return RenderLedgerSnapshot(**defaults)


def test_flag_record_and_reset_roundtrip() -> None:
    state = ThumbnailState()
    snapshot = _make_snapshot()
    signature = build_render_signature(snapshot)

    assert request_thumbnail_refresh(state, "layer-a")
    record_thumbnail_failure(state)
    assert state.pending_layer_id == "layer-a"

    assert request_thumbnail_refresh(state, "layer-a")
    record_thumbnail_success(state, signature)
    assert state.last_signature == signature
    assert not request_thumbnail_refresh(state, None)

    reset_thumbnail_state(state)
    assert state.last_signature is None
    assert state.pending_layer_id is None


def test_queue_thumbnail_refresh_marks_worker() -> None:
    state = ThumbnailState()
    calls = {"marked": False}

    def mark() -> None:
        calls["marked"] = True

    assert queue_thumbnail_refresh(state, "layer-a", mark_render_tick=mark)
    assert calls["marked"] is True
    assert state.pending_layer_id == "layer-a"
    assert state.force_refresh is True


def test_handle_render_tick_uses_pending_layer_when_default_missing() -> None:
    state = ThumbnailState()
    queue_thumbnail_refresh(state, "layer-b")
    snapshot = _make_snapshot()

    payload = handle_render_tick(
        state,
        snapshot,
        worker_ready=True,
        fetch_thumbnail=lambda layer, _: np.ones((2, 2)),
    )

    assert isinstance(payload, ThumbnailIntent)
    assert payload.layer_id == "layer-b"
    assert payload.signature == build_render_signature(snapshot)


def test_handle_render_tick_no_worker_readiness() -> None:
    state = ThumbnailState()
    snapshot = _make_snapshot()
    queue_thumbnail_refresh(state, "layer-c")

    calls = {"ticks": 0}

    def request_tick() -> None:
        calls["ticks"] += 1

    payload = handle_render_tick(
        state,
        snapshot,
        worker_ready=False,
        fetch_thumbnail=lambda layer, sig: np.ones((1, 1)),
        request_render_tick=request_tick,
    )

    assert payload is None
    assert state.pending_layer_id == "layer-c"
    assert calls["ticks"] == 1


def test_ingest_thumbnail_payload_updates_ledger() -> None:
    state = ThumbnailState()
    ledger = ServerStateLedger()

    payload = ThumbnailIntent(
        layer_id="layer-d",
        signature=build_render_signature(_make_snapshot()),
        array=np.array([[0.0, 1.0], [0.3, 0.4]], dtype=np.float32),
    )

    updated = ingest_thumbnail_payload(state, payload, ledger)
    assert updated is True
    assert state.last_signature == payload.signature

    entry = ledger.get("layer", "layer-d", "metadata")
    assert entry is not None
    assert isinstance(entry.value, dict)
    assert entry.value["thumbnail_dtype"] == "float32"
    stored = np.asarray(entry.value["thumbnail"], dtype=np.float32)
    assert stored[0, 0] == pytest.approx(0.3, rel=1e-6)
    assert stored[-1, -1] == pytest.approx(1.0, rel=1e-6)

    thumb_entry = ledger.get("layer", "layer-d", "thumbnail")
    assert thumb_entry is not None
    thumb_value = thumb_entry.value
    assert isinstance(thumb_value, dict)
    assert thumb_value["dtype"] == "float32"


def test_ingest_thumbnail_payload_skips_duplicate() -> None:
    state = ThumbnailState()
    ledger = ServerStateLedger()
    signature = build_render_signature(_make_snapshot())

    arr = np.array([[1.0, 0.5], [0.2, 0.0]], dtype=np.float32)
    payload = ThumbnailIntent(layer_id="layer-e", signature=signature, array=arr)
    assert ingest_thumbnail_payload(state, payload, ledger) is True

    # Duplicate payload should be ignored but still count as success
    again = ThumbnailIntent(layer_id="layer-e", signature=signature, array=arr)
    assert ingest_thumbnail_payload(state, again, ledger) is False
    assert state.force_refresh is False
