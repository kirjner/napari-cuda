from __future__ import annotations

import numpy as np
import pytest

import pytest


pytest.skip("legacy thumbnail path removed; tests to be rewritten for mailbox flow", allow_module_level=True)


def test_flag_record_and_reset_roundtrip() -> None:
    state = ThumbnailState()
    snapshot = _make_snapshot()
    signature = RenderSignature.from_snapshot(snapshot)

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
    assert queue_thumbnail_refresh(state, "layer-a")
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
    assert payload.signature == RenderSignature.from_snapshot(snapshot)


def test_handle_render_tick_no_worker_readiness() -> None:
    state = ThumbnailState()
    snapshot = _make_snapshot()
    queue_thumbnail_refresh(state, "layer-c")

    payload = handle_render_tick(
        state,
        snapshot,
        worker_ready=False,
        fetch_thumbnail=lambda layer, sig: np.ones((1, 1)),
    )

    assert payload is None
    assert state.pending_layer_id == "layer-c"


def test_ingest_thumbnail_payload_updates_ledger() -> None:
    state = ThumbnailState()
    ledger = ServerStateLedger()

    payload = ThumbnailIntent(
        layer_id="layer-d",
        signature=RenderSignature.from_snapshot(_make_snapshot()),
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
    signature = RenderSignature.from_snapshot(_make_snapshot())

    arr = np.array([[1.0, 0.5], [0.2, 0.0]], dtype=np.float32)
    payload = ThumbnailIntent(layer_id="layer-e", signature=signature, array=arr)
    assert ingest_thumbnail_payload(state, payload, ledger) is True

    # Duplicate payload should be ignored but still count as success
    again = ThumbnailIntent(layer_id="layer-e", signature=signature, array=arr)
    assert ingest_thumbnail_payload(state, again, ledger) is False
    assert state.force_refresh is False
