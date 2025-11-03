from __future__ import annotations

from napari_cuda.server.app.thumbnail_utils import (
    ThumbnailState,
    build_render_signature,
    flag_render_change,
    record_thumbnail_failure,
    record_thumbnail_success,
    request_thumbnail_refresh,
)
from napari_cuda.server.scene import RenderLedgerSnapshot


def test_flag_render_change_roundtrip_success_and_failure() -> None:
    state = ThumbnailState()
    snapshot = RenderLedgerSnapshot(
        op_seq=3,
        current_level=2,
        current_step=(4, 5),
        displayed=(0, 1),
        camera_versions={"plane.zoom": 7},
    )
    signature = build_render_signature(snapshot)

    assert flag_render_change(state, signature, "layer-a")
    assert state.pending_signature == signature
    assert state.pending_layer_id == "layer-a"
    assert state.force_refresh is False

    record_thumbnail_failure(state)
    assert state.pending_signature is None
    assert state.pending_layer_id is None
    assert state.last_signature is None
    assert state.force_refresh is False

    assert flag_render_change(state, signature, "layer-a")
    record_thumbnail_success(state, signature)
    assert state.last_signature == signature
    assert state.pending_signature is None
    assert state.pending_layer_id is None
    assert state.force_refresh is False

    assert not flag_render_change(state, signature, "layer-a")


def test_request_thumbnail_refresh_sets_manual_state() -> None:
    state = ThumbnailState()
    snapshot = RenderLedgerSnapshot(
        op_seq=7,
        current_level=1,
        current_step=(2, 3),
        displayed=(0, 1),
    )
    signature = build_render_signature(snapshot)
    record_thumbnail_success(state, signature)
    assert not flag_render_change(state, signature, "layer-b")

    assert request_thumbnail_refresh(state, "layer-b")
    assert state.force_refresh is True
    assert state.pending_layer_id == "layer-b"
    assert state.pending_signature is None

    assert flag_render_change(state, signature, "layer-b")
    record_thumbnail_failure(state)
    assert state.force_refresh is True

    assert flag_render_change(state, signature, "layer-b")
    record_thumbnail_success(state, signature)
    assert state.force_refresh is False


def test_build_render_signature_includes_camera_versions_and_op_seq() -> None:
    snapshot = RenderLedgerSnapshot(
        op_seq=11,
        current_level=4,
        current_step=(0, 2, 3),
        displayed=(1, 2),
        camera_versions={"plane.zoom": 5, "plane.center": 8},
    )
    signature = build_render_signature(snapshot)
    assert signature[0] == 11
    assert signature[1] == 4
    assert signature[2] == (0, 2, 3)
    assert signature[3] == (1, 2)
    assert signature[4] == (("plane.center", 8), ("plane.zoom", 5))
