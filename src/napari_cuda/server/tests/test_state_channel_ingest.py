from __future__ import annotations

import asyncio

import napari_cuda.server.data.lod as lod
from napari_cuda.protocol import build_state_update
from napari_cuda.protocol.envelopes import build_session_hello
from napari_cuda.protocol.messages import HelloClientInfo
from napari_cuda.server.runtime import worker_runtime
from napari_cuda.server.tests._helpers.state_channel import StateServerHarness


def _hello_payload(*, scene: bool = True, layers: bool = True, stream: bool = True) -> dict[str, object]:
    hello = build_session_hello(
        client=HelloClientInfo(name="tests", version="1.0", platform="pytest"),
        features={
            "notify.scene": scene,
            "notify.layers": layers,
            "notify.stream": stream,
            "notify.dims": True,
            "notify.camera": True,
            "call.command": True,
        },
        resume_tokens={},
    )
    return hello.to_dict()


def test_ingest_state_handshake_sends_baseline() -> None:
    asyncio.run(_test_ingest_state_handshake_sends_baseline())


async def _test_ingest_state_handshake_sends_baseline() -> None:
    loop = asyncio.get_running_loop()
    harness = StateServerHarness(loop)
    try:
        harness.queue_client_payload(_hello_payload())
        await harness.start()

        welcome = await harness.wait_for_frame(lambda frame: frame.get("type") == "session.welcome", timeout=3.0)
        session_info = welcome["payload"]["session"]
        session_id = session_info["id"]
        assert session_id

        await harness.wait_for_frame(lambda frame: frame.get("type") == "notify.scene", timeout=3.0)
        await harness.wait_for_frame(lambda frame: frame.get("type") == "notify.layers", timeout=3.0)
        await harness.wait_for_frame(lambda frame: frame.get("type") == "notify.dims", timeout=3.0)
        await harness.wait_for_frame(lambda frame: frame.get("type") == "notify.stream", timeout=3.0)

        await harness.drain_scheduled()
    finally:
        await harness.stop()


def test_ingest_state_rejects_missing_features() -> None:
    asyncio.run(_test_ingest_state_rejects_missing_features())


async def _test_ingest_state_rejects_missing_features() -> None:
    loop = asyncio.get_running_loop()
    harness = StateServerHarness(loop)
    try:
        harness.queue_client_payload(_hello_payload(stream=False))
        await harness.start()

        reject = await harness.wait_for_frame(lambda frame: frame.get("type") == "session.reject", timeout=3.0)
        details = reject["payload"].get("details") or {}
        assert "notify.stream" in details.get("missing", [])
    finally:
        await harness.stop()


def test_ingest_state_applies_view_update() -> None:
    asyncio.run(_test_ingest_state_applies_view_update())


async def _test_ingest_state_applies_view_update() -> None:
    loop = asyncio.get_running_loop()
    harness = StateServerHarness(loop)
    try:
        harness.queue_client_payload(_hello_payload())
        await harness.start()
        welcome = await harness.wait_for_frame(lambda frame: frame.get("type") == "session.welcome", timeout=3.0)
        session_info = welcome["payload"]["session"]
        session_id = session_info["id"]

        update = build_state_update(
            session_id=session_id,
            intent_id="view-intent",
            frame_id="frame-1",
            payload={
                "scope": "view",
                "target": "main",
                "key": "ndisplay",
                "value": 3,
            },
        )
        harness.queue_client_payload(update.to_dict())

        ack = await harness.wait_for_frame(
            lambda frame: frame.get("type") == "ack.state" and frame["payload"]["intent_id"] == "view-intent",
            timeout=3.0,
        )
        assert ack["payload"]["status"] == "accepted"
        entry = harness.server._state_ledger.get("view", "main", "ndisplay")
        assert entry is not None and int(entry.value) == 3
        assert harness.server.use_volume is True

        await harness.drain_scheduled()
        assert harness.server._ensure_keyframe_calls >= 2
    finally:
        await harness.stop()


def test_view_toggle_triggers_plane_restore_once() -> None:
    asyncio.run(_test_view_toggle_triggers_plane_restore_once())


async def _test_view_toggle_triggers_plane_restore_once() -> None:
    loop = asyncio.get_running_loop()
    harness = StateServerHarness(loop)
    from napari_cuda.server.control import control_channel_server as state_channel_handler

    orig_plane_restore = state_channel_handler.apply_plane_restore_transaction
    calls: list[dict[str, object]] = []

    def _wrapped_plane_restore(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return orig_plane_restore(**kwargs)

    state_channel_handler.apply_plane_restore_transaction = _wrapped_plane_restore  # type: ignore[assignment]
    try:
        harness.queue_client_payload(_hello_payload())
        await harness.start()
        welcome = await harness.wait_for_frame(lambda frame: frame.get("type") == "session.welcome", timeout=3.0)
        session_id = welcome["payload"]["session"]["id"]

        ledger = harness.server._state_ledger
        plane_level = 0
        plane_step = (120, 0, 0)
        plane_center = (12.5, 20.0, 3.0)
        plane_zoom = 1.5
        plane_rect = (0.0, 0.0, 256.0, 256.0)
        ledger.batch_record_confirmed(
            [
                ("view_cache", "plane", "level", plane_level),
                ("view_cache", "plane", "step", plane_step),
                ("camera_plane", "main", "center", plane_center),
                ("camera_plane", "main", "zoom", plane_zoom),
                ("camera_plane", "main", "rect", plane_rect),
                ("camera", "main", "center", plane_center),
                ("camera", "main", "zoom", plane_zoom),
                ("camera", "main", "rect", plane_rect),
                ("multiscale", "main", "level", 1),
                ("dims", "main", "current_step", (60, 0, 0)),
            ],
            origin="test.seed",
        )
        harness.server.use_volume = True
        harness.server._scene.use_volume = True

        def _make_update(frame_id: str, value: int) -> dict[str, object]:
            return build_state_update(
                session_id=session_id,
                intent_id=f"view-intent-{frame_id}",
                frame_id=frame_id,
                payload={
                    "scope": "view",
                    "target": "main",
                    "key": "ndisplay",
                    "value": value,
                },
            ).to_dict()

        harness.queue_client_payload(_make_update("frame-restore", 2))
        ack = await harness.wait_for_frame(
            lambda frame: frame.get("type") == "ack.state" and frame["payload"]["intent_id"] == "view-intent-frame-restore",
            timeout=3.0,
        )
        assert ack["payload"]["status"] == "accepted"
        assert len(calls) == 1
        scene_level = harness.server._state_ledger.get("multiscale", "main", "level")
        assert scene_level is not None and int(scene_level.value) == plane_level
        dims_step_entry = harness.server._state_ledger.get("dims", "main", "current_step")
        assert dims_step_entry is not None
        assert tuple(int(v) for v in dims_step_entry.value) == plane_step
        assert harness.server._scene.camera_deltas == []

        harness.queue_client_payload(_make_update("frame-repeat", 2))
        ack_repeat = await harness.wait_for_frame(
            lambda frame: frame.get("type") == "ack.state" and frame["payload"]["intent_id"] == "view-intent-frame-repeat",
            timeout=3.0,
        )
        assert ack_repeat["payload"]["status"] == "accepted"
        assert len(calls) == 1
    finally:
        state_channel_handler.apply_plane_restore_transaction = orig_plane_restore  # type: ignore[assignment]
        await harness.stop()


def test_view_toggle_skips_plane_restore_without_cache() -> None:
    asyncio.run(_test_view_toggle_skips_plane_restore_without_cache())


async def _test_view_toggle_skips_plane_restore_without_cache() -> None:
    loop = asyncio.get_running_loop()
    harness = StateServerHarness(loop)
    from napari_cuda.server.control import control_channel_server as state_channel_handler

    orig_plane_restore = state_channel_handler.apply_plane_restore_transaction
    calls: list[dict[str, object]] = []

    def _wrapped_plane_restore(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return orig_plane_restore(**kwargs)

    state_channel_handler.apply_plane_restore_transaction = _wrapped_plane_restore  # type: ignore[assignment]
    try:
        harness.queue_client_payload(_hello_payload())
        await harness.start()
        welcome = await harness.wait_for_frame(lambda frame: frame.get("type") == "session.welcome", timeout=3.0)
        session_id = welcome["payload"]["session"]["id"]

        update = build_state_update(
            session_id=session_id,
            intent_id="view-intent-missing",
            frame_id="frame-missing",
            payload={
                "scope": "view",
                "target": "main",
                "key": "ndisplay",
                "value": 2,
            },
        )
        harness.queue_client_payload(update.to_dict())
        ack = await harness.wait_for_frame(
            lambda frame: frame.get("type") == "ack.state" and frame["payload"]["intent_id"] == "view-intent-missing",
            timeout=3.0,
        )
        assert ack["payload"]["status"] == "accepted"
        assert calls == []
    finally:
        state_channel_handler.apply_plane_restore_transaction = orig_plane_restore  # type: ignore[assignment]
        await harness.stop()


def test_multiscale_level_switch_requests_worker_once() -> None:
    asyncio.run(_test_multiscale_level_switch_requests_worker_once())


async def _test_multiscale_level_switch_requests_worker_once() -> None:
    loop = asyncio.get_running_loop()
    harness = StateServerHarness(loop)
    try:
        harness.queue_client_payload(_hello_payload())
        await harness.start()
        welcome = await harness.wait_for_frame(lambda frame: frame.get("type") == "session.welcome", timeout=3.0)
        session_id = welcome["payload"]["session"]["id"]

        worker = harness.server._worker
        assert worker is not None
        worker.level_requests.clear()
        worker.force_idr_calls = 0

        update = build_state_update(
            session_id=session_id,
            intent_id="level-intent",
            frame_id="frame-level-1",
            payload={
                "scope": "multiscale",
                "target": "main",
                "key": "level",
                "value": 1,
            },
        )
        harness.queue_client_payload(update.to_dict())
        ack = await harness.wait_for_frame(
            lambda frame: frame.get("type") == "ack.state" and frame["payload"]["intent_id"] == "level-intent",
            timeout=3.0,
        )
        assert ack["payload"]["status"] == "accepted"

        await harness.drain_scheduled()
        assert worker.level_requests == [(1, None)]
        assert worker.force_idr_calls == 1
        level_entry = harness.server._state_ledger.get("multiscale", "main", "level")
        assert level_entry is not None and int(level_entry.value) == 1
        assert harness.server._pixel.bypass_until_key is True
    finally:
        await harness.stop()


def test_roi_applied_once_on_level_switch() -> None:
    asyncio.run(_test_roi_applied_once_on_level_switch())


async def _test_roi_applied_once_on_level_switch() -> None:
    loop = asyncio.get_running_loop()
    harness = StateServerHarness(loop)

    roi_calls: list[tuple] = []
    orig_apply_worker_level = worker_runtime.apply_worker_slice_level

    def _spy_apply_worker_slice_level(*args, **kwargs):
        roi_calls.append((args, kwargs))
        return None

    worker_runtime.apply_worker_slice_level = _spy_apply_worker_slice_level
    try:
        harness.queue_client_payload(_hello_payload())
        await harness.start()
        welcome = await harness.wait_for_frame(lambda frame: frame.get("type") == "session.welcome", timeout=3.0)
        session_id = welcome["payload"]["session"]["id"]

        update = build_state_update(
            session_id=session_id,
            intent_id="roi-intent",
            frame_id="frame-roi",
            payload={"scope": "multiscale", "target": "main", "key": "level", "value": 1},
        )
        harness.queue_client_payload(update.to_dict())
        ack = await harness.wait_for_frame(
            lambda frame: frame.get("type") == "ack.state" and frame["payload"]["intent_id"] == "roi-intent",
            timeout=5.0,
        )
        assert ack["payload"]["status"] == "accepted"

        applied = lod.LevelContext(level=1, step=(0, 0, 0), z_index=0, shape=(16, 480, 640), scale_yx=(1.0, 1.0), contrast=(0.0, 1.0), axes="zyx", dtype="float32")
        harness.server._commit_level(applied, downgraded=False)
        await harness.drain_scheduled()

        assert len(roi_calls) == 1
    finally:
        worker_runtime.apply_worker_slice_level = orig_apply_worker_level
        await harness.stop()
