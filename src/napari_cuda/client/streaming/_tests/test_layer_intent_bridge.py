from __future__ import annotations

import pytest

from napari_cuda.client.layers.registry import LayerRecord, RegistrySnapshot
from napari_cuda.client.layers.remote_image_layer import RemoteImageLayer
from napari_cuda.client.streaming.client_loop.intents import IntentState
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.layer_intent_bridge import LayerIntentBridge
from napari_cuda.client.streaming.presenter_facade import PresenterFacade
from napari_cuda.protocol.messages import LayerRenderHints, LayerSpec, LayerUpdateMessage


class DummyLoop:
    def __init__(self) -> None:
        self.posted: list[dict] = []

    def post(self, payload: dict) -> bool:
        self.posted.append(payload)
        return True


class DummyRegistry:
    def __init__(self) -> None:
        self._listeners: list = []

    def add_listener(self, callback) -> None:
        self._listeners.append(callback)

    def emit(self, snapshot: RegistrySnapshot) -> None:
        for callback in list(self._listeners):
            callback(snapshot)


@pytest.fixture
def intent_state() -> IntentState:
    state = IntentState()
    state.client_id = "client-test"
    state.settings_min_dt = 0.0
    state.last_settings_send = 0.0
    return state


def _make_layer(remote_id: str = "layer-1") -> RemoteImageLayer:
    spec = LayerSpec(
        layer_id=remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        contrast_limits=None,
        metadata={},
        render=LayerRenderHints(mode="mip"),
        controls={
            "visible": True,
            "opacity": 0.5,
            "rendering": "mip",
            "colormap": "gray",
            "gamma": 1.0,
            "contrast_limits": [0.0, 1.0],
        },
        extras={"data_id": "demo"},
    )
    return RemoteImageLayer(spec)


def test_opacity_intent_roundtrip(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer()
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))

    # Local UI change emits control command and waits for ack metadata
    layer.opacity = 0.25

    assert loop.posted, "control command not dispatched"
    payload = loop.posted[-1]
    assert payload["type"] == "control.command"
    assert payload["scope"] == "layer"
    assert payload["target"] == layer.remote_id
    assert payload["prop"] == "opacity"
    assert payload["value"] == pytest.approx(0.25)
    assert payload["phase"] == "update"
    interaction_id = payload["interaction_id"]
    assert isinstance(interaction_id, str) and interaction_id

    # Local layer reflects optimistic value until ack arrives
    assert layer.opacity == pytest.approx(0.25)

    seq = payload["client_seq"]
    assert seq in loop_state.pending_intents

    # Simulate server ack via layer.update
    ack_spec = LayerSpec(
        layer_id=layer.remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        metadata={"intent_seq": seq},
        render=LayerRenderHints(mode="mip"),
        controls={
            "visible": True,
            "opacity": 0.25,
            "rendering": "mip",
            "contrast_limits": [0.0, 1.0],
        },
    )
    message = LayerUpdateMessage(
        layer=ack_spec,
        partial=True,
        ack=True,
        intent_seq=seq,
        controls={
            "visible": True,
            "opacity": 0.25,
            "rendering": "mip",
            "contrast_limits": [0.0, 1.0],
        },
        server_seq=1,
        source_client_id="client-test",
        source_client_seq=seq,
        interaction_id=interaction_id,
        phase="update",
        control_versions={
            "opacity": {
                "server_seq": 1,
                "source_client_id": "client-test",
                "source_client_seq": seq,
                "interaction_id": interaction_id,
                "phase": "update",
            }
        },
    )
    bridge.handle_layer_update(message)

    assert layer.opacity == pytest.approx(0.25)

    # Bridge should emit a commit command after the update ack.
    assert len(loop.posted) >= 2
    commit_payload = loop.posted[-1]
    assert commit_payload["phase"] == "commit"
    assert commit_payload["prop"] == "opacity"
    commit_seq = commit_payload["client_seq"]
    assert loop_state.pending_intents.get(commit_seq)["phase"] == "commit"

    commit_ack = LayerUpdateMessage(
        layer=ack_spec,
        partial=True,
        ack=True,
        intent_seq=commit_seq,
        controls={
            "visible": True,
            "opacity": 0.25,
            "rendering": "mip",
            "contrast_limits": [0.0, 1.0],
        },
        server_seq=2,
        source_client_id="client-test",
        source_client_seq=commit_seq,
        interaction_id=interaction_id,
        phase="commit",
        control_versions={
            "opacity": {
                "server_seq": 2,
                "source_client_id": "client-test",
                "source_client_seq": commit_seq,
                "interaction_id": interaction_id,
                "phase": "commit",
            }
        },
    )
    bridge.handle_layer_update(commit_ack)

    assert not loop_state.pending_intents


def test_contrast_intent(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer("layer-contrast")
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))

    layer.contrast_limits = (0.1, 0.9)
    payload = loop.posted[-1]
    assert payload["type"] == "control.command"
    assert payload["prop"] == "contrast_limits"
    assert payload["value"] == pytest.approx((0.1, 0.9))
    assert payload["phase"] == "update"
    interaction_id = payload["interaction_id"]

    # Local layer keeps optimistic limits until ack arrives
    assert tuple(layer.contrast_limits) == pytest.approx((0.1, 0.9))

    seq = payload["client_seq"]
    ack_controls = {
        "visible": True,
        "opacity": 0.5,
        "rendering": "mip",
        "contrast_limits": [0.1, 0.9],
    }
    ack_spec = LayerSpec(
        layer_id=layer.remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        metadata={"intent_seq": seq},
        render=LayerRenderHints(mode="mip"),
        controls=ack_controls,
    )
    message = LayerUpdateMessage(
        layer=ack_spec,
        partial=True,
        ack=True,
        intent_seq=seq,
        controls=ack_controls,
        server_seq=2,
        source_client_id="client-test",
        source_client_seq=seq,
        interaction_id=interaction_id,
        phase="update",
        control_versions={
            "contrast_limits": {
                "server_seq": 2,
                "source_client_id": "client-test",
                "source_client_seq": seq,
                "interaction_id": interaction_id,
                "phase": "update",
            }
        },
    )
    bridge.handle_layer_update(message)

    assert tuple(layer.contrast_limits) == pytest.approx((0.1, 0.9))

    commit_payload = loop.posted[-1]
    assert commit_payload["phase"] == "commit"
    commit_seq = commit_payload["client_seq"]
    assert loop_state.pending_intents.get(commit_seq)["phase"] == "commit"

    bridge.handle_layer_update(
        LayerUpdateMessage(
            layer=ack_spec,
            partial=True,
            ack=True,
            intent_seq=commit_seq,
            controls=ack_controls,
            server_seq=3,
            source_client_id="client-test",
            source_client_seq=commit_seq,
            interaction_id=interaction_id,
            phase="commit",
            control_versions={
                "contrast_limits": {
                    "server_seq": 3,
                    "source_client_id": "client-test",
                    "source_client_seq": commit_seq,
                    "interaction_id": interaction_id,
                    "phase": "commit",
                }
            },
        )
    )

    assert not loop_state.pending_intents


def test_colormap_intent(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer("layer-color")
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))

    layer.colormap = "magma"
    payload = loop.posted[-1]
    assert payload["type"] == "control.command"
    assert payload["prop"] == "colormap"
    assert payload["value"] == "magma"
    assert payload["phase"] == "update"
    interaction_id = payload["interaction_id"]

    seq = payload["client_seq"]
    ack_controls = {
        "visible": True,
        "opacity": 0.5,
        "rendering": "mip",
        "colormap": "magma",
        "contrast_limits": [0.0, 1.0],
    }
    ack_spec = LayerSpec(
        layer_id=layer.remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        metadata={"intent_seq": seq},
        render=LayerRenderHints(mode="mip"),
        controls=ack_controls,
    )
    bridge.handle_layer_update(
        LayerUpdateMessage(
            layer=ack_spec,
            partial=True,
            ack=True,
            intent_seq=seq,
            controls=ack_controls,
            server_seq=3,
            source_client_id="client-test",
            source_client_seq=seq,
            interaction_id=interaction_id,
            phase="update",
            control_versions={
                "colormap": {
                    "server_seq": 3,
                    "source_client_id": "client-test",
                    "source_client_seq": seq,
                    "interaction_id": interaction_id,
                    "phase": "update",
                }
            },
        )
    )

    # Value committed after ack
    assert getattr(layer.colormap, "name", str(layer.colormap)) == "magma"
    commit_payload = loop.posted[-1]
    assert commit_payload["phase"] == "commit"
    commit_seq = commit_payload["client_seq"]
    assert loop_state.pending_intents.get(commit_seq)["phase"] == "commit"

    bridge.handle_layer_update(
        LayerUpdateMessage(
            layer=ack_spec,
            partial=True,
            ack=True,
            intent_seq=commit_seq,
            controls=ack_controls,
            server_seq=4,
            source_client_id="client-test",
            source_client_seq=commit_seq,
            interaction_id=interaction_id,
            phase="commit",
            control_versions={
                "colormap": {
                    "server_seq": 4,
                    "source_client_id": "client-test",
                    "source_client_seq": commit_seq,
                    "interaction_id": interaction_id,
                    "phase": "commit",
                }
            },
        )
    )

    assert not loop_state.pending_intents


def test_gamma_ack_out_of_order(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer("layer-gamma")
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))

    first_value = 1.2
    layer.gamma = first_value
    payload1 = loop.posted[-1]
    seq1 = payload1["client_seq"]
    assert payload1["type"] == "control.command"
    assert payload1["prop"] == "gamma"
    assert payload1["value"] == pytest.approx(first_value)
    assert payload1["phase"] == "update"
    interaction_id = payload1["interaction_id"]

    # Allow immediate second update despite default 60 Hz gating fallback
    bridge._intent_state.last_settings_send = 0.0

    second_value = 1.6
    layer.gamma = second_value

    assert len(loop.posted) == 2
    payload2 = loop.posted[-1]
    seq2 = payload2["client_seq"]
    assert payload2["value"] == pytest.approx(second_value)
    assert payload2["interaction_id"] == interaction_id

    binding = bridge._bindings[layer.remote_id]
    session = binding.sessions["gamma"]
    assert len(session.pending) == 2
    assert session.pending[0].value == pytest.approx(first_value)
    assert session.pending[1].value == pytest.approx(second_value)

    controls1 = {
        "visible": True,
        "opacity": 0.5,
        "rendering": "mip",
        "gamma": payload1["value"],
        "contrast_limits": [0.0, 1.0],
    }
    ack1 = LayerUpdateMessage(
        layer=LayerSpec(
            layer_id=layer.remote_id,
            layer_type="image",
            name="demo",
            ndim=2,
            shape=[1, 1],
            dtype="float32",
            controls=dict(controls1),
        ),
        partial=True,
        ack=True,
        intent_seq=seq1,
        controls=dict(controls1),
        server_seq=4,
        source_client_id="client-test",
        source_client_seq=seq1,
        interaction_id=interaction_id,
        phase="update",
        control_versions={
            "gamma": {
                "server_seq": 4,
                "source_client_id": "client-test",
                "source_client_seq": seq1,
                "interaction_id": interaction_id,
                "phase": "update",
            }
        },
    )
    bridge.handle_layer_update(ack1)

    # First ack should drop from session.pending but keep latest optimistic value
    assert len(session.pending) == 1
    assert session.pending[0].seq == seq2
    assert layer.gamma == pytest.approx(second_value)
    assert seq1 not in loop_state.pending_intents
    assert seq2 in loop_state.pending_intents

    controls2 = dict(controls1)
    controls2["gamma"] = payload2["value"]
    ack2 = LayerUpdateMessage(
        layer=LayerSpec(
            layer_id=layer.remote_id,
            layer_type="image",
            name="demo",
            ndim=2,
            shape=[1, 1],
            dtype="float32",
            controls=dict(controls2),
        ),
        partial=True,
        ack=True,
        intent_seq=seq2,
        controls=dict(controls2),
        server_seq=5,
        source_client_id="client-test",
        source_client_seq=seq2,
        interaction_id=interaction_id,
        phase="update",
        control_versions={
            "gamma": {
                "server_seq": 5,
                "source_client_id": "client-test",
                "source_client_seq": seq2,
                "interaction_id": interaction_id,
                "phase": "update",
            }
        },
    )
    bridge.handle_layer_update(ack2)

    assert len(session.pending) == 1
    assert session.pending[0].phase == "commit"
    assert layer.gamma == pytest.approx(second_value)

    commit_payload = loop.posted[-1]
    assert commit_payload["phase"] == "commit"
    commit_seq = commit_payload["client_seq"]
    assert loop_state.pending_intents.get(commit_seq)["phase"] == "commit"

    commit_controls = dict(controls2)
    bridge.handle_layer_update(
        LayerUpdateMessage(
            layer=LayerSpec(
                layer_id=layer.remote_id,
                layer_type="image",
                name="demo",
                ndim=2,
                shape=[1, 1],
                dtype="float32",
                controls=dict(commit_controls),
            ),
            partial=True,
            ack=True,
            intent_seq=commit_seq,
            controls=dict(commit_controls),
            server_seq=6,
            source_client_id="client-test",
            source_client_seq=commit_seq,
            interaction_id=interaction_id,
            phase="commit",
            control_versions={
                "gamma": {
                    "server_seq": 6,
                    "source_client_id": "client-test",
                    "source_client_seq": commit_seq,
                    "interaction_id": interaction_id,
                    "phase": "commit",
                }
            },
        )
    )

    assert seq2 not in loop_state.pending_intents
    assert commit_seq not in loop_state.pending_intents
    assert not session.pending


def test_gamma_latest_ack_clears_stale_pending(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer("layer-gamma-latest")
    registry.emit(RegistrySnapshot(layers=(LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer),)))

    values = [1.1, 1.3, 1.5]
    seqs: list[int] = []
    for idx, value in enumerate(values):
        if idx:
            bridge._intent_state.last_settings_send = 0.0
        layer.gamma = value
        seqs.append(loop.posted[-1]["client_seq"])

    binding = bridge._bindings[layer.remote_id]
    session = binding.sessions["gamma"]
    assert len(session.pending) == 3

    controls_latest = {
        "visible": True,
        "opacity": 0.5,
        "rendering": "mip",
        "gamma": loop.posted[-1]["value"],
        "contrast_limits": [0.0, 1.0],
    }
    ack_latest = LayerUpdateMessage(
        layer=LayerSpec(
            layer_id=layer.remote_id,
            layer_type="image",
            name="demo",
            ndim=2,
            shape=[1, 1],
            dtype="float32",
            controls=dict(controls_latest),
        ),
        partial=True,
        ack=True,
        intent_seq=seqs[-1],
        controls=dict(controls_latest),
        server_seq=7,
        source_client_id="client-test",
        source_client_seq=seqs[-1],
        interaction_id=loop.posted[-1]["interaction_id"],
        phase="update",
        control_versions={
            "gamma": {
                "server_seq": 7,
                "source_client_id": "client-test",
                "source_client_seq": seqs[-1],
                "interaction_id": loop.posted[-1]["interaction_id"],
                "phase": "update",
            }
        },
    )
    bridge.handle_layer_update(ack_latest)

    assert len(session.pending) == 1
    assert session.pending[0].phase == "commit"
    assert layer.gamma == pytest.approx(values[-1])
    assert seqs[-1] not in loop_state.pending_intents

    commit_payload = loop.posted[-1]
    assert commit_payload["phase"] == "commit"
    commit_seq = commit_payload["client_seq"]
    bridge.handle_layer_update(
        LayerUpdateMessage(
            layer=LayerSpec(
                layer_id=layer.remote_id,
                layer_type="image",
                name="demo",
                ndim=2,
                shape=[1, 1],
                dtype="float32",
                controls=dict(controls_latest),
            ),
            partial=True,
            ack=True,
            intent_seq=commit_seq,
            controls=dict(controls_latest),
            server_seq=8,
            source_client_id="client-test",
            source_client_seq=commit_seq,
            interaction_id=commit_payload["interaction_id"],
            phase="commit",
            control_versions={
                "gamma": {
                    "server_seq": 8,
                    "source_client_id": "client-test",
                    "source_client_seq": commit_seq,
                    "interaction_id": commit_payload["interaction_id"],
                    "phase": "commit",
                }
            },
        )
    )
    assert commit_seq not in loop_state.pending_intents
    assert not session.pending

    for stale_seq, gamma_value in zip(seqs[:-1], values[:-1]):
        controls = dict(controls_latest)
        controls["gamma"] = gamma_value
        bridge.handle_layer_update(
            LayerUpdateMessage(
                layer=LayerSpec(
                    layer_id=layer.remote_id,
                    layer_type="image",
                    name="demo",
                    ndim=2,
                    shape=[1, 1],
                    dtype="float32",
                    controls=dict(controls),
                ),
                partial=True,
                ack=True,
                intent_seq=stale_seq,
                controls=dict(controls),
                server_seq=stale_seq,  # any smaller seq imitates stale server echo
                source_client_id="client-test",
                source_client_seq=stale_seq,
                interaction_id=commit_payload["interaction_id"],
                phase="update",
                control_versions={
                    "gamma": {
                        "server_seq": stale_seq,
                        "source_client_id": "client-test",
                        "source_client_seq": stale_seq,
                        "interaction_id": commit_payload["interaction_id"],
                        "phase": "update",
                    }
                },
            )
        )
        assert layer.gamma == pytest.approx(values[-1])
        assert stale_seq not in loop_state.pending_intents
    

def test_gamma_multiple_pending_ack_does_not_resend(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer("layer-gamma-multi")
    registry.emit(
        RegistrySnapshot(
            layers=(LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer),)
        )
    )

    values = [1.92, 1.86, 1.95]
    seqs: list[int] = []
    interaction_id = None
    for idx, value in enumerate(values):
        if idx:
            bridge._intent_state.last_settings_send = 0.0
        layer.gamma = value
        payload = loop.posted[-1]
        seqs.append(payload["client_seq"])
        interaction_id = payload["interaction_id"]

    assert len(loop.posted) == 3

    binding = bridge._bindings[layer.remote_id]
    session = binding.sessions["gamma"]
    assert len(session.pending) == 3
    assert layer.gamma == pytest.approx(values[-1])
    initial_confirmed = session.confirmed_value

    def _ack(value: float, seq: int, server_seq: int) -> None:
        controls = {
            "visible": True,
            "opacity": 0.5,
            "rendering": "mip",
            "gamma": value,
            "contrast_limits": [0.0, 1.0],
        }
        bridge.handle_layer_update(
            LayerUpdateMessage(
                layer=LayerSpec(
                    layer_id=layer.remote_id,
                    layer_type="image",
                    name="demo",
                    ndim=2,
                    shape=[1, 1],
                    dtype="float32",
                    controls=dict(controls),
                ),
                partial=True,
                ack=True,
                intent_seq=seq,
                controls=dict(controls),
                server_seq=server_seq,
                source_client_id=intent_state.client_id,
                source_client_seq=seq,
                interaction_id=interaction_id,
                phase="update",
                control_versions={
                    "gamma": {
                        "server_seq": server_seq,
                        "source_client_id": intent_state.client_id,
                        "source_client_seq": seq,
                        "interaction_id": interaction_id,
                        "phase": "update",
                    }
                },
            )
        )

    post_count = len(loop.posted)
    _ack(values[0], seqs[0], 10)
    assert len(loop.posted) == post_count
    assert len(session.pending) == 2
    assert session.pending[0].seq == seqs[1]
    assert session.pending[1].seq == seqs[2]
    assert layer.gamma == pytest.approx(values[-1])
    assert session.confirmed_value == pytest.approx(initial_confirmed)
    assert session.last_confirmed_seq == seqs[0]

    _ack(values[1], seqs[1], 11)
    assert len(loop.posted) == post_count
    assert len(session.pending) == 1
    assert session.pending[0].seq == seqs[2]
    assert layer.gamma == pytest.approx(values[-1])
    assert session.confirmed_value == pytest.approx(initial_confirmed)
    assert session.last_confirmed_seq == seqs[1]

    _ack(values[2], seqs[2], 12)
    assert len(session.pending) == 1
    commit_payload = loop.posted[-1]
    assert commit_payload["phase"] == "commit"
    commit_seq = commit_payload["client_seq"]
    assert commit_seq in loop_state.pending_intents
    assert session.pending[0].phase == "commit"
    assert session.confirmed_value == pytest.approx(values[-1])
    assert session.last_confirmed_seq == seqs[2]

    bridge.handle_layer_update(
        LayerUpdateMessage(
            layer=LayerSpec(
                layer_id=layer.remote_id,
                layer_type="image",
                name="demo",
                ndim=2,
                shape=[1, 1],
                dtype="float32",
                controls={
                    "visible": True,
                    "opacity": 0.5,
                    "rendering": "mip",
                    "gamma": values[-1],
                    "contrast_limits": [0.0, 1.0],
                },
            ),
            partial=True,
            ack=True,
            intent_seq=commit_seq,
            controls={
                "visible": True,
                "opacity": 0.5,
                "rendering": "mip",
                "gamma": values[-1],
                "contrast_limits": [0.0, 1.0],
            },
            server_seq=13,
            source_client_id=intent_state.client_id,
            source_client_seq=commit_seq,
            interaction_id=commit_payload["interaction_id"],
            phase="commit",
            control_versions={
                "gamma": {
                    "server_seq": 13,
                    "source_client_id": intent_state.client_id,
                    "source_client_seq": commit_seq,
                    "interaction_id": commit_payload["interaction_id"],
                    "phase": "commit",
                }
            },
        )
    )

    assert not session.pending
    assert commit_seq not in loop_state.pending_intents
    assert session.confirmed_value == pytest.approx(values[-1])
    assert session.last_confirmed_seq == commit_seq
