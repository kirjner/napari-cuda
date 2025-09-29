from __future__ import annotations

import json

from napari_cuda.protocol.messages import SceneSpec, SceneSpecMessage, StateUpdateMessage
from napari_cuda.server.control.legacy_dual_emitter import encode_envelope, encode_envelope_json


def test_encode_envelope_state_update() -> None:
    message = StateUpdateMessage(scope='dims', target='z', key='step', value=5, server_seq=10)
    payload = message.to_dict()

    envelope = encode_envelope(payload)

    assert envelope is not None
    assert envelope['type'] == 'notify.state'
    assert envelope['payload']['key'] == 'step'
    assert envelope['payload']['value'] == 5


def test_encode_envelope_scene_spec_includes_capabilities() -> None:
    spec = SceneSpec(layers=[], capabilities=['notify.state'])
    scene_message = SceneSpecMessage(scene=spec, capabilities=['notify.layer'])
    payload = scene_message.to_dict()

    envelope_json = encode_envelope_json(payload)

    assert envelope_json is not None
    data = json.loads(envelope_json)
    assert data['type'] == 'notify.scene'
    assert data['payload']['scene']['capabilities'] == ['notify.state']
    assert data['payload']['state']['capabilities'] == ['notify.layer']


def test_encode_envelope_video_config() -> None:
    payload = {
        'type': 'video_config',
        'codec': 'h264',
        'fps': 60,
        'width': 1920,
        'height': 1080,
        'format': 'avcc',
        'data': 'AAA=',
    }

    envelope = encode_envelope(payload)

    assert envelope is not None
    assert envelope['type'] == 'notify.stream'
    assert envelope['payload']['codec'] == 'h264'
    assert envelope['payload']['extras']['format'] == 'avcc'

