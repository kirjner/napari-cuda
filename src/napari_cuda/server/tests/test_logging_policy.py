import json

from napari.layers.image._image_constants import Interpolation as NapariInterpolation

from napari_cuda.server.config.logging_policy import load_debug_policy


def test_debug_policy_defaults():
    policy = load_debug_policy({})
    assert policy.enabled is False
    assert policy.logging.log_camera_info is False
    assert policy.encoder.log_keyframes is False
    assert policy.dumps.enabled is False
    assert policy.dumps.frames_budget == 3
    assert policy.worker.force_tight_pitch is False
    assert policy.worker.layer_interpolation == NapariInterpolation.LINEAR.value


def test_debug_policy_env_overrides():
    env = {
        "NAPARI_CUDA_DEBUG": json.dumps(
            {
                "enabled": True,
                "flags": ["camera", "encoder-keyframes", "encoder-sps", "encoder-nals"],
                "dumps": {"frames": 5},
                "worker": {
                    "force_tight_pitch": True,
                    "roi_edge_threshold": 7,
                    "layer_interpolation": "nearest",
                },
            }
        )
    }
    policy = load_debug_policy(env)
    assert policy.enabled is True
    assert policy.dumps.enabled is True
    assert policy.dumps.frames_budget == 5
    assert policy.encoder.log_keyframes is True
    assert policy.encoder.log_nals is True
    assert policy.encoder.log_sps is True
    assert policy.worker.force_tight_pitch is True
    assert policy.worker.roi_edge_threshold == 7
    assert policy.worker.layer_interpolation == NapariInterpolation.NEAREST.value
