from napari_cuda.server.logging_policy import load_debug_policy


def test_debug_policy_defaults():
    policy = load_debug_policy({})
    assert policy.enabled is False
    assert policy.logging.log_camera_info is False
    assert policy.encoder.log_keyframes is False
    assert policy.dumps.enabled is False
    assert policy.dumps.frames_budget == 3
    assert policy.worker.force_tight_pitch is False
    assert policy.worker.layer_interpolation == "bilinear"


def test_debug_policy_env_overrides():
    env = {
        "NAPARI_CUDA_DEBUG": "1",
        "NAPARI_CUDA_DEBUG_FRAMES": "5",
        "NAPARI_CUDA_LOG_KEYFRAMES": "1",
        "NAPARI_CUDA_LOG_SPS": "1",
        "NAPARI_CUDA_FORCE_TIGHT_PITCH": "1",
        "NAPARI_CUDA_ROI_EDGE_THRESHOLD": "7",
        "NAPARI_CUDA_LOG_NALS": "1",
        "NAPARI_CUDA_INTERP": "nearest",
    }
    policy = load_debug_policy(env)
    assert policy.enabled is True
    assert policy.dumps.enabled is True
    assert policy.dumps.frames_budget == 5
    assert policy.encoder.log_keyframes is True
    assert policy.encoder.log_sps is True
    assert policy.worker.force_tight_pitch is True
    assert policy.worker.roi_edge_threshold == 7
    assert policy.worker.layer_interpolation == "nearest"
