from __future__ import annotations

import json

from napari_cuda.server.app.config import load_server_ctx


def test_load_server_ctx_honours_env_overrides() -> None:
    env = {
        "NAPARI_CUDA_HOST": "127.0.0.1",
        "NAPARI_CUDA_ENCODE_FPS": "72",
        "NAPARI_CUDA_ENCODE_BITRATE": "12500000",
        "NAPARI_CUDA_ENCODE_KEYINT": "90",
        "NAPARI_CUDA_ENCODE_CODEC": "hevc",
        "NAPARI_CUDA_ENCODER_PRESET": "P6",
        "NAPARI_CUDA_ENCODER_RC_MODE": "vbr",
        "NAPARI_CUDA_ENCODER_MAX_BITRATE": "18000000",
        "NAPARI_CUDA_ENCODER_LOOKAHEAD": "12",
        "NAPARI_CUDA_ENCODER_AQ": "2",
        "NAPARI_CUDA_ENCODER_TEMPORALAQ": "1",
        "NAPARI_CUDA_ENCODER_NON_REF_P": "1",
        "NAPARI_CUDA_ENCODER_BFRAMES": "3",
        "NAPARI_CUDA_ENCODER_IDR_PERIOD": "240",
        "NAPARI_CUDA_ENCODER_INPUT_FORMAT": "NV12",
        "NAPARI_CUDA_DUMP_BITSTREAM": "4",
        "NAPARI_CUDA_DUMP_DIR": "tmp/custom_bitstreams",
        "NAPARI_CUDA_BITSTREAM_BUILD_CYTHON": "0",
        "NAPARI_CUDA_BITSTREAM_DISABLE_FAST_PACK": "1",
        "NAPARI_CUDA_BITSTREAM_ALLOW_PY_FALLBACK": "1",
        "NAPARI_CUDA_KF_WATCHDOG_COOLDOWN": "4.5",
        "NAPARI_CUDA_POLICY_CONFIG": json.dumps(
            {
                "threshold_in": 1.1,
                "threshold_out": 1.9,
                "hysteresis": 0.05,
                "fine_threshold": 1.4,
                "cooldown_ms": 80.0,
                "sticky_contrast": False,
                "oversampling": {
                    "thresholds": {"0": 1.3, "1": 2.7},
                    "hysteresis": 0.25,
                },
            }
        ),
        "NAPARI_CUDA_DEBUG": json.dumps(
            {
                "enabled": True,
                "flags": ["encoder-keyframes"],
                "dumps": {"frames": 2, "dir": "logs/custom_frames"},
            }
        ),
    }

    ctx = load_server_ctx(env)

    assert ctx.cfg.host == "127.0.0.1"
    assert ctx.cfg.encode.fps == 72
    assert ctx.cfg.encode.bitrate == 12_500_000
    assert ctx.cfg.encode.keyint == 90
    assert ctx.cfg.encode.codec == "hevc"

    assert ctx.dump_bitstream == 4
    assert ctx.dump_dir == "tmp/custom_bitstreams"
    assert ctx.kf_watchdog_cooldown_s == 4.5

    runtime = ctx.encoder_runtime
    assert runtime.input_format == "NV12"
    assert runtime.rc_mode == "vbr"
    assert runtime.max_bitrate == 18_000_000
    assert runtime.lookahead == 12
    assert runtime.aq == 2
    assert runtime.temporalaq == 1
    assert runtime.enable_non_ref_p is True
    assert runtime.bframes == 3
    assert runtime.idr_period == 240
    assert runtime.preset == "P6"

    bitstream = ctx.bitstream
    assert bitstream.build_cython is False
    assert bitstream.disable_fast_pack is True
    assert bitstream.allow_py_fallback is True

    debug_policy = ctx.debug_policy
    assert debug_policy.enabled is True
    assert debug_policy.encoder.log_keyframes is True
    assert debug_policy.dumps.frames_budget == 2
    assert debug_policy.dumps.output_dir == "logs/custom_frames"

    policy = ctx.policy
    assert policy.threshold_in == 1.1
    assert policy.threshold_out == 1.9
    assert policy.hysteresis == 0.05
    assert policy.fine_threshold == 1.4
    assert policy.cooldown_ms == 80.0
    assert policy.sticky_contrast is False
    assert policy.oversampling_thresholds == {0: 1.3, 1: 2.7}
    assert policy.oversampling_hysteresis == 0.25
def test_load_server_ctx_defaults_use_cbr() -> None:
    ctx = load_server_ctx({})
    assert ctx.encoder_runtime.rc_mode == "cbr"
