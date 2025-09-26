from __future__ import annotations

import json

from napari_cuda.server.config import load_server_ctx


def test_load_server_ctx_hydrates_json_bundles() -> None:
    env = {
        "NAPARI_CUDA_HOST": "127.0.0.1",
        "NAPARI_CUDA_PROFILE": "quality",
        "NAPARI_CUDA_ENCODER_CONFIG": json.dumps(
            {
                "encode": {
                    "fps": 72,
                    "bitrate": 12_500_000,
                    "keyint": 90,
                    "codec": "hevc",
                },
                "runtime": {
                    "input_format": "NV12",
                    "rc_mode": "vbr",
                    "max_bitrate": 18_000_000,
                    "lookahead": 12,
                    "aq": 2,
                    "temporalaq": 1,
                    "non_ref_p": True,
                    "bframes": 3,
                    "idr_period": 240,
                    "preset": "P6",
                },
                "bitstream": {
                    "build_cython": False,
                    "disable_fast_pack": True,
                    "allow_py_fallback": True,
                },
                "dump_bitstream": 4,
                "dump_dir": "tmp/custom_bitstreams",
            }
        ),
        "NAPARI_CUDA_POLICY_CONFIG": json.dumps(
            {
                "threshold_in": 1.1,
                "threshold_out": 1.9,
                "hysteresis": 0.05,
                "fine_threshold": 1.4,
                "cooldown_ms": 80.0,
                "preserve_view_on_switch": False,
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
        "NAPARI_CUDA_KF_WATCHDOG_COOLDOWN": "4.5",
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

    assert ctx.encoder_runtime.input_format == "NV12"
    assert ctx.encoder_runtime.rc_mode == "vbr"
    assert ctx.encoder_runtime.max_bitrate == 18_000_000
    assert ctx.encoder_runtime.lookahead == 12
    assert ctx.encoder_runtime.aq == 2
    assert ctx.encoder_runtime.temporalaq == 1
    assert ctx.encoder_runtime.enable_non_ref_p is True
    assert ctx.encoder_runtime.bframes == 3
    assert ctx.encoder_runtime.idr_period == 240
    assert ctx.encoder_runtime.preset == "P6"

    assert ctx.bitstream.build_cython is False
    assert ctx.bitstream.disable_fast_pack is True
    assert ctx.bitstream.allow_py_fallback is True

    assert ctx.debug_policy.enabled is True
    assert ctx.debug_policy.encoder.log_keyframes is True
    assert ctx.debug_policy.dumps.frames_budget == 2
    assert ctx.debug_policy.dumps.output_dir == "logs/custom_frames"

    assert ctx.policy.threshold_in == 1.1
    assert ctx.policy.threshold_out == 1.9
    assert ctx.policy.hysteresis == 0.05
    assert ctx.policy.fine_threshold == 1.4
    assert ctx.policy.cooldown_ms == 80.0
    assert ctx.policy.preserve_view_on_switch is False
    assert ctx.policy.sticky_contrast is False
    assert ctx.policy.oversampling_thresholds == {0: 1.3, 1: 2.7}
    assert ctx.policy.oversampling_hysteresis == 0.25
