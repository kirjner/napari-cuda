"""Preset registry for server runtime overlays.

The registry centralizes encoder and server tweaks that were previously wired
through ad-hoc environment reads. Each preset provides structured overrides
that are merged into `ServerConfig`, `ServerCtx`, or their nested runtime
dataclasses during context construction.

Presets favour explicitness: unknown override keys raise immediately so we do
not silently drift when fields are renamed. Adding a new preset is therefore a
matter of extending `_PRESETS` with the desired mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class PresetOverrides:
    """Overrides associated with a named preset.

    The dictionaries contain concrete values ready to feed into the respective
    dataclasses (`ServerConfig`, `EncodeCfg`, `EncoderRuntime`,
    `BitstreamRuntime`). We keep them loosely typed to avoid importing those
    modules here and triggering circular dependencies during configuration
    assembly.
    """

    name: str
    description: str = ""
    server: Mapping[str, object] = field(default_factory=dict)
    encode: Mapping[str, object] = field(default_factory=dict)
    encoder_runtime: Mapping[str, object] = field(default_factory=dict)
    bitstream: Mapping[str, object] = field(default_factory=dict)


def _nvenc_preset(name: str, description: str) -> PresetOverrides:
    return PresetOverrides(
        name=name,
        description=description,
        encoder_runtime={"preset": name},
    )


def _profile_preset(
    name: str,
    description: str,
    *,
    encode: Mapping[str, object],
    server: Optional[Mapping[str, object]] = None,
    encoder_runtime: Optional[Mapping[str, object]] = None,
    bitstream: Optional[Mapping[str, object]] = None,
) -> PresetOverrides:
    return PresetOverrides(
        name=name,
        description=description,
        server=server or {},
        encode=encode,
        encoder_runtime=encoder_runtime or {},
        bitstream=bitstream or {},
    )


_PRESETS: dict[str, PresetOverrides] = {
    # NVENC performance tiers (P1 fastest, P7 highest quality). These presets
    # strictly tune the runtime encoder preset; bitrate/fps remain governed by
    # the active profile + encode config.
    "p1": _nvenc_preset("P1", "NVENC ultra-low-latency preset (fastest)."),
    "p2": _nvenc_preset("P2", "NVENC low-latency preset."),
    "p3": _nvenc_preset("P3", "NVENC balanced latency/quality preset."),
    "p4": _nvenc_preset("P4", "NVENC quality-biased preset."),
    "p5": _nvenc_preset("P5", "NVENC high-quality preset."),
    "p6": _nvenc_preset("P6", "NVENC near-max quality preset."),
    "p7": _nvenc_preset("P7", "NVENC maximum quality preset."),

    # Streaming profile bundles that layer encode + runtime hints on top of the
    # base config. These presets mirror the legacy `NAPARI_CUDA_PROFILE` modes
    # but now travel through the central registry.
    "latency": _profile_preset(
        "latency",
        "Low-latency streaming profile (baseline).",
        server={"profile": "latency"},
        encode={"codec": "h264", "bitrate": 10_000_000, "fps": 60, "keyint": 120},
        encoder_runtime={
            "preset": "P3",
            "rc_mode": "cbr",
            "lookahead": 0,
            "aq": 0,
            "temporalaq": 0,
            "enable_non_ref_p": False,
            "bframes": 0,
        },
    ),
    "quality": _profile_preset(
        "quality",
        "Higher quality streaming profile (increased bitrate + AQ).",
        server={"profile": "quality"},
        encode={"codec": "h264", "bitrate": 35_000_000, "fps": 60, "keyint": 120},
        encoder_runtime={
            "preset": "P5",
            "rc_mode": "cbr",
            "lookahead": 16,
            "aq": 1,
            "temporalaq": 1,
            "enable_non_ref_p": False,
            "bframes": 2,
        },
    ),
}


def resolve_preset(name: Optional[str]) -> Optional[PresetOverrides]:
    """Return overrides for *name* if it is a registered preset.

    Matching is case-insensitive; surrounding whitespace is stripped. Unknown
    presets return ``None`` so callers can fall back to their defaults.
    """

    if name is None:
        return None
    slug = name.strip()
    if not slug:
        return None
    preset = _PRESETS.get(slug.lower())
    if preset is not None:
        return preset
    # Accept canonical NVENC preset tokens even if they were not added to the
    # map above (e.g. caller passed "P1").
    upper = slug.upper()
    if upper.startswith("P") and upper[1:].isdigit():
        base = upper.lower()
        if base in _PRESETS:
            return _PRESETS[base]
    return None


def available_presets() -> Mapping[str, PresetOverrides]:
    """Expose the preset registry for documentation or debugging."""

    return dict(_PRESETS)
