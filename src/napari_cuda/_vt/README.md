This directory contains the macOS VideoToolbox (VT) native shim for the napari-cuda client.

Phase 1 goals:
- Own VT session + output callback in Obj-C++
- CFRetain/CFRelease CVPixelBuffer across threads
- Bounded ring queue for decoded frames
- Minimal C ABI exposed via a CPython extension (macOS only)
- Python polls frames and maps to RGB (Phase 1); zero-copy GL comes in Phase 2

Build system integration will expose the module as `napari_cuda._vt`.

