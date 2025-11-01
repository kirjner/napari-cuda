#!/usr/bin/env python
"""
Systematically investigate PyNvVideoCodec API
"""
import PyNvVideoCodec as pnvc

print("1. Checking constructor signature...")
print("   Constructor: (width, height, format_str, gpu_id, pixel_format_int, use_cpu_buffer, params_dict)")
print()

print("2. Testing format string values...")
test_formats = ['', 'NV12', 'YUV420', 'RGB', 'h264', 'H264', 'AVC']

for fmt_str in test_formats:
    try:
        enc = pnvc.PyNvEncoder(1920, 1080, fmt_str, 0, 3, False, {})
        print(f"   ✓ '{fmt_str}' worked!")
        del enc
    except Exception as e:
        error_msg = str(e).split('\n')[0]
        print(f"   ✗ '{fmt_str}' failed: {error_msg}")

print()
print("3. Looking for examples in the module...")

# Check for any example usage
if hasattr(pnvc, '__file__'):
    print(f"   Module location: {pnvc.__file__}")

# Check for example methods
for name in dir(pnvc):
    if 'example' in name.lower() or 'test' in name.lower() or 'demo' in name.lower():
        print(f"   Found: {name}")

print()
print("4. Checking if there's a codec enum...")
for attr in dir(pnvc):
    if 'codec' in attr.lower():
        obj = getattr(pnvc, attr)
        print(f"   Found: {attr} = {obj}")
        if hasattr(obj, '__dict__'):
            for sub in dir(obj):
                if not sub.startswith('_'):
                    print(f"      - {sub}")
