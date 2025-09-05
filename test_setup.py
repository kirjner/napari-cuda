#!/usr/bin/env python
"""
Test script to verify napari-cuda setup.
Run this on both laptop (CPU) and HPC (GPU) to confirm everything works.
"""

import sys
import platform
import os

def test_environment():
    """Test the environment setup."""
    print("=" * 60)
    print("NAPARI-CUDA ENVIRONMENT TEST")
    print("=" * 60)
    
    # System info
    print(f"\n📍 System Information:")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Hostname: {platform.node()}")
    
    # CUDA check
    print(f"\n🎮 CUDA Check:")
    try:
        import pycuda.driver as cuda
        cuda.init()
        print(f"  ✅ PyCUDA available")
        print(f"  Device count: {cuda.Device.count()}")
        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            print(f"  GPU {i}: {dev.name()}")
            attrs = dev.get_attributes()
            print(f"    Memory: {dev.total_memory() // 1024**3} GB")
            print(f"    Compute: {attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]}")
    except ImportError:
        print(f"  ❌ PyCUDA not installed")
    except Exception as e:
        print(f"  ❌ CUDA not available: {e}")
    
    # CuPy check
    print(f"\n📊 CuPy Check:")
    try:
        import cupy as cp
        print(f"  ✅ CuPy available")
        print(f"  Version: {cp.__version__}")
        # Try to use GPU
        arr = cp.array([1, 2, 3])
        result = cp.sum(arr)
        print(f"  GPU compute test: {result.get()} (expected: 6)")
    except ImportError:
        print(f"  ❌ CuPy not installed")
    except Exception as e:
        print(f"  ❌ CuPy error: {e}")
    
    # Napari check
    print(f"\n🔬 Napari Check:")
    try:
        import napari
        print(f"  ✅ Napari available")
        print(f"  Version: {napari.__version__}")
        
        # Try headless
        viewer = napari.Viewer(show=False)
        print(f"  ✅ Headless viewer works")
        viewer.close()
    except ImportError:
        print(f"  ❌ Napari not installed")
    except Exception as e:
        print(f"  ⚠️  Napari error: {e}")
    
    # Our module check
    print(f"\n🚀 napari-cuda Check:")
    try:
        sys.path.insert(0, 'src')
        import napari_cuda
        print(f"  ✅ Module imports")
        print(f"  CUDA available: {napari_cuda.HAS_CUDA}")
        if napari_cuda.HAS_CUDA:
            print(f"  Device: {napari_cuda.CUDA_DEVICE_NAME}")
    except Exception as e:
        print(f"  ❌ Import error: {e}")
    
    print("\n" + "=" * 60)
    
    # Summary
    if 'node' in platform.node():  # Likely on HPC
        print("📍 Looks like you're on the HPC")
        print("   You should see CUDA available above")
    else:
        print("📍 Looks like you're on a local machine")
        print("   CUDA may not be available, that's OK")
    
    print("\n💡 Next steps:")
    if 'cupy' not in sys.modules:
        print("   - Install CuPy: pip install cupy-cuda12x")
    if 'pycuda' not in sys.modules:
        print("   - Install PyCUDA: pip install pycuda")
    print("   - Sync to HPC: ./dev/sync.sh sync")
    print("   - Test on HPC: ./dev/sync.sh test")
    

if __name__ == "__main__":
    test_environment()