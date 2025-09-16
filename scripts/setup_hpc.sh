#!/bin/bash
#
# napari-cuda HPC Setup Script
# Configures environment for GPU-accelerated napari streaming
#

set -e  # Exit on error

echo "========================================="
echo "napari-cuda HPC Setup"
echo "========================================="

# Check if we're on HPC with GPU
if ! nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. Are you on the HPC with GPU?"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Load HPC modules (adjust for your system)
echo ""
echo "Step 1: Loading HPC modules..."
echo "-------------------------------"
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || echo "  ⚠ CUDA module not loaded"
module load gcc/11.2 2>/dev/null || echo "  ⚠ GCC module not loaded"

# 2. Detect CUDA version
echo ""
echo "Step 2: Detecting CUDA version..."
echo "-------------------------------"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    echo "  ✓ CUDA version: $CUDA_VERSION (major: $CUDA_MAJOR)"
else
    echo "  ✗ nvcc not found - CUDA may not be available"
    CUDA_MAJOR=""
fi

# 3. Set environment variables
echo ""
echo "Step 3: Setting environment variables..."
echo "-------------------------------"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM=offscreen
export PYOPENGL_PLATFORM=egl

echo "  CUDA_HOME=$CUDA_HOME"
echo "  QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo "  PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"

# 4. Install uv if needed
echo ""
echo "Step 4: Checking uv installation..."
echo "-------------------------------"
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo "  ✓ uv installed"
else
    echo "  ✓ uv already installed: $(which uv)"
fi

# 5. Sync base dependencies
echo ""
echo "Step 5: Installing dependencies with uv..."
echo "-------------------------------"
echo "  Running: uv sync --extra server"
uv sync --extra server

# 6. Install CUDA-specific Python packages
echo ""
echo "Step 6: Installing CUDA Python packages..."
echo "-------------------------------"
if [ -n "$CUDA_MAJOR" ]; then
    echo "  Installing packages for CUDA $CUDA_MAJOR..."
    
    # Install pycuda
    uv pip install pycuda
    
    # Install appropriate CuPy version
    if [ "$CUDA_MAJOR" = "12" ]; then
        echo "  Installing cupy-cuda12x..."
        uv pip install cupy-cuda12x
    elif [ "$CUDA_MAJOR" = "11" ]; then
        echo "  Installing cupy-cuda11x..."
        uv pip install cupy-cuda11x
    else
        echo "  WARNING: Unsupported CUDA version $CUDA_VERSION"
        echo "  You may need to manually install the correct CuPy version"
    fi
else
    echo "  ✗ Skipping CUDA packages (no CUDA detected)"
fi

# 7. Create necessary directories
echo ""
echo "Step 7: Creating project directories..."
echo "-------------------------------"
mkdir -p data logs results frames
echo "  ✓ Created: data/ logs/ results/ frames/"

# 8. Test CUDA functionality
echo ""
echo "Step 8: Testing CUDA functionality..."
echo "-------------------------------"
uv run python -c "
import sys
print('  Python:', sys.executable)
print('  Version:', sys.version)

try:
    import pycuda.driver as cuda
    cuda.init()
    device_count = cuda.Device.count()
    print(f'  ✓ PyCUDA: {device_count} device(s) found')
    for i in range(device_count):
        dev = cuda.Device(i)
        attrs = dev.get_attributes()
        mem_gb = dev.total_memory() / (1024**3)
        print(f'    Device {i}: {dev.name()} ({mem_gb:.1f} GB)')
except ImportError:
    print('  ✗ PyCUDA not installed')
except Exception as e:
    print(f'  ✗ CUDA error: {e}')

try:
    import cupy as cp
    print(f'  ✓ CuPy: version {cp.__version__}')
    x = cp.array([1, 2, 3])
    print(f'    Test computation: sum([1,2,3]) = {x.sum()}')
except ImportError:
    print('  ✗ CuPy not installed')
except Exception as e:
    print(f'  ✗ CuPy error: {e}')
"

# 9. Test OpenGL
echo ""
echo "Step 9: Testing OpenGL..."
echo "-------------------------------"
uv run python -c "
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
try:
    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QOpenGLContext
    app = QApplication([])
    ctx = QOpenGLContext()
    ctx.create()
    if ctx.isValid():
        print('  ✓ OpenGL context creation successful')
    else:
        print('  ✗ OpenGL context creation failed')
except Exception as e:
    print(f'  ✗ OpenGL error: {e}')
"

# 10. Check NVENC availability
echo ""
echo "Step 10: Checking NVENC encoder..."
echo "-------------------------------"
if nvidia-smi --query-gpu=encoder.stats.sessionCount --format=csv,noheader 2>/dev/null; then
    echo "  ✓ NVENC hardware encoder available"
else
    echo "  ⚠ NVENC not available (will use software encoding)"
fi

# 11. Create test data
echo ""
echo "Step 11: Creating test data..."
echo "-------------------------------"
uv run python scripts/create_test_data.py || echo "  ⚠ Test data script not found, creating inline..."

if [ ! -f "data/test_volume.npy" ]; then
    uv run python -c "
import numpy as np
print('  Creating test volume...')
x = np.linspace(-3, 3, 256)
y = np.linspace(-3, 3, 256)
z = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
data = []
for z_val in z:
    frame = np.sin(np.sqrt(X**2 + Y**2) - z_val*0.5) * np.exp(-0.1*(X**2 + Y**2))
    data.append(frame)
data = np.array(data, dtype=np.float32)
np.save('data/test_volume.npy', data)
print(f'  ✓ Created test volume: {data.shape} ({data.nbytes/(1024**2):.1f} MB)')
"
fi

# 12. Create launcher scripts
echo ""
echo "Step 12: Creating launcher scripts..."
echo "-------------------------------"

cat > start_server.sh << 'EOF'
#!/bin/bash
# Start napari-cuda server

# Load modules
module load cuda/12.4 2>/dev/null || true

# Load project environment if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Start server
echo "Starting napari-cuda server..."
echo "Clients can connect to:"
echo "  State port: 8081"
echo "  Pixel port: 8082"
echo ""
exec uv run python -m napari_cuda.server.headless_server ${1:-data/test_volume.npy}
EOF
chmod +x start_server.sh

cat > test_interop.sh << 'EOF'
#!/bin/bash
# Test CUDA-OpenGL interop

module load cuda/12.4 2>/dev/null || true

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

exec make verify-gl
EOF
chmod +x test_interop.sh

echo "  ✓ Created: start_server.sh test_interop.sh"

# 13. Print summary
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Environment Summary:"
echo "  CUDA Version: ${CUDA_VERSION:-Not detected}"
echo "  Python: $(uv run python --version 2>&1)"
echo "  Project: $(pwd)"
echo ""
echo "Next Steps:"
echo "  1. Test CUDA-GL interop:"
echo "     ./test_interop.sh"
echo ""
echo "  2. Start server:"
echo "     ./start_server.sh"
echo ""
echo "  3. From local machine, setup SSH tunnel:"
echo "     ssh -L 8081:localhost:8081 -L 8082:localhost:8082 $(whoami)@$(hostname)"
echo ""
echo "  4. Then run client locally:"
echo "     uv run napari-cuda-client"
echo ""
echo "Troubleshooting:"
echo "  - If CUDA not found: check 'module list' and 'module avail cuda'"
echo "  - If OpenGL fails: try export PYOPENGL_PLATFORM=osmesa"
echo "  - Check logs in: ./logs/"
echo ""
