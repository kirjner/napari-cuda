# Development Workflow: Laptop → HPC

## The Challenge
- **Development**: On your M3 Mac laptop (no CUDA)
- **Testing**: On HPC with L4 GPU (has CUDA)
- **Goal**: Smooth development cycle

## Recommended Workflow

### 1. Two-Repository Strategy
```bash
# Laptop: Main development
~/projects/napari-cuda         # Full code, docs, tests

# HPC: Testing repository  
~/napari-cuda-hpc             # Synced code for GPU testing
```

### 2. Development Cycle

```mermaid
graph LR
    A[Write Code on Laptop] --> B[Sync to HPC]
    B --> C[Test with CUDA]
    C --> D[See Results]
    D --> A
```

### 3. Project Structure for Hybrid Development

```python
napari-cuda/
├── src/napari_cuda/
│   ├── cuda/              # CUDA-specific code
│   │   ├── available.py   # Check if CUDA available
│   │   ├── screenshot.py  # CUDA screenshot (with fallback)
│   │   └── kernels.cu     # Actual CUDA kernels
│   ├── cpu/               # CPU fallbacks
│   │   └── screenshot.py  # Standard screenshot
│   └── __init__.py        # Smart imports based on availability
├── tests/
│   ├── test_cpu.py        # Can run on laptop
│   └── test_cuda.py       # Must run on HPC
└── dev/
    ├── sync.sh            # Sync script
    └── hpc_test.sh        # HPC test runner
```

### 4. Smart Code Organization

```python
# src/napari_cuda/__init__.py
import warnings

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    warnings.warn("CUDA not available, using CPU fallbacks")

if HAS_CUDA:
    from napari_cuda.cuda import CUDAScreenshot
    Screenshot = CUDAScreenshot
else:
    from napari_cuda.cpu import CPUScreenshot
    Screenshot = CPUScreenshot
```

This way your code works on both laptop (CPU) and HPC (GPU)!

## Practical Commands

### On Laptop (Development)
```bash
# Write code
code .

# Test CPU parts locally
uv run pytest tests/test_cpu.py

# When ready, sync to HPC
./dev/sync.sh sync
```

### Quick HPC Test
```bash
# One command from laptop to test on HPC
./dev/sync.sh test
```

### Interactive HPC Session
```bash
# When you need to debug on HPC
./dev/sync.sh shell

# Now you're on HPC, can run:
python -c "import cupy; print(cupy.cuda.Device(0).name)"
```

## VS Code Remote (Alternative)

```bash
# Install Remote-SSH extension in VS Code
# Add to ~/.ssh/config:
Host hpc
    HostName node2810.your-cluster.edu
    User kirjner

# Then in VS Code:
# Cmd+Shift+P → "Remote-SSH: Connect to Host" → hpc
# Now VS Code runs on HPC, full CUDA access!
```

## Jupyter Notebook on HPC (For Experimentation)

```bash
# On HPC
ssh kirjner@node2810
module load cuda/12.4
jupyter notebook --no-browser --port=8888

# On laptop (new terminal)
ssh -L 8888:localhost:8888 kirjner@node2810

# Open browser to localhost:8888
# Now you have Jupyter with GPU access!
```

## Git Branch Strategy

```bash
main           # Stable code that works CPU+GPU
├── dev-cuda   # Active CUDA development
├── dev-ui     # UI work (can do on laptop)
└── benchmark  # Performance testing on HPC
```

## Testing Strategy

```python
# tests/test_cuda.py
import pytest

pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cuda_screenshot():
    # Only runs on HPC
    pass

def test_cpu_fallback():
    # Runs everywhere
    pass
```

## Environment Management

```bash
# .env.laptop (for local dev)
NAPARI_CUDA_MODE=CPU
NAPARI_CUDA_DEBUG=1

# .env.hpc (for HPC)
NAPARI_CUDA_MODE=GPU
NAPARI_CUDA_DEVICE=0
```

## Recommended: tmux on HPC

```bash
# Start persistent session on HPC
ssh kirjner@node2810
tmux new -s napari-cuda

# Run long tests
module load cuda/12.4
python benchmark.py

# Detach: Ctrl+B, D
# Reattach later: tmux attach -t napari-cuda
```

## The "Quick Test" Pattern

```python
# quick_test.py - for rapid iteration
import sys
sys.path.insert(0, '/path/to/napari-cuda')

if __name__ == "__main__":
    # Test specific CUDA function
    from napari_cuda.cuda import test_function
    result = test_function()
    print(f"✅ Result: {result}")
```

Then: `./dev/sync.sh sync && ssh hpc "cd napari-cuda && python quick_test.py"`