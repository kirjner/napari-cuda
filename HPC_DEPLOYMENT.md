# HPC Deployment

This guide describes how to run the napari-cuda server on an HPC GPU node (e.g., L4) and connect from your local machine via SSH tunneling through the login node.

## Your Cluster Configuration

Based on your cluster discovery:
- **H100 GPUs** (80GB HBM3): Available on `ou_bcs_low` and `ou_bcs_high` partitions
- **L4 GPUs** (24GB VRAM): Available on `pi_edboyden` partition  
- **Module**: Use `nvhpc/24.5` (includes CUDA 12.4, cuDNN, NCCL, math libraries)

Recommended `.env.hpc` for your cluster:
```bash
CUDA_MODULE=nvhpc/24.5          # Comprehensive NVIDIA HPC SDK
GPU_PARTITION=ou_bcs_high        # For H100 (use pi_edboyden for L4)
SLURM_MEM=64GB                   # H100 nodes have more memory
LOGIN_HOST=login.cluster.mit.edu # Adjust to your login node
```

## 1) Configure Your Cluster
- Copy the sample config: `cp .env.hpc.example .env.hpc` and edit values (partition, memory/CPUs, login host).
- Copy `.env.example` to `.env` and update CUDA paths if they differ (e.g., `CUDA_HOME`).
- Run `make setup-gl` once per node to install Mesa/EGL headers under `build/gl`.
- `make verify-gl` confirms EGL + PyCUDA interop; it also ensures `.env` carries the right `NAPARI_CUDA_GL_PREFIX`/`NAPARI_CUDA_GL_DRIVERS` values for future runs.
- Required env on compute node: `QT_QPA_PLATFORM=offscreen`, `PYOPENGL_PLATFORM=egl` (already set via `.env`).
- Ensure CUDA module is available (e.g., `module load nvhpc/24.5`).
- Ensure CUDA module is available (e.g., `module load nvhpc/24.5`).

## 2) Submit a GPU Job
Use the helper to submit to the first available partition:

```bash
chmod +x scripts/*.sh
./scripts/submit_job.sh data/test_volume.npy
```

This submits `scripts/run_server_slurm.sh`, which will:
- Install server deps with `uv sync --extra server`
- Validate CUDA-OpenGL interop via `make verify-gl`
- Start `napari-cuda-server` bound to `127.0.0.1`

Monitor with `watch squeue -u $USER` and tail `napari-cuda-<jobid>.log`.

## 3) Open an SSH Tunnel
Forward state, pixel, and metrics ports through the login node to the compute node:

```bash
./scripts/setup_ssh_tunnel.sh <compute-node>
# (or omit argument to auto-detect your job's node)
```

## 4) Connect the Client
First validate your local environment:

```bash
uv sync --extra client
uv run python scripts/validate_local.py
```

Then launch the client:

```bash
uv run napari-cuda-client --host 127.0.0.1 --state-port 8081 --pixel-port 8082
```

Metrics:
- JSON: `curl http://127.0.0.1:8083/metrics.json`
- Dashboard: open `http://127.0.0.1:8083/dashboard` in your browser

## Tips & Troubleshooting
- Interop test fails: verify CUDA module, EGL env vars, and that you’re on a GPU node.
- No frames: ensure both 8081 (state) and 8082 (pixels) are forwarded; check server logs.
- Slow I/O: place datasets on node-local scratch if available.
- Security: server binds to `127.0.0.1` by default; use SSH tunnels rather than exposing ports.
