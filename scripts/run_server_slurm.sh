#!/usr/bin/env bash
set -euo pipefail

# SBATCH defaults (dynamic resources are passed via sbatch CLI by submit_job.sh)
#SBATCH --job-name=napari-cuda
#SBATCH --output=napari-cuda-%j.log
#SBATCH --error=napari-cuda-%j.err
#SBATCH --time=04:00:00

# Load configuration if present
if [[ -f .env.hpc ]]; then
  # shellcheck disable=SC1091
  source .env.hpc
fi

echo "=== Job Environment ==="
echo "Host: $(hostname)"
echo "User: ${USER}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-n/a}"
echo "Working dir: ${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure CUDA is available (module or preinstalled)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
  echo "nvidia-smi not found; ensure GPU drivers are available on this node"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Loading CUDA module: ${CUDA_MODULE:-cuda/12.4.0}"
  module load "${CUDA_MODULE:-cuda/12.4.0}" || true
fi

# Headless GL
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# Move to submit dir
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Installing server dependencies (uv)..."
uv sync --extra server

echo "Validating CUDA-OpenGL interop..."
if ! uv run python scripts/test_cuda_gl.py; then
  echo "❌ CUDA-OpenGL validation failed"
  exit 1
fi
echo "✅ CUDA-OpenGL validation passed"

DATASET="${1:-data/test_volume.npy}"
if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET — generating sample data..."
  uv run python scripts/create_test_data.py || true
  DATASET="data/test_volume.npy"
fi

HOST="${NAPARI_CUDA_HOST:-127.0.0.1}"
STATE_PORT="${NAPARI_CUDA_STATE_PORT:-8081}"
PIXEL_PORT="${NAPARI_CUDA_PIXEL_PORT:-8082}"
METRICS_PORT="${NAPARI_CUDA_METRICS_PORT:-8083}"

echo "Starting napari-cuda-server on $HOST:$STATE_PORT/$PIXEL_PORT (metrics $METRICS_PORT)"
exec uv run napari-cuda-server "$DATASET" 2>&1
