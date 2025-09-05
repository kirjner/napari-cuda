#!/usr/bin/env bash
set -euo pipefail

# Load configuration if present
if [[ -f .env.hpc ]]; then
  # shellcheck disable=SC1091
  source .env.hpc
fi

STATE_PORT="${NAPARI_CUDA_STATE_PORT:-8081}"
PIXEL_PORT="${NAPARI_CUDA_PIXEL_PORT:-8082}"
METRICS_PORT="${NAPARI_CUDA_METRICS_PORT:-8083}"
LOGIN_HOST="${LOGIN_HOST:-login.cluster.mit.edu}"
CLUSTER_USER="${CLUSTER_USER:-$USER}"

# Compute node can be passed as the first arg or auto-detected from running jobs
COMPUTE_NODE="${1:-}"
if [[ -z "$COMPUTE_NODE" ]]; then
  COMPUTE_NODE=$(squeue -u "$USER" -h -o '%N' | head -n1 || true)
fi
if [[ -z "$COMPUTE_NODE" || "$COMPUTE_NODE" == "(null)" ]]; then
  echo "Error: Could not detect compute node. Provide it explicitly, e.g.:"
  echo "  $0 node1803"
  exit 1
fi

echo "=== SSH Tunnel ==="
echo "Login host: $CLUSTER_USER@$LOGIN_HOST"
echo "Compute node: $COMPUTE_NODE"
echo "Forwarded ports: $STATE_PORT (state), $PIXEL_PORT (pixels), $METRICS_PORT (metrics)"
echo "Press Ctrl+C to stop tunnel."

ssh -N -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -L "$STATE_PORT":127.0.0.1:"$STATE_PORT" \
  -L "$PIXEL_PORT":127.0.0.1:"$PIXEL_PORT" \
  -L "$METRICS_PORT":127.0.0.1:"$METRICS_PORT" \
  -J "$CLUSTER_USER@$LOGIN_HOST" "${CLUSTER_USER}@${COMPUTE_NODE}"

