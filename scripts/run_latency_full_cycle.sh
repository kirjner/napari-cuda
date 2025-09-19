#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-127.0.0.1}
STATE_PORT=${STATE_PORT:-8081}
PIXEL_PORT=${PIXEL_PORT:-8082}
METRICS_PORT=${METRICS_PORT:-8083}
ZARR_PATH=${ZARR_PATH:-/orcd/data/edboyden/002/Rob/wholebrain/561_amy_resampled_raw.zarr/}
WIDTH=${WIDTH:-1920}
HEIGHT=${HEIGHT:-1080}
FPS=${FPS:-30}
FRAMES=${FRAMES:-240}
PIXEL_MAX_RUNTIME=${PIXEL_MAX_RUNTIME:-40}
IDLE_DELAY=${IDLE_DELAY:-1.0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="tmp/policy_runs/${RUN_TAG}"
LOG_PATH="${RUN_DIR}/headless_server.log"
METRICS_URL="http://${HOST}:${METRICS_PORT}/metrics.json"

mkdir -p "${RUN_DIR}"

cleanup() {
  if [[ -n "${PIXEL_PID:-}" ]] && ps -p "${PIXEL_PID}" > /dev/null 2>&1; then
    kill "${PIXEL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && ps -p "${SERVER_PID}" > /dev/null 2>&1; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if pgrep -f "napari-cuda-server" > /dev/null 2>&1; then
  echo "Found existing napari-cuda-server processes; terminating"
  pkill -f "napari-cuda-server" || true
  sleep 2
fi

printf 'Starting headless server (logs -> %s)\n' "${LOG_PATH}"
QT_QPA_PLATFORM=offscreen \
PYOPENGL_PLATFORM=egl \
uv run napari-cuda-server \
  --host "${HOST}" \
  --state-port "${STATE_PORT}" \
  --pixel-port "${PIXEL_PORT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --encoder-profile latency \
  --zarr "${ZARR_PATH}" \
  > "${LOG_PATH}" 2>&1 &
SERVER_PID=$!

wait_for_ready() {
  local attempts=0
  local max_attempts=${1:-60}
  local sleep_s=${2:-1}
  while [[ ${attempts} -lt ${max_attempts} ]]; do
    if curl -fsS "${METRICS_URL}" > /dev/null 2>&1; then
      return 0
    fi
    if ! ps -p "${SERVER_PID}" > /dev/null 2>&1; then
      echo "Server process exited unexpectedly" >&2
      return 1
    fi
    attempts=$((attempts + 1))
    sleep "${sleep_s}"
  done
  echo "Timed out waiting for metrics endpoint" >&2
  return 1
}

wait_for_ready 90 1
printf 'Server ready at %s\n' "${METRICS_URL}"

# Prime latency policy with timing samples for each multiscale level.
printf 'Launching pixel drain (frames=%s)\n' "${FRAMES}"
uv run python scripts/pixel_drain.py \
  --host "${HOST}" \
  --pixel-port "${PIXEL_PORT}" \
  --frames "${FRAMES}" \
  --idle-timeout 2.0 \
  --max-runtime "${PIXEL_MAX_RUNTIME}" \
  > "${RUN_DIR}/pixel_drain.log" 2>&1 &
PIXEL_PID=$!

sleep 3

printf 'Running combined zoom sweeps\n'
uv run python scripts/policy_intent_harness.py \
  --host "${HOST}" \
  --state-port "${STATE_PORT}" \
  --metrics-url "${METRICS_URL}" \
  --policies latency \
  --idle-delay "${IDLE_DELAY}" \
  --level-delay 0.4 \
  --levels 2 1 0 0 1 2 0 1 2 2 1 0 \
  --zoom-factors 0.6 0.9 1.2 1.8 2.5 4.0 6.0 9.0 6.0 3.5 1.8 1.0 0.7 0.6 \
  --output-prefix "${RUN_DIR}/latency_zoom"

printf 'Waiting for pixel driver to finish\n'
if ! wait "${PIXEL_PID}"; then
  echo "Pixel driver exited with non-zero status" >&2
fi

printf 'Capturing final metrics snapshot\n'
if curl -fsS "${METRICS_URL}" -o "${RUN_DIR}/metrics_final.json"; then
  echo "Metrics snapshot saved to ${RUN_DIR}/metrics_final.json"
else
  echo "Failed to fetch metrics snapshot" >&2
fi

if [[ -f tmp/policy_metrics_latest.json ]]; then
  cp tmp/policy_metrics_latest.json "${RUN_DIR}/policy_metrics_latest.json"
fi

printf '\nRun artifacts stored in %s:\n' "${RUN_DIR}"
ls -1 "${RUN_DIR}"
