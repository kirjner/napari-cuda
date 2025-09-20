#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-127.0.0.1}"
STATE_PORT="${2:-29081}"
METRICS_URL="${3:-http://127.0.0.1:8083/metrics.json}"

echo "Running latency policy sweep (zoom-in)"
uv run python scripts/policy_intent_harness.py \
  --host "$HOST" \
  --state-port "$STATE_PORT" \
  --metrics-url "$METRICS_URL" \
  --policies oversampling \
  --levels 0 1 2 \
  --zoom-factors 0.5 0.75 1.0 1.25 1.5 2.0 \
  --idle-delay 1.0 \
  --output-prefix tmp/policy_runs/latency_zoom_in

echo "Running latency policy sweep (zoom-out)"
uv run python scripts/policy_intent_harness.py \
  --host "$HOST" \
  --state-port "$STATE_PORT" \
  --metrics-url "$METRICS_URL" \
  --policies oversampling \
  --levels 2 1 0 \
  --zoom-factors 2.0 1.6 1.2 1.0 0.75 0.5 \
  --idle-delay 1.0 \
  --output-prefix tmp/policy_runs/latency_zoom_out
