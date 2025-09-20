#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-127.0.0.1}"
STATE_PORT="${2:-29081}"
METRICS_URL="${3:-http://127.0.0.1:8083/metrics.json}"

uv run python scripts/policy_intent_harness.py \
  --host "$HOST" \
  --state-port "$STATE_PORT" \
  --metrics-url "$METRICS_URL" \
  --policies oversampling \
  --levels 2 1 0 \
  --zoom-factors 2.0 1.6 1.2 1.0 0.75 0.5 \
  --idle-delay 1.0 \
  --output-prefix tmp/policy_runs/latency_zoom_out
