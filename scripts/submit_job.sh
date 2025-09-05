#!/usr/bin/env bash
set -euo pipefail

# Load configuration if present
if [[ -f .env.hpc ]]; then
  # shellcheck disable=SC1091
  source .env.hpc
fi

DATASET="${1:-data/test_volume.npy}"

# Partitions preference order (override via GPU_PARTITION if set)
PARTITIONS=()
if [[ -n "${GPU_PARTITION:-}" ]]; then
  PARTITIONS+=("$GPU_PARTITION")
else
  PARTITIONS+=(pi_edboyden ou_bcs_high ou_bcs_low)
fi

MEM="${SLURM_MEM:-32GB}"
CPUS="${SLURM_CPUS:-16}"
TIME="${SLURM_TIME:-04:00:00}"

echo "napari-cuda Job Submission"
echo "Dataset: $DATASET"
echo "Partitions (in order): ${PARTITIONS[*]}"
echo

for partition in "${PARTITIONS[@]}"; do
  echo "Trying partition: $partition"
  if sinfo -h -p "$partition" >/dev/null 2>&1; then
    echo "Submitting to $partition..."
    set -x
    sbatch \
      --partition="$partition" \
      --gres=gpu:1 \
      --mem="$MEM" \
      --cpus-per-task="$CPUS" \
      --time="$TIME" \
      scripts/run_server_slurm.sh "$DATASET"
    set +x
    exit 0
  else
    echo "Partition not available: $partition"
  fi
done

echo "No partitions accepted submission. Check availability with:"
echo "  sinfo -p ${PARTITIONS[*]}"
exit 1

