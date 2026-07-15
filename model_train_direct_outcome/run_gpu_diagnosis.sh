#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SEGMENTS="${MAX_SEGMENTS:-32}"
TARGET_RECALL="${TARGET_RECALL:-0.88}"
FIXED_THRESHOLDS="${FIXED_THRESHOLDS:-0.20,0.30,0.40,0.50,0.60,0.70,0.80}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/direct_outcome_gpu_position_aware_fp010_v1.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/diagnosis_position_aware_fp010_v1}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}"
  echo "Run ./setup_gpu_env.sh first."
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python src/diagnose_direct_outcome.py \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-segments-per-patient "${MAX_SEGMENTS}" \
  --target-recall "${TARGET_RECALL}" \
  --fixed-thresholds "${FIXED_THRESHOLDS}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
