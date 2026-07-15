#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TARGET_RECALL="${TARGET_RECALL:-0.88}"
ABNORMAL_WEIGHT="${ABNORMAL_WEIGHT:-1.0}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/clinical_only_baseline_v1.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/clinical_only_baseline_v1}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}"
  echo "Run ./setup_gpu_env.sh first."
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python src/train_clinical_baseline.py \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --target-recall "${TARGET_RECALL}" \
  --outcome-abnormal-weight "${ABNORMAL_WEIGHT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
