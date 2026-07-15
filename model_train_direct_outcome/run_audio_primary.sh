#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SEGMENTS="${MAX_SEGMENTS:-32}"
TARGET_RECALL="${TARGET_RECALL:-0.88}"
ABNORMAL_WEIGHT="${ABNORMAL_WEIGHT:-1.0}"
LR="${LR:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/audio_primary_v1.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/audio_primary_v1}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}"
  echo "Run ./setup_gpu_env.sh first."
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

if [[ "${DEVICE}" == "cuda" ]]; then
  python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA was requested but torch.cuda.is_available() is False.")
    sys.exit(1)
PY
fi

mkdir -p checkpoints results

python src/train_audio_primary.py \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-segments-per-patient "${MAX_SEGMENTS}" \
  --target-recall "${TARGET_RECALL}" \
  --outcome-abnormal-weight "${ABNORMAL_WEIGHT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
