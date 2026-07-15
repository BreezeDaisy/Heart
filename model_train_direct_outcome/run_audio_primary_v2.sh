#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-70}"
PATIENCE="${PATIENCE:-14}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SEGMENTS="${MAX_SEGMENTS:-32}"
TARGET_RECALL="${TARGET_RECALL:-0.88}"
ABNORMAL_WEIGHT="${ABNORMAL_WEIGHT:-1.10}"
LR="${LR:-0.0004}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0002}"
POSITION_DROPOUT="${POSITION_DROPOUT:-0.35}"
POSITION_EMBED_DROPOUT="${POSITION_EMBED_DROPOUT:-0.20}"
SAMPLE_NORMAL_ABSENT_WEIGHT="${SAMPLE_NORMAL_ABSENT_WEIGHT:-1.50}"
SAMPLE_COMPLETE_NORMAL_WEIGHT="${SAMPLE_COMPLETE_NORMAL_WEIGHT:-1.50}"
HARD_NEGATIVE_WEIGHT="${HARD_NEGATIVE_WEIGHT:-0.60}"
HARD_NEGATIVE_MARGIN="${HARD_NEGATIVE_MARGIN:-0.35}"
HARD_NEGATIVE_ABSENT_NORMAL_WEIGHT="${HARD_NEGATIVE_ABSENT_NORMAL_WEIGHT:-1.50}"
HARD_NEGATIVE_COMPLETE_NORMAL_WEIGHT="${HARD_NEGATIVE_COMPLETE_NORMAL_WEIGHT:-1.70}"
SOFT_FN_WEIGHT="${SOFT_FN_WEIGHT:-0.08}"
SOFT_FN_MARGIN="${SOFT_FN_MARGIN:-0.35}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/audio_primary_v2.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/audio_primary_v2}"

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
  --model-type "audio_primary_v2" \
  --position-embedding-dropout "${POSITION_EMBED_DROPOUT}" \
  --train-position-dropout-prob "${POSITION_DROPOUT}" \
  --min-positions-after-dropout 1 \
  --balanced-sampler \
  --sample-normal-absent-weight "${SAMPLE_NORMAL_ABSENT_WEIGHT}" \
  --sample-complete-normal-weight "${SAMPLE_COMPLETE_NORMAL_WEIGHT}" \
  --hard-negative-weight "${HARD_NEGATIVE_WEIGHT}" \
  --hard-negative-margin "${HARD_NEGATIVE_MARGIN}" \
  --hard-negative-absent-normal-weight "${HARD_NEGATIVE_ABSENT_NORMAL_WEIGHT}" \
  --hard-negative-complete-normal-weight "${HARD_NEGATIVE_COMPLETE_NORMAL_WEIGHT}" \
  --soft-fn-weight "${SOFT_FN_WEIGHT}" \
  --soft-fn-margin "${SOFT_FN_MARGIN}" \
  --selection-mode "recall_specificity" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
