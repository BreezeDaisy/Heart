#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SEGMENTS="${MAX_SEGMENTS:-32}"
TARGET_RECALL="${TARGET_RECALL:-0.88}"
COMPLETE_TARGET_RECALL_DELTA="${COMPLETE_TARGET_RECALL_DELTA:-0.03}"
ABSENT_ABNORMAL_TARGET_RECALL_DELTA="${ABSENT_ABNORMAL_TARGET_RECALL_DELTA:-0.03}"
ABNORMAL_WEIGHT="${ABNORMAL_WEIGHT:-1.5}"
ABSENT_ABNORMAL_WEIGHT="${ABSENT_ABNORMAL_WEIGHT:-1.25}"
FN_PENALTY="${FN_PENALTY:-0.06}"
FP_PENALTY="${FP_PENALTY:-0.12}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0002}"
ALPHA_MURMUR="${ALPHA_MURMUR:-0.00}"
ALPHA_TIMING="${ALPHA_TIMING:-0.03}"
ALPHA_GRADE="${ALPHA_GRADE:-0.03}"
ALPHA_SHAPE="${ALPHA_SHAPE:-0.02}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/direct_outcome_gpu_residual_transformer_v1.pth}"

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

python src/train_direct_outcome.py \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-segments-per-patient "${MAX_SEGMENTS}" \
  --target-recall "${TARGET_RECALL}" \
  --complete-target-recall-delta "${COMPLETE_TARGET_RECALL_DELTA}" \
  --absent-abnormal-target-recall-delta "${ABSENT_ABNORMAL_TARGET_RECALL_DELTA}" \
  --outcome-abnormal-weight "${ABNORMAL_WEIGHT}" \
  --absent-abnormal-weight "${ABSENT_ABNORMAL_WEIGHT}" \
  --fn-penalty-weight "${FN_PENALTY}" \
  --fp-penalty-weight "${FP_PENALTY}" \
  --alpha-murmur "${ALPHA_MURMUR}" \
  --alpha-timing "${ALPHA_TIMING}" \
  --alpha-grade "${ALPHA_GRADE}" \
  --alpha-shape "${ALPHA_SHAPE}" \
  --checkpoint-path "${CHECKPOINT_PATH}"
