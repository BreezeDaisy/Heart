#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-16}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEGMENT_DURATION="${SEGMENT_DURATION:-8.0}"
SEGMENT_HOP="${SEGMENT_HOP:-3.0}"
MAX_SEGMENTS="${MAX_SEGMENTS:-24}"
TARGET_RECALL="${TARGET_RECALL:-0.96}"
MAX_FN="${MAX_FN:-4}"
ABNORMAL_WEIGHT="${ABNORMAL_WEIGHT:-1.0}"
LR="${LR:-0.0004}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00015}"
POSITION_DROPOUT="${POSITION_DROPOUT:-0.08}"
POSITION_EMBED_DROPOUT="${POSITION_EMBED_DROPOUT:-0.05}"
HARD_NEGATIVE_WEIGHT="${HARD_NEGATIVE_WEIGHT:-0.12}"
HARD_NEGATIVE_MARGIN="${HARD_NEGATIVE_MARGIN:-0.36}"
HARD_NEGATIVE_ABSENT_NORMAL_WEIGHT="${HARD_NEGATIVE_ABSENT_NORMAL_WEIGHT:-1.15}"
ALPHA_SEGMENT_OUTCOME="${ALPHA_SEGMENT_OUTCOME:-0.15}"
SEGMENT_EVIDENCE_THRESHOLD="${SEGMENT_EVIDENCE_THRESHOLD:-0.55}"
MIN_EVIDENCE_SEGMENTS="${MIN_EVIDENCE_SEGMENTS:-2}"
MIN_EVIDENCE_POSITIONS="${MIN_EVIDENCE_POSITIONS:-2}"
TRIAGE_LOW_MIN="${TRIAGE_LOW_MIN:-0.06}"
TRIAGE_LOW_MAX="${TRIAGE_LOW_MAX:-0.30}"
TRIAGE_HIGH_MIN="${TRIAGE_HIGH_MIN:-0.30}"
TRIAGE_HIGH_MAX="${TRIAGE_HIGH_MAX:-0.85}"
TRIAGE_STEP="${TRIAGE_STEP:-0.02}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/audio_primary_v5_triage_8s_hop3.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/audio_primary_v5_triage_8s_hop3}"

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
  --segment-duration "${SEGMENT_DURATION}" \
  --segment-hop "${SEGMENT_HOP}" \
  --max-segments-per-patient "${MAX_SEGMENTS}" \
  --target-recall "${TARGET_RECALL}" \
  --max-fn "${MAX_FN}" \
  --outcome-abnormal-weight "${ABNORMAL_WEIGHT}" \
  --model-type "audio_primary_v5_triage_8s_hop3" \
  --position-embedding-dropout "${POSITION_EMBED_DROPOUT}" \
  --train-position-dropout-prob "${POSITION_DROPOUT}" \
  --min-positions-after-dropout 1 \
  --hard-negative-weight "${HARD_NEGATIVE_WEIGHT}" \
  --hard-negative-margin "${HARD_NEGATIVE_MARGIN}" \
  --hard-negative-scope "absent_normal" \
  --hard-negative-absent-normal-weight "${HARD_NEGATIVE_ABSENT_NORMAL_WEIGHT}" \
  --hard-negative-complete-normal-weight 1.0 \
  --soft-fn-weight 0.0 \
  --alpha-segment-outcome "${ALPHA_SEGMENT_OUTCOME}" \
  --triage-search \
  --triage-low-min "${TRIAGE_LOW_MIN}" \
  --triage-low-max "${TRIAGE_LOW_MAX}" \
  --triage-high-min "${TRIAGE_HIGH_MIN}" \
  --triage-high-max "${TRIAGE_HIGH_MAX}" \
  --triage-threshold-step "${TRIAGE_STEP}" \
  --segment-evidence-threshold "${SEGMENT_EVIDENCE_THRESHOLD}" \
  --min-evidence-segments "${MIN_EVIDENCE_SEGMENTS}" \
  --min-evidence-positions "${MIN_EVIDENCE_POSITIONS}" \
  --selection-mode "triage_fn_fp" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
