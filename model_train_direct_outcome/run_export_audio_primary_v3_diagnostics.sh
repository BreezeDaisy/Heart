#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEGMENT_DURATION="${SEGMENT_DURATION:-3.0}"
SEGMENT_HOP="${SEGMENT_HOP:-2.0}"
MAX_SEGMENTS="${MAX_SEGMENTS:-32}"
TARGET_RECALL="${TARGET_RECALL:-0.88}"
SEGMENT_EVIDENCE_THRESHOLD="${SEGMENT_EVIDENCE_THRESHOLD:-0.50}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/audio_primary_v3.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/audio_primary_v3_supplement}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}"
  echo "Run ./setup_gpu_env.sh first."
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  echo "Put audio_primary_v3.pth under checkpoints/ first."
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

mkdir -p "${OUTPUT_DIR}"

python src/export_audio_primary_diagnostics.py \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --segment-duration "${SEGMENT_DURATION}" \
  --segment-hop "${SEGMENT_HOP}" \
  --max-segments-per-patient "${MAX_SEGMENTS}" \
  --target-recall "${TARGET_RECALL}" \
  --segment-evidence-threshold "${SEGMENT_EVIDENCE_THRESHOLD}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}"
