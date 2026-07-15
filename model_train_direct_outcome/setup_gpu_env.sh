#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "[1/5] Python:"
"${PYTHON_BIN}" --version

echo "[2/5] Creating virtual environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[3/5] Activating virtual environment"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[4/5] Installing PyTorch GPU wheels"
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

echo "[5/5] Installing project dependencies"
python -m pip install -r requirements-linux.txt

echo
echo "Environment check:"
python - <<'PY'
import torch
import librosa
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
print("librosa:", librosa.__version__)
PY

echo
echo "Setup finished. Activate with:"
echo "source ${VENV_DIR}/bin/activate"
