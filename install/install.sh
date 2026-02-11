#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cpsam_gui}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
GPU_MODE="${1:---auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "GUVpose installer (macOS/Linux)"
echo "Repo: $REPO_ROOT"
echo "Env: $ENV_NAME"
echo ""

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "Environment already exists. Skipping create."
else
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

USE_GPU="false"
if [[ "$GPU_MODE" == "--gpu" ]]; then
  USE_GPU="true"
elif [[ "$GPU_MODE" == "--cpu" ]]; then
  USE_GPU="false"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    USE_GPU="true"
  fi
fi

if [[ "$USE_GPU" == "true" ]]; then
  echo "NVIDIA GPU selected. Installing PyTorch CUDA build..."
  TORCH_INDEX_GPU="${TORCH_INDEX_GPU:-https://download.pytorch.org/whl/cu121}"
  conda run -n "$ENV_NAME" pip install torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX_GPU"
else
  echo "Installing PyTorch CPU-only build..."
  conda run -n "$ENV_NAME" pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo "Pinning numpy to match reference environment..."
conda run -n "$ENV_NAME" pip install numpy==2.3.4

echo "Installing GUI dependencies..."
conda run -n "$ENV_NAME" pip install numpy scipy scikit-image pyqt5 pyqtgraph superqt natsort tifffile fastremap tqdm matplotlib

echo "Installing gRPC and optional image readers..."
conda run -n "$ENV_NAME" pip install grpcio grpcio-tools protobuf nd2 readlif

echo "Installing GUVpose project..."
conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT"

echo "Re-pinning numpy after project install..."
conda run -n "$ENV_NAME" pip install numpy==2.3.4

echo ""
echo "Done."
echo "Launch: conda run -n $ENV_NAME python -m cellpose"
