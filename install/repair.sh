#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cpsam_gui}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "GUVpose repair (macOS/Linux)"
echo "Repo: $REPO_ROOT"
echo "Env: $ENV_NAME"
echo ""

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "Environment $ENV_NAME not found. Run install.sh first."
  exit 1
fi

echo "Re-installing dependencies..."
conda run -n "$ENV_NAME" pip install numpy==2.3.4
conda run -n "$ENV_NAME" pip install numpy scipy scikit-image pyqt5 pyqtgraph superqt natsort tifffile fastremap tqdm matplotlib
conda run -n "$ENV_NAME" pip install grpcio grpcio-tools protobuf nd2 readlif

echo "Re-installing GUVpose project..."
conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT"

echo "Done."
