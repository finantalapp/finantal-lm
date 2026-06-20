#!/usr/bin/env bash
# Pretraining launcher. Set FINANTAL_DATA_ROOT to your data root (Drive or local).
# Any extra args are forwarded as --override key=value.
set -euo pipefail
cd "$(dirname "$0")/.."
export FINANTAL_DATA_ROOT="${FINANTAL_DATA_ROOT:-/content/drive/MyDrive/finantal_data}"
echo "FINANTAL_DATA_ROOT=$FINANTAL_DATA_ROOT"
python -m training.pretrain --config config/train_config.json --override "$@"
