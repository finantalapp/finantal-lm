#!/usr/bin/env bash
# SFT launcher. Loads pretrain latest.pt by default. Extra args -> --override key=value.
set -euo pipefail
cd "$(dirname "$0")/.."
export FINANTAL_DATA_ROOT="${FINANTAL_DATA_ROOT:-/content/drive/MyDrive/finantal_data}"
echo "FINANTAL_DATA_ROOT=$FINANTAL_DATA_ROOT"
python -m training.sft_train --config config/train_config.json --override "$@"
