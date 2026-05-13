#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="tools/train_net.py"
DATA_ROOT="${FASTREID_DATASETS:-datasets}"

for CONFIG_FILE in configs/CARGO/base_stage1.yml configs/CARGO/base_stage2_step5.yml; do
	echo "Start ${CONFIG_FILE}"
	"$PYTHON_BIN" "$TRAIN_SCRIPT" --config-file "$CONFIG_FILE" DATASETS.ROOT "$DATA_ROOT"
done
