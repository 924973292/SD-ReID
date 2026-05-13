#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${FASTREID_DATASETS:-datasets}"

"$PYTHON_BIN" tools/train_net.py --config-file configs/v1/base_stage1.yml DATASETS.ROOT "$DATA_ROOT"
"$PYTHON_BIN" tools/train_net.py --config-file configs/v1/base_stage2.yml DATASETS.ROOT "$DATA_ROOT"
