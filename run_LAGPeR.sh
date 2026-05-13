#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_DIR="configs/LAGPeR"
TRAIN_SCRIPT="tools/train_net.py"
DATA_ROOT="${FASTREID_DATASETS:-datasets}"

echo "Running LAGPeR training"

for STAGE in base_stage1.yml base_stage2.yml; do
    CONFIG_FILE="${CONFIG_DIR}/${STAGE}"
    OUTDIR="output/LAGPeR_${STAGE%.yml}"
    mkdir -p "$OUTDIR"

    echo "Start ${CONFIG_FILE}"
    "$PYTHON_BIN" "$TRAIN_SCRIPT" --config-file "$CONFIG_FILE" DATASETS.ROOT "$DATA_ROOT" \
        | tee "${OUTDIR}/log_$(date +%Y%m%d_%H%M%S).txt"
done

echo "LAGPeR training finished"
