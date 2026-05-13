#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_DIR="configs/G2APS_ReID"
TRAIN_SCRIPT="tools/train_net.py"
DATA_ROOT="${FASTREID_DATASETS:-datasets}"

echo "Running G2APS-ReID 128x64 training"

for STAGE in base_stage1_128.yml base_stage2_128.yml; do
    CONFIG_FILE="${CONFIG_DIR}/${STAGE}"
    OUTDIR="output/G2APS_ReID_${STAGE%.yml}"
    mkdir -p "$OUTDIR"

    echo "Start ${CONFIG_FILE}"
    "$PYTHON_BIN" "$TRAIN_SCRIPT" --config-file "$CONFIG_FILE" DATASETS.ROOT "$DATA_ROOT" \
        | tee "${OUTDIR}/log_$(date +%Y%m%d_%H%M%S).txt"
done

echo "G2APS-ReID 128x64 training finished"
