#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PARENT_FILE="${PARENT_FILE:-stack2}"
INCLUDE_LABEL2_AS_POSITIVE="${INCLUDE_LABEL2_AS_POSITIVE:-0}"
MODELS="${MODEL_LIST_CSV:-bigcode/starcoder2-15b,bigcode/starcoder2-7b,sapienzanlp/Minerva-7B-base-v1.0,JetBrains/Mellum-4b-base}"
OUTPUT="${ROOT_DIR}/results/RQ4/rq4_rq2_metrics.csv"

CMD=(
    python experiments/RQ4/aggregate_rq4_auroc.py
    --root-dir "${ROOT_DIR}"
    --parent-file "${PARENT_FILE}"
    --models "${MODELS}"
    --output "${OUTPUT}"
)
if [[ "${INCLUDE_LABEL2_AS_POSITIVE}" == "1" ]]; then
    CMD+=(--include-label2-as-positive)
fi

"${CMD[@]}"
