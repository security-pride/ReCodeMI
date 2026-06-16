#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

GPU="${1:-0}"
MODEL="${2:-bigcode/starcoder2-7b}"
DATASET_FILE="${ROOT_DIR}/datasets/stack2"
OUT_DIR="${ROOT_DIR}/results/${MODEL}/stack2/baseline"
ALL_METHODS="loss,ppl,zlib,min_k,confidence,entropy,msp,margin"

export CUDA_VISIBLE_DEVICES="${GPU}"

timestamp() { date -Is; }

run_logged() {
    local logfile="$1"; shift
    mkdir -p "$(dirname "${logfile}")"
    {
        echo "[$(timestamp)] CMD: $*"
        echo "[$(timestamp)] PWD: ${ROOT_DIR}"
        echo "[$(timestamp)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    } | tee -a "${logfile}" >/dev/null

    set +e
    "$@" 2>&1 | tee -a "${logfile}"
    local status="${PIPESTATUS[0]}"
    set -e
    echo "[$(timestamp)] DONE (exit=${status})" | tee -a "${logfile}" >/dev/null
    return "${status}"
}

for split in fit validate test; do
    [[ -f "${DATASET_FILE}/${split}.jsonl" ]] || {
        echo "Missing ${DATASET_FILE}/${split}.jsonl" >&2
        exit 1
    }
done

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run_baseline_$(date +%Y%m%d_%H%M%S).log"

run_logged "${LOG_FILE}" python "${ROOT_DIR}/experiments/baseline/loss.py" \
    --fit "${DATASET_FILE}/fit.jsonl" \
    --validate "${DATASET_FILE}/validate.jsonl" \
    --test "${DATASET_FILE}/test.jsonl" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --text-source code_generation \
    --min-k-ratio 0.2 \
    --cv-folds 5 \
    --random-state 42 \
    --methods "${ALL_METHODS}"

run_logged "${LOG_FILE}" python "${SCRIPT_DIR}/summarize_baseline_metrics.py" \
    --out-dir "${OUT_DIR}" \
    --methods "${ALL_METHODS}" \
    --model "${MODEL}"
