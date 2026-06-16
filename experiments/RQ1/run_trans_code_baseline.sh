#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-bigcode/starcoder2-7b}"

DATASET_FILE="datasets/stack2/RQ1"
BASE_DATASET_FILE="datasets/stack2"
RQ2_DATASET_FILE="datasets/stack2/RQ2/single_run_merged"
RESULT_PREFIX="stack2/RQ1"
RQ2_RESULT_PREFIX="stack2/RQ2"
OUT_DIR="results/${MODEL}/${RESULT_PREFIX}/baseline_trans_code"
RQ2_LAT_JSON="results/${MODEL}/${RQ2_RESULT_PREFIX}/single_run/LAT.json"
LOG_FILE="${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

timestamp() { date -Is; }

run_logged() {
    local logfile="$1"
    shift
    mkdir -p "$(dirname "${logfile}")"
    {
        echo "[$(timestamp)] CMD: $*"
        echo "[$(timestamp)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "[$(timestamp)] MODEL=${MODEL}"
    } | tee -a "${logfile}" >/dev/null
    "$@" 2>&1 | tee -a "${logfile}"
}

for split in fit validate test; do
    [[ -f "${BASE_DATASET_FILE}/${split}.jsonl" ]] || {
        echo "Missing ${BASE_DATASET_FILE}/${split}.jsonl" >&2
        exit 1
    }
done
for level in v1 v2 v3 v4; do
    [[ -f "${DATASET_FILE}/${level}/test.jsonl" ]] || {
        echo "Missing ${DATASET_FILE}/${level}/test.jsonl; run experiments/dataset_construction.sh first." >&2
        exit 1
    }
done

mkdir -p "${OUT_DIR}"
LEVEL_DIRS="${DATASET_FILE}/v1,${DATASET_FILE}/v2,${DATASET_FILE}/v3,${DATASET_FILE}/v4"

run_logged "${LOG_FILE}" python experiments/RQ1/eval_trans_code_baseline.py \
    --fit "${BASE_DATASET_FILE}/fit.jsonl" \
    --validate "${BASE_DATASET_FILE}/validate.jsonl" \
    --test "${BASE_DATASET_FILE}/test.jsonl" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --methods all \
    --level-dirs "${LEVEL_DIRS}" \
    --min-k-ratio 0.2 \
    --cv-folds 5 \
    --random-state 42 \
    --disable-score-orientation

if [[ ! -f "${RQ2_LAT_JSON}" ]]; then
    for split in fit validate test; do
        [[ -f "${RQ2_DATASET_FILE}/${split}.jsonl" ]] || {
            echo "Missing ${RQ2_DATASET_FILE}/${split}.jsonl; run experiments/dataset_construction.sh first." >&2
            exit 1
        }
    done
    run_logged "${LOG_FILE}" python experiments/RQ2/eval_RQ2.py \
        --dataset-root datasets/stack2/RQ2 \
        --single-run-dir "${RQ2_DATASET_FILE}" \
        --parent-prefix "${RQ2_RESULT_PREFIX}" \
        --model "${MODEL}"
fi

run_logged "${LOG_FILE}" python experiments/RQ1/plot_trans_code_baseline.py \
    --from-result "${OUT_DIR}" \
    --rq2-lat-json "${RQ2_LAT_JSON}" \
    --methods all \
    --output "${OUT_DIR}/rq1_trans_code_baseline_2x5.pdf"
