#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODELS="${MODELS:-bigcode/starcoder2-15b,sapienzanlp/Minerva-7B-base-v1.0,JetBrains/Mellum-4b-base}"

DATASET_FILE="datasets/stack2/RQ2"
SINGLE_RUN_DIR="${DATASET_FILE}/single_run_merged"
PARENT_PREFIX="stack2/RQ2"
IFS=',' read -r -a MODEL_LIST <<< "${MODELS}"

timestamp() { date -Is; }

run_logged() {
    local logfile="$1"
    shift
    mkdir -p "$(dirname "${logfile}")"
    {
        echo "[$(timestamp)] CMD: $*"
        echo "[$(timestamp)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    } | tee -a "${logfile}" >/dev/null
    "$@" 2>&1 | tee -a "${logfile}"
}

model_label() {
    case "$1" in
        "bigcode/starcoder2-15b") echo "StarCoder-15B" ;;
        "sapienzanlp/Minerva-7B-base-v1.0") echo "Minerva-7B" ;;
        "JetBrains/Mellum-4b-base") echo "Mellum-4B" ;;
        *) basename "$1" ;;
    esac
}

# for split in fit validate test; do
#     [[ -f "${SINGLE_RUN_DIR}/${split}.jsonl" ]] || {
#         echo "Missing ${SINGLE_RUN_DIR}/${split}.jsonl; run experiments/dataset_construction.sh first." >&2
#         exit 1
#     }
# done

MODEL_LABELS=()
for model in "${MODEL_LIST[@]}"; do
    model="${model#${model%%[![:space:]]*}}"
    model="${model%${model##*[![:space:]]}}"
    MODEL_LABELS+=("$(model_label "${model}")")
    OUT_DIR="results/${model}/${PARENT_PREFIX}/single_run"
    LOG_FILE="${OUT_DIR}/run_RQ2_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${OUT_DIR}"

    if [[ ! -f "${OUT_DIR}/LAT.json" ]]; then
        run_logged "${LOG_FILE}" python experiments/RQ2/eval_RQ2.py \
            --dataset-root "${DATASET_FILE}" \
            --single-run-dir "${SINGLE_RUN_DIR}" \
            --parent-prefix "${PARENT_PREFIX}" \
            --model "${model}"
    fi

    run_logged "${LOG_FILE}" python experiments/RQ2/analyze_rq2_geometry.py \
        --lat-json "${OUT_DIR}/LAT.json"
    run_logged "${LOG_FILE}" python experiments/RQ2/aggregate_rq2_results.py \
        --results-root "${OUT_DIR}"
done

if [[ "${#MODEL_LIST[@]}" -ne 3 ]]; then
    echo "plot_RQ2.py requires exactly three models; received ${#MODEL_LIST[@]}." >&2
    exit 1
fi

MODEL_LABELS_CSV="$(IFS=,; echo "${MODEL_LABELS[*]}")"
PLOT_DIR="results/${PARENT_PREFIX}"
PLOT_LOG="${PLOT_DIR}/run_RQ2_plot_$(date +%Y%m%d_%H%M%S).log"
run_logged "${PLOT_LOG}" python experiments/RQ2/plot_RQ2.py \
    --results-root results \
    --parent-prefix "${PARENT_PREFIX}" \
    --models "${MODELS}" \
    --model-labels "${MODEL_LABELS_CSV}" \
    --output "${PLOT_DIR}/rq2_score_violin_grid_2x3.pdf"
