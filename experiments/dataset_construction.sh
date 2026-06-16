#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-datasets/stack2}"
APPS_FILE="${APPS_FILE:-datasets/APPS/APPS.jsonl}"
STACK2_FILE="${STACK2_FILE:-datasets/stack2/the-stack-v2-python-90k.jsonl}"
SAMPLE_COUNT="${SAMPLE_COUNT:-800}"
DATASET_SEED="${DATASET_SEED:-42}"
NATURALCC_SHADOW_MODELS="${NATURALCC_SHADOW_MODELS:-1}"
NATURALCC_DEV_RATIO="${NATURALCC_DEV_RATIO:-0.1}"
ADV_DIR="${DATASET_ROOT}/AdvPrompt-MIA"
NATURALCC_DIR="${DATASET_ROOT}/naturalcc"
MIA_ADV_DIR="${ROOT_DIR}/experiments/RQ3/MIA_Adv"

run_step() {
    local label="$1"
    shift
    echo
    echo "== ${label} =="
    "$@"
}

run_step "[1/7] Build base LAT dataset" \
    "${PYTHON_BIN}" datasets/make_lat_dataset.py \
    --apps-file "${APPS_FILE}" \
    --stack2-file "${STACK2_FILE}" \
    --output-dir "${DATASET_ROOT}" \
    --sample-count "${SAMPLE_COUNT}" \
    --seed "${DATASET_SEED}"

run_step "[2/7] Build RQ1 cumulative variants" \
    "${PYTHON_BIN}" datasets/stack2/RQ1/build_rq1_cumulative_variants.py \
    --input-root "${DATASET_ROOT}" \
    --output-root "${DATASET_ROOT}/RQ1"

run_step "[3/7] Build RQ2 cumulative variants" \
    "${PYTHON_BIN}" datasets/stack2/RQ2/build_rq2_cumulative_variants.py \
    --input-root "${DATASET_ROOT}" \
    --output-root "${DATASET_ROOT}/RQ2"

run_step "[4/7] Build RQ2 merged single-run dataset" \
    "${PYTHON_BIN}" datasets/stack2/RQ2/build_rq2_single_run_dataset.py \
    --input-root "${DATASET_ROOT}" \
    --output-dir "${DATASET_ROOT}/RQ2/single_run_merged"

run_step "[5/7] Build AdvPrompt-MIA dataset" \
    "${PYTHON_BIN}" datasets/stack2/AdvPrompt-MIA/make_datasets.py \
    --input-jsonl "${DATASET_ROOT}/original_stack2_apps.jsonl" \
    --output-dir "${ADV_DIR}"

run_step "[6a/7] Build AdvPrompt-MIA victim files (train)" \
    "${PYTHON_BIN}" "${MIA_ADV_DIR}/perturb.py" \
    --input_dir "${ADV_DIR}" \
    --input_file train.jsonl \
    --output_file train_victim.json
run_step "[6b/7] Build AdvPrompt-MIA victim files (test)" \
    "${PYTHON_BIN}" "${MIA_ADV_DIR}/perturb.py" \
    --input_dir "${ADV_DIR}" \
    --input_file test.jsonl \
    --output_file test_victim.json

run_step "[7/7] Build NaturalCC dataset" \
    "${PYTHON_BIN}" datasets/stack2/naturalcc/make_datasets.py \
    --input-jsonl "${DATASET_ROOT}/original_stack2_apps.jsonl" \
    --output-dir "${NATURALCC_DIR}" \
    --num-shadow-models "${NATURALCC_SHADOW_MODELS}" \
    --dev-ratio "${NATURALCC_DEV_RATIO}" \
    --seed "${DATASET_SEED}"

echo
echo "Dataset construction completed: ${DATASET_ROOT}"
