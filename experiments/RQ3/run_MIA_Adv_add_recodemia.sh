#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MIA_ADV_DIR="${SCRIPT_DIR}/MIA_Adv"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${MIA_ADV_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${ROOT_DIR}/datasets/stack2/AdvPrompt-MIA"
LAT_DATA_ROOT="${ROOT_DIR}/datasets/stack2"
SHADOW_MODEL_ID="${SHADOW_MODEL_ID:-sapienzanlp/Minerva-7B-base-v1.0}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-bigcode/starcoder2-15b}"
SHADOW_MODEL_TAG="${SHADOW_MODEL_ID//\//--}"
TARGET_MODEL_TAG="${TARGET_MODEL_ID//\//--}"
MAPPER_TYPES="linear"

RESULTS_ROOT="${ROOT_DIR}/results/RQ3/MIA_Adv/${SHADOW_MODEL_TAG}_to_${TARGET_MODEL_TAG}"
TARGET_DATASET_DIR="${RESULTS_ROOT}/target_dataset"
SHADOW_DATASET_DIR="${RESULTS_ROOT}/shadow_dataset"
ARTIFACT_DIR="${RESULTS_ROOT}/artifacts"
LOG_FILE="${RESULTS_ROOT}/run.log"
mkdir -p "${TARGET_DATASET_DIR}" "${SHADOW_DATASET_DIR}" "${ARTIFACT_DIR}"

TRAIN_INPUT="train.jsonl"
TEST_INPUT="test.jsonl"
TRAIN_VICTIM_FILE="train_victim.json"
TEST_VICTIM_FILE="test_victim.json"
GROUP_SIZE="${GROUP_SIZE:-12}"
TARGET_TRUE_FILE="train_models--${TARGET_MODEL_TAG}_victim_infer.txt"
TARGET_FALSE_FILE="test_models--${TARGET_MODEL_TAG}_victim_infer.txt"
SHADOW_TRUE_FILE="train_models--${SHADOW_MODEL_TAG}_victim_infer.txt"
SHADOW_FALSE_FILE="test_models--${SHADOW_MODEL_TAG}_victim_infer.txt"

TARGET_FEATURE="${ARTIFACT_DIR}/target_feature_baseline.npz"
TARGET_SIGNALS="${ARTIFACT_DIR}/target_original_signals.jsonl"
SHADOW_SIGNALS="${ARTIFACT_DIR}/shadow_original_signals.jsonl"
SHADOW_LAT="${ARTIFACT_DIR}/shadow_lat_scores.jsonl"
SHADOW_LAT_LOG="${ARTIFACT_DIR}/shadow_lat_scores.log"
JOINT_FEATURE="${ARTIFACT_DIR}/target_feature_joint_linear.npz"
JOINT_MAPPER="${ARTIFACT_DIR}/joint_linear_mapper.joblib"
JOINT_PSEUDO="${ARTIFACT_DIR}/target_pseudo_joint_linear.jsonl"
JOINT_DIAGNOSTICS="${ARTIFACT_DIR}/joint_linear_diagnostics.json"
BASELINE_METRICS_TMP="${ARTIFACT_DIR}/.metrics_baseline.tmp"
JOINT_METRICS_TMP="${ARTIFACT_DIR}/.metrics_joint_linear.tmp"
COMPARISON_CSV="${RESULTS_ROOT}/metrics_baseline_vs_joint_linear.csv"

LAT_VAL_JSON="${LAT_VAL_JSON:-${ROOT_DIR}/results/${SHADOW_MODEL_ID}/stack2/LAT_val.json}"
CONCEPT="${CONCEPT:-familiarity}"
STIMULUS="${STIMULUS:-starcoder2-mia-test}"
LAT_BATCH_SIZE="${LAT_BATCH_SIZE:-8}"
LAT_MAX_LENGTH="${LAT_MAX_LENGTH:-512}"

GPU_DEVICES="${GPU_DEVICES:-0}"
INFER_GPU="${INFER_GPU:-${GPU_DEVICES}}"
FEATURE_GPU="${FEATURE_GPU:-${GPU_DEVICES}}"
LAT_GPU="${LAT_GPU:-${GPU_DEVICES}}"
CLS_GPU="${CLS_GPU:-${GPU_DEVICES}}"

GLOBAL_RANDOM_SEED="${GLOBAL_RANDOM_SEED:-140120031}"
RANDOM_STATE="${RANDOM_STATE:-676269283}"
RANDOM_STATE_TEST="${RANDOM_STATE_TEST:-212129145}"
N_SAMPLES_PER_CLASS="${N_SAMPLES_PER_CLASS:-2000}"
MAPPER_FOLDS="${MAPPER_FOLDS:-5}"
DROPOUT="${DROPOUT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
NUM_EPOCHS="${NUM_EPOCHS:-25}"
HIDDEN_DIMS="${HIDDEN_DIMS:-512 512 512}"

SKIP_EXISTING_INFER="${SKIP_EXISTING_INFER:-1}"
SKIP_EXISTING_FEATURES="${SKIP_EXISTING_FEATURES:-1}"
SKIP_EXISTING_SIGNALS="${SKIP_EXISTING_SIGNALS:-1}"
SKIP_EXISTING_LAT="${SKIP_EXISTING_LAT:-1}"

timestamp() { date -Is; }

run_logged() {
  {
    echo "[$(timestamp)] CMD: $*"
    echo "[$(timestamp)] PWD: ${ROOT_DIR}"
  } | tee -a "${LOG_FILE}" >/dev/null
  set +e
  "$@" 2>&1 | tee -a "${LOG_FILE}"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "[$(timestamp)] DONE (exit=${status})" | tee -a "${LOG_FILE}" >/dev/null
  return "${status}"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

prepare_victim_files() {
  require_file "${DATA_ROOT}/${TRAIN_INPUT}"
  require_file "${DATA_ROOT}/${TEST_INPUT}"
  if [[ ! -f "${DATA_ROOT}/${TRAIN_VICTIM_FILE}" ]]; then
    run_logged python "${MIA_ADV_DIR}/perturb.py" \
      --input_dir "${DATA_ROOT}" --input_file "${TRAIN_INPUT}" --output_file "${TRAIN_VICTIM_FILE}"
  fi
  if [[ ! -f "${DATA_ROOT}/${TEST_VICTIM_FILE}" ]]; then
    run_logged python "${MIA_ADV_DIR}/perturb.py" \
      --input_dir "${DATA_ROOT}" --input_file "${TEST_INPUT}" --output_file "${TEST_VICTIM_FILE}"
  fi
}

prepare_model_dataset() {
  local output_dir="$1"
  cp "${DATA_ROOT}/${TRAIN_INPUT}" "${output_dir}/${TRAIN_INPUT}"
  cp "${DATA_ROOT}/${TEST_INPUT}" "${output_dir}/${TEST_INPUT}"
  cp "${DATA_ROOT}/${TRAIN_VICTIM_FILE}" "${output_dir}/${TRAIN_VICTIM_FILE}"
  cp "${DATA_ROOT}/${TEST_VICTIM_FILE}" "${output_dir}/${TEST_VICTIM_FILE}"
}

run_inference() {
  local model_id="$1"
  local dataset_dir="$2"
  local true_file="$3"
  local false_file="$4"
  if [[ "${SKIP_EXISTING_INFER}" == "1" && -f "${dataset_dir}/${true_file}" && -f "${dataset_dir}/${false_file}" ]]; then
    echo "Inference exists for ${model_id}, skip"
    return
  fi
  run_logged env CUDA_VISIBLE_DEVICES="${INFER_GPU}" python "${MIA_ADV_DIR}/model_inference.py" \
    --data-dir "${dataset_dir}" \
    --model "${model_id}" \
    --file-types train test \
    --trust-remote-code
  require_file "${dataset_dir}/${true_file}"
  require_file "${dataset_dir}/${false_file}"
}

build_feature() {
  local dataset_dir="$1"
  local true_file="$2"
  local false_file="$3"
  local output="$4"
  if [[ "${SKIP_EXISTING_FEATURES}" == "1" && -f "${output}" ]]; then
    echo "Feature exists, skip: ${output}"
    return
  fi
  run_logged env CUDA_VISIBLE_DEVICES="${FEATURE_GPU}" python -u -c '
import sys
from feature_builder import save_features_and_labels
dataset_dir, true_file, false_file, output, group_size = sys.argv[1:]
save_features_and_labels(
    f"{dataset_dir}/{true_file}",
    f"{dataset_dir}/{false_file}",
    f"{dataset_dir}/train_victim.json",
    f"{dataset_dir}/test_victim.json",
    feature_path=output,
    group_size=int(group_size),
)
' "${dataset_dir}" "${true_file}" "${false_file}" "${output}" "${GROUP_SIZE}"
}

build_signals() {
  local dataset_dir="$1"
  local true_file="$2"
  local false_file="$3"
  local output="$4"
  if [[ "${SKIP_EXISTING_SIGNALS}" == "1" && -f "${output}" ]]; then
    echo "Signals exist, skip: ${output}"
    return
  fi
  run_logged env CUDA_VISIBLE_DEVICES="${FEATURE_GPU}" python "${MIA_ADV_DIR}/joint_linear.py" build-signals \
    --train-victim "${dataset_dir}/${TRAIN_VICTIM_FILE}" \
    --test-victim "${dataset_dir}/${TEST_VICTIM_FILE}" \
    --train-infer "${dataset_dir}/${true_file}" \
    --test-infer "${dataset_dir}/${false_file}" \
    --group-size "${GROUP_SIZE}" \
    --output "${output}"
}

run_classifier() {
  local feature="$1"
  local metrics="$2"
  local failed="$3"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${CLS_GPU}"
    python "${MIA_ADV_DIR}/feature_builder.py"
    --input_dir "${TARGET_DATASET_DIR}"
    --true_file "${TARGET_TRUE_FILE}"
    --false_file "${TARGET_FALSE_FILE}"
    --true_gt_file "${TRAIN_VICTIM_FILE}"
    --false_gt_file "${TEST_VICTIM_FILE}"
    --feature_path "${feature}"
    --failed_path "${failed}"
    --metrics_path "${metrics}"
    --n_samples_per_class "${N_SAMPLES_PER_CLASS}"
    --global_random_seed "${GLOBAL_RANDOM_SEED}"
    --random_state "${RANDOM_STATE}"
    --random_state_test "${RANDOM_STATE_TEST}"
    --dropout "${DROPOUT}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --num_epochs "${NUM_EPOCHS}"
    --hidden_dims
  )
  local hidden
  for hidden in ${HIDDEN_DIMS}; do cmd+=("${hidden}"); done
  run_logged "${cmd[@]}"
}

echo "== MIA_Adv + ReCodeMIA =="
echo "DATA_ROOT=${DATA_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SHADOW_MODEL_ID=${SHADOW_MODEL_ID}"
echo "TARGET_MODEL_ID=${TARGET_MODEL_ID}"
echo "MAPPER_TYPES=${MAPPER_TYPES}"

echo "[1/10] Prepare AdvPrompt-MIA victim files"
prepare_victim_files
prepare_model_dataset "${TARGET_DATASET_DIR}"
prepare_model_dataset "${SHADOW_DATASET_DIR}"

echo "[2/10] Run target inference"
run_inference "${TARGET_MODEL_ID}" "${TARGET_DATASET_DIR}" "${TARGET_TRUE_FILE}" "${TARGET_FALSE_FILE}"

echo "[3/10] Run shadow inference"
run_inference "${SHADOW_MODEL_ID}" "${SHADOW_DATASET_DIR}" "${SHADOW_TRUE_FILE}" "${SHADOW_FALSE_FILE}"

echo "[4/10] Build target baseline 27D features"
build_feature "${TARGET_DATASET_DIR}" "${TARGET_TRUE_FILE}" "${TARGET_FALSE_FILE}" "${TARGET_FEATURE}"

echo "[5/10] Build target original-output signals"
build_signals "${TARGET_DATASET_DIR}" "${TARGET_TRUE_FILE}" "${TARGET_FALSE_FILE}" "${TARGET_SIGNALS}"

echo "[6/10] Build shadow original-output signals"
build_signals "${SHADOW_DATASET_DIR}" "${SHADOW_TRUE_FILE}" "${SHADOW_FALSE_FILE}" "${SHADOW_SIGNALS}"

echo "[7/10] Generate shadow LAT mem/syn scores"
if [[ "${SKIP_EXISTING_LAT}" != "1" || ! -f "${SHADOW_LAT}" ]]; then
  require_file "${LAT_DATA_ROOT}/fit.jsonl"
  require_file "${LAT_DATA_ROOT}/validate.jsonl"
  require_file "${LAT_VAL_JSON}"
  run_logged env CUDA_VISIBLE_DEVICES="${LAT_GPU}" python "${MIA_ADV_DIR}/lat_scores.py" \
    --fit "${LAT_DATA_ROOT}/fit.jsonl" \
    --validate "${LAT_DATA_ROOT}/validate.jsonl" \
    --model "${SHADOW_MODEL_ID}" \
    --concept "${CONCEPT}" \
    --stimulus "${STIMULUS}" \
    --lat-val "${LAT_VAL_JSON}" \
    --train-victim "${SHADOW_DATASET_DIR}/${TRAIN_VICTIM_FILE}" \
    --test-victim "${SHADOW_DATASET_DIR}/${TEST_VICTIM_FILE}" \
    --group-size "${GROUP_SIZE}" \
    --batch-size "${LAT_BATCH_SIZE}" \
    --max-length "${LAT_MAX_LENGTH}" \
    --output "${SHADOW_LAT}" \
    --log-file "${SHADOW_LAT_LOG}" \
    --only-original
fi

echo "[8/10] Fit 2D signal-to-LAT joint linear mapper and append pseudo LAT"
run_logged python "${MIA_ADV_DIR}/joint_linear.py" fit-append \
  --shadow-signals "${SHADOW_SIGNALS}" \
  --target-signals "${TARGET_SIGNALS}" \
  --target-feature "${TARGET_FEATURE}" \
  --shadow-lat "${SHADOW_LAT}" \
  --mapper-out "${JOINT_MAPPER}" \
  --feature-out "${JOINT_FEATURE}" \
  --pseudo-out "${JOINT_PSEUDO}" \
  --diagnostics-out "${JOINT_DIAGNOSTICS}" \
  --mapper-oof \
  --mapper-folds "${MAPPER_FOLDS}" \
  --random-state "${RANDOM_STATE}" \
  --n-samples-per-class "${N_SAMPLES_PER_CLASS}"

echo "[9/10] Evaluate baseline and joint linear"
run_classifier "${TARGET_FEATURE}" "${BASELINE_METRICS_TMP}" "${ARTIFACT_DIR}/failed_baseline.jsonl"
run_classifier "${JOINT_FEATURE}" "${JOINT_METRICS_TMP}" "${ARTIFACT_DIR}/failed_joint_linear.jsonl"

echo "[10/10] Write comparison CSV"
run_logged python "${MIA_ADV_DIR}/joint_linear.py" summarize \
  --baseline-metrics "${BASELINE_METRICS_TMP}" \
  --joint-metrics "${JOINT_METRICS_TMP}" \
  --model "${TARGET_MODEL_ID}" \
  --output "${COMPARISON_CSV}" \
  --remove-inputs

echo "Comparison CSV: ${COMPARISON_CSV}"
echo "Artifacts: ${ARTIFACT_DIR}"
