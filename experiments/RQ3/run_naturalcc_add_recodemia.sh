#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXTERNAL_DIR="${SCRIPT_DIR}/naturalcc"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${ROOT_DIR}/datasets/stack2/naturalcc"
SHADOW_MODEL_ID="${SHADOW_MODEL_ID:-sapienzanlp/Minerva-7B-base-v1.0}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-bigcode/starcoder2-15b}"
SHADOW_MODEL_TAG="${SHADOW_MODEL_ID//\//--}"
TARGET_MODEL_TAG="${TARGET_MODEL_ID//\//--}"
SHADOW_PRETRAIN_DIR="${SHADOW_MODEL_ID}"
TARGET_PRETRAIN_DIR="${TARGET_MODEL_ID}"

RESULTS_ROOT="${ROOT_DIR}/results/RQ3/naturalcc/${SHADOW_MODEL_TAG}_to_${TARGET_MODEL_TAG}"
AUDIT_OUT_DIR="${RESULTS_ROOT}/audit_models"
LOG_DIR="${RESULTS_ROOT}/log"
COMPARISON_CSV="${RESULTS_ROOT}/metrics_baseline_vs_joint_linear.csv"
mkdir -p "${AUDIT_OUT_DIR}" "${LOG_DIR}"

SHADOW_RANKS_SUBDIR="${SHADOW_RANKS_SUBDIR:-ranks}"
TARGET_RANKS_SUBDIR="${TARGET_RANKS_SUBDIR:-ranks_starcoder2}"
AUDIT_SHADOW_NUM="${AUDIT_SHADOW_NUM:-1}"
OUTPUT_SIZE="${OUTPUT_SIZE:-98304}"

GPU_DEVICES="${GPU_DEVICES:-0}"
NO_CUDA="${NO_CUDA:-0}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
RANK_EVAL_BATCH_SIZE="${RANK_EVAL_BATCH_SIZE:-4}"
RANK_LOGGING_STEPS="${RANK_LOGGING_STEPS:-50}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
SKIP_EXISTING_RANKS="${SKIP_EXISTING_RANKS:-1}"
SKIP_EXISTING_SHADOW_MEM_SYN="${SKIP_EXISTING_SHADOW_MEM_SYN:-1}"

INFER_BINS="${INFER_BINS:-100}"
INFER_HIDDEN_STATES="${INFER_HIDDEN_STATES:-50}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
INFER_EPOCH="${INFER_EPOCH:-50}"
INFER_LOGGING_STEPS="${INFER_LOGGING_STEPS:-400}"
MAPPER_FOLDS="${MAPPER_FOLDS:-5}"
NUMBER_OF_QUERIES="${NUMBER_OF_QUERIES:-0}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-frequency}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-youden}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-}"
SEED="${SEED:-42}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_file "${EXTERNAL_DIR}/rank_extractor.py"
require_file "${EXTERNAL_DIR}/build_mem_syn.py"
require_file "${EXTERNAL_DIR}/joint_linear.py"
require_file "${DATA_ROOT}/lat/fit.jsonl"
require_file "${DATA_ROOT}/lat/validate.jsonl"

run_one_rank() {
  local model_name="$1"
  local pretrain_dir="$2"
  local ranks_subdir="$3"
  local split_dir="${DATA_ROOT}/${model_name}/code-gpt"
  local rank_dir="${split_dir}/${ranks_subdir}"
  require_file "${split_dir}/train.txt"
  require_file "${split_dir}/test.txt"
  if [[ "${SKIP_EXISTING_RANKS}" == "1" && -f "${rank_dir}/member_ranks.pth" && -f "${rank_dir}/non_member_ranks.pth" ]]; then
    echo "Ranks exist for ${model_name}, skip"
    return
  fi
  mkdir -p "${rank_dir}"
  local args=(
    --data-dir "${split_dir}"
    --output-dir "${rank_dir}"
    --model "${pretrain_dir}"
    --block-size "${BLOCK_SIZE}"
    --logging-steps "${RANK_LOGGING_STEPS}"
    --seed "${SEED}"
  )
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then args+=(--trust-remote-code); fi
  if [[ "${NO_CUDA}" == "1" ]]; then args+=(--no-cuda); fi
  python -u "${EXTERNAL_DIR}/rank_extractor.py" "${args[@]}" \
    2>&1 | tee "${LOG_DIR}/${model_name}_rank.log"
}

echo "== NaturalCC + ReCodeMIA =="
echo "DATA_ROOT=${DATA_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SHADOW_MODEL_ID=${SHADOW_MODEL_ID}"
echo "TARGET_MODEL_ID=${TARGET_MODEL_ID}"

echo "[1/4] Extract shadow ranks"
for ((i = 0; i < AUDIT_SHADOW_NUM; i++)); do
  run_one_rank "shadow_${i}" "${SHADOW_PRETRAIN_DIR}" "${SHADOW_RANKS_SUBDIR}"
done

echo "[2/4] Extract target ranks"
run_one_rank target "${TARGET_PRETRAIN_DIR}" "${TARGET_RANKS_SUBDIR}"

echo "[3/4] Build shadow mem/syn features"
shadow_mem_syn_ready=1
for ((i = 0; i < AUDIT_SHADOW_NUM; i++)); do
  rank_dir="${DATA_ROOT}/shadow_${i}/code-gpt/${SHADOW_RANKS_SUBDIR}"
  if [[ ! -f "${rank_dir}/member_mem_syn.pth" || ! -f "${rank_dir}/non_member_mem_syn.pth" ]]; then
    shadow_mem_syn_ready=0
  fi
done
if [[ "${SKIP_EXISTING_SHADOW_MEM_SYN}" != "1" || "${shadow_mem_syn_ready}" != "1" ]]; then
  python -u "${EXTERNAL_DIR}/build_mem_syn.py" \
    --fit "${DATA_ROOT}/lat/fit.jsonl" \
    --validate "${DATA_ROOT}/lat/validate.jsonl" \
    --data_dir "${DATA_ROOT}" \
    --model_name "${SHADOW_PRETRAIN_DIR}" \
    --num_shadow "${AUDIT_SHADOW_NUM}" \
    --shadow_ranks_subdir "${SHADOW_RANKS_SUBDIR}" \
    --target_ranks_subdir "${TARGET_RANKS_SUBDIR}" \
    --batch_size "${RANK_EVAL_BATCH_SIZE}" \
    --max_length "${BLOCK_SIZE}" \
    --seed "${SEED}" \
    --skip_target \
    2>&1 | tee "${LOG_DIR}/build_shadow_mem_syn.log"
fi

echo "[4/4] Train/evaluate baseline and joint linear"
JOINT_ARGS=(
  --data-dir "${DATA_ROOT}"
  --output-dir "${AUDIT_OUT_DIR}"
  --metrics-out "${COMPARISON_CSV}"
  --model "${TARGET_MODEL_ID}"
  --bins "${INFER_BINS}"
  --hidden-states "${INFER_HIDDEN_STATES}"
  --batch-size "${INFER_BATCH_SIZE}"
  --epoch "${INFER_EPOCH}"
  --logging-steps "${INFER_LOGGING_STEPS}"
  --seed "${SEED}"
  --number-of-shadow-model "${AUDIT_SHADOW_NUM}"
  --output-size "${OUTPUT_SIZE}"
  --number-of-queries "${NUMBER_OF_QUERIES}"
  --sampling-strategy "${SAMPLING_STRATEGY}"
  --shadow-ranks-subdir "${SHADOW_RANKS_SUBDIR}"
  --target-ranks-subdir "${TARGET_RANKS_SUBDIR}"
  --mapper-folds "${MAPPER_FOLDS}"
  --threshold-metric "${THRESHOLD_METRIC}"
)
if [[ "${NO_CUDA}" == "1" ]]; then JOINT_ARGS+=(--no-cuda); fi
if [[ -n "${FIXED_THRESHOLD}" ]]; then JOINT_ARGS+=(--fixed-threshold "${FIXED_THRESHOLD}"); fi

python -u "${EXTERNAL_DIR}/joint_linear.py" "${JOINT_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/baseline_vs_joint_linear.log"

echo "Comparison CSV: ${COMPARISON_CSV}"
echo "Audit models: ${AUDIT_OUT_DIR}"
