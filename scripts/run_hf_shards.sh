#!/usr/bin/env bash
set -euo pipefail

INPUT="${INPUT:-data/raw_main_llm_input_no_gold.jsonl}"
OUT_BASE="${OUT_BASE:-outputs/main_full_sharded}"
CONFIG="${CONFIG:-configs/model_stack.yaml}"
MODELS_ROOT="${MODELS_ROOT:-/hdd-storage/lawrencelcty/huggingface/models}"
HF_STAGE="${HF_STAGE:-full}"
CLASSIFICATION_MODE="${CLASSIFICATION_MODE:-combined}"
SHARD_COUNT="${SHARD_COUNT:-2}"
GPU_LIST="${GPU_LIST:-0,1}"
RESUME="${RESUME:-1}"
LIMIT="${LIMIT:-0}"
FILTER_BNB_WARNINGS="${FILTER_BNB_WARNINGS:-1}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-100000000}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_LIST must contain at least one device id, e.g. GPU_LIST=2,3" >&2
  exit 2
fi

mkdir -p "$OUT_BASE"

wait_for_batch() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_shard() {
  local shard_dir="$1"
  shift
  local log_file="$shard_dir/run.log"
  local exit_file="$shard_dir/exit_code.txt"
  set +e
  if [[ "$FILTER_BNB_WARNINGS" == "1" ]]; then
    "$@" 2>&1 | grep -v "MatMul8bitLt: inputs will be cast" > "$log_file"
  else
    "$@" > "$log_file" 2>&1
  fi
  local code="${PIPESTATUS[0]}"
  echo "$code" > "$exit_file"
  if [[ -f "$log_file" ]]; then
    local size
    size="$(wc -c < "$log_file")"
    if [[ "$size" -gt "$MAX_LOG_BYTES" ]]; then
      tail -c "$MAX_LOG_BYTES" "$log_file" > "$log_file.tmp"
      mv "$log_file.tmp" "$log_file"
      echo "[run_hf_shards] log truncated to last $MAX_LOG_BYTES bytes" >> "$log_file"
    fi
  fi
  return "$code"
}

pids=()
for (( shard=0; shard<SHARD_COUNT; shard++ )); do
  gpu="${GPUS[$(( shard % ${#GPUS[@]} ))]}"
  shard_name="$(printf 'shard_%03d' "$shard")"
  shard_dir="$OUT_BASE/$shard_name"
  mkdir -p "$shard_dir"

  cmd=(
    python -m dental_ai.cli run-hierarchical
    --input "$INPUT"
    --out-dir "$shard_dir"
    --backend hf
    --hf-stage "$HF_STAGE"
    --config "$CONFIG"
    --models-root "$MODELS_ROOT"
    --shard-count "$SHARD_COUNT"
    --shard-index "$shard"
    --classification-mode "$CLASSIFICATION_MODE"
  )
  if [[ "$RESUME" == "1" ]]; then
    cmd+=(--resume)
  fi
  if [[ "$LIMIT" != "0" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  echo "Launching $shard_name on CUDA_VISIBLE_DEVICES=$gpu"
  (
    export PYTHONPATH=.
    export CUDA_VISIBLE_DEVICES="$gpu"
    run_shard "$shard_dir" "${cmd[@]}"
  ) &
  pids+=("$!")

  if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
    if ! wait_for_batch "${pids[@]}"; then
      echo "At least one shard failed. Inspect $OUT_BASE/shard_*/run.log" >&2
      exit 1
    fi
    pids=()
  fi
done

if [[ "${#pids[@]}" -gt 0 ]] && ! wait_for_batch "${pids[@]}"; then
  echo "At least one shard failed. Inspect $OUT_BASE/shard_*/run.log" >&2
  exit 1
fi

echo "All shards finished. Merge with:"
echo "python scripts/merge_hf_shards.py --out-dir ${OUT_BASE}_merged ${OUT_BASE}/shard_*"
