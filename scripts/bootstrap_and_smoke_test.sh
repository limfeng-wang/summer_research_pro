#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-h-ramos}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-10}"
OUT_DIR="${OUT_DIR:-outputs/smoke_test}"

bash scripts/setup_h_ramos_env.sh "$ENV_NAME"

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  python scripts/download_models.py --config configs/model_stack.yaml
else
  echo "Skipping model download. Set DOWNLOAD_MODELS=1 to fetch configured HF models."
fi

PYTHONPATH=. python -m dental_ai.cli run-hierarchical \
  --input data/raw_eval_holdout_150.jsonl \
  --out-dir "$OUT_DIR" \
  --backend mock \
  --limit "$SMOKE_LIMIT"

echo "Smoke test complete: $OUT_DIR"
