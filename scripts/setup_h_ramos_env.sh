#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-h-ramos}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required on the target machine" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements-h-ramos.txt

PYTHONPATH=. python -m dental_ai.cli check-env
