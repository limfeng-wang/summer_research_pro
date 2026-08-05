# Local Runbook

This repository is prepared so code can be developed here and run later on the
target machine with the `h-ramos` conda environment.

## Target Hardware Assumption

- NVIDIA RTX 4060 8GB VRAM
- 18GB system RAM
- 1TB storage
- Run one LLM at a time
- Use quantized local Hugging Face models for production inference

## Model Stack

Configured in `configs/model_stack.yaml`:

- Classifier: `Qwen/Qwen3-4B-Instruct-2507`
- Extractor: `Qwen/Qwen3-8B-Instruct`
- Judge: `google/gemma-3-4b-it`
- Retriever: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`

## One-Command Setup And Smoke Test

From the repo root on the target machine:

```bash
bash scripts/bootstrap_and_smoke_test.sh
```

This installs dependencies into `h-ramos`, checks the environment, and runs a
10-row mock pipeline smoke test. It does not download model weights by default.

To also download configured Hugging Face models:

```bash
DOWNLOAD_MODELS=1 bash scripts/bootstrap_and_smoke_test.sh
```

By default, model snapshots are stored under:

```text
/hdd-storage/lawrencelcty/huggingface/models/<org>/<repo>
```

For example:

```text
/hdd-storage/lawrencelcty/huggingface/models/Qwen/Qwen3-8B-Instruct
```

To override that root:

```bash
MODELS_ROOT=/path/to/models DOWNLOAD_MODELS=1 bash scripts/bootstrap_and_smoke_test.sh
```

If the conda environment has a different name:

```bash
ENV_NAME=h-ramos DOWNLOAD_MODELS=1 bash scripts/bootstrap_and_smoke_test.sh
```

## Manual Commands

Check environment:

```bash
PYTHONPATH=. python -m dental_ai.cli check-env
```

Download models only:

```bash
python scripts/download_models.py \
  --config configs/model_stack.yaml \
  --models-root /hdd-storage/lawrencelcty/huggingface/models
```

Run a safe mock smoke test:

```bash
PYTHONPATH=. python -m dental_ai.cli run-hierarchical \
  --input data/raw_eval_holdout_150.jsonl \
  --out-dir outputs/smoke_test \
  --backend mock \
  --limit 10
```

## Current Status

The production Hugging Face backend is not implemented yet. The current runner
supports `--backend mock` to verify IO, hierarchy, validation, and output
manifests without model downloads. Real model wrappers should be added behind
the existing pipeline interfaces.
