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
- Extractor: `Qwen/Qwen3-8B`
- Judge: `google/gemma-4-E2B-it`
- Retriever: `BAAI/bge-m3`
- Optional reranker diagnostic: `BAAI/bge-reranker-v2-m3`

The default pipeline does not use or require the reranker. It can be downloaded
and tested separately, but it is not part of the active stack unless
`runtime.use_reranker` is set to `true`.

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

This downloads only the active required models by default. With the default
config, it does not download the optional reranker.

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

To also download the optional reranker for a separate diagnostic:

```bash
python scripts/download_models.py \
  --config configs/model_stack.yaml \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --include-optional
```

If Hugging Face returns an Xet/CAS `401 Unauthorized` error, first make sure
you are authenticated and have accepted gated model licenses, especially for
Gemma:

```bash
huggingface-cli login
```

Or use a token without storing it interactively:

```bash
HF_TOKEN=hf_xxx python scripts/download_models.py \
  --config configs/model_stack.yaml \
  --models-root /hdd-storage/lawrencelcty/huggingface/models
```

The downloader disables the Xet backend by default through
`HF_HUB_DISABLE_XET=1`. To force Xet, pass `--use-xet`.

Run a safe mock smoke test:

```bash
PYTHONPATH=. python -m dental_ai.cli build-splits

PYTHONPATH=. python -m dental_ai.cli run-hierarchical \
  --input data/raw_eval_holdout_150_no_gold.jsonl \
  --out-dir outputs/smoke_test \
  --backend mock \
  --limit 10
```

Check local model folders without loading weights:

```bash
PYTHONPATH=. python -m dental_ai.cli check-models \
  --models-root /hdd-storage/lawrencelcty/huggingface/models
```

Check the optional reranker in isolation before enabling it in the main run:

```bash
PYTHONPATH=. python -m dental_ai.cli check-reranker \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --backend transformers
```

The default full pipeline keeps reranking disabled until this diagnostic is
known to work on the target machine. With the default config, the defensible
method claim is BGE-M3 dense RAG few-shot retrieval plus local LLM-as-Judge,
not reranked RAG.

Run a tiny local-HF classification smoke test:

```bash
PYTHONPATH=. python -m dental_ai.cli run-hierarchical \
  --input data/raw_eval_holdout_150_no_gold.jsonl \
  --out-dir outputs/hf_classify_smoke \
  --backend hf \
  --hf-stage classify \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --limit 3
```

Run a tiny full RAG extraction + judge smoke test:

```bash
PYTHONPATH=. python -m dental_ai.cli run-hierarchical \
  --input data/raw_eval_holdout_150_no_gold.jsonl \
  --out-dir outputs/hf_full_smoke \
  --backend hf \
  --hf-stage full \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --limit 3
```

Full-stage outputs:

- `annotations.jsonl`: final post-level labels and CSM units
- `retrieval_trace.jsonl`: retrieved CSM gold examples per extracted post
- `errors.jsonl`: row-level failures without aborting the full batch
- `run_manifest.json`: attempted/succeeded/failed row counts

Run the leak-checked main automation corpus:

```bash
PYTHONPATH=. python -m dental_ai.cli run-hierarchical \
  --input data/raw_main_llm_input_no_gold.jsonl \
  --out-dir outputs/main_full_run \
  --backend hf \
  --hf-stage full \
  --models-root /hdd-storage/lawrencelcty/huggingface/models
```

## Current Status

The Hugging Face backend supports staged classification and full local
classification + CSM RAG extraction + LLM-as-Judge. Full mode loads one LLM
role at a time: classifier, then extractor, then judge.
