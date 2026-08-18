# CSM Recall-Rescue Workflow

This workflow repairs first-pass CSM false negatives without changing the original
36K outputs. The first-pass pipeline remains the high-precision baseline. The
rescue pass only reviews first-pass non-eligible rows and sends newly rescued
`R1/E1-or-E2/C1-or-C2` rows through the existing CSM extraction and judge stages.

## Why This Exists

The 150-row manually annotated holdout showed that first-pass CSM-eligibility
precision is reasonable, but recall is too low, especially for Korean. The main
failure mode is classification/gating false negatives, not CSM extraction or judge.

Therefore:

- Keep existing CSM-positive rows.
- Do not rerun all rows.
- Do not swap extractor or judge.
- Add a high-recall second-pass gate over likely false negatives.

## Step 1: Build Rescue Candidates

Use all completed merged/repaired first-pass outputs.

```bash
PYTHONPATH=. python3 scripts/build_csm_rescue_candidates.py \
  outputs/main_pilot_10000_sharded_3gpu_merged \
  outputs/main_pilot_20000_sharded_3gpu_merged_repaired \
  outputs/main_pilot_36k_sharded_3gpu_merged_repaired \
  --output data/csm_rescue_candidates_all36k.jsonl
```

If the final Japanese run has a different merged folder name, substitute that
folder. Use the repaired version if a zero-unit finalization repair was needed.

Candidate policy:

- Korean: all first-pass non-eligible rows.
- Chinese/Japanese: keyword/anchor-targeted first-pass non-eligible rows.

## Step 2: Run Rescue Classifier

Run on three GPUs with manual shard commands:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/run_csm_rescue_classifier.py \
  --input data/csm_rescue_candidates_all36k.jsonl \
  --out-dir outputs/csm_rescue_classifier_sharded/shard_000 \
  --config configs/model_stack.yaml \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --shard-count 3 \
  --shard-index 0 \
  --resume
```

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/run_csm_rescue_classifier.py \
  --input data/csm_rescue_candidates_all36k.jsonl \
  --out-dir outputs/csm_rescue_classifier_sharded/shard_001 \
  --config configs/model_stack.yaml \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --shard-count 3 \
  --shard-index 1 \
  --resume
```

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python3 scripts/run_csm_rescue_classifier.py \
  --input data/csm_rescue_candidates_all36k.jsonl \
  --out-dir outputs/csm_rescue_classifier_sharded/shard_002 \
  --config configs/model_stack.yaml \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --shard-count 3 \
  --shard-index 2 \
  --resume
```

The output of each shard is `rescue_decisions.jsonl`.

## Step 3: Prepare Extraction/Judge Run

Convert rescued positives into a normal prefilled classification checkpoint.

```bash
PYTHONPATH=. python3 scripts/prepare_rescue_extraction_run.py \
  --candidates data/csm_rescue_candidates_all36k.jsonl \
  --decisions outputs/csm_rescue_classifier_sharded/shard_000 \
  --decisions outputs/csm_rescue_classifier_sharded/shard_001 \
  --decisions outputs/csm_rescue_classifier_sharded/shard_002 \
  --out-dir outputs/csm_rescue_extraction_full_sharded \
  --shard-count 3
```

This creates:

- `outputs/csm_rescue_extraction_full_sharded/rescued_input.jsonl`
- `outputs/csm_rescue_extraction_full_sharded/shard_*/checkpoints/classified.jsonl`

## Step 4: Run Existing Extraction/Judge Pipeline

Use the normal full HF runner with `--resume`. It will skip classification
because the classified checkpoint is already present, then run RAG, extraction,
and judge for rescued rows.

```bash
GPU_LIST=2,3,4 \
SHARD_COUNT=3 \
INPUT=outputs/csm_rescue_extraction_full_sharded/rescued_input.jsonl \
OUT_BASE=outputs/csm_rescue_extraction_full_sharded \
MODELS_ROOT=/hdd-storage/lawrencelcty/huggingface/models \
bash scripts/run_hf_shards.sh
```

Alternatively, for a single-GPU small rescue run:

```bash
PYTHONPATH=. python3 scripts/prepare_rescue_extraction_run.py \
  --candidates data/csm_rescue_candidates_all36k.jsonl \
  --decisions outputs/csm_rescue_classifier_sharded/shard_000 \
  --decisions outputs/csm_rescue_classifier_sharded/shard_001 \
  --decisions outputs/csm_rescue_classifier_sharded/shard_002 \
  --out-dir outputs/csm_rescue_extraction_full
```

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -m dental_ai.cli run-hierarchical \
  --input outputs/csm_rescue_extraction_full/rescued_input.jsonl \
  --out-dir outputs/csm_rescue_extraction_full \
  --backend hf \
  --hf-stage full \
  --config configs/model_stack.yaml \
  --models-root /hdd-storage/lawrencelcty/huggingface/models \
  --classification-mode combined \
  --resume
```

## Step 5: Merge And Report

For sharded rescue extraction:

```bash
PYTHONPATH=. python3 scripts/merge_hf_shards.py \
  --out-dir outputs/csm_rescue_extraction_full_merged \
  outputs/csm_rescue_extraction_full_sharded/shard_*
```

Final analysis should combine:

- original first-pass positives
- plus rescued positives from `outputs/csm_rescue_extraction_full_merged`

Do not overwrite the original first-pass output folders.

## Holdout Tuning

Before full-corpus rescue, run the same workflow on `outputs/eval_holdout_150_latest`
and evaluate:

```bash
PYTHONPATH=. python3 scripts/build_csm_rescue_candidates.py \
  outputs/eval_holdout_150_latest \
  --output data/csm_rescue_candidates_eval150.jsonl
```

Run the rescue classifier, then:

```bash
PYTHONPATH=. python3 scripts/evaluate_rescue_decisions.py \
  --gold data/raw_eval_holdout_150_gold.xlsx \
  --first-pass outputs/eval_holdout_150_latest/annotations.jsonl \
  --decisions outputs/csm_rescue_classifier_eval150 \
  --out outputs/csm_rescue_classifier_eval150/rescue_eval_metrics.json
```

Target before full use:

- Improve Korean eligible recall substantially.
- Keep overall eligible precision around or above 70%.
- Do not let Japanese precision collapse.
