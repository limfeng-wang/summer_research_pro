# Inference and rescue workflow

The original study inputs, classification/extraction references, predictions,
and canonicalization decisions are **not distributed**. These commands are for
authorized holders of compatible private inputs or researchers supplying their
own data. Paths under `data/publication_dataset/` in the default config are
private-input placeholders, not included files. Set those paths to your own
references before real inference. Run commands from the repository root and use
a new output directory per experiment; annotation overwrites files unless
`--resume` is supplied. Generated records must stay out of public commits.

Install the tested CUDA setup and obtain model weights separately:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c configs/inference-cu128-constraints.txt -e '.[inference]'
python scripts/download_models.py --models-root models
dental-ai check-models --models-root models
```

## Tested GPU setup

A real-model smoke test passed on 2026-10-24 on two NVIDIA L20 GPUs. It used
Python 3.11, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, Transformers 5.20.2,
Accelerate 1.15.0, bitsandbytes 0.50.2, sentence-transformers 6.2.0, and
huggingface-hub 2.3.0. Selected tested dependency versions are recorded in
`configs/inference-cu128-constraints.txt`; this is not a complete dependency lock.
The CPU lock and this GPU setup serve different verification scopes.

The test checks model loading, dense retrieval without lexical fallback, full
annotation of one short evaluation post per language, byte-stable resume, exact
source evidence in direct extraction probes, and usable judge verdicts in
direct judge probes. Inputs are selected using archived outputs to exercise
the stages. Passing is an execution check, not an independent accuracy estimate.
It does not pin or reproduce the historical model checkpoint revisions.

With Conda and a compatible NVIDIA driver installed, run:

```bash
bash tests/test_gpu_smoke.sh --gpus 0,1 --models-root models --config /path/to/private_config.yaml --input /path/to/evaluation/posts.jsonl --predictions /path/to/evaluation_predictions.jsonl
```

Choose GPUs with enough free memory. The test was verified on two L20s; the
default single GPU is a device-selection default, not a claim that any GPU can
fit the models. The script preserves an existing `CUDA_VISIBLE_DEVICES` value
unless `--gpus` is supplied; otherwise it selects GPU 0. Its default model
directory is `models/` under the repository root. It creates and removes a
temporary Conda environment and downloads missing models into the model directory,
which it retains. Model access permissions still apply.

Earlier preparation runs encountered CUDA-build incompatibility, a missing
torchvision dependency for the judge processor, incomplete final annotations,
and an evidence-span failure. The final test passed with the setup above;
this does not establish that arbitrary future dependency versions will work.
Logs and reports go to ignored `outputs/gpu_smoke/` and are not public artifacts.

The original study data and references are needed to repeat that historical
smoke test; it cannot be run from public aggregate tables alone. The shell
wrapper requires explicit private config/input/prediction paths before creating
an environment or downloading models.

## First pass

The remaining examples use the historical private directory layout. Supply
those inputs separately and configure the reference paths; they are unavailable
in a public checkout. The public-only command is
`python scripts/reproduce_aggregate_results.py`.

```bash
dental-ai run-hierarchical --input data/publication_dataset/main/posts.jsonl --out-dir outputs/first_pass --backend hf --hf-stage full --models-root models
```

The input adapter discards existing automated labels and units. This command
therefore performs new annotation on the finalized source texts. It cannot
reconstruct historical collection or recover the excluded failed input.

For parallel execution, use a separate output directory per shard with
`--shard-count N --shard-index I`, where `I` runs from zero to `N-1`. Then merge:

```bash
python scripts/merge_hf_shards.py --out-dir outputs/first_pass outputs/shard_0 outputs/shard_1 outputs/shard_2
```

## Rescue pass

The rescue candidate rule reviews every non-eligible Korean post and
anchor-matched non-eligible Chinese/Japanese posts. It is implemented in
`scripts/build_csm_rescue_candidates.py`.

```bash
python scripts/build_csm_rescue_candidates.py outputs/first_pass --output outputs/rescue_candidates.jsonl
python scripts/run_csm_rescue_classifier.py --input outputs/rescue_candidates.jsonl --out-dir outputs/rescue_classified --models-root models
python scripts/prepare_rescue_extraction_run.py --candidates outputs/rescue_candidates.jsonl --decisions outputs/rescue_classified --out-dir outputs/rescue_extracted
dental-ai run-hierarchical --input outputs/rescue_extracted/rescued_input.jsonl --out-dir outputs/rescue_extracted --backend hf --hf-stage full --models-root models --resume
```

Preparation creates classified checkpoints so extraction resumes with the
rescued labels. A rescue annotation replaces its first-pass result only when
it contains at least one verdict-accepted unit.

## Forced extraction for sensitivity analyses

To retain proposed CSM units for `R1` posts outside the normal `E1/E2+C1/C2`
gate, prepare and run a separate forced pass:

```bash
python scripts/prepare_forced_csm_extraction_run.py --first-pass-dir outputs/first_pass --rescue-dir outputs/rescue_extracted --out-dir outputs/forced --shard-count 1
FORCE_EXTRACT_ALL_R1_CSM=1 dental-ai run-hierarchical --input outputs/forced/input.jsonl --out-dir outputs/forced/shard_000 --backend hf --hf-stage full --models-root models --resume
python scripts/merge_hf_shards.py --out-dir outputs/forced_merged outputs/forced/shard_000
python scripts/merge_runs.py --first-pass outputs/first_pass --rescue outputs/rescue_extracted --forced outputs/forced_merged --out-dir outputs/merged
```

Forced outputs replace earlier results even when every unit is rejected.
Zero-unit extraction checkpoints are retained as successful zero-unit results.
These rules preserve the historical merge semantics. The new merge utility
requires explicit paths and records their hashes; it never imports private
experiment directories.

Inspect `errors.jsonl`, output membership, validation reports in checkpoints,
and retrieval traces before analysis. Resume failed work with the same input,
shard, and configuration. A model path check confirms file presence only;
it does not establish that the model can be loaded on the available hardware.
