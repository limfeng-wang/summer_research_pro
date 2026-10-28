# Methods and reproducibility

This describes the original study and private-data computations. The public
release contains code, metadata, and aggregate results only. Original posts,
reference examples, annotations, predictions, retrieval traces, and reviewed
canonicalization records are not distributed. Public verification recomputes
selected aggregate calculations and figures; it cannot reproduce the original
annotation, bootstrap resampling, or derivation of all aggregate results.

## Dataset and reference use

The three groups contain 150 development, 150 evaluation, and 36,720 automated
main-corpus posts. Each development and evaluation group has 50 posts per
language. Development membership was balanced retrospectively by removing 27
unused classification-only records from a 177-post union. This preserves the
current ordered classifier examples and all 60 extraction references; it does
not establish that the historical development process was prespecified.

Classification uses 145 of the development posts. For each input, the code
selects up to eight examples, prioritizing the input language and distinct
experiencer/content-function combinations, then filling in the original order.
Other-language examples are used only if needed. The target post is excluded.
This is deterministic selection, not embedding retrieval.

Extraction retrieves five examples from the 60 extraction references using
normalized BGE-M3 embeddings and cosine similarity. All three languages are
eligible; matching language and country break score ties. No minimum similarity
threshold is applied. The optional reranker is disabled. If embedding retrieval
fails, token Jaccard similarity is used with bonuses of 0.15 for matching
language and 0.05 for matching country. The archived evaluation retrieval trace
records the selected release-format reference IDs (not publicly distributed) and scores.

The inference loader uses only source text and metadata, even when reading the
annotated main file. Evaluation labels are read only by analysis scripts. The
dataset validator checks exact source-text separation across the three groups;
this cannot exclude near duplicates or evaluation-informed tuning.

## Models and prompts

| Stage | Configured model | Maximum new tokens |
| --- | --- | ---: |
| Combined R/E/C classifier | Qwen/Qwen3-4B-Instruct-2507 | 128 |
| Rescue classifier | Same classifier model | 192 |
| Evidence extraction | Qwen/Qwen3.5-9B | 2,048 |
| Unit judge | google/gemma-4-12b-it | 768 |
| Dense retrieval | BAAI/bge-m3 | Not applicable |

Language models use 8-bit loading and greedy decoding (`temperature=0`,
`do_sample=false`). Model inputs use the preserved source-language text;
working labels may be Chinese and normalized concepts are English. The combined
classifier truncates each source-text field to 1,800 characters. Rescue uses
2,200. Classifier examples retain at most 500 source characters, 220 evidence
characters, and 180 rationale characters. The optional separate classifier mode
uses 64 tokens for relevance and 128 for experiencer/content function.

The main first-pass prompt is `COMBINED_CLASSIFICATION_PROMPT`. Rescue uses
`CSM_RESCUE_CLASSIFICATION_PROMPT`. All prompt texts are exported in
[full_prompts.md](full_prompts.md); their executable source is
`dental_ai/prompts.py`. Runtime paths and settings are in
`configs/model_stack.yaml`. Historical model commit hashes and the original
inference environment were not recorded. The CPU dependency lock records the
release verification environment only.

## Checking and failure rules

Each model response is parsed and checked against the defined JSON schema.
Invalid labels or missing required fields cause a processing error, which is
recorded with the post identifier and processing stage. Non-relevant (`R0`) or
uncertain (`RU`) posts must not receive downstream experiencer or
content-function labels or extraction units. Relevant (`R1`) posts require both
labels. Normal extraction admits `E1/E2` and `C1/C2`; the primary first-person
analysis additionally requires `E1`.

Each extraction unit must specify one of the five CSM domains, a source-language
evidence span, a working label, support and assertion status, confidence between
zero and one, and a judge verdict. An English normalized concept is required
unless the unit is explicitly marked unmapped. Evidence is checked for exact
substring containment in the authoritative source text. A failed containment or
hierarchy check is recorded in the validation report; such reports do not
automatically remove all affected historical outputs.

Missing or invalid judge verdicts are assigned `needs_human_review` and are not
accepted. The shared analysis rule accepts a unit only when the verdict is
`accept`, support is not `unsupported`, and assertion is not `negated`.
Additional language and content safeguards are implemented in
`classification_postprocess.py` and `local_models.py`. Stage exceptions are
logged in `errors.jsonl`; checkpoints allow subsequent resumption. Completeness
must be checked against the expected input IDs. Evaluation rejects missing,
extra, or duplicate prediction IDs.

The archived manuscript export instead counted all units with verdict `accept`.
Consequently, its 12,829 posts differ from the 12,817-post subset requiring the
shared acceptance rule and exact evidence containment. Both flags are retained in the private records;
the release does not silently revise historical annotations or counts.

## Evaluation and ablations

Evaluation compares archived predictions against the separate 150-post human
reference (not distributed). It reports classification, five-domain presence, normalized-concept
agreement, and evidence quality. The original annotation text differs from the
input for 51 records; both versions are retained. The historical evaluation
eligibility predicate allows `E1/E2`, whereas the gate ablation targets `E1`
plus at least one human-annotated CSM domain. These definitions yield the same
eligibility confusion counts for the archived holdout, but are not generally
interchangeable.

The gate ablation compares `R1`, `R1+E1`, `R1+C1/C2`, and the full gate. Full-gate
precision is 0.8125, recall 0.5000, and specificity 0.9388. The last two gates
produce identical results on this holdout. The unit ablation compares all
proposed units, exact-span units, accepted units, and accepted exact-span units.
Judge filtering retains 75 of 99 units; macro-domain F1 decreases from about
0.467 to 0.368, so it should not be described as an accuracy improvement.

The production grid crosses five population definitions and seven unit rules,
producing 35 configurations. It recomputes domain distributions, concept counts,
enrichment, and network stability. Its default accepted-unit rule uses the
shared schema predicate, which is stricter than the archived manuscript export.
Original English labels remain intact; the reviewed within-domain mapping
supplies the canonical layer. The network sensitivity code uses within-post
co-occurrence counts and weighted degree, not NPMI or Leiden communities.

Evaluation uses 2,000 bootstrap samples with seed 20260824; gate/unit ablations
use 2,000 with seed 20260918 and deterministic per-configuration seed strings.
`results/reference/evaluation_order.json` preserves historical resampling order
after public ID replacement. These are analysis seeds; they do not recover the
missing original date-sampling seed.
