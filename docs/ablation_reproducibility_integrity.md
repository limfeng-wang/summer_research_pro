# 4. Reproducibility and Data Integrity

This chapter reports checks of existing files. Study data and results were not changed.

## 4.1 Sampling dates and timezones

The manuscript specifies **36 dates**, one per month from July 2023 to June 2026,
shared across the three corpora.

**Missing:** the actual dates, original R script and sampling seed, and platform
query timezones. The retained input tables contain no publication timestamps.
The [36-month register](ablation_reproducibility_artifacts/sampling_dates_unverified.csv)
marks these fields as unverified.

Older cleaning code converts Japanese/Korean timestamps to UTC+09:00 and leaves
Chinese timestamps timezone-naive. This does not prove the timezone used for
search-day boundaries. The 208-post Chinese supplement also needs confirmation
that it followed the same sampling dates.

## 4.2 Dataset origins and leakage

| Resource                   | Available source                                                  |                 Size | Purpose              |
| -------------------------- | ----------------------------------------------------------------- | -------------------: | -------------------- |
| Keyword pilot              | No roster located                                                 |           100 stated | Develop search terms |
| Classification development | `三语分类金标准-Law.xlsx` / `classification_gold_172.jsonl`   |                  172 | Classifier examples  |
| RAG reference              | `csm_gold_50E1_10E2.json`                                       | 60 posts / 417 units | Extraction examples  |
| Evaluation                 | `raw_eval_holdout_150_no_gold.jsonl` and separate gold workbook | 150; 50 per language | Evaluate predictions |

**Verified:** no exact ID/text overlap between evaluation and development or
production. Development posts are also excluded from production. Examples enter
prompts programmatically; evaluation posts enter without their human labels,
which are used separately for scoring.

**Clarify before publication:**

- Classification and RAG development resources intentionally share **55 posts**.
  They are not mutually disjoint; this is development reuse, not evaluation leakage.
- The manuscript states **150 classification-development posts**, but current
  files contain **172**. Identify the version used.
- The keyword pilot cannot be audited without its roster.
- Exact-match separation does not rule out near duplicates or tuning on evaluation
  results. The holdout's freeze/development history is still needed.

Evidence: [overlap counts](ablation_reproducibility_artifacts/split_overlap_counts.csv).

## 4.3 Losses by language

| Stage                                              |    Chinese |    Japanese |     Korean |       Total |
| -------------------------------------------------- | ---------: | ----------: | ---------: | ----------: |
| Available source workbooks                         |      3,423 |      25,299 |      8,117 |      36,839 |
| **Evaluation records reserved**                   |    **50** |     **50** |    **50** |    **150** |
| **Development matches / duplicate text excluded** |    **60** |     **81** |    **35** |    **176** |
| Chinese supplement added                           |        208 |           0 |          0 |         208 |
| Production input                                   |      3,521 |      25,168 |      8,032 |      36,721 |
| **Final labeled posts**                           | **3,521** | **25,167** | **8,032** | **36,720** |
| Relevant: R1                                       |      2,932 |      15,637 |      2,214 |      20,783 |
| Own experience: R1/E1                              |      1,511 |      14,260 |      1,933 |      17,704 |
| **Eligible: R1/E1/C1–C2**                        |   **838** | **11,195** | **1,353** | **13,386** |
| Eligible with proposed units                       |        689 |      10,872 |      1,324 |      12,885 |
| **Final manuscript sample**                       |   **687** | **10,819** | **1,323** | **12,829** |

One Japanese post (`JPN_18974`) failed extraction because of an invalid temporality
value. Final retention was **19.51% Chinese, 42.99% Japanese, and 16.47% Korean**.
Screening exclusions are not necessarily annotation errors.

Losses before the available workbooks cannot be counted without collection and
cleaning logs. The supplement contained 748 rows; 208 were appended. Its remaining
540 rows were not LLM failures.

**Acceptance discrepancy:** exported counts use judge verdict `accept`, without
enforcing every schema/evidence check.

| Finding                                             | Chinese | Japanese | Korean |  Total |
| --------------------------------------------------- | ------: | -------: | -----: | -----: |
| Exported accepted units                             |   3,940 |   28,480 |  3,207 | 35,627 |
| Accepted units marked negated                       |      35 |      115 |      8 |    158 |
| Otherwise accepted units without exact source spans |      40 |      126 |     25 |    191 |

Applying the shared acceptance rule plus exact-span checking would retain
**12,817 posts / 35,278 units**. This is an audit comparison, not a revised result.
Reconcile these rules before claiming every retained unit passed all checks.

Evidence: [post attrition](ablation_reproducibility_artifacts/language_attrition.csv)
and [unit retention](ablation_reproducibility_artifacts/language_unit_retention.csv).

## 4.4 Technical settings and failure handling

| Component              | Setting                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| Classifier             | `Qwen/Qwen3-4B-Instruct-2507`, 8-bit                                    |
| Extractor              | `Qwen/Qwen3.5-9B`, 8-bit                                                |
| Judge                  | `google/gemma-4-12b-it`, 8-bit                                          |
| Retriever              | `BAAI/bge-m3`                                                           |
| RAG                    | **Top-5** examples; cosine similarity through normalized embeddings |
| Language restriction   | None; same language/country breaks dense-score ties                       |
| Reranking / cutoff     | Reranker disabled; no minimum similarity cutoff                           |
| Retrieval fallback     | Token Jaccard + 0.15 same-language / 0.05 same-country bonuses            |
| Classifier examples    | Up to **8**, preferring the same language and diverse E/C labels    |
| Decoding               | Greedy; temperature 0, sampling disabled                                  |
| Output token limits    | Combined classifier 128; rescue 192; extractor 2,048; judge 768           |
| Classifier input limit | First 1,800 characters; rescue uses 2,200                                 |

**Full prompts:** [six-prompt appendix](ablation_reproducibility_artifacts/full_prompts.md).
It snapshots current source, not historical prompt immutability. Cross-language
retrieval occurred in 32/160 evaluation trace entries. Logs do not identify whether
dense retrieval or lexical fallback was used.

**Checks and failure rules:**

- Validate schema, labels, metadata, and exact evidence spans; minor copying
  errors may be repaired using source substrings before validation.
- Judge source support, domain, and interpretation. Additional rules address
  commercial content, generic advice, experiencer, help seeking, and negation.
- Shared acceptance requires `accept`, non-unsupported support, and non-negated
  assertion. Implicit, uncertain, and planned units can remain.
- Log parsing/stage failures and resume from checkpoints. No general automatic
  retry budget exists. Missing judge verdicts remain unaccepted. Validation
  failure does not universally prevent export.

**Additional passes:** final outputs include rescue and forced extraction.
Rescue targeted all noneligible Korean posts but used language-specific anchors
for Chinese/Japanese candidates. Rescue provenance accounts for 311 Chinese,
4,306 Japanese, and 1,063 Korean eligible posts. Report these passes rather than
describing a single uniform pass. See [pass counts](ablation_reproducibility_artifacts/final_pass_lineage.csv).

**Still missing:** immutable checkpoint revisions/weight hashes, a frozen execution
environment, historical prompt/rule versions, and manifests linking rescue/repair
passes to evaluation and production. Model names alone are insufficient.

Run `python3 scripts/build_reproducibility_audit.py` to regenerate supporting tables,
prompts, and [file hashes](ablation_reproducibility_artifacts/audit_manifest.json)
without rerunning models or changing study data.
