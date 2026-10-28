# Archived aggregate results

Only study-level/group-level results are distributed here. No original posts,
evidence spans, per-post predictions, annotations, identifier lists, retrieval
traces, or reviewed concept-label mappings are included.

| File | Purpose |
| --- | --- |
| `corpus_summary.json` | Aggregate corpus and manuscript-subset counts |
| `expected_evaluation.json` | Archived holdout point estimates and bootstrap intervals |
| `aggregate_tables/*.csv` | 14 aggregate gate/unit ablation and sensitivity tables |
| `expected_ablation_tables.json` | Historical row counts and order-independent checksums |

The CSV tables were reproduced from the original private records before those
records were removed from the distributable tree. Each table matches its existing
historical checksum; neither its values nor the expected checksums were revised
for this change of release scope.

`python scripts/reproduce_aggregate_results.py` checks those signatures,
recomputes binary metrics and country-profile statistics from aggregate counts,
and redraws five figures. Figure regeneration verifies plotting from archived
tables, not pixel identity across Matplotlib versions.

This is partial computational reproducibility. It does not independently verify
how the aggregate tables were generated from posts or regenerate bootstrap
intervals. Original-data scripts are retained for inspection, but require private
inputs. No data access or separate data reuse license is granted by these files.
