# Ablation and Sensitivity Analyses

This section reports the prespecified ablation and sensitivity analyses that were
most informative for the manuscript. Component ablations were evaluated on the
independent 150-post human-labeled holdout set. Downstream sensitivity analyses
were evaluated on the final production export.

## Component Ablation: Eligibility Gates

The clearest component-level result was the role of the upstream eligibility
gates. Toothache relevance alone (`R1`) retained more potentially eligible posts,
but it also admitted many false positives. Adding the content-function
restriction to retain experience-sharing and help-seeking posts (`C1/C2`)
substantially improved precision and specificity.

| Eligibility rule    | Precision | Recall |    F1 | Specificity | TP | FP | FN | TN |
| ------------------- | --------: | -----: | ----: | ----------: | -: | -: | -: | -: |
| `R1` only         |     0.434 |  0.635 | 0.516 |       0.561 | 33 | 43 | 19 | 55 |
| `R1 + E1`         |     0.534 |  0.596 | 0.564 |       0.724 | 31 | 27 | 21 | 71 |
| `R1 + C1/C2`      |     0.813 |  0.500 | 0.619 |       0.939 | 26 |  6 | 26 | 92 |
| `R1 + E1 + C1/C2` |     0.813 |  0.500 | 0.619 |       0.939 | 26 |  6 | 26 | 92 |

![Eligibility-gate ablation](../outputs/ablation_sensitivity_analysis/eligibility_gate_ablation.png)

These results show that the content-function gate was the main safeguard against
including general knowledge, commercial, or low-information toothache-related
posts in the illness-narrative corpus. The trade-off was explicit: specificity
increased from 0.561 with `R1` alone to 0.939 with `R1 + C1/C2`, while recall
decreased from 0.635 to 0.500. Adding the `E1` gate after `C1/C2` did not change
performance in this holdout set.

Suggested manuscript wording:

> In the eligibility-gate ablation, the content-function restriction was the
> main contributor to specificity. Toothache relevance alone (`R1`) achieved
> precision of 0.434 and specificity of 0.561. Adding the `C1/C2`
> experience-sharing/help-seeking restriction increased precision to 0.813 and
> specificity to 0.939, at the cost of lower recall. This supports the use of the
> content-function gate as a conservative inclusion rule for the primary
> illness-narrative corpus.

## Unit Validation Ablation

The unit-validation ablation did not show that LLM judge filtering improved
domain classification performance. Instead, judge filtering behaved as a
conservative retention rule.

| Unit-retention rule                   | Macro precision | Macro recall | Macro F1 | Retained units |
| ------------------------------------- | --------------: | -----------: | -------: | -------------: |
| All proposed units                    |           0.739 |        0.355 |    0.467 |             99 |
| Span-supported units                  |           0.734 |        0.349 |    0.459 |             98 |
| Judge-accepted units                  |           0.741 |        0.264 |    0.368 |             75 |
| Judge-accepted + span-supported units |           0.741 |        0.264 |    0.368 |             75 |

Because the judge reduced recall and macro-F1 without materially improving
macro-precision, this result should not be framed as an accuracy gain. The more
defensible interpretation is that judge filtering produced a stricter analytic
set. Span filtering had little effect because evidence-span support was already
high: 98 of 99 proposed units had evidence spans found in the source text.

Suggested manuscript wording:

> Unit-validation ablations indicated that LLM judge filtering should be
> interpreted as conservative retention rather than improved domain
> classification. All proposed units yielded macro-F1 of 0.467, whereas
> judge-accepted units yielded macro-F1 of 0.368. Judge filtering therefore
> reduced coverage and recall while preserving a stricter set of evidence-backed
> units for descriptive analysis.

## Downstream Sensitivity of Main Findings

We next tested whether substantive country-platform findings depended on a
single corpus definition or unit-retention rule. The production sensitivity grid
crossed five corpus definitions with seven unit-retention rules, yielding 35
configurations.

Two findings were stable across all 35 configurations:

| Finding                                                                   | Configurations retaining direction |
| ------------------------------------------------------------------------- | ---------------------------------: |
| Symptom Description was the most frequent CSM domain in each corpus       |                            35 / 35 |
| China had a higher Coping and Management share than Japan and South Korea |                            35 / 35 |
| South Korea had the highest Emotional Expression share                    |                             0 / 35 |

In the primary accepted-unit analysis, Symptom Description was the most frequent
domain in every corpus. Coping and Management was also consistently more common
in the Chinese corpus than in the Japanese and Korean corpora.

| Country-platform corpus | Symptom Description | Coping and Management | Emotional Expression |
| ----------------------- | ------------------: | --------------------: | -------------------: |
| China/Xiaohongshu       |               94.5% |                 75.9% |                49.4% |
| Japan/X                 |               92.7% |                 34.1% |                36.1% |
| South Korea/X           |               84.9% |                 27.5% |                38.8% |

![Primary domain percentages](../outputs/ablation_sensitivity_analysis/main_sensitivity_heatmap.png)

The prespecified emotional-expression contrast was not supported as originally
stated. South Korea did not have the highest Emotional Expression share across
the sensitivity grid. The consistent ordering was China, then South Korea, then
Japan. Therefore, emotional-expression differences should be treated as a
secondary, cautious finding rather than a primary robust contrast.

Suggested manuscript wording:

> The downstream sensitivity grid showed that the main descriptive findings were
> not artifacts of a single filtering choice. Symptom Description remained the
> most frequent domain in all three corpora across all 35 configurations, and
> the higher share of Coping and Management in the Chinese corpus was retained
> in all 35 configurations. By contrast, the prespecified contrast that South
> Korea would show the highest Emotional Expression share was not supported;
> Emotional Expression was consistently ordered as China, South Korea, and then
> Japan.

## Recommended Interpretation

The ablations support three restrained claims:

1. The content-function gate materially improved specificity and precision for
   identifying the primary illness-narrative corpus.
2. Judge filtering should be described as conservative unit retention, not as an
   accuracy-improving component.
3. The most robust substantive findings were the dominance of symptom
   descriptions and the higher share of coping/management content in the Chinese
   corpus.

They do not support claiming that judge validation improved domain prediction,
that normalized concepts achieved high gold-standard agreement, or that South
Korea had the highest emotional-expression burden.

## 4. Sampling Provenance, Data Separation, Attrition, and Reproducibility

See the [reproducibility chapter](ablation_reproducibility_integrity.md) for sampling
records, leakage checks, losses by language, technical settings, and unresolved gaps.
