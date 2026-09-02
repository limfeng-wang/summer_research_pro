# Canonical Keyword Analysis: Team Results Brief

Generated: 2026-09-02 14:45 UTC

## Files To Review

- Processed workbook: `outputs/keyword_report/merged_label_statistics.xlsx`
- Dimension figure: `outputs/keyword_report/figure_dimension_distribution.png`
- Keyword heatmap: `outputs/keyword_report/figure_keyword_heatmap.png`
- The workbook contains the merged statistics, top keywords, enrichment results, tests, residuals, and canonicalization mapping.

## What This Analysis Uses

The analysis uses the reviewed canonical keyword layer. Original labels are preserved in the mapping sheet, but frequency summaries use `fine_canonical_keyword`. The primary count is `main_count`; `post_count` is retained but should be treated cautiously after label merging because source inputs were already aggregated before canonicalization.

Canonicalization summary:

- Total dimension-keyword labels covered: 15,323
- Labels reviewed or confirmed through cluster review: 520
- Labels changed by canonicalization: 384
- Unreviewed labels were retained as separate labels.

## Main Findings

### 1. CSM dimension profiles differ by country

The country by CSM dimension association was statistically significant: chi-square(8) = 557.59, p = 3.05e-115, Cramer's V = 0.088. The effect is statistically clear but small overall.

| dimension | CHI_main_count | CHI_percent | JPN_main_count | JPN_percent | KOR_main_count | KOR_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Coping and Management | 1015 | 31.0 | 4556 | 15.6 | 449 | 13.5 |
| Emotional Expression | 365 | 11.2 | 4587 | 15.7 | 611 | 18.3 |
| Perceived Cause | 313 | 9.6 | 3217 | 11.0 | 417 | 12.5 |
| Perceived Consequences | 350 | 10.7 | 3386 | 11.6 | 389 | 11.7 |
| Symptom Description | 1229 | 37.6 | 13380 | 45.9 | 1466 | 44.0 |

Most notable standardized residuals:

- CHI / Coping and Management: higher than expected (standardized residual 19.75)
- CHI / Emotional Expression: lower than expected (standardized residual -6.40)
- CHI / Symptom Description: lower than expected (standardized residual -6.34)
- JPN / Coping and Management: lower than expected (standardized residual -5.02)
- KOR / Coping and Management: lower than expected (standardized residual -4.74)
- KOR / Emotional Expression: higher than expected (standardized residual 4.05)

Pairwise dimension-profile tests:

| comparison | chi_square | degrees_of_freedom | p_value | cramers_v | n_main_units | bh_q_value |
| --- | --- | --- | --- | --- | --- | --- |
| CHI_vs_JPN | 499.054 | 4 | 1.07e-106 | 0.124 | 32398 | 3.22e-106 |
| CHI_vs_KOR | 318.024 | 4 | 1.40e-67 | 0.219 | 6604 | 2.10e-67 |
| JPN_vs_KOR | 29.854 | 4 | 5.24e-06 | 0.030 | 32458 | 5.24e-06 |

### 2. Top canonical keywords differ sharply across countries

- CHI top labels: Emotional distress (145, 4.4%); Sleep disturbance (106, 3.2%); Severe dental pain (102, 3.1%); Nocturnal dental pain (70, 2.1%); Referred orofacial pain (46, 1.4%)
- JPN top labels: Dental pain (4244, 14.6%); Severe dental pain (1670, 5.7%); Emotional distress (1103, 3.8%); Persistent dental pain (381, 1.3%); Eating impairment (317, 1.1%)
- KOR top labels: Dental pain (369, 11.1%); Severe dental pain (322, 9.7%); Emotional distress (256, 7.7%); Wisdom tooth (94, 2.8%); Gingival pain (67, 2.0%)

Interpretation: JPN and KOR are dominated by generic `Dental pain` and `Severe dental pain`, while CHI remains unusual after canonicalization: `Dental pain` is low and `Emotional distress`, `Sleep disturbance`, and `Severe dental pain` lead the table.

### 3. Country-enriched labels suggest interpretable differences

CHI enriched labels:

- Internal heat attribution [Perceived Cause]: 28 vs 0 other-country units; log2 rate ratio 9.37
- Postural pain relief [Coping and Management]: 11 vs 0 other-country units; log2 rate ratio 6.82
- Wisdom tooth inflammation [Perceived Cause]: 10 vs 1 other-country units; log2 rate ratio 6.34
- Severe nocturnal dental pain [Symptom Description]: 22 vs 6 other-country units; log2 rate ratio 5.38
- Pulpitis [Perceived Cause]: 9 vs 4 other-country units; log2 rate ratio 4.61

JPN enriched labels:

- NSAID use [Coping and Management]: 65 vs 0 other-country units; log2 rate ratio 5.40
- Bruxism or clenching [Perceived Cause]: 178 vs 1 other-country units; log2 rate ratio 4.76
- Dentin hypersensitivity [Perceived Cause]: 55 vs 0 other-country units; log2 rate ratio 4.66
- Bruxism [Perceived Cause]: 33 vs 0 other-country units; log2 rate ratio 3.93
- Sleep impairment [Perceived Consequences]: 29 vs 0 other-country units; log2 rate ratio 3.69

KOR enriched labels:

- Wisdom tooth extraction [Perceived Cause]: 14 vs 0 other-country units; log2 rate ratio 7.94
- Functional limitation [Perceived Consequences]: 25 vs 8 other-country units; log2 rate ratio 4.85
- Wisdom tooth [Perceived Cause]: 94 vs 43 other-country units; log2 rate ratio 4.20
- Delayed dental care [Coping and Management]: 6 vs 4 other-country units; log2 rate ratio 4.16
- Gingival pain [Symptom Description]: 67 vs 76 other-country units; log2 rate ratio 3.14

## Recommended Presentation Figures

Use the dimension distribution figure as the main result figure. Use the keyword heatmap as a secondary descriptive figure or supplementary figure; it is informative but visually dense.

## Caveats To State

- Canonicalization reduces wording fragmentation but does not make labels objective ground truth.
- Canonicalization was restricted to within-dimension labels.
- Reviewers accepted several separate clusters converging to the same canonical label; these are pooled in the final statistics.
- China still shows low generic `Dental pain` after merging, so this likely reflects upstream annotation/normalization behavior rather than only long-tail wording variation.
