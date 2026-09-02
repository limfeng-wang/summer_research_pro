# Keyword Canonicalization Protocol

This protocol defines the duplicate-label reconciliation step used after CSM
unit extraction and before comparative keyword-frequency analysis.

## Purpose

The objective is to reduce artificial long-tail fragmentation caused by
synonymous English renderings of the same concept. This step does not define
new CSM dimensions and does not replace the original extracted labels. It adds
an auditable second-stage canonical keyword layer.

## Scope

- Canonicalization is performed only within the same CSM dimension.
- Original `normalized_concept_en` labels are preserved.
- Country-level frequencies are hidden from reviewers during merge decisions.
- Computational similarity methods retrieve candidate duplicate labels only.
- Final merge decisions are made by independent human reviewers.

## Merge Rules

Merge labels only when they differ by:

- spelling or punctuation
- singular/plural morphology
- word order with no meaning change
- synonymous wording with no added clinical or behavioral meaning
- translation-equivalent English rendering of the same concept

Do not merge labels when one adds or changes:

- severity, such as mild, severe, unbearable, intense
- temporality, such as nocturnal, persistent, recurrent, acute, chronic
- cause or trigger, such as wisdom tooth, cold stimulus, extraction
- body site, such as gum, jaw, tooth, face, throat
- treatment or management, such as ibuprofen, antibiotics, dental visit
- consequence, such as sleep disturbance or eating impairment
- uncertainty, such as suspected, possible, unclear
- experiencer or attribution when analytically meaningful

## Candidate Retrieval

Within each CSM dimension, candidate pairs are retrieved with:

- character n-gram TF-IDF cosine similarity, for surface-form variants
- token Jaccard similarity, for token-overlap variants
- optional sentence-embedding cosine similarity, for semantic equivalents

These scores are not decision rules. They are used to prioritize pairs for
review.

## Review Procedure

Two reviewers independently code each blinded candidate pair as:

- `merge`
- `do_not_merge`
- `uncertain`

Reviewers should not see country, country-specific counts, or rank. When
needed, de-identified source snippets may be reviewed separately, still without
country-level frequency summaries.

Disagreements are resolved by consensus or by a third reviewer. Approved
pairwise merges are then converted into clusters, and final clusters are
manually checked for transitive errors.

## Audit Trail

The final supplement should retain:

- CSM dimension
- original keyword
- canonical keyword
- fine-grained canonical keyword, if different
- parent canonical keyword, if used
- merge rule invoked
- reviewer decisions
- consensus decision
- notes

Analyses should report both original and canonicalized keyword statistics, with
sensitivity checks showing whether substantive conclusions change.

## Paper Wording

Suggested methods wording:

> To reduce artificial fragmentation in normalized keyword labels, we performed
> a duplicate-label reconciliation step within each CSM dimension. Candidate
> duplicate labels were retrieved using character n-gram TF-IDF cosine
> similarity, token-overlap similarity, and, when available, sentence-embedding
> cosine similarity. These computational methods were used only to identify
> possible duplicate labels for review. Final merge decisions followed a
> predefined semantic-preservation codebook and were made independently by two
> reviewers blinded to country-level frequencies, with disagreements resolved
> by consensus. All original labels and final mappings were retained for audit,
> and analyses were repeated using both original and canonicalized labels.
