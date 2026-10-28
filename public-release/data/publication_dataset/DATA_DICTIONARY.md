# Private-data schema dictionary

The records described below are **not publicly distributed**. These definitions
document the formats expected by the original-data analysis and inference code.
Public aggregate metadata is in `metadata.json`.

## Shared post fields

| Field | Meaning |
| --- | --- |
| `post_id` | Release ID; unique across all three groups |
| `country`, `language`, `platform` | Corpus code, language, and platform |
| `title`, `text` | Retained original title and body |
| `source_text` | Authoritative text used for evidence checks; cleaned text when available, otherwise title/body joined with a newline |

IDs are stable for this package version and do not expose original record IDs.

## Development

`roles` contains `classification`, `extraction`, or both.

- `classification_reference` contains human E/C labels, an evidence excerpt,
  rationale, and original `example_order`. Relevance labels were not stored in
  the classification source and are not invented here.
- `extraction_reference` contains original `example_order`, human R/E/C labels,
  and annotated CSM `units`.
- An unavailable role is `null`, not an empty or negative annotation.

Preserve `example_order` when reconstructing either example pool; file order is
not the classifier or retrieval reference order. Shared posts retain their two
reference annotations separately, without forcing agreement between sources.

## Evaluation annotations

`post_id` joins each annotation to exactly one evaluation input.
`classification` contains human R/E/C labels; unavailable labels are `null`.
`reference_source_text` preserves the human workbook's title/body text, and
`reference_text_matches_input` indicates exact agreement with model input text.
There are 51 variants: 50 Chinese title/body formatting differences and one
Korean text difference. Neither version is silently overwritten.
Each of five `domains` contains:

| Field | Meaning |
| --- | --- |
| `present` | Whether the human domain-label cell is populated |
| `original_label` | Original human domain-label wording |
| `evidence_text` | Original human evidence cell; not assumed to be one exact span |
| `concepts_en_text` | Original English concept cell, preserved without splitting or renaming |

These are domain-level references, not reconstructed unit annotations.

## Main corpus

`classification` contains automated R/E/C labels. `units` preserves all available
proposed units, including rejected units. An empty list means no units are stored;
it does not independently establish absence of all CSM content.

| Label | Meaning |
| --- | --- |
| R1 / R0 / RU | Dental-pain relevant / unrelated / uncertain |
| E1 / E2 / E3 | Author / specific other person / no identifiable experiencer |
| C1 / C2 / C3 / C4 / C5 | Experience sharing / help seeking / educational / commercial / other or low-information |

The five CSM domains are Perceived Cause, Symptom Description, Perceived
Consequences, Coping and Management, and Emotional Expression.

## Unit fields and flags

Original unit fields are preserved where available: `domain`,
`evidence_span_original`, surface text/gloss, `working_language`,
`normalized_concept_en`, `concept_status`, `support_type`, `assertion`,
`temporality`, `sentiment_or_outcome`, `confidence`, and `judge_verdict`.
Missing fields remain absent. `unit_id` is replaced with a release-specific ID.
For human extraction references, verdict/confidence values are inherited metadata,
not an independent new model or human validation performed during packaging.

| Quality flag | Exact rule |
| --- | --- |
| `judge_accepted` | `judge_verdict == "accept"` |
| `schema_accepted` | Judge accepted, support not `unsupported`, assertion not `negated` |
| `evidence_span_found` | Nonempty evidence span occurs verbatim in `source_text` |

The flags describe the stored annotations, not guaranteed semantic accuracy.

| Main-post analysis flag | Exact rule |
| --- | --- |
| `eligible_first_person` | R1, E1, and C1 or C2 |
| `manuscript_primary` | Eligible and at least one verdict-accepted unit |
| `strict_primary` | Eligible and at least one schema-accepted unit with an exact source span |

Use `manuscript_primary` to reproduce the reported 12,829-post sample. Unit-level
filtering is still required: a selected post can contain rejected units. The
reported 35,627-unit count uses verdict-only acceptance within eligible posts.
