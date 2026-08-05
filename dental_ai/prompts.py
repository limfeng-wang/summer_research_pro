"""Frozen prompt templates for source-first multilingual annotation."""

RELEVANCE_PROMPT = """\
You are screening multilingual social-media posts for a toothache narrative study.

Use the original source text as authoritative. Label only relevance:
- R1: explicitly concerns toothache, tooth/gingival pain, pain-related treatment, or response to a specific toothache experience.
- R0: unrelated keyword noise, unrelated reposting, hashtag stuffing, or ambiguous non-toothache use.
- RU: relevance cannot be determined reliably from the source text.

Return strict JSON:
{"relevance_label":"R1|R0|RU"}
"""


R1_CLASSIFICATION_PROMPT = """\
You are classifying a post already screened as R1 toothache-relevant.

Assign:
1. experiencer_label:
- E1: author describes their own toothache, treatment, recovery, or care-seeking.
- E2: a specific other person has the toothache experience.
- E3: no specific experiencer; general knowledge, service, product, institution, or generalized statement.

2. content_function:
- C1: experience-sharing narrative.
- C2: genuine question or help-seeking.
- C3: reusable health knowledge.
- C4: advertising or commercial content.
- C5: other low-information toothache-related content.

Priority for content function:
C4 > C2 > C3 > C1 > C5.

Return strict JSON:
{"experiencer_label":"E1|E2|E3","content_function":"C1|C2|C3|C4|C5"}
"""


CSM_EXTRACTION_PROMPT = """\
You extract source-supported CSM narrative units from multilingual toothache posts.

Use original source text as authoritative. Chinese working-language glosses are allowed.
Every evidence_span_original must be copied exactly from the source text. No evidence span, no extraction.

CSM domains:
- Perceived Cause
- Symptom Description
- Perceived Consequences
- Coping and Management
- Emotional Expression

For surface_text_working, write a short Chinese annotator-facing gloss.
Set working_language to "zh".
Set normalized_concept_en to a concise English concept. Do not introduce diagnoses or causal claims absent from the source text.

Return strict JSON matching:
{
  "post_id": "string",
  "country": "CHI|JPN|KOR",
  "language": "zh|ja|ko",
  "units": [
    {
      "domain": "...",
      "evidence_span_original": "exact source span",
      "surface_text_working": "Chinese working gloss",
      "working_language": "zh",
      "normalized_concept_en": "English concept",
      "concept_status": "existing_dictionary|new_candidate|unmapped",
      "support_type": "explicit|implicit|unsupported",
      "assertion": "present|negated|uncertain|planned",
      "temporality": "past|current|future|unknown",
      "sentiment_or_outcome": "effective|ineffective|positive|negative|neutral|unknown",
      "confidence": 0.0,
      "judge_verdict": "needs_human_review"
    }
  ]
}
"""


JUDGE_PROMPT = """\
You are judging a proposed structured annotation for a multilingual toothache post.

Use original source text as authoritative. Check only these criteria:
1. Does every evidence_span_original appear exactly in the source text?
2. Is each CSM domain supported by its evidence span?
3. Does normalized_concept_en avoid overinterpretation?
4. Are assertion, temporality, and sentiment_or_outcome supported?
5. Are there unsupported units that should be rejected?

Return the same JSON structure with judge_verdict set per unit:
- accept
- revise
- reject
- needs_human_review
"""


__all__ = [
    "CSM_EXTRACTION_PROMPT",
    "JUDGE_PROMPT",
    "R1_CLASSIFICATION_PROMPT",
    "RELEVANCE_PROMPT",
]
