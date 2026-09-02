---
name: med-ai-paper-writing
description: Use when drafting or revising this project's JMIR-style medical AI manuscript, including Methods, Results, Discussion, limitations, abstracts, reviewer responses, and claim calibration. Enforces restrained medical-informatics tone, source-grounded AI terminology, CSM framing, evaluation caveats, and country-platform interpretation boundaries.
---

# Medical AI Paper Writing

Use this guide whenever writing manuscript text for the toothache narrative infodemiology project.

## Core Stance

Write as a medical/public-health informatics paper using AI as an annotation method, not as an AI product demo.

Default framing:

> We developed and evaluated a source-grounded multilingual AI-assisted annotation workflow to extract CSM-based toothache illness narrative units from social media posts.

Prefer restrained claims:

- "model-detected narrative patterns"
- "country-platform corpora"
- "source-supported narrative units"
- "high-confidence extracted units"
- "held-out evaluation"
- "evidence-span support"
- "human-labeled gold standard"

Avoid broad claims:

- "patients in country X"
- "prevalence"
- "the model understands"
- "AI revealed cultural differences"
- "clinical diagnosis"
- "population disease burden"

## Study Identity

Describe the study as:

- retrospective
- multilingual
- social media narrative/infodemiology research
- based on the Common-Sense Model of Self-Regulation
- comparing country-platform corpora, not national populations

Do not refer to old pipelines, prior abandoned architectures, or code history. Write only the frozen current methodology.

## CSM Framework

Use five CSM domains:

- Perceived Cause
- Symptom Description
- Perceived Consequences
- Coping and Management
- Emotional Expression

Do not include social interaction as a CSM domain.

Make clear that CSM organizes user narratives and perceived experiences. It does not establish clinical diagnoses, disease prevalence, or causal mechanisms.

## Pipeline Description

Describe the current pipeline in this order:

1. Data cleaning and harmonization with source text retained.
2. Post-level classification: relevance, experiencer, content function.
3. CSM eligibility routing.
4. RAG retrieval of human-adjudicated examples.
5. Evidence-grounded CSM unit extraction.
6. LLM-as-Judge validation.
7. Deterministic schema and evidence-span checks.
8. Optional recall-rescue pass over first-pass non-eligible rows.

Core rule:

> No extracted unit was accepted for analysis unless it was linked to a locatable source-text evidence span.

Post-level CSM labels are derived from accepted units. Do not describe CSM domain labels as independent black-box post labels.

## Evaluation Language

Always separate:

- routing/classification performance
- CSM domain detection performance
- evidence support
- normalization agreement
- human audit, if available

If recall is low but precision is acceptable, write:

> The pipeline behaved as a high-specificity, lower-sensitivity extractor. Therefore, accepted units were interpreted as high-confidence model-detected narrative signals, whereas non-extracted posts were not treated as true negatives.

For downstream corpus findings, prefer:

> among accepted high-confidence units

Avoid:

> among all users

Use bootstrap confidence intervals when reporting held-out metrics.

## Rescue Pass

Describe rescue as additive and auditable:

> A second-pass high-recall classifier was applied to selected first-pass non-eligible rows to identify likely missed CSM narratives. Rows passing the rescue gate were processed through the same extraction and judge stages as the primary pipeline. Original first-pass outputs were retained, and rescue outputs were combined additively.

Do not imply rescue reannotated all rows unless it actually did.

## Country-Platform Interpretation

Always state that China/Xiaohongshu, Japan/X-Twitter, and Korea/X-Twitter are country-platform corpora.

Acceptable:

- "The Japanese study corpus contained more model-detected symptom units..."
- "Differences among corpora may reflect platform affordances, language, search retrieval, and user behavior."

Avoid:

- "Japanese users experience more..."
- "Chinese patients are more likely to..."
- "Culture caused..."

## Methods Tone

Use precise, plain, journal-style prose.

Good:

> Cleaning was designed to preserve semantic evidence, including negation, uncertainty, emojis, punctuation, medication names, and colloquial expressions.

Too promotional:

> Our state-of-the-art AI framework accurately understands cross-cultural dental pain at scale.

## Results Tone

Lead with measured findings and uncertainty. Report validation limitations next to substantive results when relevant.

When domain/language performance is uneven, say so directly and restrict interpretation.

Do not overinterpret weak domains or low-recall outputs as prevalence estimates.

## Discussion Tone

Emphasize:

- feasibility of source-grounded multilingual annotation
- medical narrative insights from high-confidence units
- need for human audit and domain/language-specific caution
- limitations of social media data and model-detected patterns

Do not claim clinical generalizability.

## Ethics

Include:

- public or accessible social media posts
- de-identification
- no contact with users
- aggregate reporting
- no clinical decision-making
- caution with verbatim quotations

## Quick Checklist Before Finalizing Any Section

- Does the text say "country-platform corpus" where appropriate?
- Does it avoid population prevalence claims?
- Does it distinguish accepted units from true absence?
- Does it mention source-text evidence spans for extraction?
- Does it avoid acknowledging abandoned/old code paths?
- Are model claims aligned with held-out evaluation results?
- Are limitations stated plainly rather than hidden?
