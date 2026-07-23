# Project Overview

## Goal

This project studies how social media users in China, Japan, and South Korea describe dental pain. The scientific goal is to characterize and compare the cognitive, emotional, behavioral, and social domains of dental pain experience across three cultural and platform contexts.

The practical challenge is that reading and annotating thousands of multilingual posts is repetitive, slow, and expensive for human experts. This project therefore evaluates whether a small expert-labeled set can calibrate AI systems to extract reliable, culturally meaningful patterns at scale.

## Research Questions

- What do users believe caused their dental pain?
- What symptoms do they describe?
- How does pain affect sleep, eating, work, study, or mood?
- What coping actions do users report?
- What emotions do users express?
- Do users ask for advice, share reassurance, complain, warn others, or thank clinicians?
- How do these patterns differ across China, Japan, and South Korea?

## Conceptual Framework

The project uses the Common-Sense Self-Regulation Model (CSM) to organize dental pain narratives into six domains:

| CSM Domain             | Meaning                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| Perceived Cause        | What users believe caused the pain                                  |
| Symptom Description    | How users describe pain or oral conditions                          |
| Perceived Consequences | How pain disrupts life                                              |
| Coping and Management  | What users do in response                                           |
| Emotional Expression   | Feelings such as fear, anger, distress, relief                      |
| Social Interaction     | Advice-seeking, experience-sharing, gratitude, complaints, warnings |

## Core Method

The central technical object is not a free-form LLM summary. It is an evidence-grounded narrative unit.

A single post can contain multiple narrative units. Each unit represents one evidence-backed claim inside the post. For example, one post may mention a perceived cause, a symptom, a life disruption, and two coping actions. These should be stored as separate units rather than collapsed into one summary.

For example, given the post:

> 智齿发炎疼得睡不着，吃了布洛芬还是没用，明天去拔牙。

The system should extract structured units:

| Domain                 | Evidence Span | Surface Text | Normalized Concept    |
| ---------------------- | ------------- | ------------ | --------------------- |
| Perceived Cause        | 智齿发炎      | 智齿发炎     | Inflamed wisdom tooth |
| Symptom Description    | 疼得睡不着    | 疼           | Severe dental pain    |
| Perceived Consequences | 睡不着        | 睡不着       | Sleep disturbance     |
| Coping and Management  | 吃了布洛芬    | 布洛芬       | Oral analgesic        |
| Coping and Management  | 明天去拔牙    | 拔牙         | Tooth extraction      |

The key rule is:

> No evidence span, no extraction.

This keeps the analysis grounded in what users actually wrote and reduces unsupported clinical inference.

Post-level CSM labels are derived from accepted units. If a post contains at least one accepted unit assigned to `Coping and Management`, then the post is positive for the Coping and Management domain. This means Level 1 classification is not a separate black-box decision; it is grounded in the extracted Level 2 evidence.

## Formal Extraction Schema

Each accepted extraction should follow a fixed schema. The schema preserves the original user expression, the standardized analytic concept, and the exact textual evidence.

```json
{
  "post_id": "string",
  "country": "CHI | JPN | KOR",
  "language": "zh | ja | ko",
  "units": [
    {
      "unit_id": "string",
      "domain": "Perceived Cause | Symptom Description | Perceived Consequences | Coping and Management | Emotional Expression | Social Interaction",
      "evidence_span": "string copied exactly from the post",
      "surface_text": "minimal extracted phrase in the original language",
      "normalized_concept": "standardized English concept",
      "concept_status": "existing_dictionary | new_candidate | unmapped",
      "support_type": "explicit | implicit | unsupported",
      "assertion": "present | negated | uncertain | planned",
      "temporality": "past | current | future | unknown",
      "sentiment_or_outcome": "effective | ineffective | positive | negative | neutral | unknown",
      "confidence": 0.0,
      "judge_verdict": "accept | revise | reject | needs_human_review"
    }
  ]
}
```

Field definitions:

| Field                    | Level | Meaning                                                                   | Why It Is Needed                                                                           |
| ------------------------ | ----- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `post_id`              | Post  | Unique identifier for the source post                                     | Links extractions back to the original text and metadata                                   |
| `country`              | Post  | Country/platform context: China, Japan, or South Korea                    | Enables cross-country comparison                                                           |
| `language`             | Post  | Original language of the post                                             | Supports language-specific error analysis and normalization                                |
| `units`                | Post  | List of extracted narrative units from the post                           | A post may contain multiple causes, symptoms, coping actions, emotions, or social signals  |
| `unit_id`              | Unit  | Unique identifier for one extracted unit                                  | Allows auditing, deduplication, and traceability                                           |
| `domain`               | Unit  | CSM domain assigned to the unit                                           | Connects Level 2 extraction to Level 1 domain analysis                                     |
| `evidence_span`        | Unit  | Exact text copied from the original post                                  | Grounds the extraction and prevents unsupported inference                                  |
| `surface_text`         | Unit  | Minimal phrase being extracted from the evidence span                     | Preserves what the user actually wrote before normalization                                |
| `normalized_concept`   | Unit  | Standardized English concept used for analysis                            | Enables comparison across Chinese, Japanese, and Korean posts                              |
| `concept_status`       | Unit  | Whether the concept is from the dictionary, newly discovered, or unmapped | Prevents culturally specific expressions from being forced into existing categories        |
| `support_type`         | Unit  | Whether the evidence is explicit, implicit, or unsupported                | Separates high-confidence literal mentions from inferred or rejected units                 |
| `assertion`            | Unit  | Whether the concept is present, negated, uncertain, or planned            | Distinguishes "I took ibuprofen" from "I might take ibuprofen" or "ibuprofen did not help" |
| `temporality`          | Unit  | Whether the event is past, current, future, or unknown                    | Helps interpret coping and treatment narratives correctly                                  |
| `sentiment_or_outcome` | Unit  | Whether the unit carries an outcome or evaluative signal                  | Captures details such as ineffective medication, relief, dissatisfaction, or gratitude     |
| `confidence`           | Unit  | Model-estimated confidence score                                          | Used for filtering, sensitivity analysis, and audit prioritization                         |
| `judge_verdict`        | Unit  | LLM judge decision on whether the unit is acceptable                      | Records whether the extraction passed verification                                         |

Required constraints:

- `evidence_span` must appear in the original post.
- `surface_text` must be supported by the evidence span.
- `normalized_concept` must not introduce a diagnosis or causal claim absent from the post.
- `support_type = unsupported` units are not used in downstream medical or network analyses.
- The original surface phrase is always preserved, even when a normalized concept is assigned.

Schema concision:

This schema is intentionally compact. It avoids free-form reasoning text in the final analytic table because reasoning is expensive, inconsistent, and hard to aggregate. If reasoning is needed for debugging, it can be stored in a separate trace file rather than the main concept table.

The required analytic fields are:

- `domain`
- `evidence_span`
- `surface_text`
- `normalized_concept`
- `concept_status`
- `support_type`
- `assertion`
- `judge_verdict`

The recommended metadata fields are:

- `post_id`
- `country`
- `language`
- `unit_id`
- `confidence`

The optional but useful interpretation fields are:

- `temporality`
- `sentiment_or_outcome`

If implementation needs to be simplified, `temporality` and `sentiment_or_outcome` can be omitted in the first version. The rest should remain because they directly support validation, normalization, or cross-country analysis.

## Pipeline

```text
Raw social media posts
        |
        v
Data cleaning and de-identification
        |
        v
Small expert-labeled gold set
        |
        v
RAG few-shot LLM extraction
        |
        v
Open-concept discovery and normalization
        |
        v
LLM-as-Judge verification
        |
        v
Validated structured concept table
        |
        v
Cross-country comparison and network analysis
```

## Relationship Between Classification and Extraction

The project has two analytic levels, but they are not separate final goals.

Level 1 is broad CSM domain classification. It answers whether a post contains a cause, symptom, consequence, coping action, emotion, or social interaction signal.

Level 2 is grounded concept extraction. It answers what exact cause, symptom, consequence, coping action, emotion, or social interaction is present, and where the evidence appears in the text.

Operationally, the preferred design is:

```text
Post
  |
  v
Extract grounded narrative units
  |
  v
Assign each unit to a CSM domain
  |
  v
Derive post-level CSM domain labels
```

For example, if a post contains an accepted "ibuprofen" coping unit, the post is positive for Coping and Management. This avoids treating Level 1 classification as a hard gate that could block downstream extraction.

## Model Comparison

The project compares three label-efficient annotation strategies.

```text
                         100 expert gold-dev posts
                                      |
        ---------------------------------------------------------
        |                         |                             |
        v                         v                             v
RAG few-shot LLM          Silver-trained student        Gold-only student
+ LLM-as-Judge            trained on 4.8k AI labels     trained on 100 gold labels
        |                         |                             |
        ---------------------------------------------------------
                                      |
                                      v
                         100 locked expert gold-test posts
                                      |
                                      v
                         Compare performance, cost, and errors
```

### 1. RAG Few-Shot LLM + LLM-as-Judge

This system uses the expert-labeled development examples, retrieved similar examples, and a structured extraction schema to produce evidence-grounded narrative units. A separate LLM judge verifies whether each extracted unit is supported by the post, correctly assigned to a CSM domain, and normalized appropriately.

### 2. Silver-Trained Student Model

This model is trained on approximately 4.8k outputs produced by the LLM-as-Judge pipeline. It tests whether expensive LLM-based extraction can be distilled into a cheaper and faster model for large-scale use.

### 3. Gold-Only Student Model

This model is trained only on the 100 expert-labeled development posts. It tests whether a small gold set alone is sufficient for supervised learning.

Whichever method performs best becomes the main technical narrative. The others become supplementary evidence.

## Open-Concept Discovery and Normalization

The project should not force all Level 2 concepts into a closed dictionary at the moment of extraction. A closed vocabulary can erase culturally specific language, slang, platform-specific expressions, and novel coping behaviors.

The preferred approach is:

```text
Open grounded phrase extraction
        |
        v
Preserve surface phrases in the original language
        |
        v
Cluster semantically similar phrases within and across languages
        |
        v
Map clusters to existing dictionary concepts where appropriate
        |
        v
Flag unmapped clusters as new candidate concepts
        |
        v
Expert review of top concepts, new candidates, and ambiguous mappings
```

This produces two linked outputs:

1. A standardized concept table for cross-country comparison.
2. A preserved surface-phrase table for cultural and linguistic interpretation.

Normalization rules:

- Exact dictionary matches are preferred when available.
- Semantic matches may be accepted only when the evidence span clearly supports the normalized concept.
- Ambiguous, idiomatic, or culturally specific expressions should be flagged for expert review rather than forced into the nearest existing concept.
- New recurring concepts should be retained as `new_candidate` until reviewed.
- Normalization should distinguish user-attributed causes from clinical diagnoses.

Example:

| Surface Phrase | Language | Normalized Concept               | Concept Status                    |
| -------------- | -------- | -------------------------------- | --------------------------------- |
| 智齿           | zh       | Wisdom tooth                     | existing_dictionary               |
| 親知らず       | ja       | Wisdom tooth                     | existing_dictionary               |
| 사랑니         | ko       | Wisdom tooth                     | existing_dictionary               |
| 牙疼到怀疑人生 | zh       | Severe pain + emotional distress | new_candidate or reviewed mapping |

## Validation Design

The project separates development, final testing, and production-stage audit.

| Set          |      Size | Purpose                                                          |
| ------------ | --------: | ---------------------------------------------------------------- |
| Gold-dev     | 100 posts | Rubric design, examples, threshold selection, method development |
| Gold-test    | 100 posts | Locked final evaluation                                          |
| Silver-audit | 100 posts | Expert audit of production-style LLM-labeled outputs             |

The same expert-labeled posts should not be used for both method tuning and final claims.

In addition to the 100-post silver audit, experts should review:

- the most frequent extracted concepts in each country,
- all concepts highlighted in the Results or Discussion,
- new candidate concepts produced by open-concept discovery,
- and the highest-weight PMI network edges used for cross-cultural interpretation.

## Evaluation Metrics

### Level 1: CSM Domain Classification

- Accuracy
- Precision
- Recall
- F1-score
- Cohen kappa
- Per-domain performance
- Per-country and per-language performance

### Level 2: Grounded Concept Extraction

- Concept precision
- Concept recall
- Evidence-span support rate
- Normalization accuracy
- Hallucination rate
- Missed concept rate
- Per-country and per-language extraction error

### Robustness and Uncertainty

- Bootstrap confidence intervals for Level 1 and Level 2 metrics
- Confidence intervals for country-level concept prevalence
- Sensitivity analysis across confidence thresholds
- Sensitivity analysis with and without implicit extractions
- Bootstrap stability of PMI network edges
- Expert review of top concepts and top network edges
- Error analysis by country, language, platform, post length, emoji/slang use, and CSM domain

The Level 2 metrics are essential because the medical and cultural discussion depends on extracted concepts, not only on broad domain labels.

## Downstream Analysis

After validation, the accepted concept table supports the main substantive analyses:

```text
Validated extracted concepts
        |
        v
Frequency by country and CSM domain
        |
        v
High-confidence concept prevalence
        |
        v
Co-occurrence and PMI networks
        |
        v
Bootstrap stability and top-edge expert review
        |
        v
Cross-cultural interpretation
```

The analysis is designed to identify social-media narrative patterns, such as whether users in one country more often mention self-management, severe pain, treatment costs, sleep disruption, advice-seeking, or emotional distress.

Network findings should be treated as robust only when the relevant concepts and edges remain stable under threshold and bootstrap sensitivity checks.

## Interpretation Boundaries

The project analyzes self-reported social media narratives. It does not establish clinical prevalence, diagnosis, or causality.

Safe claims:

- Chinese posts more often mentioned self-management and traditional remedies.
- Japanese posts more often emphasized functional disruption or psychological burden.
- South Korean posts more often contained treatment-oriented questions.
- High-confidence extracted concepts formed distinct co-occurrence patterns across countries.

Unsafe claims:

- Chinese patients clinically rely more on traditional medicine.
- Japanese users have more stress-induced dental disease.
- South Korean users have higher wisdom-tooth disease prevalence.
- Healthcare-system differences caused the observed social media patterns.

The appropriate framing is validated narrative infodemiology, not clinical epidemiology.

## Intended Contributions

### Public Health Contribution

The project characterizes how dental pain is perceived, expressed, and managed across three East Asian social media contexts.

### Med-AI Contribution

The project evaluates a label-efficient AI pipeline for multilingual patient-narrative extraction using:

- a small expert-labeled set,
- retrieval-augmented few-shot extraction,
- LLM-as-Judge verification,
- silver-label distillation,
- locked expert testing,
- and expert audit of production-stage outputs.

## One-Sentence Summary

This study uses a small amount of expert annotation to calibrate AI systems that extract evidence-grounded dental-pain narratives from thousands of multilingual social media posts, then compares China, Japan, and South Korea in terms of perceived causes, symptoms, consequences, coping behaviors, emotions, and social interaction.
