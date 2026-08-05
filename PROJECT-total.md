# Project Overview (PROJECT-3)

## Research Objectives

This project investigates how social media users in China, Japan, and South Korea describe toothache, with a focus on perceived causes, symptom experiences, effects on daily life, coping behaviors, and emotional expressions within these narratives.

The study has two objectives:

1. **Medical and public health objective**: To describe and compare the characteristics of toothache illness narratives across three country-platform corpora.
2. **Medical artificial intelligence objective**: To evaluate whether a relatively small, high-quality human-annotated gold standard can be used to calibrate large language models (LLMs) and lightweight text classification models for reliable extraction of structured, source-supported information from thousands to tens of thousands of multilingual posts.

This study analyzes self-reported narratives on social media. It is not intended to estimate population-level toothache prevalence, make clinical diagnoses, or support causal inference.

## Research Questions

- What do users believe causes their toothache?
- How do users describe pain, oral symptoms, and related physical sensations?
- How does toothache affect sleep, eating, work, study, and everyday life?
- What self-management, healthcare-seeking, and treatment behaviors do users report?
- What toothache-related fear, anxiety, distress, anger, reassurance, or relief do users express?
- How do these narrative patterns differ across the Chinese, Japanese, and South Korean study corpora?
- Under a small human gold-standard setting, which automated annotation strategy provides the best balance among accuracy, cost, and reproducibility?

## Conceptual Framework

This project uses the Common-Sense Model of Self-Regulation (CSM) to organize toothache illness narratives. Under the current study design, the original six domains have been reduced to five by removing social interaction as an independent domain.

| CSM domain             | Operational meaning                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Perceived Cause        | What the user believes caused or triggered the toothache                                              |
| Symptom Description    | How the user describes toothache, oral abnormalities, and related physical sensations                 |
| Perceived Consequences | How toothache affects sleep, eating, work, study, activities, or quality of life                      |
| Coping and Management  | Actions taken by the user to relieve, investigate, or treat the toothache                             |
| Emotional Expression   | Expressions of fear, anxiety, helplessness, anger, distress, reassurance, relief, or related emotions |

Social interaction is no longer treated as a CSM domain because help-seeking, experience sharing, health education, commercial communication, and general interaction are already addressed by the upstream **content-function classification**. Retaining it as a CSM domain would conflate why a post was published with the medical or psychological content contained in the illness narrative.

Help-seeking behavior remains represented. It is first captured as content function `C2 Question or Help-Seeking`. If the help-seeking post also contains evidence of the author's symptoms, effects on daily life, or intended coping actions, those source-text spans may still be assigned to the corresponding five CSM domains.

## Data Sources and Sampling Framework

### Platforms and Countries

| Study corpus | Country/region | Platform    | Original language |
| ------------ | -------------- | ----------- | ----------------- |
| CHI          | China          | Xiaohongshu | Chinese           |
| JPN          | Japan          | X/Twitter   | Japanese          |
| KOR          | South Korea    | X/Twitter   | Korean            |

The same mainstream social media platform cannot be used across China, Japan, and South Korea. Therefore, the three-country comparison is, in practice, a comparison among **country-platform combinations**. The results may describe differences among the three study corpora, but not all observed differences can be directly interpreted as cultural or healthcare-system differences.

### Study Period

The common study period is:

```text
2023-07-01 to 2026-06-30, inclusive
```

Posts published before July 1, 2023, or after June 30, 2026, will be strictly excluded. Posts whose publication dates cannot be recovered reliably will not enter the primary analysis but will be retained in the audit table.

### Monthly Sampling and Retrieval

For each month within the study period, one date will be selected at random. All accessible posts from that date matching the prespecified, frozen toothache search terms will then be retrieved from each platform.

To ensure reproducibility:

- A fixed random seed will be used to generate the monthly sampling dates, and the complete date list will be retained.
- In principle, the same set of monthly dates will be used for all three study corpora.
- Core retrieval will use broad, high-recall toothache expressions.
- Targeted searches for terms such as wisdom tooth, pulpitis, and root canal treatment will be used only to expand the discovery corpus and will not be included directly in the denominator for primary proportion estimates.
- Search terms, collection time, collection batch, and collector version will be retained.
- Existing platform post IDs will be checked during collection to prevent duplicate requests and duplicate writes to the cumulative data pool.
- Each collection run will append incrementally to the existing data pool. Historical data will be retained without creating unnecessary duplicate copies.

This design is a month-stratified, date-sampled retrieval of accessible posts rather than a probability sample of all posts on each platform. Platform search ranking, interface limitations, content deletion, and account visibility may result in incomplete capture and must be reported as study limitations.

## Data Cleaning and Corpus Freezing

### Core Principles

Cleaning must be reproducible, auditable, and designed to minimize semantic loss. Raw files will remain read-only. All processed data will be written to new derivative files, with the cleaning version, source-file hash, input row count, output row count, and record-level exclusion reason retained.

Cleaning and screening are separate stages:

- **Cleaning** addresses formatting, dates, technically invalid records, and duplicate records. It does not determine whether a post belongs to the target medical narrative corpus.
- **Screening and classification** determine toothache relevance, experiencer, and content function.

### Harmonized Fields

The three corpora will be converted to a harmonized long-table structure with one post per row and at least the following fields:

| Field                        | Meaning                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| `record_id`                | Unique internal study identifier                               |
| `platform_post_id`         | Platform post identifier                                       |
| `country`                  | `CHI`, `JPN`, or `KOR`                                   |
| `platform`                 | `xiaohongshu` or `x_twitter`                               |
| `language`                 | `zh`, `ja`, or `ko`                                      |
| `published_at`             | Cleaned publication time                                       |
| `published_time_precision` | Whether the date is an exact source value or an inferred value |
| `collected_at`             | Collection time                                                |
| `query_term`               | Matched search term                                            |
| `original_title`           | Original title; may be empty for platforms without titles      |
| `original_text`            | Original post text                                             |
| `text_clean`               | Evidence-preserving cleaned text                               |
| `analysis_text_en`         | Fixed-version English translation                              |
| `author_id_hash`           | Locally salted hash of the author identifier                   |
| `source_batch`             | Collection batch                                               |
| `cleaning_version`         | Cleaning-rule version                                          |
| `text_fingerprint`         | Normalized text hash                                           |
| `analysis_included`        | Whether the record enters the cleaned candidate corpus         |
| `exclusion_reason`         | Reason for exclusion                                           |

Platform URLs are not required for data traceability because links may expire and may increase risks to platforms and users. The primary provenance fields are the internal ID, platform post ID, collection batch, source file, text hash, and cleaning log. URLs in analysis text will be replaced consistently with `[URL]`, with an optional separate `has_url` indicator.

### Evidence-Preserving Text Cleaning

Text cleaning will perform only the following deterministic operations:

1. Decode HTML entities.
2. Apply Unicode NFKC normalization.
3. Standardize newline characters.
4. Remove zero-width characters.
5. Replace URLs with `[URL]`.
6. Collapse consecutive horizontal whitespace.
7. Trim whitespace at the beginning and end of each line.
8. Reduce runs of more than two blank lines to two blank lines.
9. Combine the title and body in a fixed order.

The cleaning stage must not remove or rewrite:

- First-person pronouns or references to people.
- Negation, uncertainty, or temporal expressions.
- Question marks, exclamation marks, or other punctuation.
- Emojis, emoticons, or repeated characters.
- Hashtags, colloquial expressions, spelling errors, or platform-specific language.
- Medication names, treatment names, or the user's own causal explanations.

These elements may provide direct evidence of the experiencer, help-seeking intent, emotional intensity, or uncertainty.

### Recovery of Publication Dates

Platforms such as Xiaohongshu may provide dates in different forms, including a complete date, month and day, a relative number of days or hours ago, yesterday, or today. Date-recovery rules must be fixed in advance:

- Parse complete year-month-day values directly.
- When only month and day are available, use the collection year as the candidate year; if the inferred date is clearly later than the collection date, subtract one year.
- Calculate expressions such as "N days ago," "N hours ago," "N minutes ago," "yesterday," and "today" relative to the collection time.
- Record date precision and parsing success. Do not represent an inferred date as an exact platform-provided value.
- Mark unparseable values as `published_time_missing_or_unparsed`; these records will not enter the primary time-window analysis.

### Technical Validity and Deduplication

A technically valid record must contain at least a nonempty platform post ID and nonempty body or combined text. Records with missing author IDs may be retained but cannot be used for author-level clustering analyses.

Deduplication will proceed in the following order:

1. **Platform-ID deduplication**: retain one representative record for each identical platform and post ID.
2. **Exact normalized full-text deduplication**: collapse whitespace in the combined cleaned title and body, calculate a SHA-256 fingerprint, and retain one record for each identical fingerprint.
3. **Representative-record selection**: prioritize the earliest parseable publication date, followed by stable sorting by collection time and original row number.
4. **Near-duplicate detection**: generate similarity scores and candidate duplicate groups only. Near duplicates will not be deleted automatically in the first version, because template-like but substantively different personal experiences could otherwise be removed incorrectly. Versioned rules may be introduced after confirmation.

Deduplicated records will not disappear physically from audit data. Their duplicate-group ID, representative-record ID, and exclusion reason will be retained. Cleaning outputs will include at least:

- `cleaned_data`: unique records entering subsequent screening.
- `all_rows_audit`: all raw records and their processing status.
- `excluded_rows`: excluded records and reasons.
- `cleaning_report`: counts for input, time filtering, deduplication, and output.
- `duplicate_rows`: duplicate groups.
- `date_issues`: records with missing or unparseable dates.

### Translation (Tool to Be Determined)

Cleaning and deduplication will be completed in the original language. A fixed-version English translation, `analysis_text_en`, will then be generated for each unique record.

Translation records must include the model or service name, version, prompt version, run date, and failure status. Translation must not overwrite the source text. English translations may support coarse classification, cross-language retrieval, concept clustering, and model input; however:

- Gold-standard decisions must remain traceable to the source text.
- Evidence spans must be copied verbatim from the source text.
- Translations must not supply an experiencer, diagnosis, cause, or emotion omitted from the source text.
- Borderline Japanese and Korean samples should be reviewed by annotators who can read the source language directly or by native-language reviewers.

## Post Screening and Coarse Classification

### Level 0: Toothache Relevance

The first step is to determine whether the post concerns toothache or related oral symptoms, management, or healthcare use.

| Label             | Definition                                                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `R1 Relevant`   | The source text explicitly concerns toothache, tooth or gingival pain, pain-related treatment, or a response to a specific toothache experience |
| `R0 Irrelevant` | The post entered the data because of keyword ambiguity, hashtag stuffing, unrelated reposting, or noise                                         |
| `RU Uncertain`  | Relevance cannot be determined reliably because the translation or context is insufficient                                                      |

`R0` records will not enter subsequent analysis but will remain in the exclusion log. `RU` records require human review and must not be deleted automatically as negative cases. Content unrelated to toothache must not be assigned to `C5 Other`.

### Level 1: Experiencer

Toothache-related posts will be classified according to whose specific experience is described:

| Code   | Category                | Decision rule                                                                                                                                                                                                           |
| ------ | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `E1` | Author                  | The author explicitly describes their own current or previous toothache, healthcare visit, treatment, or recovery. The subject may be omitted in the source language, but the context must clearly identify the author. |
| `E2` | Specific other person   | The post explicitly describes the toothache-related experience of a family member, friend, identifiable patient, or person being addressed directly.                                                                    |
| `E3` | No specific experiencer | No identifiable person with a toothache experience is present; the content primarily contains general knowledge, services, products, institutional information, or generalized statements.                              |

Replying to or mentioning an account with `@` does not automatically establish `E2`. The post is labeled `E2` only when its text clearly refers to or describes that person's toothache, extraction, treatment, or recovery. Generic references such as "patients," "everyone," or "someone" do not constitute a specific other person.

### Level 2: Content Function

Each toothache-related post will receive one primary content-function label:

| Code   | Category                          | Inclusion criteria                                                                                                                                                                         | Main exclusions                                                                                                         |
| ------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `C1` | Experience-sharing narrative      | Specific symptoms, temporal progression, management, treatment, or outcomes form the main content, with emphasis on what happened                                                          | Help-seeking, reusable knowledge, or commercial solicitation dominates                                                  |
| `C2` | Question or help-seeking          | The author genuinely asks readers for a judgment, explanation, recommendation, or actionable advice that could affect a subsequent decision                                                | Rhetorical questions; interrogative titles whose body is primarily educational, narrative, or promotional               |
| `C3` | Health knowledge sharing          | Provides reusable information about causes, symptom recognition, management, prevention, or healthcare that stands independently of one person's experience                                | Only recounts a personal experience; offers reassurance or advice without reusable information                          |
| `C4` | Advertising or commercial content | The primary purpose includes purchasing, appointment booking, consultation conversion, or promotion of an institution or product, including advertorials disguised as personal experiences | Ordinary patients reporting that they booked, purchased, or visited a clinic; brand mentions without promotional intent |
| `C5` | Other                             | Toothache-related content whose main function is a brief emotional reaction, joke, well-wish, slogan, or low-information interaction                                                       | Content entirely unrelated to toothache                                                                                 |

When multiple functions are present, the following fixed priority is applied:

```text
C4 Commercial > C2 Help-Seeking > C3 Knowledge > C1 Experience > C5 Other
```

This priority determines only the post's primary communicative function and does not change the experiencer. For example, an advertorial framed as the author's own toothache experience may be labeled `E1 + C4`.

### Primary Analysis Corpus

The final analytic focus is illness narratives concerning the authors themselves. Two analysis sets should be frozen in advance:

1. **Primary set**: `R1 + E1 + (C1 or C2)`. C2 is retained because many users currently experiencing toothache and lacking professional care describe symptoms while seeking help. Excluding all C2 posts would systematically omit current suffering and unmet needs.
2. **Strict sensitivity set**: `R1 + E1 + C1`, retaining experience-sharing narratives only.

`C3`, `C4`, and `C5` will not enter the primary CSM illness-narrative analysis, but their counts and proportions will be retained to describe corpus screening and evaluate the classifier. If a C2 post contains no evidence of the author's own symptoms, management, or effects on daily life, it will not enter CSM concept extraction even if it formally constitutes help-seeking.

## Core Method

The core technical object in this project is not an unconstrained LLM-generated summary, but a **narrative unit supported by evidence from the source text**.

A post may contain multiple narrative units. Each unit represents one verifiable claim. For example, a single post may contain one perceived cause, one symptom, one effect on daily life, and two coping behaviors. These elements must be stored separately rather than compressed into an unauditable summary.

Example source text:

> My inflamed wisdom tooth hurts so much that I cannot sleep. Ibuprofen did not work, and I am having the tooth extracted tomorrow.

Structured extraction:

| CSM domain             | Source-text evidence span           | Surface expression    | Normalized concept    |
| ---------------------- | ----------------------------------- | --------------------- | --------------------- |
| Perceived Cause        | inflamed wisdom tooth               | inflamed wisdom tooth | Inflamed wisdom tooth |
| Symptom Description    | hurts so much that I cannot sleep   | hurts                 | Severe dental pain    |
| Perceived Consequences | cannot sleep                        | cannot sleep          | Sleep disturbance     |
| Coping and Management  | Ibuprofen did not work              | Ibuprofen             | Oral analgesic        |
| Coping and Management  | having the tooth extracted tomorrow | tooth extracted       | Tooth extraction      |

The core rule is:

> No valid extraction may be created without a locatable source-text evidence span.

Post-level five-domain labels will be derived from validated narrative units. For example, if a post contains at least one accepted coping unit for ibuprofen, the post is positive for Coping and Management. This approach prevents five-domain classification from becoming an untraceable black-box decision.

## Formal Extraction Structure

Each record will store both upstream classifications and narrative units:

```json
{
  "post_id": "string",
  "country": "CHI | JPN | KOR",
  "language": "zh | ja | ko",
  "relevance_label": "R1 | R0 | RU",
  "experiencer_label": "E1 | E2 | E3",
  "content_function": "C1 | C2 | C3 | C4 | C5",
  "units": [
    {
      "unit_id": "string",
      "domain": "Perceived Cause | Symptom Description | Perceived Consequences | Coping and Management | Emotional Expression",
      "evidence_span_original": "verbatim evidence span copied from the source text",
      "surface_text_original": "minimal expression in the original language",
      "normalized_concept_en": "normalized English concept for cross-language analysis",
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

The following requirements apply:

- `evidence_span_original` must be located exactly within the cleaned source text.
- `surface_text_original` must be directly supported by the evidence span.
- A normalized concept must not introduce a clinical diagnosis or causal relationship absent from the source text.
- Units with `support_type = unsupported` will not enter medical analyses or network analyses.
- Original-language expressions must be retained permanently, even after English standardization.
- Technical failure, model refusal, uncertainty, and true-negative status must use distinct values rather than all being represented as an empty value or zero.

The first version must retain the following analytic fields:

- `domain`
- `evidence_span_original`
- `surface_text_original`
- `normalized_concept_en`
- `concept_status`
- `support_type`
- `assertion`
- `judge_verdict`

`temporality` and `sentiment_or_outcome` have interpretive value but may be optional if they impose excessive implementation burden in the first version. The remaining fields should not be removed.

## Complete Research Workflow

```text
Raw multiplatform posts and cumulative data pool
        |
        v
Read-only raw archive + field harmonization + evidence-preserving cleaning
        |
        v
Strict date filtering + platform-ID/exact-text deduplication + exclusion log
        |
        v
Source-text retention + fixed-version English translation
        |
        v
Toothache relevance classification (R1/R0/RU)
        |
        v
Experiencer classification (E1/E2/E3)
        |
        v
Content-function classification (C1-C5)
        |
        v
Freeze the first-person illness-narrative corpus
        |
        v
Dual-independent-annotation gold development set and locked test set
        |
        v
RAG few-shot LLM evidence extraction + model comparison
        |
        v
Open concept discovery and cross-language standardization
        |
        v
LLM-as-Judge validation + human audit
        |
        v
Five-domain structured concept table
        |
        v
Country-platform comparisons, sensitivity analyses, and co-occurrence networks
```

## Relationship Between Classification and Extraction

The project contains three levels, but they are not competing research endpoints:

1. **Corpus eligibility level**: relevance, experiencer, and content function determine whether a post belongs to the target illness-narrative corpus.
2. **CSM domain level**: determines whether an eligible post contains a perceived cause, symptom, consequence, coping behavior, or emotion.
3. **Evidence-concept level**: extracts the specific content and its location in the source text and maps it to a cross-language normalized concept.

Operationally, evidence-supported units should be extracted first, and post-level five-domain labels should then be derived from those units:

```text
Eligible post
  |
  v
Extract source-text evidence units
  |
  v
Assign one CSM domain to each unit
  |
  v
Validate the evidence and concept mapping
  |
  v
Derive post-level five-domain multilabels
```

Upstream content-function classification cannot replace CSM extraction. For example, `C2 Question or Help-Seeking` describes why a post was published, whereas "toothache lasting for three days," "unable to sleep," and "planning to visit a hospital" belong to the symptom, consequence, and coping domains, respectively.

## Human Gold Standard

### Freeze the Codebook First

Before formal dual annotation, the current trilingual exemplar-candidate workbook should be used to discuss and freeze:

- The definition of toothache relevance.
- The `E1-E3` experiencer definitions.
- The `C1-C5` content-function definitions and conflict-resolution priority.
- Inclusion, exclusion, and borderline cases for the five CSM domains.
- The minimum-length rule for source-text evidence spans.
- The states for uncertainty, rejection, and adjudication.

The current candidate exemplars cover all marginal categories in Chinese and Japanese. Korean `C4` has only two clear commercial posts, so at least eight additional examples must be collected through targeted sampling. Ordinary patient narratives about appointments or healthcare use must not be used merely to fill this category.

### Gold-Standard Split

If approximately 300 posts can feasibly receive dual independent annotation, the following split is recommended:

| Dataset   | Recommended size | Purpose                                                                                             |
| --------- | ---------------: | --------------------------------------------------------------------------------------------------- |
| Gold-dev  |        150 posts | Definition revision, few-shot examples, prompt development, threshold selection, and error analysis |
| Gold-test |        150 posts | Final independent evaluation after pipeline freezing; must not be used to modify prompts            |

Both sets will be stratified by language, with each language contributing approximately one-third. Rare categories and five-domain-positive samples may be supplemented through moderate stratification. If sampling from the natural distribution yields too few positive examples for a critical label, a separately identified challenge set should be added, but its performance must not be combined with natural-distribution performance.

After full-corpus automated annotation, an additional 50-100 model outputs will be sampled for production-stage auditing. This audit is intended to identify errors under the deployment distribution and must not be mixed with the locked test set. If human resources are limited, two researchers may prioritize high-frequency concepts, new concepts, high-uncertainty samples, and findings that will receive emphasis in the manuscript.

### Dual Independent Annotation and Adjudication

The two annotators should work independently without access to each other's decisions.

Recommended procedure:

1. Both annotators independently read the source text; the English translation is used only as an aid.
2. Each annotator independently labels relevance, experiencer, content function, the five CSM domains, and evidence spans.
3. Calculate raw agreement and Cohen's kappa; also report Gwet's AC1 when class imbalance is substantial.
4. Export disagreements without altering the original independent annotations.
5. A prespecified adjudicator applies the frozen codebook to produce the final gold standard.
6. If agreement is unacceptably low for a category, return to the development set and revise the definition. The locked test set must not be used repeatedly to revise rules.

## Model Comparison

This project compares three annotation-efficient strategies. Coarse classification and subsequent LLM-based concept extraction are distinct tasks. They do not need to use the same model, and independent validation must not be skipped merely because an LLM is multilingual.

```text
Gold-dev
   |
   +--------------------------+--------------------------+
   |                          |                          |
   v                          v                          v
RAG few-shot LLM       Silver-label student       Gold-only student
+ LLM-as-Judge         model trained on           model trained only
                       audited AI labels           on Gold-dev
   |                          |                          |
   +--------------------------+--------------------------+
                              |
                              v
                       Locked Gold-test
                              |
                              v
             Compare performance, cost, latency,
                       and error profiles
```

### 1. RAG Few-Shot LLM + LLM-as-Judge

Adjudicated examples similar to the current post will be retrieved from Gold-dev. The frozen codebook and JSON Schema will then be supplied to generate evidence-supported narrative units. An independent judge will verify that the evidence exists, the domain is correct, and the normalized concept does not overinterpret the source.

### 2. Silver-Label Student Model

AI labels that have passed judge validation and sampled human auditing will be used to train a less expensive classification or extraction model. This experiment will test whether costly LLM capabilities can be distilled into a model suitable for batch processing. The training size will depend on the final eligible corpus and will not be fixed in advance at 4.8k.

### 3. Gold-Only Student Model

A multilingual text-classification or sequence-labeling model trained only on Gold-dev will serve as the small-sample supervised-learning baseline.

The coarse-classification stage should compare at least a fixed-prompt LLM, a multilingual classifier fine-tuned or trained on Gold-dev, and a simple rule-based baseline. If a lightweight model performs similarly to a complex multi-agent pipeline on the locked test set, the simpler, less expensive, and more reproducible approach should be preferred.

The best-performing approach will become the primary technical pipeline, while the other approaches will be retained as baselines or supplementary analyses.

## Open Concept Discovery and Standardization

Five-domain concept extraction should not begin with a completely closed dictionary. A closed vocabulary may erase colloquial language, slang, platform-specific expressions, and emerging coping behaviors in Chinese, Japanese, and Korean.

Recommended workflow:

```text
Open extraction of source-text evidence
        |
        v
Retain original expressions in all three languages
        |
        v
Cluster similar expressions within and across languages
        |
        v
Map expressions to existing normalized concepts
        |
        v
Send unmatched expressions to the new_candidate pool
        |
        v
Human review of frequent, novel, and ambiguous concepts
```

Two linked tables will be produced:

1. An English normalized-concept table for cross-country comparison.
2. An original-expression table for cultural and linguistic interpretation.

Standardization rules:

- Prioritize unambiguous exact dictionary matching.
- Multilingual embeddings will retrieve candidate mappings only; a fixed similarity threshold must not automatically write entries into the formal dictionary.
- Semantic mappings must be supported by source-text evidence and context.
- Ambiguous, idiomatic, and culturally specific expressions require human review and must not be forced into the nearest concept.
- Recurrent new concepts will remain `new_candidate` until reviewed.
- User-perceived causes must be recorded separately from clinically confirmed diagnoses.
- Formal analysis will use a frozen `core_v1`; new concepts discovered in the full corpus will form `expanded_v2` for supplementary or sensitivity analyses.

## Validation Metrics

### Upstream Classification

- Toothache relevance: precision, recall, and F1, with particular emphasis on `R1` recall.
- Experiencer: per-class precision, recall, and F1 for E1-E3, plus Macro-F1.
- Content function: per-class precision, recall, and F1 for C1-C5, plus Macro-F1.
- Confusion matrices.
- Performance reported separately by language.
- Uncertainty or refusal rate and technical failure rate.

### Five-Domain Post Classification

- Precision, recall, and F1 for each domain.
- Micro-F1 and Macro-F1.
- Subset accuracy or Hamming loss.
- Cohen's kappa, supplemented by Gwet's AC1 under class imbalance.
- Performance reported by country and language.

### Evidence and Concept Extraction

- Concept precision and recall.
- Evidence-span support rate.
- Normalization accuracy.
- Hallucination rate and omission rate.
- `UNMAPPED` rate.
- Error types by domain and language.

### Robustness and Uncertainty

- Report bootstrap 95% confidence intervals for primary performance metrics.
- Report confidence intervals for country-platform concept proportions.
- Compare different confidence thresholds.
- Compare results based only on explicit evidence with results that also include implicit evidence.
- Compare the primary set with the strict `E1+C1` set.
- Analyze errors by post length, emoji or slang use, platform, language, and time.
- Report automated-classification failures, translation failures, and API failures separately.

Cosine similarity between rationale texts must not replace evaluation of label and evidence correctness. Semantic similarity does not imply correct medical annotation.

## Downstream Statistical Analysis

### Denominators

The primary denominator is the number of unique posts in the primary analysis set:

```text
Proportion for a concept =
number of eligible posts containing the normalized concept
/
number of all eligible posts in that study corpus
```

Each post contributes at most once to a given normalized concept. If the composition of concepts among posts positive for a specific CSM domain is reported, this conditional denominator must be stated explicitly and must not be mixed with overall proportions.

### Country-Platform Comparisons

The primary comparison is among the CHI, JPN, and KOR study corpora. Analyses should consider at least:

- Month of publication.
- Text length.
- Content function (`C1/C2`).
- Clustering of repeated posts by the same author.
- Differences in platform accessibility and communication conventions.

The Benjamini-Hochberg procedure will control the false discovery rate when testing multiple domains and concepts. Confidence intervals and sensitivity analyses should be reported together with P values.

Because Xiaohongshu is used for China and X/Twitter for Japan and South Korea, results should be described as "differences among the study corpora" or "differences among country-platform cohorts." Differences must not be attributed solely to national culture without additional evidence.

## Co-Occurrence Networks

Network inputs must be validated normalized concepts from `core_v1`, rather than unreviewed free-form keywords.

Recommended rules:

1. Count each concept no more than once per post.
2. Use all eligible posts in the same study corpus as the document universe.
3. Prefer normalized pointwise mutual information (NPMI), or report both PMI and the observed co-occurrence count.
4. Prespecify a minimum document frequency and perform threshold-sensitivity analyses.
5. Construct a null distribution through permutations that preserve concept frequencies and the number of concepts per post, followed by false discovery rate control.
6. Bootstrap posts and report edge-selection rates and centrality intervals.
7. Display only stable edges in the main figures and provide the complete networks in supplementary materials.

Network size across corpora must not be compared visually alone. Prespecified measures such as node coverage, edge density, modularity, and mean weighted degree should be compared, with bootstrap intervals.

## Boundaries of Interpretation

Acceptable statements include:

- The Chinese study corpus more frequently mentions a particular type of self-management behavior.
- The Japanese study corpus more frequently includes effects on sleep or work.
- The South Korean study corpus more frequently includes a particular type of treatment-related concern.
- High-confidence normalized concepts form different stable co-occurrence patterns across the three country-platform corpora.

Unacceptable statements include:

- Chinese patients are clinically more dependent on a particular treatment.
- Stress causes more dental disease among Japanese people.
- The prevalence of wisdom-tooth disease is higher in the South Korean population.
- Differences in healthcare systems caused the observed differences among posts.

The most appropriate positioning for this study is a **validated multilingual narrative infodemiology study**, not a clinical epidemiologic survey.

## Quality Control and Reproducibility

Recommended directory structure:

```text
research_v2/
|-- configs/
|   |-- collection_protocol.yaml
|   |-- sampled_dates.csv
|   |-- cleaning_config.yaml
|   |-- annotation_schema.json
|   `-- analysis_plan.yaml
|-- data_registry/
|   |-- raw_manifest.csv
|   |-- corpus_v1_manifest.csv
|   `-- exclusion_log.csv
|-- annotation/
|   |-- codebook_v1.md
|   |-- exemplars/
|   |-- gold_dev/
|   |-- gold_test_locked/
|   `-- adjudicated_gold/
|-- translation/
|   |-- translation_manifest.csv
|   `-- failures.csv
|-- models/
|   |-- prompts/
|   |-- frozen_pipeline.json
|   `-- evaluation/
|-- ontology/
|   |-- core_v1.json
|   |-- candidate_pool.jsonl
|   `-- expanded_v2.json
|-- outputs/
|   |-- classification/
|   |-- extraction/
|   |-- statistics/
|   |-- networks/
|   `-- figures/
`-- logs/
    |-- api_calls/
    |-- failures/
    `-- run_manifest.json
```

Each run will retain the code version or commit hash, configuration hash, cleaning version, translation version, prompt hash, model name and run date, input snapshot ID, input and output row counts, exclusion count, failure count, retry count, and cost.

## Phased Implementation Sequence

### Phase A: While Data Collection Continues

1. Freeze the monthly random-date list, core search terms, and extended search terms.
2. Correct data lineage, incremental deduplication, and failure logging in the collectors.
3. Harmonize fields, the study period, and the exact-deduplication protocol across all three languages.
4. Complete and version the Chinese cleaning script, then port the same logic to the Japanese and Korean data.
5. Complete English translation for all three languages and audit translation quality.
6. Use the current exemplar candidates to freeze the relevance, E1-E3, and C1-C5 definitions, and supplement the Korean C4 exemplars.
7. Develop the codebook and borderline cases for the five CSM domains.

### Phase B: After Data Collection Is Complete

1. Freeze an immutable raw-data snapshot and `corpus_v1`.
2. Run the cleaning report and verify date exclusions, technically invalid records, and duplicate records.
3. Complete coarse classification of toothache relevance, experiencer, and content function.
4. Draw Gold-dev through language-stratified sampling, complete dual independent annotation, and adjudicate disagreements.
5. Freeze the codebook, prompts, JSON Schema, model versions, and thresholds.
6. Complete dual independent annotation of Gold-test and lock the set; models will be evaluated on it only once.
7. Compare the LLM, student models, and simple baselines on the locked test set.
8. If the prespecified performance threshold is met, run full-corpus five-domain evidence extraction. Otherwise, return to development and construct a new test set.
9. Freeze `core_v1` and complete the human audit of concept mappings.
10. Complete the full-corpus automated-annotation audit, statistical analyses, sensitivity analyses, and stable network analyses.
11. Write the Methods, Results, limitations, and AI-use disclosure according to the procedures actually performed.

## Key Decisions to Freeze in Advance

Before evaluation on the final test set and full-corpus analysis, the following must be specified and versioned:

- Monthly random dates and random seed.
- Core and extended search terms for all three languages.
- Study period.
- Cleaning, date-recovery, and deduplication rules.
- Definitions of relevance, experiencer, and content function.
- Primary set and strict sensitivity set.
- Five-domain CSM codebook.
- Gold-dev/Gold-test split.
- Models, prompts, temperature, retry strategy, and JSON Schema.
- Concept ontology `core_v1`.
- Primary statistical outcomes, covariates, multiple-testing procedure, and network-stability thresholds.

These rules must not be modified after viewing locked-test results or country comparisons for the purpose of improving performance or obtaining more statistically significant differences.

## Expected Contributions

### Medical and Public Health Contribution

The study will describe how toothache is perceived, expressed, and managed in three country-platform corpora from China, Japan, and South Korea, and will identify common symptom, daily-life impact, coping, and emotional patterns.

### Medical Artificial Intelligence Contribution

The study will evaluate an annotation-efficient workflow for multilingual patient-narrative research, including:

- Evidence-preserving cleaning.
- Relevance screening and two-axis coarse classification.
- A small, dual-independently-annotated human gold standard.
- Retrieval-augmented few-shot evidence extraction.
- LLM-as-Judge validation.
- Silver-label distillation.
- Locked testing and production-stage human auditing.
- Open concept discovery and versioned cross-language standardization.

## One-Sentence Summary

Through auditable data cleaning, toothache-relevance screening, and experiencer/content-function classification, this study will construct a corpus of first-person toothache illness narratives from China, Japan, and South Korea, use a small dual-annotated gold standard to calibrate AI systems, extract source-supported concepts across five CSM domains, and conduct cautious country-platform comparisons.
