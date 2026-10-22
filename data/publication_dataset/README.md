# Dental pain narrative dataset

Three separate groups support multilingual analysis of Chinese Xiaohongshu posts
and Japanese/Korean X posts. `CHI`, `JPN`, and `KOR` identify these country-platform
corpora; languages are `zh`, `ja`, and `ko`.

| Group | Chinese | Japanese | Korean | Total |
| --- | ---: | ---: | ---: | ---: |
| Development | 50 | 50 | 50 | **150** |
| Evaluation | 50 | 50 | 50 | **150** |
| Main automated corpus | 3,521 | 25,167 | 8,032 | **36,720** |

## Files

- [development/posts.jsonl](development/posts.jsonl): human-annotated examples,
  stored once per post with explicit classification/extraction roles.
- [evaluation/posts.jsonl](evaluation/posts.jsonl): evaluation inputs without labels.
- [evaluation/annotations.jsonl](evaluation/annotations.jsonl): separate human
  reference annotations, joined to inputs by `post_id`.
- [main/posts.jsonl](main/posts.jsonl): all 36,720 posts with automated labels,
  extraction units, and analysis/quality flags.
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md): field definitions and inclusion rules.
- [manifest.json](manifest.json): file hashes, counts, and source hashes.

JSONL stores one JSON object per line. The 150 evaluation annotation rows describe
the same 150 evaluation posts; they are not additional posts.
The human workbook and model-input text differ for 51 records (50 Chinese and
one Korean). Both versions are preserved explicitly in the evaluation files;
annotations record whether their reference text matches the model input.

## Development roles

The development set contains **145 classification examples and 60 extraction
examples, with 55 shared posts**: 90 classification-only, 5 extraction-only, and
55 serving both purposes. Classification uses up to eight examples selected by
language and label coverage; extraction retrieves five similar examples from the
60-post reference pool. Order fields preserve the original example ordering.

This balanced set was curated retrospectively from 177 development posts by
removing 12 Chinese, 10 Japanese, and 5 Korean classification-only posts. The
current classifier selects identical ordered examples after removal, and the
entire RAG reference is preserved. This does not establish that historical
prompts or code were unchanged throughout development.

## Main corpus and analytical subset

The full corpus contains **36,720 labeled posts**, not just eligible narratives.
The original automated input had 36,721 posts; one Japanese extraction failed.
`analysis_flags.manuscript_primary` identifies the **12,829-post** manuscript
sample, containing **35,627 verdict-accepted units**. The stricter schema/span
rule identifies **12,817 posts**. Existing labels and units are preserved rather
than silently corrected; unit flags make the differing rules explicit.

## Scope and identifiers

Development, evaluation, and main groups have no exact source-text overlap in the
audited files. This is not proof against near duplicates or evaluation-informed
tuning. Human reference annotations and automated outputs are kept distinct.

Release-specific post/unit IDs replace source record IDs. Internal source paths,
spreadsheet coordinates, and run names are omitted. Original narrative text and
annotation wording are retained; this packaging step does not newly anonymize
identifying information that may occur within that text. Publication timestamps
and the original 36-date sampling log are unavailable in these source files.

## Verify

From this folder, run `python3 validate.py`. The validator uses only the Python
standard library and checks hashes, counts, role membership, disjoint source
texts, annotation joins, unit IDs, and analysis/quality flags.

The folder is self-contained for reading and validation. In the working
repository, `python3 scripts/build_publication_dataset.py` rebuilds its data files
from the archived sources. The private source-to-release ID mapping is stored
outside this folder under `outputs/publication_dataset_build/` and is not part
of this package.
