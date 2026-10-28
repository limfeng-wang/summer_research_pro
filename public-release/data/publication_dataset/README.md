# Study metadata — no public dataset

The original Chinese Xiaohongshu and Japanese/Korean X datasets are not
publicly distributed. This directory contains only:

- `metadata.json`: group sizes, language/platform definitions, reference roles,
  retrospective curation notes, and aggregate analytical subset counts.
- `DATA_DICTIONARY.md`: the private file schemas expected by the retained code.
- `validate.py`: a validator for separately supplied private records.

It does not contain development/evaluation/main posts, human annotations,
source/evidence text, individual predictions, or a per-record identifier list.
Researchers may request access from the corresponding authors, Kun Wang
([2511110190@pku.edu.cn](mailto:2511110190@pku.edu.cn)) or Beibei Xu
([xubeibei@bjmu.edu.cn](mailto:xubeibei@bjmu.edu.cn)). Requests are considered
individually and do not guarantee access.

## Public checks

From the repository root run `python scripts/verify_release.py` after installing
the package and dependencies. This tests software and aggregate calculations;
it does not validate private corpus membership or textual evidence.

## Private validation

An authorized holder of a compatible private dataset can run:

```bash
python data/publication_dataset/validate.py --dataset /path/to/private_dataset
```

The supplied directory must include its private `manifest.json`, development,
evaluation, and main JSONL files. The validator checks hashes, joins, membership,
source-text separation, reference roles, unit IDs, and historical inclusion flags.
These checks cannot be run against the public metadata alone.

## Historical study composition

Development and evaluation each contain 150 posts, 50 per language. Development
roles comprise 145 classification references and 60 extraction references, with
55 shared posts. The main corpus contains 36,720 posts; the archived manuscript
subset contains 12,829 posts and 35,627 verdict-accepted units. The stricter
schema-plus-exact-span rule selects 12,817 posts. These are aggregate metadata,
not a distributed dataset or a substitute for independent source-data validation.
