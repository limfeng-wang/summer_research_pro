# Multilingual dental pain narratives

Software, study metadata, and aggregate results for a study of Chinese
Xiaohongshu and Japanese/Korean X posts about dental pain.

**The study dataset is not publicly released.** This repository contains no
post-level corpus, human reference annotations, evidence spans, individual model
predictions, or retrieval traces. Public results are aggregate tables and metrics.

| Study group (not distributed) | Chinese | Japanese | Korean | Total |
| --- | ---: | ---: | ---: | ---: |
| Development | 50 | 50 | 50 | 150 |
| Independent evaluation | 50 | 50 | 50 | 150 |
| Main automated corpus | 3,521 | 25,167 | 8,032 | 36,720 |

The development set supplies 145 classification references and 60 extraction
references; 55 posts serve both roles. The historical manuscript subset contains
12,829 posts and 35,627 verdict-accepted units. See the
[metadata guide](data/publication_dataset/README.md) and
[private-data field definitions](data/publication_dataset/DATA_DICTIONARY.md).

## Public verification

Run from the repository root with Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/verify_release.py
```

Or test installation in a fresh Conda environment:

```bash
bash tests/test_release.sh --cleanup-env
```

Verification checks the release inventory and hashes, runs software tests using
small illustrative fixtures, checks 14 archived aggregate tables, recomputes
112 binary metric rows and 35 country-profile tests from aggregate counts, and
regenerates five figures (PNG and PDF). It does not download models or require
study records. Reports go to `outputs/verification/` and figures to
`outputs/public_reproduction/`, both ignored by Git.

To regenerate only the aggregate calculations and figures:

```bash
python scripts/reproduce_aggregate_results.py
```

These checks **do not reproduce** collection, model inference on the original
posts, source-text validation, bootstrap resampling, or derivation of the
aggregate tables from individual observations. Confidence intervals and
concept/network stability statistics are archived results. Their display and
file integrity can be checked publicly; their derivation requires private data.

## Inference and original-data analyses

The inference and analysis code is included for inspection and use with
separately supplied compatible inputs. The private study references and inputs
named in `configs/model_stack.yaml` are not included. Configure reference paths
before using the real-model pipeline; downloading models alone is insufficient.
See [inference instructions](docs/inference.md) for the tested GPU setup and
private-data commands. Public verification does not run GPU inference.

A separate three-language GPU smoke test passed on two NVIDIA L20 GPUs on
2026-10-24 using private inputs. It establishes execution on that setup, not
model accuracy or exact historical prediction reproduction.

## Contents

- `dental_ai/`: prompts, schemas, annotation, retrieval, judging, validation, and CLI.
- `data/publication_dataset/`: aggregate metadata, schema documentation, and a
  validator for separately supplied private records; no study records.
- `results/reference/`: aggregate corpus/evaluation summaries and 14 aggregate
  ablation/sensitivity tables with historical checksums.
- `scripts/`: public aggregate reproduction and original-data analysis utilities.
- `tests/`: software regression tests and a private-input GPU smoke test.
- `docs/`: methods, full prompts, and release/inference scope.

## Data availability

The underlying dataset is not publicly released. This release provides code,
metadata, and aggregate results sufficient for the public checks described above.
Researchers may request access to the underlying dataset by emailing either
corresponding author: Kun Wang ([2511110190@pku.edu.cn](mailto:2511110190@pku.edu.cn))
or Beibei Xu ([xubeibei@bjmu.edu.cn](mailto:xubeibei@bjmu.edu.cn)). Requests will
be considered individually; emailing does not guarantee access. See
[release scope](docs/release_scope.md).

To prepare a clean public folder, run:

```bash
python scripts/export_public_release.py --out-dir /tmp/dental-pain-public
```

Choose a new destination outside this working tree. The export copies only the
verified manifest-listed files and the manifest. It excludes Git history and
local ignored files. **Do not publish the whole working folder:** ignored
`outputs/` can contain private originals and logs. Review and initialize the
export as a fresh Git repository before publication.

## License

The original software code and associated technical documentation are licensed
under the [MIT License](LICENSE), with the paper's 12 authors named in the
copyright notice. The software license does not cover underlying social-media
content, annotations, or the aggregate research artifacts under `data/` and
`results/`; no separate data reuse license is granted here. Third-party software
and model weights retain their own licenses.
