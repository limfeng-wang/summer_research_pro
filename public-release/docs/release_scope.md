# Release scope

This is a software and aggregate-results release candidate for a separate public
GitHub repository. The owner has decided not to release the underlying dataset
for the paper. No commit, push, or publication has been performed here.

## Public contents and reproducibility

- Original software, model configuration, prompts, deterministic rules, and tests.
- Aggregate group sizes, language/platform metadata, and private-data schemas.
- Aggregate corpus/evaluation summaries and 14 historical ablation/sensitivity
  tables. Their historical checksums are preserved.
- Public verification of inventory/hashes, software contracts, binary metric
  arithmetic, country-profile statistics, and regeneration of five figures.

No post-level data, source posts, evidence spans, human annotations, individual
predictions, retrieval traces, identifier lists, canonicalization decision
records, or private collection archives are distributed. Small illustrative
regression fixtures test behavior; they do not constitute a research dataset or
support the reported empirical results. Known longer verbatim study excerpts
found in tests were replaced with illustrative wording during this scope change.

The public package cannot independently reproduce collection, source-text
checks, original inference/reference retrieval, post-level evaluation, bootstrap
resampling, or derivation of the aggregate tables and network/concept statistics.
Archived confidence intervals and stability measures are not re-estimated from
public data. These limits must also be reflected in the manuscript.

## Licensing and data availability

Original software and associated technical documentation use MIT, with the
paper's 12 authors named in `LICENSE`. The software license excludes underlying
social-media content, annotations, and research artifacts under `data/` and
`results/`. No separate data reuse license is granted here. Model weights are
not distributed and third-party licenses remain separate.

The decision not to distribute the dataset is final for this paper release.
Researchers may request access by emailing either corresponding author:
Kun Wang (2511110190@pku.edu.cn) or Beibei Xu (xubeibei@bjmu.edu.cn).
Requests are considered individually; access is not guaranteed. This repository does not claim
an ethics waiver, consent determination, legal restriction, or platform permission
that has not been supplied.
Journal/editorial verification requests must be handled according to the actual
institutional permissions and journal requirements.

## Historical validation and remaining provenance limits

Complete private-data CPU reproduction previously passed: 89 original tests,
dataset validation, mock annotation, evaluation with 2,000 bootstrap samples,
35 sensitivity configurations, and 14 archived aggregate-table comparisons.
Subsequent software inventory tests increased the test count. The current public
suite has a different scope: its PASS does not imply access to the original data.

A real-model smoke test passed on 2026-10-24 using two NVIDIA L20 GPUs and
private inputs. It covered model loading, dense retrieval, three-language full
annotation, stable resume, and direct extraction/judge probes. This establishes
execution on that setup, not accuracy or historical prediction equivalence.
See [inference.md](inference.md).

Historical checkpoint commits, original inference dependency versions, the 36
date-sampling log/seed, collection timezones/timestamps, and complete cleaning
attrition logs remain unavailable. The main input originally had 36,721 posts;
one Japanese extraction failed, leaving 36,720 finalized records. The retained
network code uses co-occurrence counts and weighted degree; a separate
NPMI/Leiden implementation was not located.

The runner retains a correction that writes successful zero-unit extractions
immediately. Historical annotations and results were not altered by this fix or
by the change to an aggregate-only public release.

## Publication boundary

Publish a clean export of the manifest-listed files plus `release_manifest.json`,
not a copy or zip of the entire working folder. Local ignored `outputs/` includes
private originals, historical logs, and environments and must never be published.
Local `models/` and personal configuration directories must also stay excluded.
Use `python scripts/export_public_release.py --out-dir /tmp/dental-pain-public`
with a new destination outside the working tree to make that export.
The manifest builder rejects post-level JSONL artifacts and private configuration;
the verifier checks both the complete distributable inventory and file hashes.
Regenerating a manifest does not replace reviewing file contents.

Use a fresh Git history if the source repository has private commits. The CPU
wrapper accepts any branch and rejects a nested release checkout. Git metadata
is unavailable in the current workspace, so tracked files, remotes, and private
history remain unchecked. Publishing requires a separate explicit action.
