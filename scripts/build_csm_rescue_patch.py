#!/usr/bin/env python3
"""Build a targeted CSM rescue patch without changing pipeline code.

The patch contains two groups:

1. CHI first-pass non-eligible rows that were not rescue candidates under the
   original keyword-targeted CHI policy. These rows should go through the
   high-recall rescue classifier.
2. Rows that already passed the prior rescue classifier but did not finalize
   into merged rescue annotations. These rows skip reclassification and are
   represented by synthetic rescue decisions derived from the prior classified
   checkpoint.

The generated files are designed to feed the existing
`run_csm_rescue_classifier.py` and `prepare_rescue_extraction_run.py` scripts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scripts.build_csm_rescue_candidates import _candidate_payload, _candidate_reasons, _is_csm_eligible, _post_id


DEFAULT_FIRST_PASS_RUNS = [
    "outputs/main_pilot_10000_sharded_3gpu_merged",
    "outputs/main_pilot_20000_sharded_3gpu_merged_repaired",
    "outputs/main_pilot_36k_sharded_3gpu_merged",
]
DEFAULT_PRIOR_RESCUE_MERGED = "outputs/csm_rescue_extraction_full_sharded_merged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-pass-run",
        action="append",
        default=[],
        help="First-pass merged run directory. Repeat to override defaults.",
    )
    parser.add_argument("--prior-rescue-merged", default=DEFAULT_PRIOR_RESCUE_MERGED)
    parser.add_argument("--out-dir", default="data/csm_rescue_patch_chi_noncandidate_unfinished")
    args = parser.parse_args()

    first_pass_runs = [Path(item) for item in (args.first_pass_run or DEFAULT_FIRST_PASS_RUNS)]
    prior_rescue = Path(args.prior_rescue_merged)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_pass_rows, first_pass_source = _load_first_pass_rows(first_pass_runs)
    original_candidates = _original_policy_candidates(first_pass_rows, first_pass_source)
    first_pass_eligible_ids = {
        post_id for post_id, row in first_pass_rows.items() if _is_csm_eligible(row)
    }
    chi_noncandidate_rows = []
    for post_id, row in sorted(first_pass_rows.items()):
        if row.get("country") != "CHI":
            continue
        if post_id in first_pass_eligible_ids or post_id in original_candidates:
            continue
        chi_noncandidate_rows.append(
            _candidate_payload(
                row,
                first_pass_source[post_id],
                ["chi_all_noneligible_patch"],
            )
        )

    prior_classified = _load_checkpoint_results(prior_rescue / "checkpoints" / "classified.jsonl")
    prior_final_ids = {
        _post_id(row)
        for row in _read_jsonl(prior_rescue / "annotations.jsonl")
        if _post_id(row)
    }
    unfinished_positive_rows = []
    unfinished_positive_decisions = []
    for post_id, result in sorted(prior_classified.items()):
        if post_id in prior_final_ids:
            continue
        source_row = first_pass_rows.get(post_id)
        if source_row is None:
            continue
        experiencer = result.get("experiencer_label")
        content = result.get("content_function")
        if result.get("relevance_label") != "R1" or experiencer not in {"E1", "E2"} or content not in {"C1", "C2"}:
            continue
        candidate = _candidate_payload(
            source_row,
            first_pass_source[post_id],
            ["prior_rescue_positive_not_finalized"],
        )
        unfinished_positive_rows.append(candidate)
        unfinished_positive_decisions.append(
            {
                "post_id": post_id,
                "country": candidate.get("country"),
                "language": candidate.get("language"),
                "first_pass": {
                    "relevance_label": candidate.get("first_pass_relevance_label"),
                    "experiencer_label": candidate.get("first_pass_experiencer_label"),
                    "content_function": candidate.get("first_pass_content_function"),
                    "candidate_reasons": candidate.get("rescue_candidate_reasons") or [],
                },
                "decision": {
                    "rescue_csm_eligible": True,
                    "rescued_experiencer_label": experiencer,
                    "rescued_content_function": content,
                    "rescue_evidence": "",
                    "reason": "Previously passed rescue gate but did not finalize into merged rescue annotations.",
                },
                "raw_model_text": "",
            }
        )

    combined_candidates = _dedupe_rows([*chi_noncandidate_rows, *unfinished_positive_rows])

    paths = {
        "candidates_to_classify": out_dir / "candidates_to_classify.jsonl",
        "unfinished_positive_candidates": out_dir / "unfinished_positive_candidates.jsonl",
        "unfinished_positive_decisions": out_dir / "unfinished_positive_decisions.jsonl",
        "candidates_all": out_dir / "candidates_all.jsonl",
        "manifest": out_dir / "manifest.json",
    }
    _write_jsonl(paths["candidates_to_classify"], chi_noncandidate_rows)
    _write_jsonl(paths["unfinished_positive_candidates"], unfinished_positive_rows)
    _write_jsonl(paths["unfinished_positive_decisions"], unfinished_positive_decisions)
    _write_jsonl(paths["candidates_all"], combined_candidates)

    manifest = {
        "first_pass_runs": [str(path) for path in first_pass_runs],
        "prior_rescue_merged": str(prior_rescue),
        "out_dir": str(out_dir),
        "first_pass_unique_rows": len(first_pass_rows),
        "original_policy_candidate_rows": len(original_candidates),
        "first_pass_csm_eligible_rows": len(first_pass_eligible_ids),
        "chi_noncandidate_to_classify_rows": len(chi_noncandidate_rows),
        "unfinished_prior_rescue_positive_rows": len(unfinished_positive_rows),
        "combined_candidate_rows": len(combined_candidates),
        "country_counts": {
            "candidates_to_classify": dict(Counter(row.get("country") for row in chi_noncandidate_rows)),
            "unfinished_positive": dict(Counter(row.get("country") for row in unfinished_positive_rows)),
            "combined": dict(Counter(row.get("country") for row in combined_candidates)),
        },
        "files": {key: str(path) for key, path in paths.items() if key != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _load_first_pass_rows(run_dirs: Iterable[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for run_dir in run_dirs:
        for row in _read_jsonl(run_dir / "annotations.jsonl"):
            post_id = _post_id(row)
            if not post_id or post_id in rows:
                continue
            rows[post_id] = row
            sources[post_id] = run_dir
    return rows, sources


def _original_policy_candidates(
    first_pass_rows: dict[str, dict[str, Any]],
    first_pass_source: dict[str, Path],
) -> set[str]:
    candidates = set()
    for post_id, row in first_pass_rows.items():
        if _candidate_reasons(row):
            candidates.add(post_id)
    return candidates


def _load_checkpoint_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        result = row.get("result", row)
        post_id = _post_id(result)
        if post_id:
            results[post_id] = result
    return results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        post_id = _post_id(row)
        if not post_id or post_id in seen:
            continue
        seen.add(post_id)
        output.append(row)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
