#!/usr/bin/env python3
"""Summarize completed HF shard progress from annotations/checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize dental_ai HF shard progress.")
    parser.add_argument("run_dir", help="Run directory containing shard_* subdirectories or one run directory")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    shard_dirs = sorted(path for path in run_dir.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        shard_dirs = [run_dir]

    summary = summarize(shard_dirs)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_summary(summary)
    return 0


def summarize(shard_dirs: list[Path]) -> dict[str, Any]:
    classified_seen: set[str] = set()
    final_seen: set[str] = set()
    extracted_seen: set[str] = set()
    label_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    eligible_csm = 0
    rule_counts: Counter[str] = Counter()
    shard_rows = []
    errors = 0
    malformed_jsonl_lines = 0

    for shard_dir in shard_dirs:
        classified_rows, bad_classified = _read_jsonl(shard_dir / "checkpoints" / "classified.jsonl")
        extracted_rows, bad_extracted = _read_jsonl(shard_dir / "checkpoints" / "extracted.jsonl")
        final_rows, bad_final = _read_jsonl(shard_dir / "annotations.jsonl")
        error_rows, bad_errors = _read_jsonl(shard_dir / "errors.jsonl")
        shard_bad_lines = bad_classified + bad_extracted + bad_final + bad_errors
        malformed_jsonl_lines += shard_bad_lines
        errors += len(error_rows)

        shard_classified = 0
        shard_eligible = 0
        for row in classified_rows:
            result = row.get("result", row)
            post_id = str(result.get("post_id", ""))
            if not post_id or post_id in classified_seen:
                continue
            classified_seen.add(post_id)
            shard_classified += 1
            country_counts[str(result.get("country", ""))] += 1
            language_counts[str(result.get("language", ""))] += 1
            rel = result.get("relevance_label")
            exp = result.get("experiencer_label")
            content = result.get("content_function")
            label_counts[f"{rel}|{exp}|{content}"] += 1
            if rel == "R1" and exp in {"E1", "E2"} and content in {"C1", "C2"}:
                eligible_csm += 1
                shard_eligible += 1
            trace = row.get("trace", {})
            for rule in trace.get("postprocessing_rules", []):
                if isinstance(rule, dict) and rule.get("rule"):
                    rule_counts[str(rule["rule"])] += 1

        for row in extracted_rows:
            result = row.get("result", row)
            post_id = str(result.get("post_id", ""))
            if post_id:
                extracted_seen.add(post_id)

        for row in final_rows:
            post_id = str(row.get("post_id", ""))
            if post_id:
                final_seen.add(post_id)

        shard_rows.append(
            {
                "shard_dir": str(shard_dir),
                "classified_rows": shard_classified,
                "eligible_csm_rows": shard_eligible,
                "extracted_checkpoint_rows": len(extracted_rows),
                "final_annotation_rows": len(final_rows),
                "error_rows": len(error_rows),
                "malformed_jsonl_lines": shard_bad_lines,
            }
        )

    return {
        "shards": len(shard_dirs),
        "classified_rows": len(classified_seen),
        "eligible_csm_rows": eligible_csm,
        "eligible_csm_rate": eligible_csm / len(classified_seen) if classified_seen else 0.0,
        "extracted_checkpoint_rows": len(extracted_seen),
        "final_annotation_rows": len(final_seen),
        "error_rows": errors,
        "malformed_jsonl_lines": malformed_jsonl_lines,
        "country_counts": dict(sorted(country_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "top_label_counts": dict(label_counts.most_common(20)),
        "postprocessing_rule_counts": dict(sorted(rule_counts.items())),
        "shards_detail": shard_rows,
    }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    rows = []
    bad_lines = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines += 1
    return rows, bad_lines


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"shards: {summary['shards']}")
    print(f"classified_rows: {summary['classified_rows']}")
    print(f"eligible_csm_rows: {summary['eligible_csm_rows']} ({summary['eligible_csm_rate']:.1%})")
    print(f"extracted_checkpoint_rows: {summary['extracted_checkpoint_rows']}")
    print(f"final_annotation_rows: {summary['final_annotation_rows']}")
    print(f"error_rows: {summary['error_rows']}")
    print(f"malformed_jsonl_lines: {summary['malformed_jsonl_lines']}")
    print("top_label_counts:")
    for label, count in summary["top_label_counts"].items():
        print(f"  {label}: {count}")
    print("postprocessing_rule_counts:")
    for rule, count in summary["postprocessing_rule_counts"].items():
        print(f"  {rule}: {count}")
    print("shards_detail:")
    for shard in summary["shards_detail"]:
        print(
            "  {shard_dir}: classified={classified_rows}, eligible_csm={eligible_csm_rows}, "
            "extracted={extracted_checkpoint_rows}, final={final_annotation_rows}, errors={error_rows}, "
            "malformed={malformed_jsonl_lines}".format(**shard)
        )


if __name__ == "__main__":
    raise SystemExit(main())
