#!/usr/bin/env python3
"""Prepare a forced CSM extraction run for R1 rows skipped by the normal gate.

The normal extraction gate only parses R1 E1/E2 C1/C2 rows. This script builds
an input JSONL and sharded classified checkpoints for R1 rows that are E3 or
C3-C5, preserving their existing labels so --resume can skip classification and
run only extraction/judging when FORCE_EXTRACT_ALL_R1_CSM=1 is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dental_ai.schemas import ExtractionResult, SourcePost
from dental_ai.validate import validate_hierarchical_result

COUNTRIES = ("CHI", "JPN", "KOR")
SOURCE_FIELDS = set(SourcePost.model_fields)
RESULT_FIELDS = set(ExtractionResult.model_fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument(
        "--post-id",
        action="append",
        default=[],
        help="Restrict preparation to one or more specific post IDs.",
    )
    parser.add_argument(
        "--first-pass-dir",
        action="append",
        default=[
            "outputs/main_pilot_10000_sharded_3gpu_merged",
            "outputs/main_pilot_20000_sharded_3gpu_merged",
            "outputs/main_pilot_36k_sharded_3gpu_merged",
        ],
    )
    parser.add_argument(
        "--rescue-dir",
        action="append",
        default=[
            "outputs/csm_rescue_extraction_full_sharded_merged",
            "outputs/csm_rescue_patch_extraction_sharded_merged",
        ],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.jsonl"

    corpus: dict[str, dict[str, Any]] = {}
    for run_dir in [Path(path) for path in args.first_pass_dir]:
        for row in read_jsonl(run_dir / "annotations.jsonl"):
            post_id = get_post_id(row)
            if post_id and post_id not in corpus:
                corpus[post_id] = row
    for run_dir in [Path(path) for path in args.rescue_dir]:
        for row in read_jsonl(run_dir / "annotations.jsonl"):
            post_id = get_post_id(row)
            if post_id and has_accepted_unit(row):
                corpus[post_id] = row

    if args.post_id:
        post_ids = set(args.post_id)
        targets = [row for post_id, row in corpus.items() if post_id in post_ids]
    else:
        targets = [row for row in corpus.values() if should_force_extract(row)]
    targets.sort(key=lambda row: get_post_id(row))

    with input_path.open("w", encoding="utf-8") as handle:
        for row in targets:
            handle.write(json.dumps(source_payload(row), ensure_ascii=False) + "\n")

    for shard_index in range(args.shard_count):
        shard_rows = select_shard(targets, args.shard_count, shard_index)
        shard_dir = out_dir / f"shard_{shard_index:03d}"
        checkpoint_dir = shard_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name in ["annotations.jsonl", "errors.jsonl", "retrieval_trace.jsonl"]:
            (shard_dir / name).write_text("", encoding="utf-8")
        (checkpoint_dir / "extracted.jsonl").write_text("", encoding="utf-8")
        with (checkpoint_dir / "classified.jsonl").open("w", encoding="utf-8") as handle:
            for row in shard_rows:
                post = SourcePost.model_validate(source_payload(row))
                result = ExtractionResult.model_validate(result_payload(row)).model_copy(update={"units": []})
                validation = validate_hierarchical_result(result, post)
                payload = {
                    "result": result.model_dump(mode="json"),
                    "trace": {
                        "stages": ["loaded_forced_csm_classification"],
                        "raw_labels": {
                            "relevance_label": row.get("relevance_label"),
                            "experiencer_label": row.get("experiencer_label"),
                            "content_function": row.get("content_function"),
                        },
                        "postprocessing_rules": [],
                        "validation": {
                            "ok": validation.ok,
                            "issues": [
                                {"code": issue.code, "message": issue.message}
                                for issue in validation.issues
                            ],
                        },
                    },
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "out_dir": str(out_dir),
        "input": str(input_path),
        "target_rows": len(targets),
        "target_rows_by_country": counts_by_country(targets),
        "normal_gate_skipped_definition": "R1 and (E3 or C3/C4/C5)",
        "shard_count": args.shard_count,
    }
    (out_dir / "forced_csm_preparation_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def get_post_id(row: dict[str, Any]) -> str:
    return str(row.get("post_id") or row.get("record_id") or "").strip()


def has_accepted_unit(row: dict[str, Any]) -> bool:
    return any(unit.get("judge_verdict") == "accept" for unit in row.get("units") or [])


def should_force_extract(row: dict[str, Any]) -> bool:
    if row.get("relevance_label") != "R1":
        return False
    if has_accepted_unit(row):
        return False
    return row.get("experiencer_label") == "E3" or row.get("content_function") in {"C3", "C4", "C5"}


def source_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in SOURCE_FIELDS if key in row}


def result_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {key: row[key] for key in RESULT_FIELDS if key in row}
    payload.setdefault("units", [])
    return payload


def select_shard(rows: list[dict[str, Any]], shard_count: int, shard_index: int) -> list[dict[str, Any]]:
    start = len(rows) * shard_index // shard_count
    end = len(rows) * (shard_index + 1) // shard_count
    return rows[start:end]


def counts_by_country(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {country: sum(1 for row in rows if row.get("country") == country) for country in COUNTRIES}


if __name__ == "__main__":
    raise SystemExit(main())
