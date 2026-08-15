#!/usr/bin/env python3
"""Repair merged outputs missing zero-unit extracted rows in annotations.

Some first-pass HF full runs checkpoint CSM-eligible rows with zero extracted
units but do not write them to annotations.jsonl until a resume pass. This
script creates a repaired copy of a merged output directory by appending only
missing extracted checkpoint rows whose result has zero units.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Merged output directory to repair")
    parser.add_argument("--out-dir", required=True, help="Repaired output directory")
    parser.add_argument(
        "--input",
        default="",
        help="Optional input JSONL. If provided, report expected input IDs missing from repaired annotations.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing out-dir: {out_dir}")
    if not (run_dir / "annotations.jsonl").exists():
        raise SystemExit(f"Missing annotations.jsonl in {run_dir}")
    if not (run_dir / "checkpoints" / "extracted.jsonl").exists():
        raise SystemExit(f"Missing checkpoints/extracted.jsonl in {run_dir}")

    shutil.copytree(run_dir, out_dir)

    annotations_path = out_dir / "annotations.jsonl"
    extracted_path = out_dir / "checkpoints" / "extracted.jsonl"
    manifest_path = out_dir / "run_manifest.json"

    final_ids = _load_annotation_ids(annotations_path)
    extracted = _load_extracted_results(extracted_path)
    missing = [result for post_id, result in extracted.items() if post_id not in final_ids]
    non_zero_missing = [result for result in missing if result.get("units")]
    if non_zero_missing:
        examples = [result.get("post_id") for result in non_zero_missing[:10]]
        raise SystemExit(
            "Missing final rows include extracted rows with non-zero units; refusing repair. "
            f"Examples: {examples}"
        )

    with annotations_path.open("a", encoding="utf-8") as handle:
        for result in missing:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    repaired_ids = _load_annotation_ids(annotations_path)
    expected_missing = []
    if args.input:
        input_ids = _load_input_ids(Path(args.input))
        expected_missing = sorted(input_ids - repaired_ids)

    repair_summary = {
        "source_run_dir": str(run_dir),
        "repaired_out_dir": str(out_dir),
        "zero_unit_rows_appended": len(missing),
        "final_rows_before": len(final_ids),
        "final_rows_after": len(repaired_ids),
        "expected_input": args.input,
        "expected_input_missing_after_repair": len(expected_missing),
        "expected_input_missing_examples": expected_missing[:20],
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["rows_completed_before_zero_unit_repair"] = manifest.get("rows_completed")
        manifest["rows_completed"] = len(repaired_ids)
        manifest["zero_unit_missing_final_repair"] = repair_summary
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out_dir / "zero_unit_missing_final_repair.json").write_text(
        json.dumps(repair_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(repair_summary, ensure_ascii=False, indent=2))
    return 0


def _load_annotation_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _loads_jsonl(line, path, line_number)
            post_id = str(row.get("post_id") or row.get("record_id") or "").strip()
            if post_id:
                ids.add(post_id)
    return ids


def _load_extracted_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _loads_jsonl(line, path, line_number)
            result = row.get("result", row)
            post_id = str(result.get("post_id") or result.get("record_id") or "").strip()
            if post_id:
                results[post_id] = result
    return results


def _load_input_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _loads_jsonl(line, path, line_number)
            post_id = str(row.get("post_id") or row.get("record_id") or "").strip()
            if post_id:
                ids.add(post_id)
    return ids


def _loads_jsonl(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())

