#!/usr/bin/env python3
"""Merge sharded dental_ai run outputs.

Each shard is a normal run directory produced by:

    python -m dental_ai.cli run-hierarchical ... --shard-count N --shard-index I

This script preserves the first final annotation per post_id, concatenates
diagnostic traces, and writes a small manifest so sharded production runs can be
reported as one corpus-level run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded dental_ai HF run directories.")
    parser.add_argument("--out-dir", required=True, help="Merged output directory")
    parser.add_argument("shard_dirs", nargs="+", help="Shard output directories")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    shard_dirs = [Path(path) for path in args.shard_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations, duplicate_annotations = _merge_annotations(shard_dirs, out_dir / "annotations.jsonl")
    errors = _concat_jsonl(shard_dirs, "errors.jsonl", out_dir / "errors.jsonl", add_shard_dir=True)
    retrieval_rows = _concat_jsonl(
        shard_dirs,
        "retrieval_trace.jsonl",
        out_dir / "retrieval_trace.jsonl",
        add_shard_dir=True,
    )
    classified_checkpoints = _concat_checkpoint(
        shard_dirs,
        "checkpoints/classified.jsonl",
        out_dir / "checkpoints" / "classified.jsonl",
        add_shard_dir=True,
    )
    extracted_checkpoints = _concat_checkpoint(
        shard_dirs,
        "checkpoints/extracted.jsonl",
        out_dir / "checkpoints" / "extracted.jsonl",
        add_shard_dir=True,
    )
    shard_manifests = _load_manifests(shard_dirs)
    rule_counts = _sum_rule_counts(shard_manifests)

    manifest = {
        "merged_from": [str(path) for path in shard_dirs],
        "shards": len(shard_dirs),
        "rows_completed": len(annotations),
        "duplicate_annotations_skipped": duplicate_annotations,
        "stage_errors": len(errors),
        "retrieval_trace_rows": len(retrieval_rows),
        "classified_checkpoint_rows": len(classified_checkpoints),
        "extracted_checkpoint_rows": len(extracted_checkpoints),
        "postprocessing_rule_counts": rule_counts,
        "shard_manifests": shard_manifests,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "shard_manifests"}, ensure_ascii=False, indent=2))
    return 0


def _merge_annotations(shard_dirs: Iterable[Path], out_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    records: dict[str, dict[str, Any]] = {}
    duplicates = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for shard_dir in shard_dirs:
            for record in _read_jsonl(shard_dir / "annotations.jsonl"):
                post_id = str(record.get("post_id") or record.get("record_id") or "")
                if not post_id:
                    continue
                if post_id in records:
                    duplicates += 1
                    continue
                records[post_id] = record
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records, duplicates


def _concat_jsonl(shard_dirs: Iterable[Path], relative_path: str, out_path: Path, *, add_shard_dir: bool) -> list[dict[str, Any]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as out_file:
        for shard_dir in shard_dirs:
            for record in _read_jsonl(shard_dir / relative_path):
                if add_shard_dir:
                    record = {"shard_dir": str(shard_dir), **record}
                records.append(record)
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _concat_checkpoint(shard_dirs: Iterable[Path], relative_path: str, out_path: Path, *, add_shard_dir: bool) -> list[dict[str, Any]]:
    return _concat_jsonl(shard_dirs, relative_path, out_path, add_shard_dir=add_shard_dir)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _load_manifests(shard_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    manifests = []
    for shard_dir in shard_dirs:
        path = shard_dir / "run_manifest.json"
        if not path.exists():
            manifests.append({"shard_dir": str(shard_dir), "missing": True})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests.append({"shard_dir": str(shard_dir), **payload})
    return manifests


def _sum_rule_counts(manifests: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for manifest in manifests:
        rule_counts = manifest.get("postprocessing_rule_counts", {})
        if not isinstance(rule_counts, dict):
            continue
        for rule, count in rule_counts.items():
            counts[str(rule)] = counts.get(str(rule), 0) + int(count)
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
