"""Merge first-pass, accepted rescue, and forced-extraction outputs explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [record.get("result", record) for record in records]


def has_accepted_unit(record: dict[str, Any]) -> bool:
    return any(unit.get("judge_verdict") == "accept" for unit in record.get("units", []))


def merge_runs(first_pass: list[Path], rescue: list[Path], forced: list[Path]) -> tuple[list[dict], list[dict]]:
    corpus = {}
    inputs = []

    def load(run: Path, relative: str) -> list[dict]:
        path = run / relative
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return read_rows(path)

    for run in first_pass:
        for record in load(run, "annotations.jsonl"):
            corpus.setdefault(record["post_id"], record)
    for run in rescue:
        for record in load(run, "annotations.jsonl"):
            if record["post_id"] in corpus and has_accepted_unit(record):
                corpus[record["post_id"]] = record
    for run in forced:
        for record in load(run, "annotations.jsonl"):
            if record["post_id"] in corpus:
                corpus[record["post_id"]] = record
    for run in forced:
        checkpoint = run / "checkpoints/extracted.jsonl"
        if checkpoint.exists():
            for record in load(run, "checkpoints/extracted.jsonl"):
                if record["post_id"] in corpus and not record.get("units"):
                    corpus[record["post_id"]] = record
    return list(corpus.values()), inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass", type=Path, action="append", required=True)
    parser.add_argument("--rescue", type=Path, action="append", default=[])
    parser.add_argument("--forced", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, inputs = merge_runs(args.first_pass, args.rescue, args.forced)
    primary = [
        record for record in rows
        if record.get("relevance_label") == "R1"
        and record.get("experiencer_label") == "E1"
        and record.get("content_function") in {"C1", "C2"}
        and has_accepted_unit(record)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("annotations.jsonl", rows), ("primary.jsonl", primary)):
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "inputs": inputs,
        "posts": len(rows),
        "countries": dict(Counter(record["country"] for record in rows)),
        "primary_posts": len(primary),
        "primary_countries": dict(Counter(record["country"] for record in primary)),
        "primary_verdict_accepted_units": sum(
            unit.get("judge_verdict") == "accept" for record in primary for unit in record.get("units", [])
        ),
        "rule": "First occurrence in first pass; rescue overlays only with accepted units; forced overlays also retain rejected-only and zero-unit results.",
    }
    (args.out_dir / "merge_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
