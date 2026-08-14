#!/usr/bin/env python3
"""Build a follow-up 10k input prioritizing remaining KOR and CHI rows.

The default selection is:
- all unprocessed KOR rows
- all unprocessed CHI rows
- enough unprocessed JPN rows to reach 10,000 total

Previously processed post IDs are read from an annotations JSONL file or an
output directory containing annotations.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw_main_llm_input_no_gold.jsonl")
    parser.add_argument("--processed", required=True, help="Prior output dir or annotations.jsonl to exclude")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--target-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--jpn-mode",
        choices=("random", "next"),
        default="random",
        help="'random' samples remaining JPN rows deterministically; 'next' takes earliest remaining JPN rows.",
    )
    parser.add_argument(
        "--preserve-output-order",
        action="store_true",
        help="Keep selected rows grouped by KOR, CHI, then JPN. Default shuffles rows to balance contiguous shards.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    processed_ids = _load_processed_ids(Path(args.processed))

    remaining: dict[str, list[dict[str, Any]]] = {"KOR": [], "CHI": [], "JPN": []}
    total_rows = 0
    skipped_processed = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            post_id = str(row.get("post_id") or row.get("record_id") or "").strip()
            if post_id in processed_ids:
                skipped_processed += 1
                continue
            country = str(row.get("country") or "")
            if country in remaining:
                remaining[country].append(row)

    selected = list(remaining["KOR"]) + list(remaining["CHI"])
    jpn_needed = args.target_rows - len(selected)
    if jpn_needed < 0:
        raise SystemExit(
            f"KOR+CHI remaining rows ({len(selected)}) exceed target rows ({args.target_rows}); "
            "increase --target-rows or change selection logic."
        )
    if jpn_needed > len(remaining["JPN"]):
        raise SystemExit(
            f"Need {jpn_needed} JPN rows but only {len(remaining['JPN'])} unprocessed JPN rows are available."
        )

    if args.jpn_mode == "random":
        rng = random.Random(args.seed)
        selected_jpn = rng.sample(remaining["JPN"], jpn_needed)
        selected_jpn.sort(key=lambda row: str(row.get("post_id") or row.get("record_id") or ""))
    else:
        selected_jpn = remaining["JPN"][:jpn_needed]

    selected.extend(selected_jpn)
    if not args.preserve_output_order:
        rng = random.Random(args.seed + 1)
        rng.shuffle(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path),
        "processed": str(args.processed),
        "output": str(output_path),
        "target_rows": args.target_rows,
        "seed": args.seed,
        "jpn_mode": args.jpn_mode,
        "preserve_output_order": args.preserve_output_order,
        "input_rows": total_rows,
        "processed_ids_excluded": len(processed_ids),
        "input_rows_skipped_as_processed": skipped_processed,
        "remaining_by_country": {country: len(rows) for country, rows in remaining.items()},
        "selected_by_country": dict(Counter(str(row.get("country") or "") for row in selected)),
        "selected_rows": len(selected),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_processed_ids(path: Path) -> set[str]:
    candidates = [path]
    if path.is_dir():
        shard_dirs = sorted(item for item in path.glob("shard_*") if item.is_dir())
        if shard_dirs:
            candidates = [item / "annotations.jsonl" for item in shard_dirs]
        else:
            candidates = [path / "annotations.jsonl"]

    ids: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f"Processed annotations file does not exist: {candidate}")
        with candidate.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {candidate}:{line_number}: {exc}") from exc
                post_id = str(row.get("post_id") or row.get("record_id") or "").strip()
                if post_id:
                    ids.add(post_id)
    return ids


if __name__ == "__main__":
    raise SystemExit(main())
