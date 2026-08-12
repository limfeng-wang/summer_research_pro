#!/usr/bin/env python3
"""Drop stale R1->R0 relevance-safeguard demotions from classified checkpoints.

This is an emergency repair for runs produced before the Japanese relevance
cue expansion. The old classifier checkpoint cannot recover E/C labels for
demoted rows because the combined classifier returned early after R0. The
safe repair is to remove only those checkpoint rows and let --resume classify
them again with the patched code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RULE = "relevance.toothache_evidence_required"


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair stale relevance demotions in classified.jsonl.")
    parser.add_argument("paths", nargs="+", help="classified.jsonl file(s) to inspect")
    parser.add_argument("--drop", action="store_true", help="Rewrite files, dropping repair-target rows")
    parser.add_argument("--backup-suffix", default=".bak_relevance_demotions", help="Backup suffix for --drop")
    args = parser.parse_args()

    total_seen = 0
    total_bad_json = 0
    total_targets = 0
    for raw_path in args.paths:
        path = Path(raw_path)
        stats = inspect_or_repair(path, drop=args.drop, backup_suffix=args.backup_suffix)
        total_seen += stats["seen"]
        total_bad_json += stats["bad_json"]
        total_targets += stats["targets"]
        print(json.dumps({"path": str(path), **stats}, ensure_ascii=False))

    print(
        json.dumps(
            {
                "mode": "drop" if args.drop else "dry_run",
                "seen": total_seen,
                "bad_json": total_bad_json,
                "targets": total_targets,
            },
            ensure_ascii=False,
        )
    )
    return 0


def inspect_or_repair(path: Path, *, drop: bool, backup_suffix: str) -> dict[str, int]:
    kept_lines: list[str] = []
    seen = 0
    bad_json = 0
    targets = 0

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                if not drop:
                    kept_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            if is_repair_target(row):
                targets += 1
                if not drop:
                    kept_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            kept_lines.append(line if line.endswith("\n") else line + "\n")

    if drop:
        backup = path.with_name(path.name + backup_suffix)
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(kept_lines), encoding="utf-8")
        tmp.replace(path)

    return {
        "seen": seen,
        "kept": len(kept_lines),
        "bad_json": bad_json,
        "targets": targets,
        "dropped": targets + bad_json if drop else 0,
    }


def is_repair_target(row: dict[str, Any]) -> bool:
    result = row.get("result", row)
    trace = row.get("trace", {})
    if result.get("relevance_label") != "R0":
        return False
    if trace.get("raw_labels", {}).get("relevance_label") != "R1":
        return False
    for rule in trace.get("postprocessing_rules", []):
        if not isinstance(rule, dict):
            continue
        if (
            rule.get("rule") == RULE
            and rule.get("field") == "relevance_label"
            and rule.get("before") == "R1"
            and rule.get("after") == "R0"
        ):
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
