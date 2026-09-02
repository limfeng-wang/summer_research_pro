#!/usr/bin/env python3
"""Build a keyword mapping workbook from reviewed cluster decisions."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_REVIEW = Path("data/keyword_cluster_candidates_blinded.csv")
DEFAULT_INVENTORY = Path("outputs/keyword_report/canonical_keyword_mapping.xlsx")
DEFAULT_OUT = Path("outputs/keyword_report/canonical_keyword_mapping.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def split_labels(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split("\n") if part.strip()]


def parse_split_mapping(value: object, labels: list[str] | None = None) -> dict[str, str]:
    """Parse lines like `Canonical: label A; label B` or `Canonical = A + B`."""
    text = "" if pd.isna(value) else str(value).strip()
    mapping: dict[str, str] = {}
    if not text:
        return mapping
    if "separate labels" in text.casefold():
        return {label: label for label in labels or []}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            canonical, members = line.split(":", 1)
        elif "=" in line:
            canonical, members = line.split("=", 1)
        else:
            raise ValueError(f"Split mapping line lacks ':' or '=': {line!r}")
        canonical = canonical.strip()
        if not canonical:
            raise ValueError(f"Split mapping line has blank canonical label: {line!r}")
        for member in re.split(r";|\||\+", members):
            member = member.strip()
            if member:
                mapping[member] = canonical
    return mapping


def main() -> None:
    args = parse_args()
    review = pd.read_csv(args.review)
    inventory = pd.read_excel(args.inventory)
    for col in ["dimension", "keyword"]:
        inventory[col] = inventory[col].fillna("").astype(str).str.strip()

    canonical_by_key: dict[tuple[str, str], tuple[str, str, str]] = {}
    errors: list[str] = []
    for row in review.itertuples(index=False):
        cluster_id = str(row.cluster_id)
        dimension = str(row.csm_dimension).strip()
        decision = str(row.consensus_decision).strip()
        labels = split_labels(row.labels_in_proposed_family)
        canonical_value = "" if pd.isna(row.consensus_canonical_labels) else str(row.consensus_canonical_labels).strip()
        rule = "" if pd.isna(row.merge_rule_invoked) else str(row.merge_rule_invoked).strip()
        notes = "" if pd.isna(row.review_notes) else str(row.review_notes).strip()

        if decision == "merge_all":
            if not canonical_value:
                errors.append(f"{cluster_id}: merge_all missing consensus_canonical_labels")
                continue
            for label in labels:
                canonical_by_key[(dimension, label)] = (canonical_value, rule or "cluster_merge_all", notes)
        elif decision == "split":
            split_map = parse_split_mapping(canonical_value, labels)
            missing = [label for label in labels if label not in split_map]
            extra = [label for label in split_map if label not in labels]
            if missing:
                errors.append(f"{cluster_id}: split mapping missing labels: {missing}")
            if extra:
                errors.append(f"{cluster_id}: split mapping includes unknown labels: {extra}")
            for label, canonical in split_map.items():
                canonical_by_key[(dimension, label)] = (canonical, rule or "cluster_split", notes)
        elif decision == "do_not_merge":
            continue
        elif decision == "uncertain":
            errors.append(f"{cluster_id}: consensus_decision is uncertain")
        else:
            errors.append(f"{cluster_id}: unsupported consensus_decision {decision!r}")

    if errors:
        raise SystemExit("Cannot build mapping:\n" + "\n".join(errors[:40]))

    rows = []
    for row in inventory.itertuples(index=False):
        key = (str(row.dimension), str(row.keyword))
        canonical, rule, notes = canonical_by_key.get(key, (key[1], "", ""))
        rows.append(
            {
                "dimension": key[0],
                "keyword": key[1],
                "fine_canonical_keyword": canonical,
                "parent_canonical_keyword": canonical,
                "merge_rule_invoked": rule,
                "consensus_source": "cluster_review" if key in canonical_by_key else "retained_unreviewed_or_unmerged",
                "notes": notes,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame(rows)
    mapping.to_excel(args.out, sheet_name="canonical_mapping", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "review": str(args.review),
        "inventory": str(args.inventory),
        "out": str(args.out),
        "labels_total": int(len(mapping)),
        "labels_changed": int((mapping["keyword"] != mapping["fine_canonical_keyword"]).sum()),
        "reviewed_labels_changed_or_confirmed": int((mapping["consensus_source"] == "cluster_review").sum()),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
