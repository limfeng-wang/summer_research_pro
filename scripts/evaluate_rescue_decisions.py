#!/usr/bin/env python3
"""Evaluate first-pass + rescue eligibility against the 150-row gold workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/raw_eval_holdout_150_gold.xlsx")
    parser.add_argument("--first-pass", required=True, help="First-pass annotations.jsonl")
    parser.add_argument(
        "--decisions",
        action="append",
        required=True,
        help="rescue_decisions.jsonl file or directory containing rescue_decisions.jsonl. Repeat for shards.",
    )
    parser.add_argument("--out", default="", help="Optional JSON metrics output")
    args = parser.parse_args()

    gold = _load_gold(Path(args.gold))
    first = _load_first_pass(Path(args.first_pass))
    decisions = _load_decisions([Path(item) for item in args.decisions])
    rescued = {
        _post_id(row)
        for row in decisions
        if row.get("decision", {}).get("rescue_csm_eligible")
        and row.get("decision", {}).get("rescued_experiencer_label") in {"E1", "E2"}
        and row.get("decision", {}).get("rescued_content_function") in {"C1", "C2"}
    }
    ids = sorted(set(gold) & set(first))
    metrics = {
        "gold": args.gold,
        "first_pass_annotations": args.first_pass,
        "decisions": args.decisions,
        "matched_rows": len(ids),
        "first_pass": _metrics(gold, first, set(), ids),
        "first_pass_plus_rescue": _metrics(gold, first, rescued, ids),
        "rescued_rows": len(rescued),
        "rescued_gold_ids": len(rescued & set(gold)),
        "rescued_true_eligible": sum(gold[post_id]["eligible"] for post_id in rescued if post_id in gold),
        "rescued_false_eligible": sum(not gold[post_id]["eligible"] for post_id in rescued if post_id in gold),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _metrics(
    gold: dict[str, dict[str, Any]],
    first: dict[str, dict[str, Any]],
    rescued: set[str],
    ids: list[str],
) -> dict[str, Any]:
    overall = _eligibility_metrics(gold, first, rescued, ids)
    by_country = {}
    for country in sorted({gold[post_id]["country"] for post_id in ids}):
        country_ids = [post_id for post_id in ids if gold[post_id]["country"] == country]
        by_country[country] = _eligibility_metrics(gold, first, rescued, country_ids)
    return {**overall, "by_country": by_country}


def _eligibility_metrics(
    gold: dict[str, dict[str, Any]],
    first: dict[str, dict[str, Any]],
    rescued: set[str],
    ids: list[str],
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for post_id in ids:
        pred_eligible = first[post_id]["eligible"] or post_id in rescued
        gold_eligible = gold[post_id]["eligible"]
        if gold_eligible and pred_eligible:
            tp += 1
        elif (not gold_eligible) and pred_eligible:
            fp += 1
        elif gold_eligible and not pred_eligible:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(ids),
        "eligible_tp": tp,
        "eligible_fp": fp,
        "eligible_fn": fn,
        "eligible_tn": tn,
        "eligible_precision": precision,
        "eligible_recall": recall,
        "eligible_f1": f1,
    }


def _load_gold(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_excel(path, sheet_name="1")
    rows = {}
    for _, row in frame.iterrows():
        post_id = str(row["record_id"]).strip()
        label = (
            str(row["牙痛相关性"]).strip(),
            _clean_label(row["经历主体"]),
            _clean_label(row["内容功能"]),
        )
        rows[post_id] = {"country": str(row["国家"]).strip(), "eligible": _is_eligible(label)}
    return rows


def _load_first_pass(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _read_jsonl(path):
        post_id = _post_id(row)
        label = (row.get("relevance_label"), row.get("experiencer_label"), row.get("content_function"))
        rows[post_id] = {"country": row.get("country"), "eligible": _is_eligible(label)}
    return rows


def _load_decisions(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        decision_path = path / "rescue_decisions.jsonl" if path.is_dir() else path
        rows.extend(_read_jsonl(decision_path))
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _post_id(row: dict[str, Any]) -> str:
    return str(row.get("post_id") or row.get("record_id") or "").strip()


def _is_eligible(label: tuple[Any, Any, Any]) -> bool:
    return label[0] == "R1" and label[1] in {"E1", "E2"} and label[2] in {"C1", "C2"}


def _clean_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


if __name__ == "__main__":
    raise SystemExit(main())
