#!/usr/bin/env python3
"""Evaluate model annotations against raw_eval_holdout_150_gold.xlsx."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/raw_eval_holdout_150_gold.xlsx")
    parser.add_argument("--pred", required=True, help="Predicted annotations.jsonl")
    parser.add_argument("--out-dir", default="", help="Optional directory for metrics/disagreement files")
    args = parser.parse_args()

    gold_rows = _load_gold(Path(args.gold))
    pred_rows = _load_predictions(Path(args.pred))
    common_ids = sorted(set(gold_rows) & set(pred_rows))
    missing_pred = sorted(set(gold_rows) - set(pred_rows))
    extra_pred = sorted(set(pred_rows) - set(gold_rows))

    metrics = _metrics(gold_rows, pred_rows, common_ids)
    metrics["gold"] = args.gold
    metrics["pred"] = args.pred
    metrics["gold_rows"] = len(gold_rows)
    metrics["pred_rows"] = len(pred_rows)
    metrics["matched_rows"] = len(common_ids)
    metrics["missing_prediction_rows"] = len(missing_pred)
    metrics["extra_prediction_rows"] = len(extra_pred)
    metrics["missing_prediction_examples"] = missing_pred[:20]
    metrics["extra_prediction_examples"] = extra_pred[:20]

    disagreements = _disagreements(gold_rows, pred_rows, common_ids)
    metrics["top_confusions"] = [
        {"gold": _label_string(gold), "pred": _label_string(pred), "count": count}
        for (gold, pred), count in Counter(
            (gold_rows[post_id]["label"], pred_rows[post_id]["label"])
            for post_id in common_ids
            if gold_rows[post_id]["label"] != pred_rows[post_id]["label"]
        ).most_common(25)
    ]

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "holdout_eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(out_dir / "holdout_eval_disagreements.csv", disagreements)
    return 0


def _load_gold(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_excel(path, sheet_name="1")
    rows: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        post_id = str(row["record_id"]).strip()
        label = (
            str(row["牙痛相关性"]).strip(),
            _clean_label(row["经历主体"]),
            _clean_label(row["内容功能"]),
        )
        rows[post_id] = {
            "post_id": post_id,
            "country": str(row["国家"]).strip(),
            "language": str(row["语言"]).strip(),
            "title": _clean_text(row.get("原文标题", "")),
            "text": _clean_text(row.get("原文正文", "")),
            "label": label,
            "eligible": _is_eligible(label),
            "domain_presence": _gold_domain_presence(row),
        }
    return rows


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            post_id = str(row.get("post_id") or row.get("record_id") or "").strip()
            if not post_id:
                continue
            label = (
                row.get("relevance_label"),
                row.get("experiencer_label"),
                row.get("content_function"),
            )
            units = row.get("units") or []
            rows[post_id] = {
                "post_id": post_id,
                "country": row.get("country"),
                "language": row.get("language"),
                "label": label,
                "eligible": _is_eligible(label),
                "unit_count": len(units),
                "accepted_unit_count": sum(1 for unit in units if unit.get("judge_verdict") == "accept"),
                "unit_domains": Counter(unit.get("domain") for unit in units),
                "accepted_unit_domains": Counter(
                    unit.get("domain") for unit in units if unit.get("judge_verdict") == "accept"
                ),
            }
    return rows


def _metrics(
    gold_rows: dict[str, dict[str, Any]],
    pred_rows: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    label_metrics = _label_metrics(gold_rows, pred_rows, ids)
    by_country = {}
    for country in sorted({gold_rows[post_id]["country"] for post_id in ids}):
        country_ids = [post_id for post_id in ids if gold_rows[post_id]["country"] == country]
        by_country[country] = _label_metrics(gold_rows, pred_rows, country_ids)
    return {
        **label_metrics,
        "by_country": by_country,
        "gold_label_counts": _counter_dict(Counter(_label_string(gold_rows[post_id]["label"]) for post_id in ids)),
        "pred_label_counts": _counter_dict(Counter(_label_string(pred_rows[post_id]["label"]) for post_id in ids)),
    }


def _label_metrics(
    gold_rows: dict[str, dict[str, Any]],
    pred_rows: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    n = len(ids)
    if not n:
        return {}
    relevance_correct = sum(gold_rows[post_id]["label"][0] == pred_rows[post_id]["label"][0] for post_id in ids)
    experiencer_correct = sum(gold_rows[post_id]["label"][1] == pred_rows[post_id]["label"][1] for post_id in ids)
    content_correct = sum(gold_rows[post_id]["label"][2] == pred_rows[post_id]["label"][2] for post_id in ids)
    combo_correct = sum(gold_rows[post_id]["label"] == pred_rows[post_id]["label"] for post_id in ids)
    tp = sum(gold_rows[post_id]["eligible"] and pred_rows[post_id]["eligible"] for post_id in ids)
    fp = sum((not gold_rows[post_id]["eligible"]) and pred_rows[post_id]["eligible"] for post_id in ids)
    fn = sum(gold_rows[post_id]["eligible"] and not pred_rows[post_id]["eligible"] for post_id in ids)
    tn = sum((not gold_rows[post_id]["eligible"]) and not pred_rows[post_id]["eligible"] for post_id in ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "relevance_accuracy": relevance_correct / n,
        "experiencer_accuracy": experiencer_correct / n,
        "content_accuracy": content_correct / n,
        "exact_rec_accuracy": combo_correct / n,
        "eligible_tp": tp,
        "eligible_fp": fp,
        "eligible_fn": fn,
        "eligible_tn": tn,
        "eligible_precision": precision,
        "eligible_recall": recall,
        "eligible_f1": f1,
    }


def _disagreements(
    gold_rows: dict[str, dict[str, Any]],
    pred_rows: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for post_id in ids:
        gold = gold_rows[post_id]
        pred = pred_rows[post_id]
        if gold["label"] == pred["label"]:
            continue
        rows.append(
            {
                "post_id": post_id,
                "country": gold["country"],
                "language": gold["language"],
                "gold_label": _label_string(gold["label"]),
                "pred_label": _label_string(pred["label"]),
                "gold_eligible": gold["eligible"],
                "pred_eligible": pred["eligible"],
                "pred_unit_count": pred["unit_count"],
                "pred_accepted_unit_count": pred["accepted_unit_count"],
                "gold_domains_present": ",".join(gold["domain_presence"]),
                "title": gold["title"],
                "text": gold["text"],
            }
        )
    return rows


def _gold_domain_presence(row: Any) -> list[str]:
    domains = {
        "Perceived Cause": "感知病因_标签",
        "Symptom Description": "症状描述_标签",
        "Perceived Consequences": "感知后果_标签",
        "Coping and Management": "应对与管理_标签",
        "Emotional Expression": "情绪表达_标签",
    }
    return [domain for domain, column in domains.items() if _clean_label(row[column]) is not None]


def _is_eligible(label: tuple[Any, Any, Any]) -> bool:
    return label[0] == "R1" and label[1] in {"E1", "E2"} and label[2] in {"C1", "C2"}


def _label_string(label: tuple[Any, Any, Any]) -> str:
    return "|".join("None" if item is None else str(item) for item in label)


def _clean_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(counter.most_common())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

