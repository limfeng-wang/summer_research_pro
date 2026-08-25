#!/usr/bin/env python3
"""Publication-level evaluation for the locked 150-row holdout set.

This script compares a frozen `annotations.jsonl` run against
`data/raw_eval_holdout_150_gold.xlsx` and writes paper-ready metric tables:

- classification precision/recall/F1 by language
- CSM domain-level precision/recall/F1 by language
- evidence-span support and rejection/hallucination proxy rates
- normalized concept agreement against gold English keyword/concept columns
- bootstrap confidence intervals over posts
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


DOMAINS = {
    "Perceived Cause": {
        "gold_label": "感知病因_标签",
        "gold_evidence": "感知病因_证据",
        "gold_concept": "感知病因_关键词_EN",
    },
    "Symptom Description": {
        "gold_label": "症状描述_标签",
        "gold_evidence": "症状描述_证据",
        "gold_concept": "症状描述_关键词_EN",
    },
    "Perceived Consequences": {
        "gold_label": "感知后果_标签",
        "gold_evidence": "感知后果_证据",
        "gold_concept": "感知后果_关键词_EN",
    },
    "Coping and Management": {
        "gold_label": "应对与管理_标签",
        "gold_evidence": "应对与管理_证据",
        "gold_concept": "应对与管理_关键词_EN",
    },
    "Emotional Expression": {
        "gold_label": "情绪表达_标签",
        "gold_evidence": "情绪表达_证据",
        "gold_concept": "情绪表达_关键词_EN",
    },
}

PUNCT_TABLE = str.maketrans({char: " " for char in string.punctuation})
SPLIT_RE = re.compile(r"[,;，；、|/\n\r]+")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/raw_eval_holdout_150_gold.xlsx")
    parser.add_argument("--pred", required=True, help="Predicted annotations.jsonl from the frozen model run")
    parser.add_argument("--out-dir", required=True, help="Directory for evaluation tables")
    parser.add_argument("--manifest", default="data/project_split_manifest.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)

    gold = load_gold(Path(args.gold))
    pred = load_predictions(Path(args.pred))
    ids = sorted(set(gold) & set(pred))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluation = build_evaluation(gold, pred, ids)
    ci = bootstrap_confidence_intervals(
        gold,
        pred,
        ids,
        iterations=args.bootstrap,
        seed=args.seed,
    )
    leakage = load_leakage_summary(Path(args.manifest))

    write_csv(out_dir / "classification_metrics.csv", evaluation["classification_metrics"])
    write_csv(out_dir / "csm_domain_metrics.csv", evaluation["domain_metrics"])
    write_csv(out_dir / "normalization_metrics.csv", evaluation["normalization_metrics"])
    write_csv(out_dir / "unit_quality_by_language.csv", evaluation["unit_quality_by_language"])
    write_csv(out_dir / "classification_disagreements.csv", evaluation["classification_disagreements"])
    write_csv(out_dir / "csm_domain_disagreements.csv", evaluation["domain_disagreements"])
    write_csv(out_dir / "normalization_mismatches.csv", evaluation["normalization_mismatches"])
    write_csv(out_dir / "unit_quality_rows.csv", evaluation["unit_quality_rows"])

    summary = {
        "gold": args.gold,
        "pred": args.pred,
        "manifest": args.manifest,
        "gold_rows": len(gold),
        "pred_rows": len(pred),
        "matched_rows": len(ids),
        "missing_prediction_rows": sorted(set(gold) - set(pred)),
        "extra_prediction_rows": sorted(set(pred) - set(gold)),
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_seed": args.seed,
        "overall": evaluation["overall"],
        "confidence_intervals": ci,
        "leakage_summary": leakage,
    }
    (out_dir / "publication_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_evaluation(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    classification_metrics = classification_metric_rows(gold, pred, ids)
    domain_metrics = domain_metric_rows(gold, pred, ids)
    normalization_metrics, normalization_mismatches = normalization_metric_rows(gold, pred, ids)
    unit_quality_rows = unit_quality_detail_rows(gold, pred, ids)
    unit_quality_by_language = unit_quality_summary_rows(unit_quality_rows)
    classification_disagreements = classification_disagreement_rows(gold, pred, ids)
    domain_disagreements = domain_disagreement_rows(gold, pred, ids)

    overall = {
        "eligibility": eligibility_metrics(gold, pred, ids),
        "classification_exact_accuracy": exact_label_accuracy(gold, pred, ids),
        "csm_domain_macro_f1": macro_domain_f1(gold, pred, ids),
        "normalization_micro": concept_set_metrics(gold, pred, ids, group_name="overall"),
        "unit_quality": unit_quality_summary(unit_quality_rows),
    }
    return {
        "overall": overall,
        "classification_metrics": classification_metrics,
        "domain_metrics": domain_metrics,
        "normalization_metrics": normalization_metrics,
        "normalization_mismatches": normalization_mismatches,
        "unit_quality_rows": unit_quality_rows,
        "unit_quality_by_language": unit_quality_by_language,
        "classification_disagreements": classification_disagreements,
        "domain_disagreements": domain_disagreements,
    }


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_excel(path, sheet_name="1")
    rows: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        post_id = str(row["record_id"]).strip()
        domain_presence = {
            domain: is_positive_label(row[config["gold_label"]])
            for domain, config in DOMAINS.items()
        }
        domain_concepts = {
            domain: parse_concept_set(row[config["gold_concept"]])
            for domain, config in DOMAINS.items()
        }
        rows[post_id] = {
            "post_id": post_id,
            "country": clean_string(row["国家"]),
            "language": clean_string(row["语言"]),
            "title": clean_string(row.get("原文标题", "")),
            "text": clean_string(row.get("原文正文", "")),
            "relevance_label": clean_label(row["牙痛相关性"]),
            "experiencer_label": clean_label(row["经历主体"]),
            "content_function": clean_label(row["内容功能"]),
            "domain_presence": domain_presence,
            "domain_concepts": domain_concepts,
        }
    return rows


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
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
            source_text = source_text_from_prediction(row)
            units = row.get("units") or []
            accepted_units = [unit for unit in units if is_accepted_unit(unit)]
            domain_presence = {
                domain: any(unit.get("domain") == domain for unit in accepted_units)
                for domain in DOMAINS
            }
            domain_concepts = {
                domain: {
                    normalize_concept(unit.get("normalized_concept_en"))
                    for unit in accepted_units
                    if unit.get("domain") == domain and normalize_concept(unit.get("normalized_concept_en"))
                }
                for domain in DOMAINS
            }
            rows[post_id] = {
                "post_id": post_id,
                "country": clean_string(row.get("country")),
                "language": clean_string(row.get("language")),
                "relevance_label": clean_label(row.get("relevance_label")),
                "experiencer_label": clean_label(row.get("experiencer_label")),
                "content_function": clean_label(row.get("content_function")),
                "source_text": source_text,
                "units": units,
                "accepted_units": accepted_units,
                "domain_presence": domain_presence,
                "domain_concepts": domain_concepts,
            }
    return rows


def classification_metric_rows(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = [
        ("relevance", "relevance_label"),
        ("experiencer", "experiencer_label"),
        ("content_function", "content_function"),
        ("eligibility", "eligibility"),
    ]
    groups = [("overall", "overall", ids)] + language_groups(gold, ids)
    for group_type, group_value, group_ids in groups:
        for task, field in tasks:
            gold_values = [classification_value(gold[post_id], field) for post_id in group_ids]
            pred_values = [classification_value(pred[post_id], field) for post_id in group_ids]
            rows.extend(multiclass_metric_rows(task, group_type, group_value, gold_values, pred_values))
    return rows


def domain_metric_rows(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    groups = [("overall", "overall", ids)] + language_groups(gold, ids)
    for group_type, group_value, group_ids in groups:
        for domain in DOMAINS:
            gold_values = [gold[post_id]["domain_presence"][domain] for post_id in group_ids]
            pred_values = [pred[post_id]["domain_presence"][domain] for post_id in group_ids]
            counts = binary_counts(gold_values, pred_values)
            rows.append({"group_type": group_type, "group": group_value, "domain": domain, **prf_row(counts)})
    return rows


def normalization_metric_rows(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    mismatches = []
    groups = [("overall", "overall", ids)] + language_groups(gold, ids)
    for group_type, group_value, group_ids in groups:
        rows.append({"group_type": group_type, "group": group_value, **concept_set_metrics(gold, pred, group_ids, "all_domains")})
        for domain in DOMAINS:
            rows.append(
                {
                    "group_type": group_type,
                    "group": group_value,
                    **concept_set_metrics(gold, pred, group_ids, domain, domain=domain),
                }
            )
    for post_id in ids:
        for domain in DOMAINS:
            gold_set = gold[post_id]["domain_concepts"][domain]
            pred_set = pred[post_id]["domain_concepts"][domain]
            if gold_set == pred_set:
                continue
            mismatches.append(
                {
                    "post_id": post_id,
                    "country": gold[post_id]["country"],
                    "language": gold[post_id]["language"],
                    "domain": domain,
                    "gold_concepts": "; ".join(sorted(gold_set)),
                    "pred_concepts": "; ".join(sorted(pred_set)),
                    "missing_gold_concepts": "; ".join(sorted(gold_set - pred_set)),
                    "extra_pred_concepts": "; ".join(sorted(pred_set - gold_set)),
                    "text": combined_gold_text(gold[post_id]),
                }
            )
    return rows, mismatches


def concept_set_metrics(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
    group_name: str,
    *,
    domain: str | None = None,
) -> dict[str, Any]:
    tp = fp = fn = 0
    domains = [domain] if domain else list(DOMAINS)
    for post_id in ids:
        for item_domain in domains:
            gold_set = gold[post_id]["domain_concepts"][item_domain]
            pred_set = pred[post_id]["domain_concepts"][item_domain]
            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)
    return {"metric": "normalization", "domain": group_name, **prf_row({"tp": tp, "fp": fp, "fn": fn, "tn": 0})}


def unit_quality_detail_rows(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for post_id in ids:
        pred_row = pred[post_id]
        source_text = pred_row["source_text"] or combined_gold_text(gold[post_id])
        for unit in pred_row["units"]:
            span = clean_string(unit.get("evidence_span_original"))
            judge_verdict = clean_string(unit.get("judge_verdict"))
            support_type = clean_string(unit.get("support_type"))
            assertion = clean_string(unit.get("assertion"))
            span_found = bool(span and span in source_text)
            accepted = is_accepted_unit(unit)
            hallucination_proxy = (
                not span_found
                or judge_verdict == "reject"
                or support_type == "unsupported"
            )
            rows.append(
                {
                    "post_id": post_id,
                    "country": gold[post_id]["country"],
                    "language": gold[post_id]["language"],
                    "unit_id": unit.get("unit_id", ""),
                    "domain": unit.get("domain", ""),
                    "judge_verdict": judge_verdict,
                    "support_type": support_type,
                    "assertion": assertion,
                    "accepted_for_analysis": int(accepted),
                    "span_found_in_source": int(span_found),
                    "hallucination_proxy": int(hallucination_proxy),
                    "normalized_concept_en": unit.get("normalized_concept_en", ""),
                    "evidence_span_original": span,
                }
            )
    return rows


def unit_quality_summary_rows(unit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {("overall", "overall"): unit_rows}
    for language in sorted({row["language"] for row in unit_rows if row["language"]}):
        groups[("language", language)] = [row for row in unit_rows if row["language"] == language]
    for domain in sorted({row["domain"] for row in unit_rows if row["domain"]}):
        groups[("domain", domain)] = [row for row in unit_rows if row["domain"] == domain]
    for (group_type, group_value), rows_for_group in groups.items():
        rows.append({"group_type": group_type, "group": group_value, **unit_quality_summary(rows_for_group)})
    return rows


def unit_quality_summary(unit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(unit_rows)
    accepted = sum(int(row["accepted_for_analysis"]) for row in unit_rows)
    span_found = sum(int(row["span_found_in_source"]) for row in unit_rows)
    hallucination = sum(int(row["hallucination_proxy"]) for row in unit_rows)
    rejected = sum(1 for row in unit_rows if row["judge_verdict"] == "reject")
    unsupported = sum(1 for row in unit_rows if row["support_type"] == "unsupported")
    return {
        "unit_count": total,
        "accepted_unit_count": accepted,
        "accepted_unit_rate": safe_div(accepted, total),
        "evidence_span_support_rate": safe_div(span_found, total),
        "rejected_unit_rate": safe_div(rejected, total),
        "unsupported_unit_rate": safe_div(unsupported, total),
        "hallucination_proxy_rate": safe_div(hallucination, total),
    }


def classification_disagreement_rows(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for post_id in ids:
        gold_label = label_tuple(gold[post_id])
        pred_label = label_tuple(pred[post_id])
        if gold_label == pred_label:
            continue
        rows.append(
            {
                "post_id": post_id,
                "country": gold[post_id]["country"],
                "language": gold[post_id]["language"],
                "gold_relevance": gold_label[0],
                "pred_relevance": pred_label[0],
                "gold_experiencer": gold_label[1],
                "pred_experiencer": pred_label[1],
                "gold_content_function": gold_label[2],
                "pred_content_function": pred_label[2],
                "gold_eligible": int(is_eligible(gold[post_id])),
                "pred_eligible": int(is_eligible(pred[post_id])),
                "text": combined_gold_text(gold[post_id]),
            }
        )
    return rows


def domain_disagreement_rows(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for post_id in ids:
        for domain in DOMAINS:
            gold_value = gold[post_id]["domain_presence"][domain]
            pred_value = pred[post_id]["domain_presence"][domain]
            if gold_value == pred_value:
                continue
            rows.append(
                {
                    "post_id": post_id,
                    "country": gold[post_id]["country"],
                    "language": gold[post_id]["language"],
                    "domain": domain,
                    "error_type": "FN" if gold_value and not pred_value else "FP",
                    "gold_present": int(gold_value),
                    "pred_present": int(pred_value),
                    "gold_concepts": "; ".join(sorted(gold[post_id]["domain_concepts"][domain])),
                    "pred_concepts": "; ".join(sorted(pred[post_id]["domain_concepts"][domain])),
                    "text": combined_gold_text(gold[post_id]),
                }
            )
    return rows


def bootstrap_confidence_intervals(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    rng = random.Random(seed)
    metric_functions: dict[str, Callable[[list[str]], float | None]] = {
        "eligibility_f1": lambda sample: eligibility_metrics(gold, pred, sample)["f1"],
        "classification_exact_accuracy": lambda sample: exact_label_accuracy(gold, pred, sample),
        "csm_domain_macro_f1": lambda sample: macro_domain_f1(gold, pred, sample),
        "normalization_micro_f1": lambda sample: concept_set_metrics(gold, pred, sample, "overall")["f1"],
    }
    for domain in DOMAINS:
        metric_functions[f"domain_f1::{domain}"] = (
            lambda sample, item_domain=domain: domain_f1(gold, pred, sample, item_domain)
        )
    values: dict[str, list[float]] = {name: [] for name in metric_functions}
    if not ids or iterations <= 0:
        return {name: {"mean": None, "ci_low": None, "ci_high": None} for name in metric_functions}
    for _ in range(iterations):
        sample = [rng.choice(ids) for _ in ids]
        for name, func in metric_functions.items():
            value = func(sample)
            if value is not None:
                values[name].append(float(value))
    return {name: percentile_interval(sample_values) for name, sample_values in values.items()}


def eligibility_metrics(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    counts = binary_counts([is_eligible(gold[post_id]) for post_id in ids], [is_eligible(pred[post_id]) for post_id in ids])
    return prf_row(counts)


def exact_label_accuracy(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> float:
    return safe_div(sum(label_tuple(gold[post_id]) == label_tuple(pred[post_id]) for post_id in ids), len(ids))


def macro_domain_f1(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> float:
    values = [domain_f1(gold, pred, ids, domain) for domain in DOMAINS]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def domain_f1(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
    domain: str,
) -> float:
    counts = binary_counts(
        [gold[post_id]["domain_presence"][domain] for post_id in ids],
        [pred[post_id]["domain_presence"][domain] for post_id in ids],
    )
    return prf_row(counts)["f1"]


def multiclass_metric_rows(task: str, group_type: str, group_value: str, gold_values: list[str], pred_values: list[str]) -> list[dict[str, Any]]:
    rows = []
    labels = sorted(set(gold_values) | set(pred_values))
    for label in labels:
        counts = binary_counts([value == label for value in gold_values], [value == label for value in pred_values])
        rows.append(
            {
                "task": task,
                "group_type": group_type,
                "group": group_value,
                "label": label,
                **prf_row(counts),
            }
        )
    macro_f1 = sum(row["f1"] for row in rows) / len(rows) if rows else 0.0
    rows.append(
        {
            "task": task,
            "group_type": group_type,
            "group": group_value,
            "label": "macro_avg",
            "tp": "",
            "fp": "",
            "fn": "",
            "tn": "",
            "support": len(gold_values),
            "precision": "",
            "recall": "",
            "f1": macro_f1,
        }
    )
    rows.append(
        {
            "task": task,
            "group_type": group_type,
            "group": group_value,
            "label": "accuracy",
            "tp": "",
            "fp": "",
            "fn": "",
            "tn": "",
            "support": len(gold_values),
            "precision": "",
            "recall": "",
            "f1": safe_div(sum(g == p for g, p in zip(gold_values, pred_values)), len(gold_values)),
        }
    )
    return rows


def binary_counts(gold_values: Iterable[bool], pred_values: Iterable[bool]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for gold_value, pred_value in zip(gold_values, pred_values):
        if gold_value and pred_value:
            tp += 1
        elif (not gold_value) and pred_value:
            fp += 1
        elif gold_value and not pred_value:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf_row(counts: dict[str, int]) -> dict[str, Any]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "support": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def percentile_interval(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    ordered = sorted(values)
    low_index = max(0, int(0.025 * (len(ordered) - 1)))
    high_index = min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)))
    return {
        "mean": sum(ordered) / len(ordered),
        "ci_low": ordered[low_index],
        "ci_high": ordered[high_index],
    }


def language_groups(gold: dict[str, dict[str, Any]], ids: list[str]) -> list[tuple[str, str, list[str]]]:
    return [
        ("language", language, [post_id for post_id in ids if gold[post_id]["language"] == language])
        for language in sorted({gold[post_id]["language"] for post_id in ids})
    ]


def classification_value(row: dict[str, Any], field: str) -> str:
    if field == "eligibility":
        return "eligible" if is_eligible(row) else "not_eligible"
    return str(row.get(field) or "None")


def label_tuple(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        classification_value(row, "relevance_label"),
        classification_value(row, "experiencer_label"),
        classification_value(row, "content_function"),
    )


def is_eligible(row: dict[str, Any]) -> bool:
    return (
        row.get("relevance_label") == "R1"
        and row.get("experiencer_label") in {"E1", "E2"}
        and row.get("content_function") in {"C1", "C2"}
    )


def is_accepted_unit(unit: dict[str, Any]) -> bool:
    return (
        unit.get("judge_verdict") == "accept"
        and unit.get("support_type") != "unsupported"
        and unit.get("assertion") != "negated"
    )


def is_positive_label(value: Any) -> bool:
    label = clean_label(value)
    if label is None:
        return False
    return label.lower() not in {"0", "false", "no", "negative", "none", "na", "nan", "r0"}


def parse_concept_set(value: Any) -> set[str]:
    text = clean_string(value)
    if not text:
        return set()
    return {
        normalized
        for raw in SPLIT_RE.split(text)
        if (normalized := normalize_concept(raw))
    }


def normalize_concept(value: Any) -> str:
    text = clean_string(value).lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"", "none", "na", "nan", "unmapped"}:
        return ""
    return text


def clean_label(value: Any) -> str | None:
    text = clean_string(value)
    if not text or text.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return None
    return text


def clean_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def combined_gold_text(row: dict[str, Any]) -> str:
    return "\n".join(part for part in [row.get("title", ""), row.get("text", "")] if part).strip()


def source_text_from_prediction(row: dict[str, Any]) -> str:
    if row.get("text_clean"):
        return clean_string(row.get("text_clean"))
    return "\n".join(
        part
        for part in [clean_string(row.get("original_title")), clean_string(row.get("original_text"))]
        if part
    ).strip()


def load_leakage_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"manifest_found": False}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "manifest_found": True,
        "disjointness_ok": manifest.get("disjointness_audit", {}).get("ok"),
        "overlaps": manifest.get("disjointness_audit", {}).get("overlaps", []),
        "eval_holdout_no_gold_rows": manifest.get("eval_holdout_no_gold_rows"),
        "main_excluded_rows": manifest.get("main_excluded_rows"),
        "main_exclusion_reasons": manifest.get("main_exclusion_reasons", {}),
    }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
