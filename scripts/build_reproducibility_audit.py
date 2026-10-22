"""Audit the available manuscript artifacts without modifying study data."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dental_ai.classification_gold import canonical_post_id, source_text_fingerprint


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identifier(row):
    return str(row.get("post_id") or row.get("record_id") or "").strip()


def body(row):
    return str(row.get("original_text") or row.get("text") or "").strip()


def source(row):
    return str(row.get("text_clean") or "").strip() or "\n".join(
        part for part in [str(row.get("original_title") or "").strip(), body(row)] if part
    )


def keys(rows):
    return {
        "country_id": {f"{row['country']}:{canonical_post_id(identifier(row))}" for row in rows},
        "source_sha256": {source_text_fingerprint("", source(row)) for row in rows if source(row)},
        "body_sha256": {source_text_fingerprint("", body(row)) for row in rows if body(row)},
    }


def accepted(unit):
    return (
        unit.get("judge_verdict") == "accept"
        and unit.get("support_type") != "unsupported"
        and unit.get("assertion") != "negated"
    )


def primary(row):
    return (
        row.get("relevance_label") == "R1"
        and row.get("experiencer_label") == "E1"
        and row.get("content_function") in {"C1", "C2"}
    )


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/ablation_reproducibility_artifacts"))
    args = parser.parse_args()
    root = REPO_ROOT
    out = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "classification_development": root / "data/classification_gold_172.jsonl",
        "rag_reference": root / "data/csm_gold_50E1_10E2.json",
        "evaluation_input": root / "data/raw_eval_holdout_150_no_gold.jsonl",
        "production_input": root / "data/raw_main_llm_input_no_gold.jsonl",
        "final_labeled": root / "outputs/final_export/36k_labelled.jsonl",
        "final_primary": root / "outputs/final_export/r1_e1_c1c2_with_accepted_units.jsonl",
        "evaluation_predictions": root / "outputs/eval_holdout_150_latest/annotations.jsonl",
    }
    sets = {
        name: json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else read_jsonl(path)
        for name, path in paths.items()
    }
    set_keys = {name: keys(rows) for name, rows in sets.items()}
    overlap_rows = []
    overlap_details = []
    for left, right in itertools.combinations(sets, 2):
        entry = {"left": left, "right": right}
        details = {"left": left, "right": right}
        for key in set_keys[left]:
            overlap = sorted(set_keys[left][key] & set_keys[right][key])
            entry[f"{key}_overlap"] = len(overlap)
            details[f"{key}_count"] = len(overlap)
            details[f"{key}_sample"] = overlap[:10]
        overlap_rows.append(entry)
        overlap_details.append(details)
    write_csv(out / "split_overlap_counts.csv", overlap_rows)

    attrition = []
    unit_rows = []
    validation_rows = []
    lineage_rows = []
    for country in ["CHI", "JPN", "KOR"]:
        inputs = [row for row in sets["production_input"] if row["country"] == country]
        rows = [row for row in sets["final_labeled"] if row["country"] == country]
        eligible = [row for row in rows if primary(row)]
        stages = [
            ("production_input", len(inputs)),
            ("final_labeled", len(rows)),
            ("R1", sum(row.get("relevance_label") == "R1" for row in rows)),
            ("R1_E1", sum(row.get("relevance_label") == "R1" and row.get("experiencer_label") == "E1" for row in rows)),
            ("R1_E1_C1C2", len(eligible)),
            ("eligible_with_proposed_units", sum(bool(row.get("units")) for row in eligible)),
            ("eligible_with_verdict_accept", sum(any(unit.get("judge_verdict") == "accept" for unit in row.get("units", [])) for row in eligible)),
            ("eligible_with_schema_accepted_units", sum(any(accepted(unit) for unit in row.get("units", [])) for row in eligible)),
            ("eligible_with_accepted_and_span_supported_units", sum(any(accepted(unit) and unit.get("evidence_span_original", "") and unit["evidence_span_original"] in source(row) for unit in row.get("units", [])) for row in eligible)),
        ]
        previous = None
        for stage, retained in stages:
            attrition.append({
                "country": country, "stage": stage, "retained_posts": retained,
                "loss_from_previous_stage": "" if previous is None else previous - retained,
                "loss_pct_previous_stage": "" if previous in {None, 0} else round(100 * (previous - retained) / previous, 4),
                "retention_pct_input": round(100 * retained / len(inputs), 4),
            })
            previous = retained
        units = [(row, unit) for row in eligible for unit in row.get("units", [])]
        unit_rows.append({
            "country": country, "proposed_units": len(units),
            "verdict_accept_units": sum(unit.get("judge_verdict") == "accept" for _, unit in units),
            "schema_accepted_units": sum(accepted(unit) for _, unit in units),
            "accepted_implicit_units": sum(accepted(unit) and unit.get("support_type") == "implicit" for _, unit in units),
            "accepted_missing_span_units": sum(accepted(unit) and (not unit.get("evidence_span_original") or unit["evidence_span_original"] not in source(row)) for row, unit in units),
            "accepted_negated_units": sum(unit.get("judge_verdict") == "accept" and unit.get("assertion") == "negated" for _, unit in units),
        })
        for source_pass, count in sorted(Counter(row.get("_source_pass", "unrecorded") for row in rows).items()):
            lineage_rows.append({"country": country, "source_pass": source_pass, "all_labeled_posts": count,
                                 "primary_posts": sum(primary(row) and row.get("_source_pass", "unrecorded") == source_pass for row in rows)})
    for row in sets["evaluation_predictions"]:
        for unit in row.get("units", []):
            if not unit.get("evidence_span_original") or unit["evidence_span_original"] not in source(row):
                validation_rows.append({"post_id": identifier(row), "unit_id": unit.get("unit_id"),
                                        "judge_verdict": unit.get("judge_verdict"), "accepted": accepted(unit)})
    write_csv(out / "language_attrition.csv", attrition)
    write_csv(out / "language_unit_retention.csv", unit_rows)
    write_csv(out / "final_pass_lineage.csv", lineage_rows)

    input_ids = {identifier(row) for row in sets["production_input"]}
    final_ids = {identifier(row) for row in sets["final_labeled"]}
    reference_ids = {identifier(row) for row in sets["rag_reference"]}
    trace_paths = sorted((root / "outputs").glob("*/retrieval_trace.jsonl"))
    retrieval_rows = []
    for path in trace_paths:
        counts = Counter()
        outside = set()
        target_languages = {identifier(row): row.get("language") for row in sets["production_input"] + sets["evaluation_input"]}
        cross_language = 0
        scores = []
        for row in read_jsonl(path):
            counts[row["post_id"]] += 1
            if row["gold_post_id"] not in reference_ids:
                outside.add(row["gold_post_id"])
            if row.get("gold_language") != target_languages.get(row["post_id"]):
                cross_language += 1
            scores.append(float(row["score"]))
        retrieval_rows.append({"path": str(path.relative_to(root)), "trace_rows": sum(counts.values()),
                               "target_posts": len(counts), "examples_per_target": dict(sorted(Counter(counts.values()).items())),
                               "cross_language_trace_rows": cross_language, "nonreference_gold_ids": sorted(outside),
                               "score_min": min(scores) if scores else None, "score_max": max(scores) if scores else None})

    prompt_path = root / "dental_ai/prompts.py"
    prompt_sections = ["# Full prompt snapshot\n\nGenerated from the current source; not proof of historical prompt immutability.\n"]
    for node in ast.parse(prompt_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            names = [target.id for target in node.targets if isinstance(target, ast.Name) and target.id.endswith("PROMPT")]
            for name in names:
                prompt_sections.append(f"\n## {name}\n\n```text\n{node.value.value}\n```\n")
    (out / "full_prompts.md").write_text("".join(prompt_sections), encoding="utf-8")

    import pandas as pd

    raw_counts = {}
    for country in ["CHI", "JPN", "KOR"]:
        path = root / f"data/raw_data/{country}_2023-07-01_2026-06-30.xlsx"
        frame = pd.read_excel(path)
        included_ids = {identifier(row) for name in ["production_input", "evaluation_input"] for row in sets[name] if row["country"] == country}
        excluded_ids = {str(value) for value in frame["record_id"]} - included_ids
        development_ids = {identifier(row) for name in ["classification_development", "rag_reference"] for row in sets[name] if row["country"] == country}
        development_canonical_ids = {canonical_post_id(value) for value in development_ids}
        development_fingerprints = set_keys["classification_development"]["source_sha256"] | set_keys["rag_reference"]["source_sha256"]
        included_fingerprints = set_keys["production_input"]["source_sha256"] | set_keys["evaluation_input"]["source_sha256"]
        exclusion_reasons = Counter()
        unexplained = []
        for raw_row in frame.fillna("").to_dict(orient="records"):
            raw_id = str(raw_row["record_id"])
            if raw_id not in excluded_ids:
                continue
            fingerprint = source_text_fingerprint(str(raw_row.get("title", "")), str(raw_row.get("text", "")))
            if canonical_post_id(raw_id) in development_canonical_ids:
                exclusion_reasons["development_id_match"] += 1
            elif fingerprint in development_fingerprints:
                exclusion_reasons["development_exact_text_match"] += 1
            elif fingerprint in included_fingerprints:
                exclusion_reasons["duplicate_exact_text_of_included_record"] += 1
            else:
                exclusion_reasons["unresolved"] += 1
                unexplained.append(raw_id)
        raw_counts[country] = {"rows": len(frame), "columns": list(frame.columns),
                               "production_input_rows_from_file": sum(row.get("source_file") == str(path.relative_to(root)) for row in sets["production_input"]),
                               "evaluation_rows_from_file": sum(row.get("source_file") == str(path.relative_to(root)) for row in sets["evaluation_input"]),
                               "excluded_raw_ids": len(excluded_ids),
                               "excluded_raw_ids_matching_development": sum(canonical_post_id(value) in development_canonical_ids for value in excluded_ids),
                               "exclusion_reconciliation": dict(exclusion_reasons),
                               "unresolved_excluded_raw_ids": sorted(unexplained)}
        paths[f"raw_{country}"] = path
    paths.update({"prompts": prompt_path, "model_stack": root / "configs/model_stack.yaml",
                  "split_manifest": root / "data/project_split_manifest.json",
                  "gold_eval_workbook": root / "data/raw_eval_holdout_150_gold.xlsx",
                  "audit_script": Path(__file__).resolve()})
    for module in ["local_models", "rag", "run", "schemas", "validate", "classification_postprocess"]:
        paths[module] = root / f"dental_ai/{module}.py"
    paths["export_script"] = root / "scripts/merge_export.py"
    for name in ["outputs/final_export/export_summary.json", "outputs/eval_holdout_150_latest/run_manifest.json",
                 "outputs/ablation_sensitivity_analysis/analysis_configuration_manifest.json"]:
        paths[name] = root / name
    months = []
    for offset in range(36):
        month_index = 2023 * 12 + 6 + offset
        months.append({"month": f"{month_index // 12:04d}-{month_index % 12 + 1:02d}",
                       "sampled_date": "", "sampling_seed": "", "CHI_query_timezone": "",
                       "JPN_query_timezone": "", "KOR_query_timezone": "",
                       "status": "not_recoverable_from_available_collection_artifacts"})
    write_csv(out / "sampling_dates_unverified.csv", months)
    manifest = {
        "audit_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Current local artifacts; no inference or source-data modification. Historical collection provenance not reconstructed.",
        "sets": {name: {"rows": len(rows), "countries": dict(Counter(row["country"] for row in rows))} for name, rows in sets.items()},
        "rag_reference_units": sum(len(row.get("units", [])) for row in sets["rag_reference"]),
        "split_overlap_details": overlap_details,
        "production_ids_missing_final": sorted(input_ids - final_ids),
        "final_ids_not_in_input": sorted(final_ids - input_ids),
        "evaluation_units_with_missing_spans": validation_rows,
        "raw_workbooks": raw_counts, "retrieval_trace_audit": retrieval_rows,
        "files": {name: {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in paths.items()},
    }
    (out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "sets": manifest["sets"], "production_ids_missing_final": manifest["production_ids_missing_final"], "raw_workbooks": raw_counts, "unit_retention": unit_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
