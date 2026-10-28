"""Check archived aggregate tables and redraw figures without post-level data.

This does not regenerate the tables, model predictions, or bootstrap intervals
from original observations. Those records are deliberately not distributed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from scipy.stats import chi2_contingency

import run_ablation_sensitivity_analysis as ablation
from verify_release import table_signature


def require_close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Aggregate arithmetic mismatch: {label}: {actual} != {expected}")


def load_table(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def check_binary_metrics(rows):
    checked = 0
    for row in rows:
        if row.get("domain") == "macro_avg":
            continue  # Macro averages cannot be calculated from pooled counts.
        tp, fp, fn = (row[key] for key in ["tp", "fp", "fn"])
        for key, numerator, denominator in [
            ("precision", tp, tp + fp), ("recall", tp, tp + fn),
            ("f1", 2 * tp, 2 * tp + fp + fn),
        ]:
            require_close(row[key], numerator / denominator if denominator else 0, key)
        if "specificity" in row:
            tn = row["tn"]
            require_close(row["specificity"], tn / (tn + fp) if tn + fp else 0, "specificity")
        checked += 1
    return checked


def check_country_profiles(tables):
    distributions = tables["domain_distributions_35_configs.csv"]
    for profile in tables["country_profile_tests.csv"]:
        rows = [row for row in distributions if row["config_id"] == profile["config_id"]]
        countries = sorted({row["country"] for row in rows})
        matrix = [[next(row["posts_with_domain"] for row in rows
                        if row["country"] == country and row["domain"] == domain)
                   for domain in ablation.DOMAINS] for country in countries]
        chi2, p_value, dof, _ = chi2_contingency(matrix)
        total = sum(sum(row) for row in matrix)
        for key, value in [("chi2", chi2), ("p_value", p_value), ("dof", dof),
                           ("total_domain_post_presences", total),
                           ("cramers_v", math.sqrt(chi2 / (total * min(len(countries) - 1, 4))))]:
            require_close(profile[key], value, f"{profile['config_id']}:{key}")
    return len(tables["country_profile_tests.csv"])


def reproduce(reference_dir: Path, out_dir: Path):
    expected = json.loads((reference_dir / "expected_ablation_tables.json").read_text())
    tables = {}
    for name, signature in expected.items():
        path = reference_dir / "aggregate_tables" / name
        if table_signature(path) != signature:
            raise ValueError(f"Archived aggregate table mismatch: {name}")
        tables[name] = load_table(path)
    binary_rows = sum(check_binary_metrics(tables[name]) for name in [
        "eligibility_gate_metrics.csv", "unit_validation_metrics.csv",
        "unit_validation_normalization_metrics.csv",
    ])
    profiles = check_country_profiles(tables)
    evaluation = json.loads((reference_dir / "expected_evaluation.json").read_text())
    full_gate = next(row for row in tables["eligibility_gate_metrics.csv"]
                     if row["group"] == "overall" and row["config_id"] == "A4_full_gate")
    for key in ["tp", "fp", "fn", "tn", "precision", "recall", "f1"]:
        require_close(full_gate[key], evaluation["overall"]["eligibility"][key], f"holdout:{key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    os.environ.setdefault("MPLBACKEND", "Agg")
    production = {
        "domain_distributions": tables["domain_distributions_35_configs.csv"],
        "domain_stability": tables["domain_profile_stability.csv"],
        "concept_stability": tables["concept_rank_topk_stability.csv"],
    }
    status = ablation.write_plots(out_dir, production)
    if not status["generated"]:
        raise ValueError(f"Figure generation failed: {status}")
    files = status["files"] + ablation.write_component_plots(
        out_dir, tables["eligibility_gate_metrics.csv"], tables["unit_validation_metrics.csv"])
    for name in files:
        if not (out_dir / name).is_file():
            raise ValueError(f"Missing generated figure: {name}")
    report = {
        "scope": "Calculations and figures from archived aggregates; no post-level dataset distributed.",
        "aggregate_tables_checked": len(tables), "binary_metric_rows_recomputed": binary_rows,
        "country_profiles_recomputed": profiles, "figures_generated": files,
        "bootstrap_intervals": "Archived only; not regenerated without original records.",
        "network_and_concept_stability": "Archived only; not rederived from post-level units.",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path("results/reference"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/public_reproduction"))
    args = parser.parse_args()
    print(json.dumps(reproduce(args.reference_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
