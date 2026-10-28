#!/usr/bin/env python3
"""Run the ablation and sensitivity protocol.

This implements docs/methods.md using archived row-level annotations and the
publication JSONL references. No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import networkx as nx
from scipy.stats import chi2_contingency, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_publication_holdout as pub  # noqa: E402


DOMAINS = list(pub.DOMAINS)
PRIMARY_POPULATION_ID = "C2_primary_r1_e1_c1c2"
PRIMARY_UNIT_RULE_ID = "D3_judge_accepted"


@dataclass(frozen=True)
class PopulationConfig:
    config_id: str
    label: str
    predicate: Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class UnitRuleConfig:
    config_id: str
    label: str
    predicate: Callable[[dict[str, Any], dict[str, Any]], bool]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = load_jsonl(args.final_export)
    eval_pred = load_jsonl_by_id(args.eval_pred)
    mapping = load_canonical_mapping(args.canonical_decisions)
    populations = population_configs()
    unit_rules = unit_rule_configs()

    gold = None
    gold_status: dict[str, Any] = {"loaded": False, "skipped": args.skip_gold}
    if not args.skip_gold:
        try:
            gold = pub.load_gold(args.gold)
            gold_status = {"loaded": True, "skipped": False, "rows": len(gold)}
        except Exception as exc:  # noqa: BLE001 - manifest should preserve exact blocker.
            gold_status = {
                "loaded": False,
                "skipped": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            raise

    manifest = build_manifest(args, final_rows, eval_pred, mapping, gold_status, populations, unit_rules)
    write_json(out_dir / "analysis_configuration_manifest.json", manifest)

    gold_outputs: dict[str, Any] = {}
    if gold:
        pub.require_matching_ids(gold, eval_pred)
        ids = pub.evaluation_order(set(gold), args.evaluation_order)
        gate_rows, gate_ci_rows = eligibility_gate_ablation(gold, eval_pred, ids, args.bootstrap, args.seed)
        unit_rows, unit_ci_rows, unit_retention_rows, concept_rows = unit_validation_ablation(
            gold,
            eval_pred,
            ids,
            args.bootstrap,
            args.seed,
        )
        negative_control_rows = negative_control_analysis(gold, eval_pred, ids)
        write_csv(out_dir / "eligibility_gate_metrics.csv", gate_rows)
        write_csv(out_dir / "eligibility_gate_confidence_intervals.csv", gate_ci_rows)
        write_csv(out_dir / "unit_validation_metrics.csv", unit_rows)
        write_csv(out_dir / "unit_validation_confidence_intervals.csv", unit_ci_rows)
        write_csv(out_dir / "unit_validation_retention.csv", unit_retention_rows)
        write_csv(out_dir / "unit_validation_normalization_metrics.csv", concept_rows)
        write_csv(out_dir / "negative_control_gold_r1_nonprimary.csv", negative_control_rows)
        gold_outputs = {
            "matched_eval_rows": len(ids),
            "eligibility_gate_metrics": "eligibility_gate_metrics.csv",
            "unit_validation_metrics": "unit_validation_metrics.csv",
            "negative_control": "negative_control_gold_r1_nonprimary.csv",
        }
    else:
        write_json(out_dir / "gold_workstreams_skipped.json", gold_status)

    production = production_sensitivity(final_rows, populations, unit_rules, mapping)
    write_csv(out_dir / "population_denominators_by_country.csv", production["population_denominators"])
    write_csv(out_dir / "domain_distributions_35_configs.csv", production["domain_distributions"])
    write_csv(out_dir / "country_profile_tests.csv", production["country_profile_tests"])
    write_csv(out_dir / "standardized_residuals.csv", production["standardized_residuals"])
    write_csv(out_dir / "concept_counts_long.csv", production["concept_counts"])
    write_csv(out_dir / "concept_rank_topk_stability.csv", production["concept_stability"])
    write_csv(out_dir / "concept_enrichment.csv", production["concept_enrichment"])
    write_csv(out_dir / "network_stability_results.csv", production["network_stability"])
    write_csv(out_dir / "domain_profile_stability.csv", production["domain_stability"])
    write_csv(out_dir / "principal_findings_stability.csv", production["principal_findings"])

    plot_status = write_plots(out_dir, production)
    if gold:
        plot_status["files"].extend(write_component_plots(out_dir, gate_rows, unit_rows))
    write_results_brief(out_dir, gold_status, gold_outputs, production, plot_status)
    print(json.dumps({"out_dir": str(out_dir), "gold_status": gold_status, "plot_status": plot_status}, indent=2))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-export", type=Path, default=Path("data/publication_dataset/main/posts.jsonl"))
    parser.add_argument("--primary-export", type=Path, default=Path("data/publication_dataset/main/posts.jsonl"), help="Provenance only; populations are selected from --final-export")
    parser.add_argument("--export-summary", type=Path, default=Path("data/publication_dataset/manifest.json"))
    parser.add_argument("--gold", type=Path, default=Path("data/publication_dataset/evaluation/annotations.jsonl"))
    parser.add_argument("--eval-pred", type=Path, default=Path("results/reference/evaluation_predictions.jsonl"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("results/reference/provenance.json"))
    parser.add_argument("--canonical-decisions", type=Path, default=Path("data/canonicalization/reviewed_cluster_decisions.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/methods.md"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ablation_sensitivity_analysis"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--evaluation-order", type=Path, default=Path("results/reference/evaluation_order.json"))
    parser.add_argument("--skip-gold", action="store_true", help="Skip Workstreams A, B, and E.")
    parser.add_argument("--require-gold", action="store_true", help="Compatibility option: invalid gold input always fails unless --skip-gold is used.")
    args = parser.parse_args(argv)
    required = [args.final_export, args.eval_pred, args.canonical_decisions]
    if not args.skip_gold:
        required.extend([args.gold, args.evaluation_order])
    if any(not path.is_file() for path in required):
        parser.error("Private post-level inputs and canonicalization decisions are not distributed. Supply compatible paths or use scripts/reproduce_aggregate_results.py for public checks.")
    return args


def build_manifest(
    args: argparse.Namespace,
    final_rows: list[dict[str, Any]],
    eval_pred: dict[str, dict[str, Any]],
    mapping: dict[tuple[str, str], list[str]],
    gold_status: dict[str, Any],
    populations: list[PopulationConfig],
    unit_rules: list[UnitRuleConfig],
) -> dict[str, Any]:
    paths = {
        "protocol": args.protocol,
        "final_export": args.final_export,
        "primary_export": args.primary_export,
        "export_summary": args.export_summary,
        "gold": args.gold,
        "eval_pred": args.eval_pred,
        "eval_manifest": args.eval_manifest,
        "canonical_decisions": args.canonical_decisions,
        "evaluation_order": args.evaluation_order,
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_seed": args.seed,
        "inputs": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "final_export_rows": len(final_rows),
        "eval_prediction_rows": len(eval_pred),
        "canonical_mapping_entries": len(mapping),
        "gold_status": gold_status,
        "population_configs": [{"config_id": item.config_id, "label": item.label} for item in populations],
        "unit_rule_configs": [{"config_id": item.config_id, "label": item.label} for item in unit_rules],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if "analysis_flags" in row and "classification" in row:
                        from dental_ai.publication import extraction_payload

                        row = extraction_payload(row)
                    rows.append(row)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in load_jsonl(path):
        post_id = str(row.get("post_id") or row.get("record_id") or "").strip()
        if post_id:
            if post_id in rows:
                raise ValueError(f"Duplicate prediction ID: {post_id}")
            row["source_text"] = pub.source_text_from_prediction(row)
            rows[post_id] = row
    return rows


def load_canonical_mapping(path: Path) -> dict[tuple[str, str], list[str]]:
    mapping: dict[tuple[str, str], list[str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            domain = clean(row.get("csm_dimension"))
            labels = split_multiline(row.get("labels_in_proposed_family"))
            decision = clean(row.get("consensus_decision")).lower()
            consensus = clean(row.get("consensus_canonical_labels"))
            if not domain or not labels:
                continue
            if decision == "merge_all" and consensus:
                for label in labels:
                    mapping[(domain, concept_key(label))] = [consensus]
            elif decision == "split" and consensus:
                for canonical, source_labels in parse_split_consensus(consensus).items():
                    for label in source_labels:
                        mapping[(domain, concept_key(label))] = [canonical]
    return mapping


def parse_split_consensus(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in split_multiline(text):
        if "=" not in line:
            continue
        canonical, rhs = line.split("=", 1)
        canonical = canonical.strip().strip(";.")
        if not canonical:
            continue
        out[canonical] = [part.strip().strip(";.") for part in rhs.split("+") if part.strip()]
    return out


def population_configs() -> list[PopulationConfig]:
    return [
        PopulationConfig("C1_strict_r1_e1_c1", "Strict R1 + E1 + C1", lambda row: is_r1(row) and is_e1(row) and is_c(row, {"C1"})),
        PopulationConfig(PRIMARY_POPULATION_ID, "Primary R1 + E1 + C1/C2", lambda row: is_r1(row) and is_e1(row) and is_c(row, {"C1", "C2"})),
        PopulationConfig("C3_all_r1_e1", "All R1 + E1", lambda row: is_r1(row) and is_e1(row)),
        PopulationConfig("C4_r1_c1c2_any_experiencer", "R1 + C1/C2, any experiencer", lambda row: is_r1(row) and is_c(row, {"C1", "C2"})),
        PopulationConfig("C5_all_r1", "All R1", lambda row: is_r1(row)),
    ]


def unit_rule_configs() -> list[UnitRuleConfig]:
    return [
        UnitRuleConfig("D1_all_proposed", "All proposed units", lambda row, unit: True),
        UnitRuleConfig("D2_span_supported", "Evidence span found in source text", lambda row, unit: span_supported(row, unit)),
        UnitRuleConfig(PRIMARY_UNIT_RULE_ID, "Judge-accepted units", lambda row, unit: accepted_for_analysis(unit)),
        UnitRuleConfig("D4_judge_accepted_explicit", "Judge-accepted and explicit", lambda row, unit: accepted_for_analysis(unit) and unit.get("support_type") == "explicit"),
        UnitRuleConfig("D5a_conf_ge_070", "Judge-accepted with confidence >= 0.70", lambda row, unit: accepted_for_analysis(unit) and unit_confidence(unit) >= 0.70),
        UnitRuleConfig("D5b_conf_ge_080", "Judge-accepted with confidence >= 0.80", lambda row, unit: accepted_for_analysis(unit) and unit_confidence(unit) >= 0.80),
        UnitRuleConfig("D5c_conf_ge_090", "Judge-accepted with confidence >= 0.90", lambda row, unit: accepted_for_analysis(unit) and unit_confidence(unit) >= 0.90),
    ]


def eligibility_gate_ablation(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("A1_r1_only", "R1 only", is_r1),
        ("A2_r1_e1", "R1 + E1", lambda row: is_r1(row) and is_e1(row)),
        ("A3_r1_c1c2", "R1 + C1/C2", lambda row: is_r1(row) and is_c(row, {"C1", "C2"})),
        ("A4_full_gate", "R1 + E1 + C1/C2", lambda row: is_r1(row) and is_e1(row) and is_c(row, {"C1", "C2"})),
    ]
    groups = [("overall", "overall", ids)] + pub.language_groups(gold, ids)
    rows = []
    ci_rows = []
    for group_type, group_value, group_ids in groups:
        gold_values = [gold_primary_target(gold[post_id]) for post_id in group_ids]
        for config_id, label, predicate in gates:
            pred_values = [predicate(pred[post_id]) for post_id in group_ids]
            metric = with_specificity(pub.binary_counts(gold_values, pred_values))
            rows.append({"config_id": config_id, "gate": label, "group_type": group_type, "group": group_value, **metric})
            ci_rows.extend(binary_metric_bootstrap_rows(config_id, label, group_type, group_value, gold_values, pred_values, bootstrap, seed))
    return rows, ci_rows


def unit_validation_ablation(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rules = [
        ("B1_extractor_only", "All proposed units", lambda row, unit: True),
        ("B2_span_supported", "Span-supported units", lambda row, unit: span_supported(row, unit)),
        ("B3_judge_accepted", "Judge-accepted units", lambda row, unit: accepted_for_analysis(unit)),
        ("B4_judge_span_supported", "Judge accepted + span supported", lambda row, unit: accepted_for_analysis(unit) and span_supported(row, unit)),
    ]
    metric_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    concept_rows: list[dict[str, Any]] = []
    groups = [("overall", "overall", ids)] + pub.language_groups(gold, ids)

    for config_id, label, predicate in rules:
        adapted_pred = {
            post_id: prediction_with_units(pred[post_id], retained_units(pred[post_id], predicate))
            for post_id in ids
        }
        for group_type, group_value, group_ids in groups:
            domain_metrics = []
            for domain in DOMAINS:
                gold_values = [gold[post_id]["domain_presence"][domain] for post_id in group_ids]
                pred_values = [adapted_pred[post_id]["domain_presence"][domain] for post_id in group_ids]
                metric = pub.prf_row(pub.binary_counts(gold_values, pred_values))
                domain_metrics.append(metric)
                metric_rows.append({"config_id": config_id, "rule": label, "group_type": group_type, "group": group_value, "domain": domain, **metric})
                ci_rows.extend(binary_metric_bootstrap_rows(config_id, f"{label}::{domain}", group_type, group_value, gold_values, pred_values, bootstrap, seed))
            macro = macro_metrics(domain_metrics)
            metric_rows.append({"config_id": config_id, "rule": label, "group_type": group_type, "group": group_value, "domain": "macro_avg", **macro})
            concept_rows.append({"config_id": config_id, "rule": label, "group_type": group_type, "group": group_value, **pub.concept_set_metrics(gold, adapted_pred, group_ids, "all_domains")})
        retention_rows.extend(unit_retention_summary_rows(config_id, label, pred, ids, predicate))
    return metric_rows, ci_rows, retention_rows, concept_rows


def negative_control_analysis(
    gold: dict[str, dict[str, Any]],
    pred: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    targets = [post_id for post_id in ids if gold[post_id].get("relevance_label") == "R1" and not gold_primary_target(gold[post_id])]
    groups: dict[tuple[str, str], list[str]] = {("overall", "overall"): targets}
    for post_id in targets:
        stratum = f"{gold[post_id].get('experiencer_label') or 'None'} + {gold[post_id].get('content_function') or 'None'}"
        groups.setdefault(("gold_ec_stratum", stratum), []).append(post_id)
        groups.setdefault(("language", gold[post_id].get("language") or "None"), []).append(post_id)
        groups.setdefault(("country", gold[post_id].get("country") or "None"), []).append(post_id)
    for (group_type, group), group_ids in groups.items():
        proposed = [unit for post_id in group_ids for unit in pred[post_id].get("units", [])]
        accepted = [unit for unit in proposed if accepted_for_analysis(unit)]
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "posts": len(group_ids),
                "predicted_primary_eligible_posts": sum(1 for post_id in group_ids if is_primary_pred(pred[post_id])),
                "posts_with_proposed_units": sum(1 for post_id in group_ids if pred[post_id].get("units")),
                "posts_with_accepted_units": sum(1 for post_id in group_ids if any(accepted_for_analysis(unit) for unit in pred[post_id].get("units", []))),
                "proposed_units": len(proposed),
                "accepted_units": len(accepted),
                **{f"accepted_domain_{domain}": sum(1 for unit in accepted if unit.get("domain") == domain) for domain in DOMAINS},
            }
        )
    return rows


def production_sensitivity(
    rows: list[dict[str, Any]],
    populations: list[PopulationConfig],
    unit_rules: list[UnitRuleConfig],
    mapping: dict[tuple[str, str], list[str]],
) -> dict[str, list[dict[str, Any]]]:
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    population_denominators: list[dict[str, Any]] = []
    domain_distributions: list[dict[str, Any]] = []
    country_profile_tests: list[dict[str, Any]] = []
    standardized_residuals: list[dict[str, Any]] = []
    concept_counts: list[dict[str, Any]] = []
    concept_enrichment: list[dict[str, Any]] = []
    network_summaries: dict[tuple[str, str], dict[str, Any]] = {}

    countries = sorted({clean(row.get("country")) for row in rows if clean(row.get("country"))})
    for population in populations:
        population_rows = [row for row in rows if population.predicate(row)]
        population_denominators.extend(population_denominator_rows(population, population_rows, countries))
        for unit_rule in unit_rules:
            key = (population.config_id, unit_rule.config_id)
            config_id = config_key(population.config_id, unit_rule.config_id)
            analysis = summarize_production_config(population, unit_rule, population_rows, countries, mapping)
            configs[key] = analysis
            domain_distributions.extend(analysis["domain_distribution"])
            country_profile_tests.append(analysis["country_profile_test"])
            standardized_residuals.extend(analysis["standardized_residuals"])
            concept_counts.extend(analysis["concept_counts"])
            concept_enrichment.extend(analysis["concept_enrichment"])
            network_summaries[key] = analysis["network_summary"]
            assert all(row["config_id"] == config_id for row in analysis["domain_distribution"])

    primary = configs[(PRIMARY_POPULATION_ID, PRIMARY_UNIT_RULE_ID)]
    domain_stability = domain_profile_stability(configs, primary)
    concept_stability = concept_rank_stability(configs, primary)
    network_stability = network_stability_rows(network_summaries, network_summaries[(PRIMARY_POPULATION_ID, PRIMARY_UNIT_RULE_ID)])
    principal_findings = principal_findings_stability(configs, primary)
    return {
        "population_denominators": population_denominators,
        "domain_distributions": domain_distributions,
        "country_profile_tests": country_profile_tests,
        "standardized_residuals": standardized_residuals,
        "concept_counts": concept_counts,
        "concept_enrichment": concept_enrichment,
        "domain_stability": domain_stability,
        "concept_stability": concept_stability,
        "network_stability": network_stability,
        "principal_findings": principal_findings,
    }


def summarize_production_config(
    population: PopulationConfig,
    unit_rule: UnitRuleConfig,
    population_rows: list[dict[str, Any]],
    countries: list[str],
    mapping: dict[tuple[str, str], list[str]],
) -> dict[str, Any]:
    config_id = config_key(population.config_id, unit_rule.config_id)
    post_domain_presence: dict[str, Counter[str]] = {country: Counter() for country in countries}
    denominators: Counter[str] = Counter()
    population_counts: Counter[str] = Counter()
    unit_counts: dict[str, Counter[str]] = {country: Counter() for country in countries}
    concept_unit_counts: Counter[tuple[str, str, str]] = Counter()
    concept_post_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    post_concepts: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in population_rows:
        country = clean(row.get("country"))
        if not country:
            continue
        population_counts[country] += 1
        units = canonical_retained_units(row, unit_rule.predicate, mapping)
        if not units:
            continue
        denominators[country] += 1
        domains = {unit["domain"] for unit in units if unit.get("domain") in DOMAINS}
        for domain in domains:
            post_domain_presence[country][domain] += 1
        for unit in units:
            domain = unit["domain"]
            concept = unit["canonical_concept"]
            if domain not in DOMAINS or not concept:
                continue
            unit_counts[country][domain] += 1
            concept_unit_counts[(country, domain, concept)] += 1
            concept_post_sets[(country, domain, concept)].add(str(row.get("post_id")))
            post_concepts[(country, str(row.get("post_id")))].add(f"{domain}::{concept}")

    domain_rows = []
    for country in countries:
        denom = denominators[country]
        for domain in DOMAINS:
            count = post_domain_presence[country][domain]
            domain_rows.append(
                {
                    "config_id": config_id,
                    "population_id": population.config_id,
                    "unit_rule_id": unit_rule.config_id,
                    "country": country,
                    "population_posts": population_counts[country],
                    "analysis_posts_with_retained_units": denom,
                    "domain": domain,
                    "posts_with_domain": count,
                    "post_percentage": pub.safe_div(count, denom),
                    "retained_units_in_domain": unit_counts[country][domain],
                }
            )

    chi_row, residual_rows = chi_square_rows(config_id, population, unit_rule, countries, post_domain_presence)
    concept_rows = concept_count_rows(config_id, population, unit_rule, concept_unit_counts, concept_post_sets)
    enrichment_rows = enrichment_rows_for_config(config_id, population, unit_rule, concept_unit_counts, denominators, countries)
    network_summary = network_summary_for_config(config_id, population, unit_rule, countries, post_concepts)
    return {
        "domain_distribution": domain_rows,
        "country_profile_test": chi_row,
        "standardized_residuals": residual_rows,
        "concept_counts": concept_rows,
        "concept_enrichment": enrichment_rows,
        "network_summary": network_summary,
        "post_domain_presence": post_domain_presence,
        "denominators": denominators,
        "concept_unit_counts": concept_unit_counts,
        "network_post_concepts": post_concepts,
    }


def population_denominator_rows(
    population: PopulationConfig,
    rows: list[dict[str, Any]],
    countries: list[str],
) -> list[dict[str, Any]]:
    counts = Counter(clean(row.get("country")) for row in rows)
    out = []
    for country in countries:
        out.append({"population_id": population.config_id, "population": population.label, "country": country, "posts_before_unit_rule": counts[country]})
    out.append({"population_id": population.config_id, "population": population.label, "country": "ALL", "posts_before_unit_rule": len(rows)})
    return out


def chi_square_rows(
    config_id: str,
    population: PopulationConfig,
    unit_rule: UnitRuleConfig,
    countries: list[str],
    domain_counts: dict[str, Counter[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table = [[domain_counts[country][domain] for domain in DOMAINS] for country in countries]
    total = sum(sum(row) for row in table)
    if total == 0 or len(countries) < 2:
        return (
            {"config_id": config_id, "population_id": population.config_id, "unit_rule_id": unit_rule.config_id, "chi2": "", "p_value": "", "dof": "", "cramers_v": "", "total_domain_post_presences": total},
            [],
        )
    chi2, p_value, dof, expected = chi2_contingency(table)
    min_dim = min(len(countries) - 1, len(DOMAINS) - 1)
    cramers_v = math.sqrt(chi2 / (total * min_dim)) if min_dim > 0 else 0.0
    residuals = []
    for i, country in enumerate(countries):
        for j, domain in enumerate(DOMAINS):
            exp = expected[i][j]
            obs = table[i][j]
            residuals.append(
                {
                    "config_id": config_id,
                    "population_id": population.config_id,
                    "unit_rule_id": unit_rule.config_id,
                    "country": country,
                    "domain": domain,
                    "observed": obs,
                    "expected": exp,
                    "standardized_residual": (obs - exp) / math.sqrt(exp) if exp else "",
                }
            )
    return (
        {
            "config_id": config_id,
            "population_id": population.config_id,
            "unit_rule_id": unit_rule.config_id,
            "chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "cramers_v": cramers_v,
            "total_domain_post_presences": total,
        },
        residuals,
    )


def concept_count_rows(
    config_id: str,
    population: PopulationConfig,
    unit_rule: UnitRuleConfig,
    unit_counts: Counter[tuple[str, str, str]],
    post_sets: dict[tuple[str, str, str], set[str]],
) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for (country, domain, concept), unit_count in unit_counts.items():
        grouped[(country, domain)].append((concept, unit_count, len(post_sets[(country, domain, concept)])))
    for (country, domain), values in grouped.items():
        for rank, (concept, unit_count, post_count) in enumerate(sorted(values, key=lambda item: (-item[1], -item[2], item[0])), start=1):
            rows.append(
                {
                    "config_id": config_id,
                    "population_id": population.config_id,
                    "unit_rule_id": unit_rule.config_id,
                    "country": country,
                    "domain": domain,
                    "concept": concept,
                    "unit_count": unit_count,
                    "post_count": post_count,
                    "rank_within_country_domain": rank,
                }
            )
    return rows


def enrichment_rows_for_config(
    config_id: str,
    population: PopulationConfig,
    unit_rule: UnitRuleConfig,
    unit_counts: Counter[tuple[str, str, str]],
    denominators: Counter[str],
    countries: list[str],
) -> list[dict[str, Any]]:
    concepts = sorted({(domain, concept) for _, domain, concept in unit_counts})
    rows = []
    for domain, concept in concepts:
        total_other_denominator_cache = sum(denominators.values())
        for country in countries:
            count = unit_counts[(country, domain, concept)]
            other_count = sum(unit_counts[(other, domain, concept)] for other in countries if other != country)
            country_den = denominators[country]
            other_den = total_other_denominator_cache - country_den
            rate = (count + 0.5) / (country_den + 1.0) if country_den else 0.0
            other_rate = (other_count + 0.5) / (other_den + 1.0) if other_den else 0.0
            rows.append(
                {
                    "config_id": config_id,
                    "population_id": population.config_id,
                    "unit_rule_id": unit_rule.config_id,
                    "country": country,
                    "domain": domain,
                    "concept": concept,
                    "unit_count": count,
                    "analysis_posts_with_retained_units": country_den,
                    "rate_per_post": pub.safe_div(count, country_den),
                    "log2_rate_ratio_vs_other_countries": math.log2(rate / other_rate) if rate and other_rate else "",
                }
            )
    return rows


def network_summary_for_config(
    config_id: str,
    population: PopulationConfig,
    unit_rule: UnitRuleConfig,
    countries: list[str],
    post_concepts: dict[tuple[str, str], set[str]],
) -> dict[str, dict[str, Any]]:
    out = {}
    for country in countries:
        graph = nx.Graph()
        for (item_country, _post_id), concepts in post_concepts.items():
            if item_country != country:
                continue
            concept_list = sorted(concepts)
            for concept in concept_list:
                graph.add_node(concept)
            for index, source in enumerate(concept_list):
                for target in concept_list[index + 1 :]:
                    if graph.has_edge(source, target):
                        graph[source][target]["weight"] += 1
                    else:
                        graph.add_edge(source, target, weight=1)
        strengths = dict(graph.degree(weight="weight"))
        top_nodes = [node for node, _ in sorted(strengths.items(), key=lambda item: (-item[1], item[0]))[:20]]
        out[country] = {
            "config_id": config_id,
            "population_id": population.config_id,
            "unit_rule_id": unit_rule.config_id,
            "country": country,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
            "node_strength": strengths,
            "top20_nodes": top_nodes,
        }
    return out


def domain_profile_stability(configs: dict[tuple[str, str], dict[str, Any]], primary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    primary_pct = {
        (row["country"], row["domain"]): row["post_percentage"]
        for row in primary["domain_distribution"]
    }
    primary_v = primary["country_profile_test"].get("cramers_v") or 0.0
    for (population_id, unit_rule_id), config in configs.items():
        test = config["country_profile_test"]
        for row in config["domain_distribution"]:
            key = (row["country"], row["domain"])
            rows.append(
                {
                    "config_id": config_key(population_id, unit_rule_id),
                    "population_id": population_id,
                    "unit_rule_id": unit_rule_id,
                    "country": row["country"],
                    "domain": row["domain"],
                    "primary_post_percentage": primary_pct.get(key, 0.0),
                    "comparison_post_percentage": row["post_percentage"],
                    "percentage_point_change": row["post_percentage"] - primary_pct.get(key, 0.0),
                    "primary_cramers_v": primary_v,
                    "comparison_cramers_v": test.get("cramers_v") or 0.0,
                    "cramers_v_change": (test.get("cramers_v") or 0.0) - primary_v,
                }
            )
    return rows


def concept_rank_stability(configs: dict[tuple[str, str], dict[str, Any]], primary: dict[str, Any]) -> list[dict[str, Any]]:
    primary_counts = primary["concept_unit_counts"]
    primary_groups = grouped_concept_counts(primary_counts)
    rows = []
    for (population_id, unit_rule_id), config in configs.items():
        groups = grouped_concept_counts(config["concept_unit_counts"])
        keys = sorted(set(primary_groups) | set(groups))
        for country, domain in keys:
            primary_map = primary_groups.get((country, domain), {})
            compare_map = groups.get((country, domain), {})
            concepts = sorted(set(primary_map) | set(compare_map))
            if len(concepts) >= 2:
                corr = spearmanr([primary_map.get(c, 0) for c in concepts], [compare_map.get(c, 0) for c in concepts]).statistic
                corr = 0.0 if math.isnan(corr) else corr
            else:
                corr = ""
            rows.append(
                {
                    "config_id": config_key(population_id, unit_rule_id),
                    "population_id": population_id,
                    "unit_rule_id": unit_rule_id,
                    "country": country,
                    "domain": domain,
                    "concepts_in_union": len(concepts),
                    "spearman_count_correlation": corr,
                    "top10_jaccard": topk_jaccard(primary_map, compare_map, 10),
                    "top20_jaccard": topk_jaccard(primary_map, compare_map, 20),
                }
            )
    return rows


def network_stability_rows(
    summaries: dict[tuple[str, str], dict[str, dict[str, Any]]],
    primary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (population_id, unit_rule_id), country_summaries in summaries.items():
        for country, summary in country_summaries.items():
            primary_summary = primary.get(country, {})
            primary_strength = primary_summary.get("node_strength", {})
            compare_strength = summary.get("node_strength", {})
            nodes = sorted(set(primary_strength) | set(compare_strength))
            if len(nodes) >= 2:
                corr = spearmanr([primary_strength.get(node, 0) for node in nodes], [compare_strength.get(node, 0) for node in nodes]).statistic
                corr = 0.0 if math.isnan(corr) else corr
            else:
                corr = ""
            rows.append(
                {
                    "config_id": config_key(population_id, unit_rule_id),
                    "population_id": population_id,
                    "unit_rule_id": unit_rule_id,
                    "country": country,
                    "primary_node_count": primary_summary.get("node_count", 0),
                    "comparison_node_count": summary.get("node_count", 0),
                    "primary_edge_count": primary_summary.get("edge_count", 0),
                    "comparison_edge_count": summary.get("edge_count", 0),
                    "primary_density": primary_summary.get("density", 0.0),
                    "comparison_density": summary.get("density", 0.0),
                    "node_strength_spearman": corr,
                    "top20_node_jaccard": jaccard(set(primary_summary.get("top20_nodes", [])), set(summary.get("top20_nodes", []))),
                }
            )
    return rows


def principal_findings_stability(configs: dict[tuple[str, str], dict[str, Any]], primary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    primary_domain = domain_percentage_lookup(primary)
    primary_coping_chi = primary_domain.get(("CHI", "Coping and Management"), 0.0)
    primary_emotion_kor = primary_domain.get(("KOR", "Emotional Expression"), 0.0)
    for (population_id, unit_rule_id), config in configs.items():
        lookup = domain_percentage_lookup(config)
        symptoms_top = all(
            lookup.get((country, "Symptom Description"), -1.0)
            >= max(lookup.get((country, domain), -1.0) for domain in DOMAINS)
            for country in {country for country, _ in lookup}
        )
        coping_chi = lookup.get(("CHI", "Coping and Management"), 0.0)
        emotion_kor = lookup.get(("KOR", "Emotional Expression"), 0.0)
        rows.extend(
            [
                {
                    "config_id": config_key(population_id, unit_rule_id),
                    "population_id": population_id,
                    "unit_rule_id": unit_rule_id,
                    "finding": "Symptom Description is most frequent in each corpus",
                    "stable_direction": int(symptoms_top),
                    "primary_value": "",
                    "comparison_value": "",
                },
                {
                    "config_id": config_key(population_id, unit_rule_id),
                    "population_id": population_id,
                    "unit_rule_id": unit_rule_id,
                    "finding": "China has higher Coping and Management share than Japan and South Korea",
                    "stable_direction": int(coping_chi >= lookup.get(("JPN", "Coping and Management"), 0.0) and coping_chi >= lookup.get(("KOR", "Coping and Management"), 0.0)),
                    "primary_value": primary_coping_chi,
                    "comparison_value": coping_chi,
                },
                {
                    "config_id": config_key(population_id, unit_rule_id),
                    "population_id": population_id,
                    "unit_rule_id": unit_rule_id,
                    "finding": "South Korea has higher Emotional Expression share than China and Japan",
                    "stable_direction": int(emotion_kor >= lookup.get(("CHI", "Emotional Expression"), 0.0) and emotion_kor >= lookup.get(("JPN", "Emotional Expression"), 0.0)),
                    "primary_value": primary_emotion_kor,
                    "comparison_value": emotion_kor,
                },
            ]
        )
    return rows


def write_component_plots(out_dir: Path, gate_rows: list[dict], unit_rows: list[dict]) -> list[str]:
    import matplotlib.pyplot as plt

    gates = [row for row in gate_rows if row["group_type"] == "overall"]
    fig, axis = plt.subplots(figsize=(12, 6))
    for metric, label in [("precision", "Precision"), ("recall", "Recall"), ("specificity", "Specificity")]:
        axis.plot(range(len(gates)), [row[metric] for row in gates], marker="o", label=label)
    axis.set_xticks(range(len(gates)), ["R1", "R1+E1", "R1+C1/C2", "Full gate"])
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("Metric value")
    axis.set_title("Eligibility-gate ablation on 150-post holdout")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right", ncols=3)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out_dir / f"eligibility_gate_ablation.{extension}", dpi=200)
    plt.close(fig)
    units = [row for row in unit_rows if row["group_type"] == "overall" and row["domain"] == "macro_avg"]
    fig, axis = plt.subplots(figsize=(10, 6))
    for metric, label in [("precision", "Macro precision"), ("recall", "Macro recall"), ("f1", "Macro F1")]:
        axis.plot(range(len(units)), [row[metric] for row in units], marker="o", label=label)
    axis.set_xticks(range(len(units)), ["All proposed", "Exact span", "Accepted", "Accepted + exact span"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Metric value")
    axis.set_title("Unit-validation ablation on 150-post holdout")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(out_dir / f"unit_validation_ablation.{extension}", dpi=200)
    plt.close(fig)
    return ["eligibility_gate_ablation.png", "eligibility_gate_ablation.pdf", "unit_validation_ablation.png", "unit_validation_ablation.pdf"]


def write_plots(out_dir: Path, production: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        status = {"generated": False, "reason": f"{type(exc).__name__}: {exc}"}
        write_json(out_dir / "figure_generation_skipped.json", status)
        return status

    primary_rows = [
        row
        for row in production["domain_distributions"]
        if row["population_id"] == PRIMARY_POPULATION_ID and row["unit_rule_id"] == PRIMARY_UNIT_RULE_ID
    ]
    countries = sorted({row["country"] for row in primary_rows})
    matrix = [[next((r["post_percentage"] for r in primary_rows if r["country"] == country and r["domain"] == domain), 0.0) for domain in DOMAINS] for country in countries]
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(DOMAINS)), DOMAINS, rotation=35, ha="right")
    ax.set_yticks(range(len(countries)), countries)
    ax.set_title("Primary Domain Percentages")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "main_sensitivity_heatmap.png", dpi=200)
    fig.savefig(out_dir / "main_sensitivity_heatmap.pdf")
    plt.close(fig)

    stability = production["domain_stability"]
    changes = [float(row["percentage_point_change"]) for row in stability]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(changes, bins=30)
    ax.set_title("Distribution of Domain Percentage-Point Changes")
    ax.set_xlabel("Comparison minus primary")
    ax.set_ylabel("Cells")
    fig.tight_layout()
    fig.savefig(out_dir / "effect_size_stability_plot.png", dpi=200)
    fig.savefig(out_dir / "effect_size_stability_plot.pdf")
    plt.close(fig)

    rank_rows = [row for row in production["concept_stability"] if row["spearman_count_correlation"] != ""]
    corrs = [float(row["spearman_count_correlation"]) for row in rank_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(corrs, bins=30)
    ax.set_title("Concept-Rank Stability")
    ax.set_xlabel("Spearman correlation with primary")
    ax.set_ylabel("Country-domain cells")
    fig.tight_layout()
    fig.savefig(out_dir / "concept_rank_stability_plot.png", dpi=200)
    fig.savefig(out_dir / "concept_rank_stability_plot.pdf")
    plt.close(fig)

    return {
        "generated": True,
        "files": [
            "main_sensitivity_heatmap.png",
            "main_sensitivity_heatmap.pdf",
            "effect_size_stability_plot.png",
            "effect_size_stability_plot.pdf",
            "concept_rank_stability_plot.png",
            "concept_rank_stability_plot.pdf",
        ],
    }


def write_results_brief(
    out_dir: Path,
    gold_status: dict[str, Any],
    gold_outputs: dict[str, Any],
    production: dict[str, list[dict[str, Any]]],
    plot_status: dict[str, Any],
) -> None:
    stable_counts = Counter(row["finding"] for row in production["principal_findings"] if int(row["stable_direction"]) == 1)
    total_counts = Counter(row["finding"] for row in production["principal_findings"])
    lines = [
        "# Ablation and Sensitivity Analysis Outputs",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Gold-Set Workstreams",
        "",
    ]
    if gold_status.get("loaded"):
        lines.append(f"Gold workstreams completed on {gold_outputs.get('matched_eval_rows')} matched evaluation rows.")
        for label, filename in gold_outputs.items():
            if label != "matched_eval_rows":
                lines.append(f"- {label}: `{filename}`")
    elif gold_status.get("skipped"):
        lines.append("Gold workstreams were intentionally skipped for this run.")
    else:
        lines.append("Gold workstreams were skipped because the gold file could not be loaded.")
        lines.append(f"- Error: `{gold_status.get('error_type')}: {gold_status.get('error')}`")
    lines.extend(
        [
            "",
            "## Production Sensitivity",
            "",
            "- Domain distributions: `domain_distributions_35_configs.csv`",
            "- Concept stability: `concept_rank_topk_stability.csv`",
            "- Network stability: `network_stability_results.csv`",
            "- Principal findings stability: `principal_findings_stability.csv`",
            "",
            "## Principal Finding Direction Counts",
            "",
        ]
    )
    for finding in sorted(total_counts):
        lines.append(f"- {finding}: {stable_counts[finding]} / {total_counts[finding]} configurations retained direction")
    lines.extend(["", "## Figure Status", "", json.dumps(plot_status, indent=2)])
    (out_dir / "results_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prediction_with_units(row: dict[str, Any], units: list[dict[str, Any]]) -> dict[str, Any]:
    clone = dict(row)
    clone["units"] = units
    clone["accepted_units"] = units
    clone["domain_presence"] = {domain: any(unit.get("domain") == domain for unit in units) for domain in DOMAINS}
    clone["domain_concepts"] = {
        domain: {
            pub.normalize_concept(unit.get("normalized_concept_en"))
            for unit in units
            if unit.get("domain") == domain and pub.normalize_concept(unit.get("normalized_concept_en"))
        }
        for domain in DOMAINS
    }
    return clone


def retained_units(row: dict[str, Any], predicate: Callable[[dict[str, Any], dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [unit for unit in row.get("units", []) if unit.get("domain") in DOMAINS and predicate(row, unit)]


def canonical_retained_units(
    row: dict[str, Any],
    predicate: Callable[[dict[str, Any], dict[str, Any]], bool],
    mapping: dict[tuple[str, str], list[str]],
) -> list[dict[str, str]]:
    out = []
    for unit in retained_units(row, predicate):
        domain = clean(unit.get("domain"))
        concept = clean(unit.get("normalized_concept_en"))
        if not concept:
            continue
        canonical_values = mapping.get((domain, concept_key(concept)), [concept])
        for canonical in canonical_values:
            if canonical:
                out.append({"domain": domain, "canonical_concept": canonical})
    return out


def unit_retention_summary_rows(
    config_id: str,
    label: str,
    pred: dict[str, dict[str, Any]],
    ids: list[str],
    predicate: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    groups = {"overall": ids}
    for post_id in ids:
        groups.setdefault(f"language::{pred[post_id].get('language') or 'None'}", []).append(post_id)
    rows = []
    for group, group_ids in groups.items():
        all_units = [unit for post_id in group_ids for unit in pred[post_id].get("units", [])]
        kept = [unit for post_id in group_ids for unit in pred[post_id].get("units", []) if predicate(pred[post_id], unit)]
        span_supported_count = sum(1 for post_id in group_ids for unit in pred[post_id].get("units", []) if span_supported(pred[post_id], unit))
        rows.append(
            {
                "config_id": config_id,
                "rule": label,
                "group": group,
                "posts": len(group_ids),
                "all_proposed_units": len(all_units),
                "retained_units": len(kept),
                "retained_unit_percentage": pub.safe_div(len(kept), len(all_units)),
                "span_support_rate_all_units": pub.safe_div(span_supported_count, len(all_units)),
            }
        )
    return rows


def binary_metric_bootstrap_rows(
    config_id: str,
    label: str,
    group_type: str,
    group: str,
    gold_values: list[bool],
    pred_values: list[bool],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not gold_values or iterations <= 0:
        return []
    rng = random.Random(f"{seed}:{config_id}:{group_type}:{group}:{label}")
    values: dict[str, list[float]] = defaultdict(list)
    indexes = list(range(len(gold_values)))
    for _ in range(iterations):
        sample = [rng.choice(indexes) for _ in indexes]
        counts = pub.binary_counts([gold_values[i] for i in sample], [pred_values[i] for i in sample])
        metric = with_specificity(counts)
        for key in ["precision", "recall", "f1", "specificity"]:
            values[key].append(float(metric[key]))
    return [
        {"config_id": config_id, "label": label, "group_type": group_type, "group": group, "metric": metric, **pub.percentile_interval(samples)}
        for metric, samples in values.items()
    ]


def grouped_concept_counts(counts: Counter[tuple[str, str, str]]) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for (country, domain, concept), count in counts.items():
        out[(country, domain)][concept] = count
    return out


def macro_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"tp": "", "fp": "", "fn": "", "tn": "", "support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "tp": "",
        "fp": "",
        "fn": "",
        "tn": "",
        "support": sum(int(item["support"]) for item in metrics),
        "precision": sum(float(item["precision"]) for item in metrics) / len(metrics),
        "recall": sum(float(item["recall"]) for item in metrics) / len(metrics),
        "f1": sum(float(item["f1"]) for item in metrics) / len(metrics),
    }


def with_specificity(counts: dict[str, int]) -> dict[str, Any]:
    out = pub.prf_row(counts)
    out["specificity"] = pub.safe_div(counts["tn"], counts["tn"] + counts["fp"])
    return out


def gold_primary_target(row: dict[str, Any]) -> bool:
    return is_r1(row) and is_e1(row) and is_c(row, {"C1", "C2"}) and any(row.get("domain_presence", {}).values())


def is_primary_pred(row: dict[str, Any]) -> bool:
    return is_r1(row) and is_e1(row) and is_c(row, {"C1", "C2"})


def is_r1(row: dict[str, Any]) -> bool:
    return row.get("relevance_label") == "R1"


def is_e1(row: dict[str, Any]) -> bool:
    return row.get("experiencer_label") == "E1"


def is_c(row: dict[str, Any], allowed: set[str]) -> bool:
    return row.get("content_function") in allowed


def accepted_for_analysis(unit: dict[str, Any]) -> bool:
    return pub.is_accepted_unit(unit)


def span_supported(row: dict[str, Any], unit: dict[str, Any]) -> bool:
    span = clean(unit.get("evidence_span_original"))
    return bool(span and span in source_text(row))


def source_text(row: dict[str, Any]) -> str:
    return clean(row.get("source_text")) or pub.source_text_from_prediction(row)


def unit_confidence(unit: dict[str, Any]) -> float:
    try:
        return float(unit.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def domain_percentage_lookup(config: dict[str, Any]) -> dict[tuple[str, str], float]:
    return {(row["country"], row["domain"]): row["post_percentage"] for row in config["domain_distribution"]}


def topk_jaccard(left: dict[str, int], right: dict[str, int], k: int) -> float:
    return jaccard(topk_set(left, k), topk_set(right, k))


def topk_set(values: dict[str, int], k: int) -> set[str]:
    return {key for key, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:k]}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def config_key(population_id: str, unit_rule_id: str) -> str:
    return f"{population_id}__{unit_rule_id}"


def concept_key(value: str) -> str:
    return pub.normalize_concept(value)


def split_multiline(value: Any) -> list[str]:
    return [line.strip() for line in clean(value).splitlines() if line.strip()]


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
