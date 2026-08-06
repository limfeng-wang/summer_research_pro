"""Command line utilities for the dental pain annotation pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from dental_ai.data_io import write_extractions_jsonl, write_unit_table
from dental_ai.classification_gold import classification_gold_summary, load_classification_gold_xlsx
from dental_ai.goldset import (
    load_csm_gold_jsons,
    primary_csm_results,
    proxy_csm_results,
    summarize_csm_gold,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dental_ai command line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dental_ai.cli",
        description="Utilities for validating and exporting dental pain annotation data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser(
        "summarize-gold",
        help="Validate CSM gold JSON files and print count summaries.",
    )
    summarize.add_argument("paths", nargs="+", help="CSM gold JSON file paths")
    summarize.add_argument(
        "--no-validate-spans",
        action="store_true",
        help="Load files without checking evidence spans against source text.",
    )
    summarize.add_argument(
        "--json",
        action="store_true",
        help="Print summary as JSON instead of a readable table.",
    )
    summarize.set_defaults(func=_cmd_summarize_gold)

    export = subparsers.add_parser(
        "export-gold-units",
        help="Validate CSM gold JSON files and export flattened unit tables.",
    )
    export.add_argument("paths", nargs="+", help="CSM gold JSON file paths")
    export.add_argument(
        "--out-dir",
        required=True,
        help="Directory for exported files.",
    )
    export.add_argument(
        "--format",
        choices=("csv", "xlsx"),
        default="csv",
        help="Export table format.",
    )
    export.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rejected/negated/unsupported units in flattened unit exports.",
    )
    export.add_argument(
        "--jsonl",
        action="store_true",
        help="Also export post-level JSONL files for all/primary/proxy records.",
    )
    export.add_argument(
        "--no-validate-spans",
        action="store_true",
        help="Load files without checking evidence spans against source text.",
    )
    export.set_defaults(func=_cmd_export_gold_units)

    classify_gold = subparsers.add_parser(
        "summarize-classification-gold",
        help="Load the multilingual R/E/C classification workbook and print count summaries.",
    )
    classify_gold.add_argument(
        "--xlsx",
        default="data/三语分类金标准-Law.xlsx",
        help="Classification gold workbook.",
    )
    classify_gold.add_argument(
        "--json",
        action="store_true",
        help="Print summary as JSON.",
    )
    classify_gold.set_defaults(func=_cmd_summarize_classification_gold)

    splits = subparsers.add_parser(
        "build-splits",
        help="Export classification gold and rebuild leak-checked eval/main splits.",
    )
    splits.add_argument("--classification-gold-xlsx", default="data/三语分类金标准-Law.xlsx")
    splits.add_argument("--classification-gold-jsonl", default="data/classification_gold_172.jsonl")
    splits.add_argument("--csm-gold", default="data/csm_gold_50E1_10E2.json")
    splits.add_argument("--eval", default="data/raw_eval_holdout_150_no_gold.jsonl")
    splits.add_argument("--eval-out", default="data/raw_eval_holdout_150_no_gold.jsonl")
    splits.add_argument("--main-in", default="data/raw_main_llm_input_no_gold.jsonl")
    splits.add_argument("--main-out", default="data/raw_main_llm_input_no_gold.jsonl")
    splits.add_argument("--manifest", default="data/project_split_manifest.json")
    splits.set_defaults(func=_cmd_build_splits)

    env = subparsers.add_parser(
        "check-env",
        help="Report local h-ramos model environment readiness without downloading model weights.",
    )
    env.set_defaults(func=_cmd_check_env)

    models = subparsers.add_parser(
        "check-models",
        help="Check configured local model paths without loading weights.",
    )
    models.add_argument("--config", default="configs/model_stack.yaml", help="Model stack config")
    models.add_argument("--models-root", default="/hdd-storage/lawrencelcty/huggingface/models")
    models.set_defaults(func=_cmd_check_models)

    run = subparsers.add_parser(
        "run-hierarchical",
        help="Run hierarchical annotation. Mock backend is available before HF model integration.",
    )
    run.add_argument("--input", required=True, help="Input source-post JSONL")
    run.add_argument("--out-dir", required=True, help="Output directory")
    run.add_argument("--backend", choices=("mock", "hf"), default="mock")
    run.add_argument("--config", default="configs/model_stack.yaml", help="Model stack config")
    run.add_argument("--models-root", default="/hdd-storage/lawrencelcty/huggingface/models")
    run.add_argument("--hf-stage", choices=("classify", "full"), default="classify")
    run.add_argument("--limit", type=int, default=0, help="Optional max rows for smoke tests")
    run.set_defaults(func=_cmd_run_hierarchical)

    return parser


def _cmd_summarize_gold(args: argparse.Namespace) -> int:
    results = load_csm_gold_jsons(args.paths, validate_spans=not args.no_validate_spans)
    summary = summarize_csm_gold(results)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_summary(summary)
    return 0


def _cmd_export_gold_units(args: argparse.Namespace) -> int:
    results = load_csm_gold_jsons(args.paths, validate_spans=not args.no_validate_spans)
    primary = primary_csm_results(results)
    proxy = proxy_csm_results(results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.format
    accepted_only = not args.include_rejected
    write_unit_table(results, out_dir / f"all_units.{suffix}", accepted_only=accepted_only)
    write_unit_table(primary, out_dir / f"primary_units.{suffix}", accepted_only=accepted_only)
    write_unit_table(proxy, out_dir / f"proxy_units.{suffix}", accepted_only=accepted_only)

    if args.jsonl:
        write_extractions_jsonl(results, out_dir / "all_posts.jsonl")
        write_extractions_jsonl(primary, out_dir / "primary_posts.jsonl")
        write_extractions_jsonl(proxy, out_dir / "proxy_posts.jsonl")

    summary = summarize_csm_gold(results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_summary(summary)
    print(f"\nexported: {out_dir}")
    return 0


def _cmd_summarize_classification_gold(args: argparse.Namespace) -> int:
    records = load_classification_gold_xlsx(args.xlsx)
    summary = classification_gold_summary(records)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


def _cmd_build_splits(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_project_splits.py"
    argv = [
        sys.executable,
        str(script),
        "--classification-gold-xlsx",
        args.classification_gold_xlsx,
        "--classification-gold-jsonl",
        args.classification_gold_jsonl,
        "--csm-gold",
        args.csm_gold,
        "--eval",
        args.eval,
        "--eval-out",
        args.eval_out,
        "--main-in",
        args.main_in,
        "--main-out",
        args.main_out,
        "--manifest",
        args.manifest,
    ]
    proc = subprocess.run(argv, check=False)
    return proc.returncode


def _cmd_check_env(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_h_ramos_env.py"
    proc = subprocess.run([sys.executable, str(script)], check=False)
    return proc.returncode


def _cmd_check_models(args: argparse.Namespace) -> int:
    from dental_ai.model_config import check_model_paths, load_model_stack_config

    config = load_model_stack_config(args.config)
    report = check_model_paths(config, args.models_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["exists"] for item in report.values()) else 1


def _cmd_run_hierarchical(args: argparse.Namespace) -> int:
    from dental_ai.run import main as run_main

    argv = [
        "--input",
        args.input,
        "--out-dir",
        args.out_dir,
        "--backend",
        args.backend,
        "--config",
        args.config,
        "--models-root",
        args.models_root,
        "--hf-stage",
        args.hf_stage,
    ]
    if args.limit:
        argv.extend(["--limit", str(args.limit)])
    return run_main(argv)


def _print_summary(summary: dict[str, object]) -> None:
    scalar_keys = [
        "posts",
        "units",
        "accepted_units",
        "span_exact_matches",
        "primary_posts",
        "proxy_posts",
    ]
    for key in scalar_keys:
        print(f"{key}: {summary[key]}")

    section_keys = [
        "countries",
        "languages",
        "relevance_labels",
        "experiencer_labels",
        "content_functions",
        "domains",
        "accepted_domains",
        "concept_statuses",
        "support_types",
        "assertions",
        "judge_verdicts",
    ]
    for key in section_keys:
        print(f"\n{key}:")
        values = summary[key]
        if not isinstance(values, dict):
            print(f"  {values}")
            continue
        for label, count in values.items():
            printable_label = label if label else "<missing>"
            print(f"  {printable_label}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
