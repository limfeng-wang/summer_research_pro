"""Command line utilities for the dental pain annotation pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from dental_ai.data_io import write_extractions_jsonl, write_unit_table
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

    env = subparsers.add_parser(
        "check-env",
        help="Report local h-ramos model environment readiness without downloading model weights.",
    )
    env.set_defaults(func=_cmd_check_env)

    run = subparsers.add_parser(
        "run-hierarchical",
        help="Run hierarchical annotation. Mock backend is available before HF model integration.",
    )
    run.add_argument("--input", required=True, help="Input source-post JSONL")
    run.add_argument("--out-dir", required=True, help="Output directory")
    run.add_argument("--backend", choices=("mock", "hf"), default="mock")
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


def _cmd_check_env(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_h_ramos_env.py"
    proc = subprocess.run([sys.executable, str(script)], check=False)
    return proc.returncode


def _cmd_run_hierarchical(args: argparse.Namespace) -> int:
    from dental_ai.run import main as run_main

    argv = [
        "--input",
        args.input,
        "--out-dir",
        args.out_dir,
        "--backend",
        args.backend,
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
