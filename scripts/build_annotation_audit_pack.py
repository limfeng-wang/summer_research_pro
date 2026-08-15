#!/usr/bin/env python3
"""Build manual-audit CSVs for LLM annotation quality checks.

The exported buckets target:
- possible false negatives in top-level routing
- CSM-eligible rows with zero extracted units
- evidence spans that are not exact source substrings
- accepted-unit spot checks
- rejected/unsupported-unit spot checks
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable


KEYWORD_PATTERNS = {
    "CHI": re.compile(r"(牙疼|牙痛|牙龈痛|牙齿痛|牙酸|智齿.*痛)"),
    "JPN": re.compile(r"(歯が痛|歯痛|歯.*痛い|奥歯.*痛|虫歯.*痛)"),
    "KOR": re.compile(r"(치통|이가\s*아|이빨\s*아|치아\s*통증|잇몸\s*통증|이\s*아파|아픈\s*이|아픈\s*치아|잇몸.*아파)"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Merged/repaired run directory containing annotations.jsonl")
    parser.add_argument("--out-dir", default="", help="Audit output directory")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "audit_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(run_dir / "annotations.jsonl")
    rng = random.Random(args.seed)

    keyword_hits = [row for row in rows if _keyword_hits(row)]
    keyword_r0 = [row for row in keyword_hits if row.get("relevance_label") == "R0"]
    keyword_noneligible = [
        row
        for row in keyword_hits
        if row.get("relevance_label") == "R1" and not _is_csm_eligible(row)
    ]
    zero_unit_eligible = [row for row in rows if _is_csm_eligible(row) and not row.get("units")]
    accepted_unit_rows = _unit_rows(rows, lambda unit: unit.get("judge_verdict") == "accept")
    rejected_unit_rows = _unit_rows(
        rows,
        lambda unit: unit.get("judge_verdict") == "reject" or unit.get("support_type") == "unsupported",
    )
    span_failure_rows = _span_failure_rows(rows)

    exports = {
        "keyword_hit_classified_R0_possible_false_negatives.csv": _post_rows(
            _sample(keyword_r0, args.sample_size, rng)
        ),
        "keyword_hit_R1_noneligible_possible_routing_errors.csv": _post_rows(
            _sample(keyword_noneligible, args.sample_size, rng)
        ),
        "eligible_zero_units_possible_missed_extractions.csv": _post_rows(
            _sample(zero_unit_eligible, args.sample_size, rng)
        ),
        "accepted_units_spot_check.csv": _sample(accepted_unit_rows, args.sample_size, rng),
        "rejected_or_unsupported_units_spot_check.csv": _sample(rejected_unit_rows, args.sample_size, rng),
        "evidence_span_failures_all.csv": span_failure_rows,
    }

    for filename, export_rows in exports.items():
        _write_csv(out_dir / filename, export_rows)

    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "rows": len(rows),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "country_counts": dict(Counter(row.get("country") for row in rows)),
        "keyword_hit_rows": len(keyword_hits),
        "keyword_hit_classified_R0_rows": len(keyword_r0),
        "keyword_hit_R1_noneligible_rows": len(keyword_noneligible),
        "eligible_zero_unit_rows": len(zero_unit_eligible),
        "accepted_unit_rows": len(accepted_unit_rows),
        "rejected_or_unsupported_unit_rows": len(rejected_unit_rows),
        "evidence_span_failure_units": len(span_failure_rows),
        "exports": {filename: len(export_rows) for filename, export_rows in exports.items()},
    }
    (out_dir / "audit_pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


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


def _source_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field) or "")
        for field in ("original_title", "text_clean", "original_text")
        if row.get(field)
    ).strip()


def _keyword_hits(row: dict[str, Any]) -> list[str]:
    pattern = KEYWORD_PATTERNS.get(str(row.get("country") or ""))
    if not pattern:
        return []
    return pattern.findall(_source_text(row))


def _is_csm_eligible(row: dict[str, Any]) -> bool:
    return (
        row.get("relevance_label") == "R1"
        and row.get("experiencer_label") in {"E1", "E2"}
        and row.get("content_function") in {"C1", "C2"}
    )


def _post_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "post_id": row.get("post_id") or row.get("record_id"),
            "country": row.get("country"),
            "language": row.get("language"),
            "relevance_label": row.get("relevance_label"),
            "experiencer_label": row.get("experiencer_label"),
            "content_function": row.get("content_function"),
            "unit_count": len(row.get("units") or []),
            "keyword_hits": " | ".join(_keyword_hits(row)),
            "text": _source_text(row),
            "manual_correct_relevance": "",
            "manual_correct_experiencer": "",
            "manual_correct_content": "",
            "manual_should_have_csm_units": "",
            "manual_notes": "",
        }
        for row in rows
    ]


def _unit_rows(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        text = _source_text(row)
        for unit in row.get("units") or []:
            if not predicate(unit):
                continue
            span = str(unit.get("evidence_span_original") or "")
            output.append(
                {
                    "post_id": row.get("post_id") or row.get("record_id"),
                    "country": row.get("country"),
                    "language": row.get("language"),
                    "relevance_label": row.get("relevance_label"),
                    "experiencer_label": row.get("experiencer_label"),
                    "content_function": row.get("content_function"),
                    "unit_id": unit.get("unit_id"),
                    "domain": unit.get("domain"),
                    "evidence_span_original": span,
                    "span_exactly_in_source": span in text,
                    "surface_text_working": unit.get("surface_text_working"),
                    "normalized_concept_en": unit.get("normalized_concept_en"),
                    "support_type": unit.get("support_type"),
                    "assertion": unit.get("assertion"),
                    "temporality": unit.get("temporality"),
                    "sentiment_or_outcome": unit.get("sentiment_or_outcome"),
                    "confidence": unit.get("confidence"),
                    "judge_verdict": unit.get("judge_verdict"),
                    "text": text,
                    "manual_unit_supported": "",
                    "manual_unit_domain_correct": "",
                    "manual_unit_should_keep": "",
                    "manual_corrected_span": "",
                    "manual_notes": "",
                }
            )
    return output


def _span_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _unit_rows(
        rows,
        lambda unit: False,
    ) + [
        unit_row
        for unit_row in _unit_rows(rows, lambda unit: True)
        if not unit_row["span_exactly_in_source"]
    ]


def _sample(rows: list[dict[str, Any]], sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) <= sample_size:
        return list(rows)
    return rng.sample(rows, sample_size)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

