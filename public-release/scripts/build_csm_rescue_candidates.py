#!/usr/bin/env python3
"""Build second-pass CSM rescue candidates from first-pass outputs.

This script does not modify the input runs. It exports non-CSM-eligible rows
that deserve a high-recall rescue review:
- KOR: all non-eligible rows.
- CHI/JPN: keyword/anchor-targeted non-eligible rows.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOOTHACHE_PATTERNS = {
    "CHI": re.compile(r"(牙疼|牙痛|牙龈痛|牙齿痛|牙酸|智齿.*痛|牙髓炎|根管.*痛|拔牙.*痛)"),
    "JPN": re.compile(r"(歯が痛|歯痛|歯.*痛い|奥歯.*痛|虫歯.*痛|親知らず.*痛|抜歯.*痛|歯茎.*痛)"),
    "KOR": re.compile(
        r"(치통|이가\s*아|이빨\s*아|치아\s*통증|잇몸\s*통증|이\s*아파|아픈\s*이|"
        r"아픈\s*치아|잇몸.*아파|사랑니.*아파|발치.*아파|치과.*통증)"
    ),
}

PERSONAL_OR_PROXY_PATTERNS = {
    "CHI": re.compile(r"(我|我的|本人|妈妈|爸爸|孩子|朋友|老公|老婆|患者|疼到|痛到|睡不着|吃不了|止痛|布洛芬|看牙|牙医)"),
    "JPN": re.compile(r"(私|自分|母|父|子ども|友達|夫|妻|痛くて|眠れ|食べられ|歯医者|痛み止め|ロキソニン)"),
    "KOR": re.compile(r"(나|나는|제가|내|우리|엄마|아빠|친구|남편|아내|딸|아들|아파서|잠|못 먹|치과|진통제|타이레놀|이부프로펜)"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Merged first-pass run directories")
    parser.add_argument("--output", required=True, help="Candidate JSONL output")
    parser.add_argument("--manifest", default="", help="Optional manifest path")
    args = parser.parse_args()

    rows = []
    seen: set[str] = set()
    for run_dir in [Path(item) for item in args.run_dirs]:
        for row in _read_annotations(run_dir):
            post_id = _post_id(row)
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            reasons = _candidate_reasons(row)
            if not reasons:
                continue
            rows.append(_candidate_payload(row, run_dir, reasons))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "run_dirs": args.run_dirs,
        "output": str(output_path),
        "candidate_rows": len(rows),
        "country_counts": dict(Counter(row.get("country") for row in rows)),
        "reason_counts": dict(Counter(reason for row in rows for reason in row["rescue_candidate_reasons"])),
        "first_pass_label_counts": dict(
            Counter(
                "|".join(
                    str(row.get(field))
                    for field in [
                        "first_pass_relevance_label",
                        "first_pass_experiencer_label",
                        "first_pass_content_function",
                    ]
                )
                for row in rows
            )
        ),
    }
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _read_annotations(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "annotations.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing annotations.jsonl: {path}")
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


def _candidate_reasons(row: dict[str, Any]) -> list[str]:
    if _is_csm_eligible(row):
        return []
    country = str(row.get("country") or "")
    label = (row.get("relevance_label"), row.get("experiencer_label"), row.get("content_function"))
    text = _source_text(row)
    reasons = []
    if country == "KOR":
        reasons.append("kor_all_noneligible")
    if _has_toothache_keyword(country, text):
        if row.get("relevance_label") == "R0":
            reasons.append("keyword_hit_r0")
        if row.get("relevance_label") == "R1" and label[1] in {"E1", "E2"} and label[2] in {"C3", "C4", "C5"}:
            reasons.append("keyword_hit_specific_noneligible_content")
        if row.get("relevance_label") == "R1" and label[1] == "E3" and _has_personal_or_proxy_anchor(country, text):
            reasons.append("keyword_hit_e3_with_personal_proxy_anchor")
    return sorted(set(reasons))


def _candidate_payload(row: dict[str, Any], run_dir: Path, reasons: list[str]) -> dict[str, Any]:
    fields = [
        "post_id",
        "record_id",
        "country",
        "language",
        "platform",
        "original_title",
        "original_text",
        "text_clean",
        "text_length_chars",
        "text_fingerprint",
        "analysis_text_en",
        "source_file",
        "source_sheet",
        "source_row_number",
        "length_band",
        "country_length_tertile_cutpoints",
    ]
    payload = {field: row.get(field) for field in fields if field in row}
    payload["post_id"] = _post_id(row)
    payload["first_pass_run_dir"] = str(run_dir)
    payload["first_pass_relevance_label"] = row.get("relevance_label")
    payload["first_pass_experiencer_label"] = row.get("experiencer_label")
    payload["first_pass_content_function"] = row.get("content_function")
    payload["first_pass_unit_count"] = len(row.get("units") or [])
    payload["rescue_candidate_reasons"] = reasons
    return payload


def _is_csm_eligible(row: dict[str, Any]) -> bool:
    return (
        row.get("relevance_label") == "R1"
        and row.get("experiencer_label") in {"E1", "E2"}
        and row.get("content_function") in {"C1", "C2"}
    )


def _has_toothache_keyword(country: str, text: str) -> bool:
    pattern = TOOTHACHE_PATTERNS.get(country)
    return bool(pattern and pattern.search(text))


def _has_personal_or_proxy_anchor(country: str, text: str) -> bool:
    pattern = PERSONAL_OR_PROXY_PATTERNS.get(country)
    return bool(pattern and pattern.search(text))


def _source_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get(field) or "")
        for field in ["original_title", "text_clean", "original_text", "analysis_text_en"]
        if row.get(field)
    )


def _post_id(row: dict[str, Any]) -> str:
    return str(row.get("post_id") or row.get("record_id") or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())

