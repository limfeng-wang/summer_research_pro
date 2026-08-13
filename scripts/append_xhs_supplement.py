#!/usr/bin/env python3
"""Append distinct-body Xiaohongshu supplement rows to the main corpus.

The supplement workbook is intentionally handled as a post-split addition:
rows are deduplicated by exact ``正文`` body text, skipped if that body or
record_id already exists in the main corpus, and then written in the same JSONL
schema as ``data/raw_main_llm_input_no_gold.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dental_ai.classification_gold import source_text_fingerprint


DEFAULT_SUPPLEMENT = Path("data/xhs_牙痛数据_补充.xlsx")
DEFAULT_MAIN = Path("data/raw_main_llm_input_no_gold.jsonl")
DEFAULT_SOURCE_SHEET = "牙痛数据"
CHI_CUTPOINTS = {"short_max": 99, "medium_max": 341}
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append unique-body XHS supplement rows to the main JSONL corpus.")
    parser.add_argument("--supplement", default=str(DEFAULT_SUPPLEMENT), help="Supplement .xlsx path")
    parser.add_argument("--main", default=str(DEFAULT_MAIN), help="Main corpus JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without appending rows")
    args = parser.parse_args(argv)

    supplement_path = Path(args.supplement)
    main_path = Path(args.main)
    main_rows = _read_jsonl(main_path)
    append_rows, summary = build_append_rows(
        supplement_path=supplement_path,
        main_rows=main_rows,
    )

    if not args.dry_run and append_rows:
        with main_path.open("a", encoding="utf-8") as file:
            for row in append_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary | {"dry_run": args.dry_run}, ensure_ascii=False, indent=2))
    return 0


def build_append_rows(*, supplement_path: Path, main_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    main_bodies = {str(row.get("original_text", "")).strip() for row in main_rows}
    main_record_ids = {str(row.get("record_id", "")).strip() for row in main_rows}

    seen_bodies: set[str] = set()
    append_rows: list[dict[str, Any]] = []
    skipped_duplicate_body_in_supplement = 0
    skipped_existing_main_body = 0
    skipped_existing_main_id = 0
    supplement_rows = _xlsx_rows(supplement_path)

    for excel_row, row in supplement_rows:
        note_id = str(row.get("笔记ID", "")).strip()
        title = str(row.get("标题", "")).strip()
        body = str(row.get("正文", "")).strip()
        if body in seen_bodies:
            skipped_duplicate_body_in_supplement += 1
            continue
        seen_bodies.add(body)

        record_id = f"CHI_{note_id}"
        if record_id in main_record_ids:
            skipped_existing_main_id += 1
            continue
        if body in main_bodies:
            skipped_existing_main_body += 1
            continue

        text_clean = "\n".join(part for part in [title, body] if part).strip()
        append_rows.append(
            {
                "record_id": record_id,
                "country": "CHI",
                "language": "zh",
                "platform": "xiaohongshu",
                "original_title": title,
                "original_text": body,
                "text_clean": text_clean,
                "text_length_chars": len(text_clean),
                "text_fingerprint": source_text_fingerprint(title, body),
                "source_file": str(supplement_path),
                "source_sheet": DEFAULT_SOURCE_SHEET,
                "source_row_number": excel_row,
                "length_band": _length_band(len(text_clean)),
                "country_length_tertile_cutpoints": dict(CHI_CUTPOINTS),
            }
        )

    summary = {
        "main_rows_before": len(main_rows),
        "supplement_rows": len(supplement_rows),
        "unique_bodies_in_supplement": len(seen_bodies),
        "skipped_duplicate_body_in_supplement": skipped_duplicate_body_in_supplement,
        "skipped_existing_main_body": skipped_existing_main_body,
        "skipped_existing_main_id": skipped_existing_main_id,
        "appended_rows": len(append_rows),
        "main_rows_after": len(main_rows) + len(append_rows),
    }
    return append_rows, summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _length_band(length: int) -> str:
    if length <= CHI_CUTPOINTS["short_max"]:
        return "short"
    if length <= CHI_CUTPOINTS["medium_max"]:
        return "medium"
    return "long"


def _xlsx_rows(path: Path) -> list[tuple[int, dict[str, str]]]:
    parsed = _xlsx_cells(path)
    header = parsed[0][1]
    return [(excel_row, dict(zip(header, cells))) for excel_row, cells in parsed[1:]]


def _xlsx_cells(path: Path) -> list[tuple[int, list[str]]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        parsed = []
        for row in sheet.findall(".//a:sheetData/a:row", XLSX_NS):
            excel_row = int(row.attrib["r"])
            values: dict[int, str] = {}
            for cell in row.findall("a:c", XLSX_NS):
                index = _column_index(cell.attrib.get("r", ""))
                values[index] = _cell_value(cell, shared_strings)
            if values:
                parsed.append((excel_row, [values.get(index, "") for index in range(max(values) + 1)]))
    return parsed


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS)) for item in root.findall("a:si", XLSX_NS)]


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find("a:v", XLSX_NS)
    if cell.attrib.get("t") == "s" and value is not None and value.text is not None:
        return shared_strings[int(value.text)]
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", XLSX_NS))
    if value is not None and value.text is not None:
        return value.text
    return ""


def _column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + ord(char.upper()) - 64
    return index - 1


if __name__ == "__main__":
    raise SystemExit(main())
