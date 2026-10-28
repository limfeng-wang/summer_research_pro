"""Translate the publication records without exposing annotations to inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def source_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "post_id": record["post_id"],
        "country": record["country"],
        "language": record["language"],
        "platform": record.get("platform", ""),
        "original_title": record.get("title", ""),
        "original_text": record.get("text", ""),
        "text_clean": record["source_text"],
    }


def extraction_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = source_payload(record)
    reference = record.get("extraction_reference") or record
    payload.update(reference["classification"])
    payload["units"] = [
        {key: value for key, value in unit.items() if key != "quality_flags"}
        for unit in reference["units"]
    ]
    return payload
