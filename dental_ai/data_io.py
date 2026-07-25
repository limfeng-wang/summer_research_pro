"""Input/output helpers for posts and extraction results.

The functions here are deliberately format-light. They load tabular source
posts into the schema models and persist extraction outputs as JSONL/CSV
without assuming a specific gold-set layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from dental_ai.schemas import Country, ExtractionResult, Language, SourcePost, flatten_units


POST_ID_CANDIDATES = ("post_id", "Post_ID", "id", "ID", "note_id", "tweet_id")
TEXT_CANDIDATES = ("text", "Text", "content", "Content", "post_text", "clean_text", "正文")
COUNTRY_CANDIDATES = ("country", "Country", "country_code", "Country_Code")
LANGUAGE_CANDIDATES = ("language", "Language", "lang", "Lang")


@dataclass(frozen=True)
class PostColumnMap:
    """Column names that map a table into SourcePost fields.

    `country` and `language` are optional because some files may be split by
    country/language and receive those values as explicit constants.
    """

    post_id: str
    text: str
    country: str | None = None
    language: str | None = None


def read_table(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read a CSV, Excel, or JSONL table into a DataFrame."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def infer_post_column_map(columns: Iterable[str]) -> PostColumnMap:
    """Infer a post column map only when required columns are unambiguous."""

    column_names = list(columns)
    return PostColumnMap(
        post_id=_infer_one(column_names, POST_ID_CANDIDATES, "post_id"),
        text=_infer_one(column_names, TEXT_CANDIDATES, "text"),
        country=_infer_optional(column_names, COUNTRY_CANDIDATES, "country"),
        language=_infer_optional(column_names, LANGUAGE_CANDIDATES, "language"),
    )


def load_posts(
    path: str | Path,
    *,
    column_map: PostColumnMap | None = None,
    country: Country | str | None = None,
    language: Language | str | None = None,
    sheet_name: str | int = 0,
    drop_empty_text: bool = False,
) -> list[SourcePost]:
    """Load source posts from a table.

    Args:
        path: CSV, Excel, or JSONL table path.
        column_map: Explicit source column mapping. If omitted, required
            columns are inferred only when unambiguous.
        country: Constant country value used when no country column is mapped.
        language: Constant language value used when no language column is mapped.
        sheet_name: Excel sheet name or index.
        drop_empty_text: If True, rows with empty text are skipped. Strict mode
            keeps the default False and raises validation errors instead.
    """

    frame = read_table(path, sheet_name=sheet_name)
    column_map = column_map or infer_post_column_map(frame.columns)

    if column_map.country is None and country is None:
        raise ValueError("country must be provided either as a column or an explicit constant")
    if column_map.language is None and language is None:
        raise ValueError("language must be provided either as a column or an explicit constant")

    posts: list[SourcePost] = []
    for row_index, row in frame.iterrows():
        text = _clean_cell(row[column_map.text])
        if not text and drop_empty_text:
            continue

        try:
            posts.append(
                SourcePost(
                    post_id=_clean_cell(row[column_map.post_id]),
                    country=_row_or_constant(row, column_map.country, country),
                    language=_row_or_constant(row, column_map.language, language),
                    text=text,
                )
            )
        except Exception as exc:
            raise ValueError(f"Invalid source post at row {row_index}: {exc}") from exc

    return posts


def write_posts_jsonl(posts: Iterable[SourcePost], path: str | Path) -> None:
    """Write source posts as JSONL using schema field names."""

    _write_models_jsonl(posts, path)


def read_posts_jsonl(path: str | Path) -> list[SourcePost]:
    """Read source posts from JSONL previously written with schema field names."""

    return [SourcePost.model_validate(item) for item in _read_jsonl(path)]


def write_extractions_jsonl(results: Iterable[ExtractionResult], path: str | Path) -> None:
    """Write extraction results as one JSON object per line."""

    _write_models_jsonl(results, path)


def read_extractions_jsonl(path: str | Path) -> list[ExtractionResult]:
    """Read extraction results from JSONL and validate them against schemas."""

    return [ExtractionResult.model_validate(item) for item in _read_jsonl(path)]


def write_unit_table(
    results: Iterable[ExtractionResult],
    path: str | Path,
    *,
    accepted_only: bool = True,
) -> None:
    """Write a flattened unit-level table as CSV or Excel."""

    path = Path(path)
    rows = flatten_units(results, accepted_only=accepted_only)
    frame = pd.DataFrame(rows)

    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix in {".xlsx", ".xls"}:
        frame.to_excel(path, index=False)
        return
    raise ValueError(f"Unsupported unit table format: {path.suffix}")


def _infer_one(columns: list[str], candidates: Iterable[str], field_name: str) -> str:
    matches = [column for column in columns if column in candidates]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Could not infer {field_name} column; pass PostColumnMap explicitly")
    raise ValueError(f"Ambiguous {field_name} columns {matches!r}; pass PostColumnMap explicitly")


def _infer_optional(columns: list[str], candidates: Iterable[str], field_name: str) -> str | None:
    matches = [column for column in columns if column in candidates]
    if len(matches) <= 1:
        return matches[0] if matches else None
    raise ValueError(f"Ambiguous {field_name} columns {matches!r}; pass PostColumnMap explicitly")


def _row_or_constant(row: pd.Series, column: str | None, constant: Any) -> Any:
    if column is not None:
        return _clean_cell(row[column])
    return constant


def _clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _write_models_jsonl(models: Iterable[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for model in models:
            handle.write(json.dumps(model.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


__all__ = [
    "PostColumnMap",
    "infer_post_column_map",
    "load_posts",
    "read_extractions_jsonl",
    "read_posts_jsonl",
    "read_table",
    "write_extractions_jsonl",
    "write_posts_jsonl",
    "write_unit_table",
]
