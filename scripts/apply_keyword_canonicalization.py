#!/usr/bin/env python3
"""Apply an approved canonical keyword mapping to the statistics workbook."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_STATS = Path("data/keyword_statistics_CHI_JPN_KOR.xlsx")
DEFAULT_MAPPING = Path("outputs/keyword_report/canonical_keyword_mapping.xlsx")
DEFAULT_OUT = Path("outputs/keyword_report/canonicalized_keyword_statistics_full.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_stats(path: Path) -> pd.DataFrame:
    rows = []
    for sheet_name in pd.ExcelFile(path).sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        country = sheet_name.split("_")[-1]
        body = raw.iloc[3:].copy()
        body.columns = ["rank", "dimension", "keyword", "main_count", "post_count"]
        body = body.dropna(subset=["dimension", "keyword"])
        for _, row in body.iterrows():
            rows.append(
                {
                    "source_sheet": sheet_name,
                    "country": country,
                    "dimension": str(row["dimension"]).strip(),
                    "keyword": str(row["keyword"]).strip(),
                    "main_count": int(pd.to_numeric(row["main_count"], errors="coerce") or 0),
                    "post_count": int(pd.to_numeric(row["post_count"], errors="coerce") or 0),
                }
            )
    return pd.DataFrame(rows)


def read_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_excel(path)
    required = {"dimension", "keyword", "fine_canonical_keyword", "parent_canonical_keyword"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file is missing required columns: {sorted(missing)}")
    for col in required:
        mapping[col] = mapping[col].fillna("").astype(str).str.strip()
    mapping["fine_canonical_keyword"] = mapping.apply(
        lambda row: row["fine_canonical_keyword"] or row["keyword"],
        axis=1,
    )
    mapping["parent_canonical_keyword"] = mapping.apply(
        lambda row: row["parent_canonical_keyword"] or row["fine_canonical_keyword"],
        axis=1,
    )
    duplicates = mapping.duplicated(["dimension", "keyword"], keep=False)
    if duplicates.any():
        dupes = mapping.loc[duplicates, ["dimension", "keyword"]].drop_duplicates()
        raise ValueError(f"Mapping contains duplicate dimension/keyword rows: {dupes.head(20).to_dict('records')}")
    return mapping


def summarize_level(df: pd.DataFrame, keyword_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["country", "dimension", keyword_col], as_index=False)
        .agg(main_count=("main_count", "sum"), post_count=("post_count", "sum"))
        .rename(columns={keyword_col: "keyword"})
    )
    grouped["rank"] = (
        grouped.sort_values(["country", "dimension", "main_count", "post_count", "keyword"], ascending=[True, True, False, False, True])
        .groupby(["country", "dimension"])
        .cumcount()
        + 1
    )
    return grouped.sort_values(["country", "dimension", "rank"])


def sensitivity_summary(original: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    before = (
        original.groupby(["country", "dimension"], as_index=False)
        .agg(original_labels=("keyword", "nunique"), original_units=("main_count", "sum"))
    )
    fine = (
        merged.groupby(["country", "dimension"], as_index=False)
        .agg(fine_canonical_labels=("fine_canonical_keyword", "nunique"), fine_units=("main_count", "sum"))
    )
    parent = (
        merged.groupby(["country", "dimension"], as_index=False)
        .agg(parent_canonical_labels=("parent_canonical_keyword", "nunique"), parent_units=("main_count", "sum"))
    )
    out = before.merge(fine, on=["country", "dimension"], how="left").merge(parent, on=["country", "dimension"], how="left")
    out["fine_label_reduction"] = out["original_labels"] - out["fine_canonical_labels"]
    out["parent_label_reduction"] = out["original_labels"] - out["parent_canonical_labels"]
    return out.sort_values(["country", "dimension"])


def top20(df: pd.DataFrame, keyword_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["country", keyword_col], as_index=False)
        .agg(main_count=("main_count", "sum"), post_count=("post_count", "sum"))
        .rename(columns={keyword_col: "keyword"})
    )
    grouped["rank"] = (
        grouped.sort_values(["country", "main_count", "post_count", "keyword"], ascending=[True, False, False, True])
        .groupby("country")
        .cumcount()
        + 1
    )
    return grouped[grouped["rank"].le(20)].sort_values(["country", "rank"])


def main() -> None:
    args = parse_args()
    stats = read_stats(args.stats)
    mapping = read_mapping(args.mapping)
    merged = stats.merge(mapping, on=["dimension", "keyword"], how="left", validate="many_to_one")
    missing = merged["fine_canonical_keyword"].isna()
    if missing.any():
        missing_rows = merged.loc[missing, ["dimension", "keyword"]].drop_duplicates()
        raise ValueError(f"Mapping is missing {len(missing_rows)} labels; first rows: {missing_rows.head(20).to_dict('records')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stats": str(args.stats),
        "mapping": str(args.mapping),
        "out": str(args.out),
        "input_rows": int(len(stats)),
        "mapped_rows": int(len(merged)),
    }
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        sensitivity_summary(stats, merged).to_excel(writer, sheet_name="sensitivity_summary", index=False)
        top20(stats, "keyword").to_excel(writer, sheet_name="top20_original", index=False)
        top20(merged, "fine_canonical_keyword").to_excel(writer, sheet_name="top20_fine_canonical", index=False)
        top20(merged, "parent_canonical_keyword").to_excel(writer, sheet_name="top20_parent_canonical", index=False)
        summarize_level(merged, "fine_canonical_keyword").to_excel(writer, sheet_name="fine_canonical_by_dimension", index=False)
        summarize_level(merged, "parent_canonical_keyword").to_excel(writer, sheet_name="parent_canonical_by_dimension", index=False)
        mapping.to_excel(writer, sheet_name="mapping_used", index=False)
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
