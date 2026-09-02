#!/usr/bin/env python3
"""Build descriptive analysis tables from canonicalized keyword statistics."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/keyword_report/merged_label_statistics.xlsx")
DEFAULT_OUT_DIR = Path("outputs/keyword_report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def read_canonical(path: Path) -> pd.DataFrame:
    sheet_names = pd.ExcelFile(path).sheet_names
    if "fine_canonical_by_dimension" in sheet_names:
        sheet_name = "fine_canonical_by_dimension"
    elif "canonical_stats_by_dimension" in sheet_names:
        sheet_name = "canonical_stats_by_dimension"
    else:
        raise ValueError(f"{path} does not contain a canonical keyword statistics sheet")
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = {"country", "dimension", "keyword", "main_count", "post_count", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns in fine_canonical_by_dimension: {sorted(missing)}")
    df["main_count"] = pd.to_numeric(df["main_count"], errors="coerce").fillna(0).astype(int)
    df["post_count"] = pd.to_numeric(df["post_count"], errors="coerce").fillna(0).astype(int)
    return df


def dimension_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["country", "dimension"], as_index=False).agg(main_count=("main_count", "sum"))
    totals = out.groupby("country", as_index=False).agg(country_main_total=("main_count", "sum"))
    out = out.merge(totals, on="country", how="left")
    out["country_percent"] = out["main_count"] / out["country_main_total"] * 100
    return out.sort_values(["country", "country_percent"], ascending=[True, False])


def dimension_distribution_wide(dist: pd.DataFrame) -> pd.DataFrame:
    counts = dist.pivot(index="dimension", columns="country", values="main_count").fillna(0).astype(int)
    percents = dist.pivot(index="dimension", columns="country", values="country_percent").fillna(0)
    pieces = []
    for country in sorted(dist["country"].unique()):
        pieces.append(counts[[country]].rename(columns={country: f"{country}_main_count"}))
        pieces.append(percents[[country]].rename(columns={country: f"{country}_percent"}))
    return pd.concat(pieces, axis=1).reset_index()


def top_by_country_dimension(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    scoped = df.copy()
    totals = scoped.groupby(["country", "dimension"], as_index=False).agg(dimension_total=("main_count", "sum"))
    scoped = scoped.merge(totals, on=["country", "dimension"], how="left")
    scoped["within_dimension_percent"] = scoped["main_count"] / scoped["dimension_total"] * 100
    scoped = scoped.sort_values(["country", "dimension", "main_count", "post_count", "keyword"], ascending=[True, True, False, False, True])
    scoped["within_dimension_rank"] = scoped.groupby(["country", "dimension"]).cumcount() + 1
    return scoped[scoped["within_dimension_rank"].le(top_n)][
        [
            "country",
            "dimension",
            "within_dimension_rank",
            "keyword",
            "main_count",
            "within_dimension_percent",
            "post_count",
        ]
    ]


def top_overall_by_country(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    grouped = df.groupby(["country", "keyword"], as_index=False).agg(main_count=("main_count", "sum"), post_count=("post_count", "sum"))
    totals = grouped.groupby("country", as_index=False).agg(country_main_total=("main_count", "sum"))
    grouped = grouped.merge(totals, on="country", how="left")
    grouped["country_percent"] = grouped["main_count"] / grouped["country_main_total"] * 100
    grouped = grouped.sort_values(["country", "main_count", "post_count", "keyword"], ascending=[True, False, False, True])
    grouped["country_rank"] = grouped.groupby("country").cumcount() + 1
    return grouped[grouped["country_rank"].le(top_n)][
        ["country", "country_rank", "keyword", "main_count", "country_percent", "post_count"]
    ]


def cross_country_enrichment(df: pd.DataFrame, min_total_count: int) -> pd.DataFrame:
    grouped = df.groupby(["dimension", "keyword", "country"], as_index=False).agg(main_count=("main_count", "sum"))
    dimension_totals = df.groupby(["dimension", "country"], as_index=False).agg(dimension_country_total=("main_count", "sum"))
    grouped = grouped.merge(dimension_totals, on=["dimension", "country"], how="left")
    total_lookup = {
        (str(row.dimension), str(row.country)): int(row.dimension_country_total)
        for row in dimension_totals.itertuples(index=False)
    }

    all_countries = sorted(df["country"].unique())
    rows = []
    for (dimension, keyword), sub in grouped.groupby(["dimension", "keyword"]):
        counts = {country: 0 for country in all_countries}
        totals = {country: total_lookup.get((str(dimension), country), 0) for country in all_countries}
        for row in sub.itertuples(index=False):
            counts[row.country] = int(row.main_count)
        total_count = sum(counts.values())
        if total_count < min_total_count:
            continue
        for country in all_countries:
            country_count = counts[country]
            country_total = totals.get(country, 0)
            other_count = total_count - country_count
            other_total = sum(totals.get(c, 0) for c in all_countries if c != country)
            # Add 0.5 smoothing to avoid infinite ratios for country-specific labels.
            country_rate = (country_count + 0.5) / (country_total + 1) if country_total else 0.0
            other_rate = (other_count + 0.5) / (other_total + 1) if other_total else 0.0
            rows.append(
                {
                    "dimension": dimension,
                    "keyword": keyword,
                    "country": country,
                    "country_count": country_count,
                    "other_countries_count": other_count,
                    "total_count": total_count,
                    "country_rate_per_1000_dimension_units": country_rate * 1000,
                    "other_rate_per_1000_dimension_units": other_rate * 1000,
                    "rate_difference_per_1000": (country_rate - other_rate) * 1000,
                    "log2_rate_ratio_vs_others": math.log2(country_rate / other_rate) if other_rate > 0 and country_rate > 0 else "",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["country", "log2_rate_ratio_vs_others", "rate_difference_per_1000", "total_count"],
        ascending=[True, False, False, False],
    )


def keyword_country_matrix(df: pd.DataFrame, top_overall: pd.DataFrame) -> pd.DataFrame:
    selected = sorted(top_overall["keyword"].unique(), key=str.casefold)
    sub = df[df["keyword"].isin(selected)]
    matrix = sub.pivot_table(index="keyword", columns="country", values="main_count", aggfunc="sum", fill_value=0)
    for country in sorted(df["country"].unique()):
        if country not in matrix.columns:
            matrix[country] = 0
    return matrix.reset_index()


def write_outputs(out_dir: Path, tables: dict[str, pd.DataFrame], manifest: dict[str, object], *, write_csv: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    workbook = out_dir / "descriptive_analysis_tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)
    if write_csv:
        for name, table in tables.items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "canonical_keyword_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    df = read_canonical(args.input)
    dist = dimension_distribution(df)
    top_dim = top_by_country_dimension(df, args.top_n)
    top_country = top_overall_by_country(df, args.top_n)
    enrich = cross_country_enrichment(df, args.min_total_count)
    matrix = keyword_country_matrix(df, top_country)
    tables = {
        "dimension_distribution": dist,
        "dimension_distribution_wide": dimension_distribution_wide(dist),
        "top_by_country_dimension": top_dim,
        "top_overall_by_country": top_country,
        "cross_country_enrichment": enrich,
        "top_keyword_country_matrix": matrix,
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "out_dir": str(args.out_dir),
        "top_n": args.top_n,
        "min_total_count_for_enrichment": args.min_total_count,
        "primary_count_field": "main_count",
        "post_count_note": "post_count is retained from aggregated inputs and may overcount after canonical label merges; main_count is used for comparative analysis.",
        "countries": sorted(df["country"].unique().tolist()),
        "dimensions": sorted(df["dimension"].unique().tolist()),
        "rows": int(len(df)),
        "total_main_count": int(df["main_count"].sum()),
    }
    write_outputs(args.out_dir, tables, manifest, write_csv=args.write_csv)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
