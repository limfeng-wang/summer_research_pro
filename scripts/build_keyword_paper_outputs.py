#!/usr/bin/env python3
"""Create paper-oriented tables and figures for keyword analysis."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


DEFAULT_CANONICAL = Path("outputs/keyword_report/canonicalized_keyword_statistics_full.xlsx")
DEFAULT_ANALYSIS = Path("outputs/keyword_report/descriptive_analysis_tables.xlsx")
DEFAULT_OUT_DIR = Path("outputs/keyword_report")

COLORS = {
    "Symptom Description": "#2C7FB8",
    "Emotional Expression": "#D95F02",
    "Coping and Management": "#1B9E77",
    "Perceived Cause": "#7570B3",
    "Perceived Consequences": "#E6AB02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--heatmap-top-n", type=int, default=30)
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def read_fine(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="fine_canonical_by_dimension")
    df["main_count"] = pd.to_numeric(df["main_count"], errors="coerce").fillna(0).astype(int)
    df["post_count"] = pd.to_numeric(df["post_count"], errors="coerce").fillna(0).astype(int)
    return df


def dimension_contingency(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    table = df.pivot_table(index="country", columns="dimension", values="main_count", aggfunc="sum", fill_value=0)
    table = table.reindex(index=["CHI", "JPN", "KOR"])
    table = table.reindex(columns=sorted(table.columns), fill_value=0)
    chi2, p_value, dof, expected = chi2_contingency(table.values)
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    residuals = (table - expected_df) / np.sqrt(expected_df)
    n = table.values.sum()
    r, c = table.shape
    cramers_v = math.sqrt(chi2 / (n * min(r - 1, c - 1)))
    stats = {
        "chi_square": float(chi2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
        "n_main_units": int(n),
    }
    return table.reset_index(), expected_df.reset_index(), residuals.reset_index(), stats


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
        value = min(prev, p_value * n / rank)
        adjusted[idx] = value
        prev = value
    return adjusted


def pairwise_dimension_tests(table: pd.DataFrame) -> pd.DataFrame:
    countries = ["CHI", "JPN", "KOR"]
    matrix = table.set_index("country")
    rows = []
    for i, country_a in enumerate(countries):
        for country_b in countries[i + 1 :]:
            sub = matrix.loc[[country_a, country_b]]
            chi2, p_value, dof, _ = chi2_contingency(sub.values)
            n = sub.values.sum()
            cramers_v = math.sqrt(chi2 / n)
            rows.append(
                {
                    "comparison": f"{country_a}_vs_{country_b}",
                    "chi_square": float(chi2),
                    "degrees_of_freedom": int(dof),
                    "p_value": float(p_value),
                    "cramers_v": float(cramers_v),
                    "n_main_units": int(n),
                }
            )
    adjusted = benjamini_hochberg([row["p_value"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["bh_q_value"] = q_value
    return pd.DataFrame(rows)


def dimension_interpretation(residuals: pd.DataFrame) -> pd.DataFrame:
    long = residuals.melt(id_vars="country", var_name="dimension", value_name="std_residual")
    long["direction"] = np.where(long["std_residual"] > 0, "higher_than_expected", "lower_than_expected")
    return long.reindex(long["std_residual"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def top_country_keywords(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    grouped = df.groupby(["country", "keyword"], as_index=False).agg(main_count=("main_count", "sum"))
    totals = grouped.groupby("country", as_index=False).agg(country_total=("main_count", "sum"))
    grouped = grouped.merge(totals, on="country", how="left")
    grouped["percent"] = grouped["main_count"] / grouped["country_total"] * 100
    grouped = grouped.sort_values(["country", "main_count", "keyword"], ascending=[True, False, True])
    grouped["rank"] = grouped.groupby("country").cumcount() + 1
    return grouped[grouped["rank"].le(n)][["country", "rank", "keyword", "main_count", "percent"]]


def svg_text(x: float, y: float, text: object, *, size: int = 12, weight: str = "400", anchor: str = "start") -> str:
    safe = html.escape(str(text))
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" font-family="Arial, sans-serif">{safe}</text>'


def write_dimension_svg(dist: pd.DataFrame, path: Path) -> None:
    width = 940
    height = 290
    left = 120
    bar_w = 650
    bar_h = 34
    gap = 38
    top = 62
    countries = ["CHI", "JPN", "KOR"]
    dimensions = list(COLORS)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(24, 30, "CSM Dimension Distribution by Country", size=18, weight="700"),
    ]
    for row_i, country in enumerate(countries):
        y = top + row_i * (bar_h + gap)
        parts.append(svg_text(24, y + 23, country, size=13, weight="700"))
        x = left
        sub = dist[dist["country"].eq(country)].set_index("dimension")
        for dimension in dimensions:
            pct = float(sub.loc[dimension, "country_percent"]) if dimension in sub.index else 0.0
            w = bar_w * pct / 100
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" fill="{COLORS[dimension]}"/>')
            if w > 52:
                parts.append(svg_text(x + w / 2, y + 22, f"{pct:.1f}%", size=11, weight="700", anchor="middle"))
            x += w
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#333" stroke-width="0.7"/>')
    legend_x = left
    legend_y = top + 3 * (bar_h + gap) + 8
    for dimension in dimensions:
        parts.append(f'<rect x="{legend_x:.1f}" y="{legend_y:.1f}" width="12" height="12" fill="{COLORS[dimension]}"/>')
        parts.append(svg_text(legend_x + 18, legend_y + 11, dimension, size=11))
        legend_x += 165
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_heatmap_svg(df: pd.DataFrame, path: Path, top_n: int) -> pd.DataFrame:
    grouped = df.groupby(["country", "keyword"], as_index=False).agg(main_count=("main_count", "sum"))
    totals = grouped.groupby("country", as_index=False).agg(country_total=("main_count", "sum"))
    grouped = grouped.merge(totals, on="country", how="left")
    grouped["percent"] = grouped["main_count"] / grouped["country_total"] * 100
    top_keywords = (
        grouped.groupby("keyword", as_index=False)
        .agg(total=("main_count", "sum"))
        .sort_values(["total", "keyword"], ascending=[False, True])
        .head(top_n)["keyword"]
        .tolist()
    )
    heat = grouped[grouped["keyword"].isin(top_keywords)].pivot_table(index="keyword", columns="country", values="percent", fill_value=0)
    heat["total_order"] = heat.sum(axis=1)
    heat = heat.sort_values("total_order", ascending=False).drop(columns=["total_order"])

    cell_w = 84
    cell_h = 22
    left = 280
    top = 56
    width = left + cell_w * 3 + 60
    height = top + cell_h * len(heat) + 48
    max_pct = float(heat.max().max()) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(24, 30, f"Top {top_n} Canonical Keywords by Country Share", size=18, weight="700"),
    ]
    for col_i, country in enumerate(["CHI", "JPN", "KOR"]):
        parts.append(svg_text(left + col_i * cell_w + cell_w / 2, top - 12, country, size=12, weight="700", anchor="middle"))
    for row_i, (keyword, row) in enumerate(heat.iterrows()):
        y = top + row_i * cell_h
        parts.append(svg_text(24, y + 15, keyword, size=10))
        for col_i, country in enumerate(["CHI", "JPN", "KOR"]):
            pct = float(row.get(country, 0.0))
            intensity = min(1.0, pct / max_pct)
            red = int(245 - 175 * intensity)
            green = int(248 - 115 * intensity)
            blue = int(250 - 85 * intensity)
            x = left + col_i * cell_w
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w}" height="{cell_h}" fill="rgb({red},{green},{blue})" stroke="white"/>')
            parts.append(svg_text(x + cell_w / 2, y + 15, f"{pct:.1f}", size=10, anchor="middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return heat.reset_index()


def write_dimension_matplotlib(dist: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    countries = ["CHI", "JPN", "KOR"]
    dimensions = list(COLORS)
    fig, ax = plt.subplots(figsize=(8.6, 3.0), constrained_layout=True)
    left = np.zeros(len(countries))
    y = np.arange(len(countries))
    dist_idx = dist.set_index(["country", "dimension"])
    for dimension in dimensions:
        values = np.array(
            [
                float(dist_idx.loc[(country, dimension), "country_percent"])
                if (country, dimension) in dist_idx.index
                else 0.0
                for country in countries
            ]
        )
        ax.barh(y, values, left=left, label=dimension, color=COLORS[dimension], height=0.58)
        for yi, start, value in zip(y, left, values):
            if value >= 7:
                ax.text(start + value / 2, yi, f"{value:.1f}%", ha="center", va="center", fontsize=8)
        left += values
    ax.set_yticks(y, countries)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of accepted CSM units (%)")
    ax.set_ylabel("")
    ax.set_title("CSM Dimension Distribution by Country", loc="left", fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
    sns.despine(ax=ax, left=True, bottom=False)
    fig.savefig(out_dir / "figure_dimension_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_dimension_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def write_heatmap_matplotlib(heatmap_table: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    heat = heatmap_table.set_index("keyword")[["CHI", "JPN", "KOR"]]
    fig_h = max(5.8, 0.22 * len(heat) + 1.5)
    fig, ax = plt.subplots(figsize=(5.9, fig_h), constrained_layout=True)
    sns.heatmap(
        heat,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Country share of main units (%)"},
    )
    ax.set_title("Top Canonical Keywords by Country Share", loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(out_dir / "figure_keyword_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figure_keyword_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_fine(args.canonical)
    dist = pd.read_excel(args.analysis, sheet_name="dimension_distribution")
    table, expected, residuals, stats = dimension_contingency(df)
    pairwise = pairwise_dimension_tests(table)
    residual_long = dimension_interpretation(residuals)
    top_keywords = top_country_keywords(df)
    heatmap_table = write_heatmap_svg(df, out_dir / "figure_keyword_heatmap.svg", args.heatmap_top_n)
    write_dimension_svg(dist, out_dir / "figure_dimension_distribution.svg")
    write_dimension_matplotlib(dist, out_dir)
    write_heatmap_matplotlib(heatmap_table, out_dir)

    workbook = out_dir / "statistical_tests_and_paper_tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="dimension_counts", index=False)
        expected.to_excel(writer, sheet_name="dimension_expected", index=False)
        residuals.to_excel(writer, sheet_name="dimension_std_residuals", index=False)
        residual_long.to_excel(writer, sheet_name="dimension_residuals_long", index=False)
        pairwise.to_excel(writer, sheet_name="pairwise_dimension_tests", index=False)
        top_keywords.to_excel(writer, sheet_name="top15_keywords_by_country", index=False)
        heatmap_table.to_excel(writer, sheet_name="heatmap_values_percent", index=False)

    if args.write_csv:
        for name, table_df in {
            "dimension_counts": table,
            "dimension_expected": expected,
            "dimension_std_residuals": residuals,
            "dimension_residuals_long": residual_long,
            "pairwise_dimension_tests": pairwise,
            "top15_keywords_by_country": top_keywords,
            "heatmap_values_percent": heatmap_table,
        }.items():
            table_df.to_csv(out_dir / f"{name}.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_input": str(args.canonical),
        "analysis_input": str(args.analysis),
        "out_dir": str(out_dir),
        "global_dimension_chi_square": stats,
        "outputs": [
            "statistical_tests_and_paper_tables.xlsx",
            "figure_dimension_distribution.svg",
            "figure_dimension_distribution.png",
            "figure_dimension_distribution.pdf",
            "figure_keyword_heatmap.svg",
            "figure_keyword_heatmap.png",
            "figure_keyword_heatmap.pdf",
        ],
    }
    (out_dir / "keyword_paper_outputs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
