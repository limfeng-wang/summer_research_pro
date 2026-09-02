#!/usr/bin/env python3
"""Build a team-facing keyword-analysis report and processed workbook."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CANONICAL_STATS = Path("outputs/keyword_report/merged_label_statistics.xlsx")
ANALYSIS_PACK = Path("outputs/keyword_report/merged_label_statistics.xlsx")
PAPER_TABLES = Path("outputs/keyword_report/statistical_tests_and_paper_tables.xlsx")
PAPER_DIR = Path("outputs/keyword_report")
MAPPING = Path("outputs/keyword_report/merged_label_statistics.xlsx")
OUT_DIR = Path("outputs/keyword_report")


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def markdown_table(df: pd.DataFrame, float_digits: int = 1) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for row in df.itertuples(index=False):
        cells = []
        for col, value in zip(headers, row):
            if isinstance(value, float) and math.isnan(value):
                cells.append("")
            elif col in {"p_value", "bh_q_value"} and isinstance(value, float):
                cells.append(fmt_p(value))
            elif isinstance(value, float):
                cells.append(f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        rows.append(cells)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def read_first_available_sheet(path: Path, sheets: list[str]) -> pd.DataFrame:
    available = set(pd.ExcelFile(path).sheet_names)
    for sheet in sheets:
        if sheet in available:
            return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"{path} does not contain any of these sheets: {sheets}")


def write_processed_workbook(out_path: Path) -> None:
    canonical_sheet = "canonical_stats_by_dimension"
    if CANONICAL_STATS.exists():
        canonical_sheets = pd.ExcelFile(CANONICAL_STATS).sheet_names
        if "fine_canonical_by_dimension" in canonical_sheets:
            canonical_sheet = "fine_canonical_by_dimension"
    sheets = {
        "README": pd.DataFrame(
            {
                "field": [
                    "primary_count",
                    "post_count_note",
                    "canonicalization_scope",
                    "review_scope",
                    "recommended_use",
                ],
                "value": [
                    "main_count",
                    "post_count is retained from aggregated input and should be interpreted cautiously after label merging.",
                    "Labels were merged only within the same CSM dimension.",
                    "Cluster-reviewed labels were canonicalized; unreviewed labels were retained as separate labels.",
                    "Use this workbook for team review, descriptive tables, and manuscript drafting.",
                ],
            }
        ),
        "dimension_distribution": read_sheet(ANALYSIS_PACK, "dimension_distribution"),
        "dimension_distribution_wide": read_sheet(ANALYSIS_PACK, "dimension_distribution_wide"),
        "top_overall_by_country": read_sheet(ANALYSIS_PACK, "top_overall_by_country"),
        "top_by_country_dimension": read_sheet(ANALYSIS_PACK, "top_by_country_dimension"),
        "cross_country_enrichment": read_sheet(ANALYSIS_PACK, "cross_country_enrichment"),
        "dimension_tests": read_sheet(PAPER_TABLES, "pairwise_dimension_tests"),
        "dimension_std_residuals": read_sheet(PAPER_TABLES, "dimension_std_residuals"),
        "canonical_stats_by_dimension": read_sheet(CANONICAL_STATS, canonical_sheet),
        "canonicalization_mapping": read_first_available_sheet(MAPPING, ["canonical_mapping", "canonicalization_mapping"]),
    }
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def top_list(top_df: pd.DataFrame, country: str, n: int = 5) -> str:
    rows = top_df[top_df["country"].eq(country)].head(n)
    return "; ".join(
        f"{row.keyword} ({int(row.main_count)}, {fmt_pct(float(row.country_percent))})"
        for row in rows.itertuples(index=False)
    )


def enrichment_list(enrich: pd.DataFrame, country: str, n: int = 5) -> str:
    sub = enrich[(enrich["country"].eq(country)) & (enrich["country_count"] >= 3)].head(n)
    lines = []
    for row in sub.itertuples(index=False):
        ratio = float(row.log2_rate_ratio_vs_others)
        ratio_text = "country-specific in this table" if math.isnan(ratio) else f"log2 rate ratio {ratio:.2f}"
        lines.append(
            f"- {row.keyword} [{row.dimension}]: {int(row.country_count)} vs "
            f"{int(row.other_countries_count)} other-country units; {ratio_text}"
        )
    return "\n".join(lines)


def build_markdown(out_path: Path) -> None:
    dist = read_sheet(ANALYSIS_PACK, "dimension_distribution")
    dist_wide = read_sheet(ANALYSIS_PACK, "dimension_distribution_wide")
    top = read_sheet(ANALYSIS_PACK, "top_overall_by_country")
    enrich = read_sheet(ANALYSIS_PACK, "cross_country_enrichment")
    pairwise = read_sheet(PAPER_TABLES, "pairwise_dimension_tests")
    residuals_long = read_sheet(PAPER_TABLES, "dimension_residuals_long")
    mapping = read_first_available_sheet(MAPPING, ["canonical_mapping", "canonicalization_mapping"])
    manifest = json.loads((PAPER_DIR / "keyword_paper_outputs_manifest.json").read_text(encoding="utf-8"))
    global_stats = manifest["global_dimension_chi_square"]

    changed = int((mapping["keyword"] != mapping["fine_canonical_keyword"]).sum())
    reviewed = int(mapping["consensus_source"].eq("cluster_review").sum())
    total_labels = int(len(mapping))

    lines = [
        "# Canonical Keyword Analysis: Team Results Brief",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Files To Review",
        "",
        "- Processed workbook: `outputs/keyword_report/merged_label_statistics.xlsx`",
        "- Dimension figure: `outputs/keyword_report/figure_dimension_distribution.png`",
        "- Keyword heatmap: `outputs/keyword_report/figure_keyword_heatmap.png`",
        "- The workbook contains the merged statistics, top keywords, enrichment results, tests, residuals, and canonicalization mapping.",
        "",
        "## What This Analysis Uses",
        "",
        "The analysis uses the reviewed canonical keyword layer. Original labels are preserved in the mapping sheet, but frequency summaries use `fine_canonical_keyword`. The primary count is `main_count`; `post_count` is retained but should be treated cautiously after label merging because source inputs were already aggregated before canonicalization.",
        "",
        "Canonicalization summary:",
        "",
        f"- Total dimension-keyword labels covered: {total_labels:,}",
        f"- Labels reviewed or confirmed through cluster review: {reviewed:,}",
        f"- Labels changed by canonicalization: {changed:,}",
        "- Unreviewed labels were retained as separate labels.",
        "",
        "## Main Findings",
        "",
        "### 1. CSM dimension profiles differ by country",
        "",
        f"The country by CSM dimension association was statistically significant: chi-square({global_stats['degrees_of_freedom']}) = {global_stats['chi_square']:.2f}, p = {fmt_p(global_stats['p_value'])}, Cramer's V = {global_stats['cramers_v']:.3f}. The effect is statistically clear but small overall.",
        "",
        markdown_table(dist_wide, float_digits=1),
        "",
        "Most notable standardized residuals:",
        "",
    ]
    for row in residuals_long.head(6).itertuples(index=False):
        lines.append(
            f"- {row.country} / {row.dimension}: {row.direction.replace('_', ' ')} "
            f"(standardized residual {float(row.std_residual):.2f})"
        )
    lines.extend(
        [
            "",
            "Pairwise dimension-profile tests:",
            "",
            markdown_table(pairwise, float_digits=3),
            "",
            "### 2. Top canonical keywords differ sharply across countries",
            "",
            f"- CHI top labels: {top_list(top, 'CHI')}",
            f"- JPN top labels: {top_list(top, 'JPN')}",
            f"- KOR top labels: {top_list(top, 'KOR')}",
            "",
            "Interpretation: JPN and KOR are dominated by generic `Dental pain` and `Severe dental pain`, while CHI remains unusual after canonicalization: `Dental pain` is low and `Emotional distress`, `Sleep disturbance`, and `Severe dental pain` lead the table.",
            "",
            "### 3. Country-enriched labels suggest interpretable differences",
            "",
            "CHI enriched labels:",
            "",
            enrichment_list(enrich, "CHI"),
            "",
            "JPN enriched labels:",
            "",
            enrichment_list(enrich, "JPN"),
            "",
            "KOR enriched labels:",
            "",
            enrichment_list(enrich, "KOR"),
            "",
            "## Recommended Presentation Figures",
            "",
            "Use the dimension distribution figure as the main result figure. Use the keyword heatmap as a secondary descriptive figure or supplementary figure; it is informative but visually dense.",
            "",
            "## Caveats To State",
            "",
            "- Canonicalization reduces wording fragmentation but does not make labels objective ground truth.",
            "- Canonicalization was restricted to within-dimension labels.",
            "- Reviewers accepted several separate clusters converging to the same canonical label; these are pooled in the final statistics.",
            "- China still shows low generic `Dental pain` after merging, so this likely reflects upstream annotation/normalization behavior rather than only long-tail wording variation.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def copy_figures() -> None:
    copies = {
        PAPER_DIR / "figure_dimension_distribution.png": OUT_DIR / "figure_dimension_distribution.png",
        PAPER_DIR / "figure_dimension_distribution.pdf": OUT_DIR / "figure_dimension_distribution.pdf",
        PAPER_DIR / "figure_keyword_heatmap.png": OUT_DIR / "figure_keyword_heatmap.png",
        PAPER_DIR / "figure_keyword_heatmap.pdf": OUT_DIR / "figure_keyword_heatmap.pdf",
        Path("data/keyword_cluster_candidates_blinded.csv"): OUT_DIR / "reviewed_cluster_decisions.csv",
        Path("docs/keyword_canonicalization_protocol.md"): OUT_DIR / "canonicalization_protocol.md",
    }
    for src, dst in copies.items():
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_processed_workbook(OUT_DIR / "merged_label_statistics.xlsx")
    copy_figures()
    build_markdown(OUT_DIR / "README_keyword_results.md")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(OUT_DIR),
        "outputs": [
            "README_keyword_results.md",
            "merged_label_statistics.xlsx",
            "reviewed_cluster_decisions.csv",
            "canonicalization_protocol.md",
            "figure_dimension_distribution.png",
            "figure_dimension_distribution.pdf",
            "figure_keyword_heatmap.png",
            "figure_keyword_heatmap.pdf",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
