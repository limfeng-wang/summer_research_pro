import json
from pathlib import Path
from collections import Counter
from typing import Any

FIRST_PASS_DIRS = [
    Path("outputs/main_pilot_10000_sharded_3gpu_merged"),
    Path("outputs/main_pilot_20000_sharded_3gpu_merged"),
    Path("outputs/main_pilot_36k_sharded_3gpu_merged"),
]
ACCEPTED_RESCUE_DIRS = [
    Path("outputs/csm_rescue_extraction_full_sharded_merged"),
    Path("outputs/csm_rescue_patch_extraction_sharded_merged"),
]
FORCED_CSM_DIRS = [
    Path("outputs/forced_csm_r1_e3_c3c5_sharded_merged"),
    Path("outputs/repair_jpn_2777_forced_csm_merged"),
]
EXTRACTION_CHECKPOINT_DIRS = FIRST_PASS_DIRS + ACCEPTED_RESCUE_DIRS + FORCED_CSM_DIRS
OUT_DIR = Path("outputs/final_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_EXPORT = OUT_DIR / "full_36k_with_labels_and_units.jsonl"
PRIMARY_EXPORT = OUT_DIR / "r1_e1_c1c2_with_accepted_units.jsonl"
SUMMARY_EXPORT = OUT_DIR / "export_summary.json"
REDUNDANT_EXPORTS = [
    OUT_DIR / "r1_anyE_anyC_with_accepted_units.jsonl",
    OUT_DIR / "r1_anyE_c1c2_with_accepted_units.jsonl",
    OUT_DIR / "r1_e1_anyC_with_accepted_units.jsonl",
    OUT_DIR / "r1_anyE_c1c2_export_summary.json",
]
COUNTRIES = ["CHI", "JPN", "KOR"]
ANY_E = {"E1", "E2", "E3"}
E1_ONLY = {"E1"}
ANY_C = {"C1", "C2", "C3", "C4", "C5"}
C1C2 = {"C1", "C2"}


def post_id(row: dict[str, Any]) -> str:
    return str(row.get("post_id") or row.get("record_id") or "").strip()


def unwrap(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    return result if isinstance(result, dict) else record


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_annotations(run_dir: Path):
    yield from (unwrap(record) for record in read_jsonl(run_dir / "annotations.jsonl"))


def read_extracted_checkpoints(run_dir: Path):
    yield from (unwrap(record) for record in read_jsonl(run_dir / "checkpoints" / "extracted.jsonl"))


def has_accepted_unit(row: dict[str, Any]) -> bool:
    return any(unit.get("judge_verdict") == "accept" for unit in row.get("units") or [])


def matches_filter(row: dict[str, Any], *, experiencers: set[str], content_functions: set[str]) -> bool:
    return (
        row.get("relevance_label") == "R1"
        and row.get("experiencer_label") in experiencers
        and row.get("content_function") in content_functions
        and has_accepted_unit(row)
    )


def rows_by_country(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("country") for row in rows)
    return {country: counts[country] for country in COUNTRIES if counts[country]}


def rows_by_ec(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter((row.get("experiencer_label"), row.get("content_function")) for row in rows)
    return {f"{e}|{c}": count for (e, c), count in sorted(counts.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])) if e and c}


def rows_by_country_ec(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter((row.get("country"), row.get("experiencer_label"), row.get("content_function")) for row in rows)
    return {f"{country}|{e}|{c}": count for (country, e, c), count in sorted(counts.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])) if country and e and c}


def accepted_units(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (row.get("country"), unit)
        for row in rows
        for unit in (row.get("units") or [])
        if unit.get("judge_verdict") == "accept"
    ]


def unit_verdict_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(unit.get("judge_verdict"))
        for row in rows
        for unit in (row.get("units") or [])
    )
    return dict(sorted(counts.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])))


def accepted_units_by_domain(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(unit.get("domain") for _, unit in accepted_units(rows))
    return dict(sorted(counts.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])))


def accepted_units_by_country(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(country for country, _ in accepted_units(rows))
    return {country: counts[country] for country in COUNTRIES if counts[country]}


def accepted_units_by_country_domain(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {country: Counter() for country in COUNTRIES}
    for country, unit in accepted_units(rows):
        counts[country][unit.get("domain")] += 1
    return {country: dict(sorted(domain_counts.items())) for country, domain_counts in counts.items() if domain_counts}


def label_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rows": len(rows),
        "rows_by_country": rows_by_country(rows),
        "rows_by_E_C": rows_by_ec(rows),
        "rows_by_country_E_C": rows_by_country_ec(rows),
    }


def accepted_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rows_with_at_least_one_accepted_csm_unit": len(rows),
        "rows_by_country": rows_by_country(rows),
        "rows_by_E_C": rows_by_ec(rows),
        "rows_by_country_E_C": rows_by_country_ec(rows),
        "accepted_unit_count": len(accepted_units(rows)),
        "accepted_units_by_country": accepted_units_by_country(rows),
        "accepted_units_by_domain": accepted_units_by_domain(rows),
        "accepted_units_by_country_domain": accepted_units_by_country_domain(rows),
        "all_unit_count_on_these_rows": sum(len(row.get("units") or []) for row in rows),
        "source_run_rows": dict(sorted(Counter(row.get("_source_run", "first_pass") for row in rows).items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


corpus: dict[str, dict[str, Any]] = {}
source_order: list[str] = []

for run_dir in FIRST_PASS_DIRS:
    for row in read_annotations(run_dir):
        pid = post_id(row)
        if pid and pid not in corpus:
            row["_source_pass"] = "first_pass"
            row["_source_run"] = str(run_dir)
            corpus[pid] = row
            source_order.append(pid)

# Rescue passes can update the final corpus only when they recovered at least
# one accepted CSM unit; otherwise they may represent failed/experimental labels.
for run_dir in ACCEPTED_RESCUE_DIRS:
    for row in read_annotations(run_dir):
        pid = post_id(row)
        if pid and pid in corpus and has_accepted_unit(row):
            row["_source_pass"] = "csm_rescue_pass"
            row["_source_run"] = str(run_dir)
            corpus[pid] = row

# Forced CSM runs were prepared from the finalized R1 label corpus, so all
# successful forced rows are valid overlays, including rejected-only rows.
for run_dir in FORCED_CSM_DIRS:
    for row in read_annotations(run_dir):
        pid = post_id(row)
        if pid and pid in corpus:
            row["_source_pass"] = "forced_csm_pass"
            row["_source_run"] = str(run_dir)
            corpus[pid] = row

# For forced R1 rows where extraction found zero units, annotations may be absent;
# keep the extraction checkpoint as proof that the CSM stage ran.
for run_dir in FORCED_CSM_DIRS:
    for row in read_extracted_checkpoints(run_dir):
        pid = post_id(row)
        if pid and pid in corpus and not (row.get("units") or []):
            row["_source_pass"] = "forced_csm_zero_units"
            row["_source_run"] = str(run_dir)
            corpus[pid] = row

all_rows = [corpus[pid] for pid in source_order if pid in corpus]
r1_rows = [row for row in all_rows if row.get("relevance_label") == "R1"]
r1_e1_rows = [row for row in r1_rows if row.get("experiencer_label") == "E1"]
r1_anyE_c1c2_label = [row for row in r1_rows if row.get("experiencer_label") in ANY_E and row.get("content_function") in C1C2]
r1_e1_c1c2_label = [row for row in r1_rows if row.get("experiencer_label") == "E1" and row.get("content_function") in C1C2]

accepted_sets = {
    "r1_anyE_anyC": [row for row in all_rows if matches_filter(row, experiencers=ANY_E, content_functions=ANY_C)],
    "r1_anyE_c1c2": [row for row in all_rows if matches_filter(row, experiencers=ANY_E, content_functions=C1C2)],
    "r1_e1_anyC": [row for row in all_rows if matches_filter(row, experiencers=E1_ONLY, content_functions=ANY_C)],
    "r1_e1_c1c2": [row for row in all_rows if matches_filter(row, experiencers=E1_ONLY, content_functions=C1C2)],
}

write_jsonl(FULL_EXPORT, all_rows)
write_jsonl(PRIMARY_EXPORT, accepted_sets["r1_e1_c1c2"])
for path in REDUNDANT_EXPORTS:
    if path.exists():
        path.unlink()

extracted_by_post_id: dict[str, dict[str, Any]] = {}
for run_dir in EXTRACTION_CHECKPOINT_DIRS:
    for row in read_extracted_checkpoints(run_dir):
        pid = post_id(row)
        if pid:
            extracted_by_post_id[pid] = row

r1_post_ids = {post_id(row) for row in r1_rows}
r1_extracted_post_ids = r1_post_ids & set(extracted_by_post_id)
r1_with_units = [row for row in r1_rows if row.get("units")]
r1_zero_unit_extractions = [
    extracted_by_post_id[pid]
    for pid in r1_extracted_post_ids
    if not (extracted_by_post_id[pid].get("units") or [])
]

summary = {
    "annotation_phase_finalized": True,
    "export_files": {
        "full_labeled_corpus": {
            "path": str(FULL_EXPORT),
            "description": "All 36,720 unique posts with final R/E/C labels; R1 rows include judged units where units were extracted and empty units where extraction returned none.",
            "rows": len(all_rows),
        },
        "primary_analysis_export": {
            "path": str(PRIMARY_EXPORT),
            "description": "R1 E1 C1/C2 rows with at least one judge-accepted CSM unit. This is the only filtered row-level JSONL export retained.",
            "rows": len(accepted_sets["r1_e1_c1c2"]),
        },
        "summary": {
            "path": str(SUMMARY_EXPORT),
            "description": "Single summary file containing all other combination counts without duplicating row-level exports.",
        },
    },
    "removed_redundant_files": [str(path) for path in REDUNDANT_EXPORTS],
    "combined_unique_posts": len(all_rows),
    "corpus_funnel_label_counts": {
        "grand_total_rows": label_bucket(all_rows),
        "r1": label_bucket(r1_rows),
        "r1_e1": label_bucket(r1_e1_rows),
        "r1_anyE_c1c2": label_bucket(r1_anyE_c1c2_label),
        "r1_e1_c1c2": label_bucket(r1_e1_c1c2_label),
    },
    "r1_csm_pipeline_coverage": {
        "r1_rows": len(r1_rows),
        "r1_rows_with_csm_extraction_output": len(r1_extracted_post_ids),
        "r1_rows_without_csm_extraction_output": len(r1_post_ids - r1_extracted_post_ids),
        "r1_rows_with_extracted_units": len(r1_with_units),
        "r1_rows_with_zero_extracted_units": len(r1_zero_unit_extractions),
        "r1_rows_with_at_least_one_accepted_csm_unit": len(accepted_sets["r1_anyE_anyC"]),
        "r1_unit_judge_verdict_counts": unit_verdict_counts(r1_rows),
    },
    "accepted_csm_combination_counts": {name: accepted_bucket(rows) for name, rows in accepted_sets.items()},
}
SUMMARY_EXPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
