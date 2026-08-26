import json
from pathlib import Path
from collections import Counter

FIRST_PASS_DIRS = [
    Path("outputs/main_pilot_10000_sharded_3gpu_merged"),
    Path("outputs/main_pilot_20000_sharded_3gpu_merged"),
    Path("outputs/main_pilot_36k_sharded_3gpu_merged"),
]
RESCUE_DIRS = [
    Path("outputs/csm_rescue_extraction_full_sharded_merged"),
    Path("outputs/csm_rescue_patch_extraction_sharded_merged"),
]
OUT_DIR = Path("outputs/final_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def post_id(row):
    return str(row.get("post_id") or row.get("record_id") or "").strip()

def read_annotations(run_dir):
    path = run_dir / "annotations.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def has_accepted_unit(row):
    return any(u.get("judge_verdict") == "accept" for u in row.get("units") or [])

def matches_filter(row, *, experiencers, content_functions):
    return (
        row.get("relevance_label") == "R1"
        and row.get("experiencer_label") in experiencers
        and row.get("content_function") in content_functions
        and has_accepted_unit(row)
    )

def rows_by_country(rows):
    counts = Counter(row.get("country") for row in rows)
    return {country: counts[country] for country in ["CHI", "JPN", "KOR"] if counts[country]}

def accepted_units(rows):
    return [
        (row.get("country"), unit)
        for row in rows
        for unit in (row.get("units") or [])
        if unit.get("judge_verdict") == "accept"
    ]

def accepted_units_by_domain(rows):
    counts = Counter(unit.get("domain") for _, unit in accepted_units(rows))
    return dict(sorted(counts.items()))

def accepted_units_by_country(rows):
    counts = Counter(country for country, _ in accepted_units(rows))
    return {country: counts[country] for country in ["CHI", "JPN", "KOR"] if counts[country]}

def accepted_units_by_country_domain(rows):
    counts = {country: Counter() for country in ["CHI", "JPN", "KOR"]}
    for country, unit in accepted_units(rows):
        counts[country][unit.get("domain")] += 1
    return {country: dict(sorted(domain_counts.items())) for country, domain_counts in counts.items() if domain_counts}

def label_bucket(rows):
    return {
        "total_rows": len(rows),
        "rows_by_country": rows_by_country(rows),
    }

def bucket(rows):
    return {
        "total_rows": len(rows),
        "rows_by_country": rows_by_country(rows),
        "accepted_unit_count": len(accepted_units(rows)),
        "accepted_units_by_country": accepted_units_by_country(rows),
        "accepted_units_by_domain": accepted_units_by_domain(rows),
        "accepted_units_by_country_domain": accepted_units_by_country_domain(rows),
        "all_unit_count": sum(len(row.get("units") or []) for row in rows),
        "source_run_rows": dict(sorted(Counter(row.get("_source_run", "first_pass") for row in rows).items())),
    }

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

corpus = {}
source_pass = {}

for run_dir in FIRST_PASS_DIRS:
    for row in read_annotations(run_dir):
        pid = post_id(row)
        if pid and pid not in corpus:
            row["_source_pass"] = "first_pass"
            corpus[pid] = row
            source_pass[pid] = "first_pass"

# Rescue rows override first-pass rows only when they contain accepted CSM units.
for run_dir in RESCUE_DIRS:
    for row in read_annotations(run_dir):
        pid = post_id(row)
        if pid and has_accepted_unit(row):
            row["_source_pass"] = "rescue_pass"
            row["_source_run"] = str(run_dir)
            corpus[pid] = row
            source_pass[pid] = "rescue_pass"

ANY_E = {"E1", "E2", "E3"}
E1_ONLY = {"E1"}
ANY_C = {"C1", "C2", "C3", "C4", "C5"}
C1C2 = {"C1", "C2"}

all_rows = list(corpus.values())
r1_rows = [row for row in all_rows if row.get("relevance_label") == "R1"]
r1_e1_rows = [row for row in r1_rows if row.get("experiencer_label") == "E1"]
r1_anyE_c1c2_label = [
    row
    for row in r1_rows
    if row.get("experiencer_label") in ANY_E and row.get("content_function") in C1C2
]
r1_e1_c1c2_label = [
    row
    for row in r1_rows
    if row.get("experiencer_label") == "E1" and row.get("content_function") in C1C2
]

r1_anyE_anyC = [
    row for row in corpus.values() if matches_filter(row, experiencers=ANY_E, content_functions=ANY_C)
]
r1_anyE_c1c2 = [
    row for row in corpus.values() if matches_filter(row, experiencers=ANY_E, content_functions=C1C2)
]
r1_e1_anyC = [
    row for row in corpus.values() if matches_filter(row, experiencers=E1_ONLY, content_functions=ANY_C)
]
r1_e1_c1c2 = [
    row for row in corpus.values() if matches_filter(row, experiencers=E1_ONLY, content_functions=C1C2)
]

write_jsonl(OUT_DIR / "r1_anyE_anyC_with_accepted_units.jsonl", r1_anyE_anyC)
write_jsonl(OUT_DIR / "r1_anyE_c1c2_with_accepted_units.jsonl", r1_anyE_c1c2)
write_jsonl(OUT_DIR / "r1_e1_anyC_with_accepted_units.jsonl", r1_e1_anyC)
write_jsonl(OUT_DIR / "r1_e1_c1c2_with_accepted_units.jsonl", r1_e1_c1c2)

summary = {
    "annotation_phase_finalized": True,
    "corpus_funnel_label_counts": {
        "grand_total_rows": label_bucket(all_rows),
        "r1": label_bucket(r1_rows),
        "r1_e1": label_bucket(r1_e1_rows),
        "r1_anyE_c1c2": label_bucket(r1_anyE_c1c2_label),
        "r1_e1_c1c2": label_bucket(r1_e1_c1c2_label),
    },
    "accepted_csm_exports": {
        "r1_anyE_anyC": bucket(r1_anyE_anyC),
        "r1_anyE_c1c2": bucket(r1_anyE_c1c2),
        "r1_e1_anyC": bucket(r1_e1_anyC),
        "r1_e1_c1c2": bucket(r1_e1_c1c2),
    },
    "combined_unique_posts": len(corpus),
}
(OUT_DIR / "r1_anyE_c1c2_export_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
