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

def bucket(rows):
    return {
        "total_rows": len(rows),
        "rows_by_country": rows_by_country(rows),
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
    "combined_unique_posts": len(corpus),
    "r1_anyE_anyC": bucket(r1_anyE_anyC),
    "r1_anyE_c1c2": bucket(r1_anyE_c1c2),
    "r1_e1_anyC": bucket(r1_e1_anyC),
    "r1_e1_c1c2": bucket(r1_e1_c1c2),
}
(OUT_DIR / "r1_anyE_c1c2_export_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
