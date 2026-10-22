"""Validate the dataset package using only the Python standard library."""

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def accepted(unit):
    return unit.get("judge_verdict") == "accept" and unit.get("support_type") != "unsupported" and unit.get("assertion") != "negated"


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    groups = {}
    for name, expected in manifest["files"].items():
        path = root / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], f"Hash mismatch: {name}")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        require(len(rows) == expected["rows"], f"Row count mismatch: {name}")
        require(len({row["post_id"] for row in rows}) == len(rows), f"Duplicate post IDs: {name}")
        if "countries" in expected:
            require(dict(Counter(row["country"] for row in rows)) == expected["countries"], f"Country counts: {name}")
        groups[name] = rows
    expected_sizes = {"development/posts.jsonl": 150, "evaluation/posts.jsonl": 150,
                      "evaluation/annotations.jsonl": 150, "main/posts.jsonl": 36720}
    require({name: len(rows) for name, rows in groups.items()} == expected_sizes, "Unexpected package composition")
    development = groups["development/posts.jsonl"]
    role_counts = Counter(role for row in development for role in row["roles"])
    require(role_counts == Counter({"classification": 145, "extraction": 60}), "Development role counts")
    require(sum(len(row["roles"]) == 2 for row in development) == 55, "Development overlap")
    require(Counter(row["country"] for row in development) == Counter({"CHI": 50, "JPN": 50, "KOR": 50}), "Unbalanced development set")
    require(Counter(row["country"] for row in groups["evaluation/posts.jsonl"]) == Counter({"CHI": 50, "JPN": 50, "KOR": 50}), "Unbalanced evaluation set")
    require(Counter(row["country"] for row in groups["main/posts.jsonl"]) == Counter({"CHI": 3521, "JPN": 25167, "KOR": 8032}), "Main language counts")
    for row in development:
        require(set(row["roles"]) <= {"classification", "extraction"} and row["roles"], "Invalid role")
        for role in ["classification", "extraction"]:
            require((role in row["roles"]) == (row[f"{role}_reference"] is not None), "Role/reference mismatch")
    for role in ["classification", "extraction"]:
        orders = [row[f"{role}_reference"]["example_order"] for row in development if role in row["roles"]]
        require(len(set(orders)) == len(orders), f"Duplicate {role} example ordering")
    evaluation_ids = {row["post_id"] for row in groups["evaluation/posts.jsonl"]}
    require(evaluation_ids == {row["post_id"] for row in groups["evaluation/annotations.jsonl"]}, "Evaluation join mismatch")
    evaluation_texts = {row["post_id"]: row["source_text"] for row in groups["evaluation/posts.jsonl"]}
    for row in groups["evaluation/annotations.jsonl"]:
        require(row["reference_text_matches_input"] == (row["reference_source_text"] == evaluation_texts[row["post_id"]]), "Reference text flag mismatch")
    require(sum(not row["reference_text_matches_input"] for row in groups["evaluation/annotations.jsonl"]) == manifest["evaluation"]["reference_text_variants"], "Reference text variant count")
    for row in groups["evaluation/posts.jsonl"]:
        require(not set(row) & {"classification", "domains", "units", "annotations"}, "Labels in evaluation input")
    post_groups = {name: rows for name, rows in groups.items() if name.endswith("posts.jsonl")}
    for left, right in itertools.combinations(post_groups, 2):
        for field in ["post_id", "source_text"]:
            require(not {row[field] for row in post_groups[left]} & {row[field] for row in post_groups[right]}, f"Overlapping {field}: {left}, {right}")
    unit_ids = set()
    for name, rows in post_groups.items():
        for row in rows:
            require(set(row) <= {"post_id", "country", "language", "platform", "title", "text", "source_text", "roles",
                                 "classification_reference", "extraction_reference", "classification", "units", "analysis_flags"}, "Unexpected/internal post metadata")
            require(row["source_text"].strip(), "Empty source text")
            units = row.get("units", []) if name == "main/posts.jsonl" else (row.get("extraction_reference") or {}).get("units", [])
            for unit in units:
                require(unit["unit_id"].startswith(row["post_id"] + "_U"), "Unexpected unit ID")
                require(unit["unit_id"] not in unit_ids, "Duplicate unit ID")
                unit_ids.add(unit["unit_id"])
                expected_flags = {"judge_accepted": unit.get("judge_verdict") == "accept", "schema_accepted": accepted(unit),
                                  "evidence_span_found": bool(unit.get("evidence_span_original") and unit["evidence_span_original"] in row["source_text"])}
                require(unit["quality_flags"] == expected_flags, "Incorrect quality flags")
            if name == "main/posts.jsonl":
                labels = row["classification"]
                eligible = labels["relevance_label"] == "R1" and labels["experiencer_label"] == "E1" and labels["content_function"] in {"C1", "C2"}
                flags = {"eligible_first_person": eligible,
                         "manuscript_primary": eligible and any(unit["quality_flags"]["judge_accepted"] for unit in units),
                         "strict_primary": eligible and any(unit["quality_flags"]["schema_accepted"] and unit["quality_flags"]["evidence_span_found"] for unit in units)}
                require(row["analysis_flags"] == flags, "Incorrect analysis flags")
    main_rows = groups["main/posts.jsonl"]
    require(sum(row["analysis_flags"]["manuscript_primary"] for row in main_rows) == 12829, "Manuscript sample count")
    require(sum(row["analysis_flags"]["strict_primary"] for row in main_rows) == 12817, "Strict sample count")
    require(sum(unit["quality_flags"]["judge_accepted"] for row in main_rows if row["analysis_flags"]["manuscript_primary"] for unit in row["units"]) == 35627, "Manuscript unit count")
    print("PASS: 150 development, 150 evaluation, 36,720 main posts; roles, separation, hashes, and annotation flags verified.")


if __name__ == "__main__":
    main()
