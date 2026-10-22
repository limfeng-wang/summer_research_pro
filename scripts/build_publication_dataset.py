"""Build the self-contained publication dataset without changing source data."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dental_ai.classification_gold import canonical_post_id, load_classification_gold_jsonl
from dental_ai.local_models import _classification_fewshot_payload
from dental_ai.schemas import SourcePost

COUNTRIES = ["CHI", "JPN", "KOR"]
DOMAINS = {
    "Perceived Cause": "感知病因",
    "Symptom Description": "症状描述",
    "Perceived Consequences": "感知后果",
    "Coping and Management": "应对与管理",
    "Emotional Expression": "情绪表达",
}
PLATFORMS = {"CHI": "xiaohongshu", "JPN": "x", "KOR": "x"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row):
    return str(row["country"]), canonical_post_id(row.get("post_id") or row.get("record_id"))


def source_text(row):
    return str(row.get("text_clean") or "").strip() or "\n".join(
        str(row.get(field) or "").strip() for field in ["original_title", "original_text"]
        if str(row.get(field) or "").strip()
    )


def clean_label(value):
    text = str(value).strip()
    return None if text.upper() in {"", "NA", "NAN", "NONE", "NULL"} else text


def classification(row):
    return {field: clean_label(row.get(field, "")) for field in ["relevance_label", "experiencer_label", "content_function"]}


def accepted(unit):
    return unit.get("judge_verdict") == "accept" and unit.get("support_type") != "unsupported" and unit.get("assertion") != "negated"


def unit_record(unit, release_id, index, text):
    fields = ["domain", "evidence_span_original", "surface_text_original", "surface_text_working", "working_language",
              "normalized_concept_en", "concept_status", "support_type", "assertion", "temporality",
              "sentiment_or_outcome", "confidence", "judge_verdict"]
    record = {field: unit[field] for field in fields if field in unit}
    record["unit_id"] = f"{release_id}_U{index:03d}"
    record["quality_flags"] = {
        "judge_accepted": unit.get("judge_verdict") == "accept",
        "schema_accepted": accepted(unit),
        "evidence_span_found": bool(unit.get("evidence_span_original") and unit["evidence_span_original"] in text),
    }
    return record


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    destination = ROOT / "data/publication_dataset"
    private = ROOT / "outputs/publication_dataset_build"
    private.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "classification_development": ROOT / "data/classification_gold_172.jsonl",
        "rag_reference": ROOT / "data/csm_gold_50E1_10E2.json",
        "removal_roster": ROOT / "docs/ablation_reproducibility_artifacts/development_27_removal_candidates.csv",
        "evaluation_input": ROOT / "data/raw_eval_holdout_150_no_gold.jsonl",
        "evaluation_reference": ROOT / "data/raw_eval_holdout_150_gold.xlsx",
        "main_corpus": ROOT / "outputs/final_export/36k_labelled.jsonl",
    }
    examples = load_classification_gold_jsonl(source_paths["classification_development"])
    development = [example.to_json() for example in examples]
    rag = json.loads(source_paths["rag_reference"].read_text(encoding="utf-8"))
    with source_paths["removal_roster"].open(encoding="utf-8") as handle:
        removals = {(row["country"], row["canonical_id"]) for row in csv.DictReader(handle)}
    assert len(removals) == 27
    assert Counter(country for country, identifier in removals) == Counter({"CHI": 12, "JPN": 10, "KOR": 5})
    assert removals <= {key(row) for row in development}
    assert not removals & {key(row) for row in rag}
    retained_examples = [example for example in examples if (str(example.country), example.canonical_id) not in removals]
    queries = [example.as_source_post() for example in examples]
    queries.extend(SourcePost(post_id=f"{country}_publication_audit", country=country, language=language, original_text="audit")
                   for country, language in [("CHI", "zh"), ("JPN", "ja"), ("KOR", "ko")])
    for query in queries:
        assert _classification_fewshot_payload(query, examples, k=8) == _classification_fewshot_payload(query, retained_examples, k=8)
    classification_map = {key(row): (index, row) for index, row in enumerate(development, 1) if key(row) not in removals}
    rag_map = {key(row): (index, row) for index, row in enumerate(rag, 1)}
    evaluation = read_jsonl(source_paths["evaluation_input"])
    main_rows = read_jsonl(source_paths["main_corpus"])
    development_keys = set(classification_map) | set(rag_map)
    assert Counter(country for country, identifier in development_keys) == Counter({country: 50 for country in COUNTRIES})
    original_classification_keys = {key(row) for row in development}
    assert not original_classification_keys & {key(row) for row in evaluation + main_rows}
    sets = {"development": development_keys, "evaluation": {key(row) for row in evaluation}, "main": {key(row) for row in main_rows}}
    assert all(not sets[left] & sets[right] for left, right in [("development", "evaluation"), ("development", "main"), ("evaluation", "main")])
    counters = Counter()
    mapping = []

    def base_record(row, split):
        country = str(row["country"])
        counters[(split, country)] += 1
        release_id = f"{split.upper()}_{country}_{counters[(split, country)]:05d}"
        mapping.append({"split": split, "country": country, "source_canonical_id": key(row)[1], "release_post_id": release_id})
        return {"post_id": release_id, "country": country, "language": row["language"], "platform": PLATFORMS[country],
                "title": str(row.get("original_title") or ""), "text": str(row.get("original_text") or ""),
                "source_text": source_text(row)}

    development_out = []
    for record_key in sorted(development_keys):
        classification_entry = classification_map.get(record_key)
        extraction_entry = rag_map.get(record_key)
        row = classification_entry[1] if classification_entry else extraction_entry[1]
        record = base_record(row, "development")
        record["roles"] = []
        record["classification_reference"] = None
        record["extraction_reference"] = None
        if classification_entry:
            order, reference = classification_entry
            record["roles"].append("classification")
            record["classification_reference"] = {
                "example_order": order, "experiencer_label": reference["experiencer_label"],
                "content_function": reference["content_function"], "evidence_excerpt": reference["evidence_excerpt"],
                "rationale": reference["rationale"],
            }
        if extraction_entry:
            order, reference = extraction_entry
            assert source_text(reference) == record["source_text"]
            record["roles"].append("extraction")
            record["extraction_reference"] = {
                "example_order": order, "classification": classification(reference),
                "units": [unit_record(unit, record["post_id"], index, record["source_text"]) for index, unit in enumerate(reference["units"], 1)],
            }
        development_out.append(record)

    gold = pd.read_excel(source_paths["evaluation_reference"], sheet_name="1").fillna("")
    gold_map = {str(row["record_id"]): row for row in gold.to_dict(orient="records")}
    assert set(gold_map) == {row["record_id"] for row in evaluation}
    evaluation_out = []
    annotations_out = []
    for row in sorted(evaluation, key=key):
        record = base_record(row, "evaluation")
        reference = gold_map[row["record_id"]]
        assert str(reference["国家"]) == record["country"] and str(reference["语言"]) == record["language"]
        reference_source_text = source_text({"original_title": reference["原文标题"], "original_text": reference["原文正文"]})
        evaluation_out.append(record)
        annotations_out.append({
            "post_id": record["post_id"],
            "reference_source_text": reference_source_text,
            "reference_text_matches_input": reference_source_text == record["source_text"],
            "classification": {"relevance_label": clean_label(reference["牙痛相关性"]), "experiencer_label": clean_label(reference["经历主体"]),
                               "content_function": clean_label(reference["内容功能"])},
            "domains": {domain: {"present": bool(clean_label(reference[f"{prefix}_标签"])),
                                  "original_label": str(reference[f"{prefix}_标签"]),
                                  "evidence_text": str(reference[f"{prefix}_证据"]),
                                  "concepts_en_text": str(reference[f"{prefix}_关键词_EN"])} for domain, prefix in DOMAINS.items()},
        })

    main_out = []
    for row in sorted(main_rows, key=key):
        record = base_record(row, "main")
        record["classification"] = classification(row)
        record["units"] = [unit_record(unit, record["post_id"], index, record["source_text"]) for index, unit in enumerate(row.get("units", []), 1)]
        eligible = record["classification"]["relevance_label"] == "R1" and record["classification"]["experiencer_label"] == "E1" and record["classification"]["content_function"] in {"C1", "C2"}
        record["analysis_flags"] = {
            "eligible_first_person": eligible,
            "manuscript_primary": eligible and any(unit["quality_flags"]["judge_accepted"] for unit in record["units"]),
            "strict_primary": eligible and any(unit["quality_flags"]["schema_accepted"] and unit["quality_flags"]["evidence_span_found"] for unit in record["units"]),
        }
        main_out.append(record)
    outputs = {"development/posts.jsonl": development_out, "evaluation/posts.jsonl": evaluation_out,
               "evaluation/annotations.jsonl": annotations_out, "main/posts.jsonl": main_out}
    for filename, rows in outputs.items():
        write_jsonl(destination / filename, rows)
    public_files = {}
    for filename, rows in outputs.items():
        public_files[filename] = {"rows": len(rows), "sha256": hashlib.sha256((destination / filename).read_bytes()).hexdigest()}
        if filename != "evaluation/annotations.jsonl":
            public_files[filename]["countries"] = dict(sorted(Counter(row["country"] for row in rows).items()))
    manifest = {
        "dataset_version": "1.0.0", "format": "UTF-8 JSONL", "files": public_files,
        "development": {"unique_posts": 150, "classification": 145, "extraction": 60, "both": 55,
                        "curation": "Retrospective balanced subset of 177 development posts; 27 classification-only posts removed. Current ordered classifier selections and all 60 RAG examples preserved."},
        "evaluation": {"posts": 150, "reference_text_variants": sum(not row["reference_text_matches_input"] for row in annotations_out)},
        "main": {"posts": len(main_out), "manuscript_primary_posts": sum(row["analysis_flags"]["manuscript_primary"] for row in main_out),
                 "strict_primary_posts": sum(row["analysis_flags"]["strict_primary"] for row in main_out),
                 "manuscript_primary_verdict_accepted_units": sum(unit["quality_flags"]["judge_accepted"] for row in main_out if row["analysis_flags"]["manuscript_primary"] for unit in row["units"])},
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()},
        "identifiers": "Split-specific release IDs replace original record/unit IDs. Original source text and annotations retained; internal file paths, spreadsheet locations, and run names omitted.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (private / "private_id_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    (private / "build_sources.json").write_text(json.dumps({name: str(path.relative_to(ROOT)) for name, path in source_paths.items()}, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
