"""Identify a balanced development subset without changing active resources."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dental_ai.classification_gold import canonical_post_id, load_classification_gold_jsonl
from dental_ai.local_models import _classification_example_payload, _classification_fewshot_payload
from dental_ai.schemas import SourcePost


def record_key(country, post_id):
    return str(country), canonical_post_id(post_id)


def example_key(example):
    return record_key(example.country, example.post_id)


def selected_keys(post, examples):
    payloads = _classification_fewshot_payload(post, examples, k=8)
    return {
        example_key(example)
        for example in examples
        if _classification_example_payload(example) in payloads
    }


def main():
    root = REPO_ROOT
    out = root / "docs/ablation_reproducibility_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    classification_path = root / "data/classification_gold_172.jsonl"
    rag_path = root / "data/csm_gold_50E1_10E2.json"
    examples = load_classification_gold_jsonl(classification_path)
    rag = json.loads(rag_path.read_text(encoding="utf-8"))
    rag_keys = {record_key(row["country"], row["post_id"]) for row in rag}
    original_keys = {example_key(example) for example in examples}
    query_posts = [example.as_source_post() for example in examples]
    external_queries = [
        SourcePost(post_id=f"{country}_balance_audit", country=country, language=language, original_text="audit")
        for country, language in [("CHI", "zh"), ("JPN", "ja"), ("KOR", "ko")]
    ]
    query_posts.extend(external_queries)
    protected = set().union(*(selected_keys(post, examples) for post in query_posts))
    main_selected = {str(post.country): selected_keys(post, examples) for post in external_queries}
    removed = set()
    candidate_counts = {}
    for country, required in [("CHI", 12), ("JPN", 10), ("KOR", 5)]:
        candidates = [example_key(example) for example in examples
                      if str(example.country) == country and example_key(example) not in protected | rag_keys]
        candidate_counts[country] = len(candidates)
        assert len(candidates) >= required
        removed.update(candidates[-required:])
    retained = [example for example in examples if example_key(example) not in removed]
    assert len(removed) == 27 and not removed & rag_keys
    for post in query_posts:
        assert _classification_fewshot_payload(post, examples, k=8) == _classification_fewshot_payload(post, retained, k=8)
    input_counts = {}
    input_paths = [root / "data/raw_main_llm_input_no_gold.jsonl", root / "data/raw_eval_holdout_150_no_gold.jsonl"]
    for path in input_paths:
        counts = Counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                assert record_key(row["country"], row.get("record_id") or row.get("post_id")) not in original_keys
                counts[row["country"]] += 1
        input_counts[str(path.relative_to(root))] = dict(counts)
    retained_union = {example_key(example) for example in retained} | rag_keys
    assert Counter(country for country, post_id in retained_union) == Counter({"CHI": 50, "JPN": 50, "KOR": 50})
    rows = []
    for country, post_id in sorted(original_keys | rag_keys):
        record = next((example for example in examples if example_key(example) == (country, post_id)), None)
        rag_record = next((row for row in rag if record_key(row["country"], row["post_id"]) == (country, post_id)), None)
        rows.append({
            "country": country,
            "canonical_id": post_id,
            "classification_post_id": record.post_id if record else "",
            "rag_post_id": rag_record["post_id"] if rag_record else "",
            "selected_for_main_eval_classifier": int((country, post_id) in main_selected[country]),
            "protected_for_any_development_target": int((country, post_id) in protected),
            "in_rag_reference": int((country, post_id) in rag_keys),
            "proposal": "remove_from_development_pool" if (country, post_id) in removed else "retain",
        })
    for filename, selected_rows in [("development_balance_roster.csv", rows),
                                    ("development_27_removal_candidates.csv", [row for row in rows if row["proposal"] == "remove_from_development_pool"])]:
        with (out / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)
    paths = [classification_path, rag_path, *input_paths, root / "dental_ai/local_models.py", Path(__file__).resolve()]
    manifest = {
        "scope": "Proposal only. Active example files, configuration, and study outputs are unchanged.",
        "evidence_limit": "Classifier usage reconstructed from current deterministic selector; historical rendered classifier prompts are not archived. All RAG examples are preserved, irrespective of historical retrieval frequency.",
        "unused_candidate_counts_preserving_all_rag_and_classifier_selections": candidate_counts,
        "removed_counts": dict(Counter(country for country, post_id in removed)),
        "retained_classification_counts": dict(Counter(str(example.country) for example in retained)),
        "retained_rag_counts": dict(Counter(row["country"] for row in rag)),
        "retained_development_union_counts": dict(Counter(country for country, post_id in retained_union)),
        "main_eval_classifier_ids": {country: sorted(post_id for item_country, post_id in values) for country, values in main_selected.items()},
        "verified_input_counts": input_counts,
        "selection_invariance": "Identical ordered eight-example payloads for all original classification targets and one nondevelopment target per language. All actual main/evaluation IDs are outside development, so their selections match the respective nondevelopment target.",
        "rag_invariance": "All 60 RAG records remain; retrieval pool, ordering, and content are unchanged.",
        "candidate_selection_rule": "Last required eligible records in original classifier-file order, after protecting all RAG records and all classifier selections for development/nondevelopment targets. No outcome metrics used.",
        "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (out / "development_balance_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
