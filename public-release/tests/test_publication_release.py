import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dental_ai.data_io import read_posts_jsonl
from dental_ai.publication import read_jsonl
from dental_ai.schemas import ExtractionResult, SourcePost

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/reference"
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_publication_holdout as evaluation
import merge_runs


def test_publication_inference_discards_existing_annotations(tmp_path):
    original = {"post_id": "synthetic_input", "country": "CHI", "language": "zh",
                "platform": "synthetic", "title": "示例", "text": "牙疼", "source_text": "示例\n牙疼"}
    contaminated = {
        **original,
        "classification": {"relevance_label": "R0"},
        "units": [{"normalized_concept_en": "never expose this annotation"}],
        "domains": {"secret": True},
    }
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(contaminated), encoding="utf-8")
    post = read_posts_jsonl(path)[0]
    assert post.combined_source_text == original["source_text"]
    assert post.original_title == original["title"]
    assert not {"classification", "units", "domains"} & set(post.model_dump())


def test_archived_gate_confusion_counts_from_public_aggregates():
    import csv
    with (REFERENCE / "aggregate_tables/eligibility_gate_metrics.csv").open() as handle:
        gates = list(csv.DictReader(handle))
    counts = [tuple(int(row[field]) for field in ["tp", "fp", "fn", "tn"])
              for row in gates if row["group"] == "overall"]
    assert counts == [(33, 43, 19, 55), (31, 27, 21, 71), (26, 6, 26, 92), (26, 6, 26, 92)]


def test_public_metadata_and_archived_totals_are_consistent():
    metadata = json.loads((ROOT / "data/publication_dataset/metadata.json").read_text())
    corpus = json.loads((REFERENCE / "corpus_summary.json").read_text())
    development = metadata["development"]
    assert development["classification"] + development["extraction"] - development["both"] == development["unique_posts"]
    assert sum(metadata["main"]["posts_by_country"].values()) == corpus["posts"] == 36720
    assert sum(corpus["primary_countries"].values()) == corpus["primary_posts"] == 12829
    assert metadata["main"]["manuscript_primary_verdict_accepted_units"] == corpus["primary_verdict_accepted_units"] == 35627
    evaluation_result = json.loads((REFERENCE / "expected_evaluation.json").read_text())
    assert evaluation_result["matched_rows"] == metadata["evaluation"]["posts"] == 150


def test_evaluation_rejects_incomplete_duplicate_and_wrong_order_inputs(tmp_path):
    with pytest.raises(ValueError, match="IDs differ"):
        evaluation.require_matching_ids({"a": {}}, {})
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"post_id":"duplicate"}\n{"post_id":"duplicate"}\n')
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluation.load_predictions(path)
    path.write_text('["a", "a"]')
    with pytest.raises(ValueError, match="every evaluation ID"):
        evaluation.evaluation_order({"a", "b"}, path)


def test_rescue_merge_preserves_rejected_and_zero_unit_forced_outputs(tmp_path):
    def write(run, records, relative="annotations.jsonl"):
        path = tmp_path / run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return tmp_path / run

    base = [{"post_id": post_id, "units": [], "version": "first"} for post_id in ["a", "b", "c"]]
    first = write("first", base)
    rescue = write("rescue", [
        {"post_id": "a", "units": [{"judge_verdict": "reject"}], "version": "ignored"},
        {"post_id": "b", "units": [{"judge_verdict": "accept"}], "version": "rescued"},
    ])
    forced = write("forced", [{"post_id": "c", "units": [{"judge_verdict": "reject"}], "version": "forced"}])
    rows, _ = merge_runs.merge_runs([first], [rescue], [forced])
    assert [row["version"] for row in rows] == ["first", "rescued", "forced"]
    write("forced", [{"result": {"post_id": "c", "units": [], "version": "zero"}}], "checkpoints/extracted.jsonl")
    rows, _ = merge_runs.merge_runs([first], [rescue], [forced])
    assert rows[-1]["version"] == "zero"


def test_full_runner_writes_successful_zero_unit_extraction_without_resuming(tmp_path, monkeypatch):
    import dental_ai.local_models as models
    import dental_ai.rag as rag
    from dental_ai.run import main

    class EmptyModel:
        def generate_json_text(self, prompt, payload, config):
            if config.max_new_tokens == 128:
                return '{"relevance_label":"R1","experiencer_label":"E1","content_function":"C1"}'
            return json.dumps({"post_id": payload["post"]["post_id"], "country": "CHI", "language": "zh", "units": []})

        def close(self):
            pass

    monkeypatch.setattr(models, "local_lm_for_role", lambda *args, **kwargs: EmptyModel())
    monkeypatch.setattr(rag.GoldRAGRetriever, "from_config", lambda *args, **kwargs: SimpleNamespace(retrieve_with_scores=lambda *args, **kwargs: []))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_stack": {}, "runtime": {}, "paths": {}}))
    post = SourcePost(post_id="empty", country="CHI", language="zh", text_clean="我牙疼，睡不着。")
    path = tmp_path / "input.jsonl"
    path.write_text(post.model_dump_json() + "\n", encoding="utf-8")
    out_dir = tmp_path / "output"
    assert main(["--input", str(path), "--out-dir", str(out_dir), "--backend", "hf", "--hf-stage", "full", "--config", str(config)]) == 0
    result = ExtractionResult.model_validate(read_jsonl(out_dir / "annotations.jsonl")[0])
    assert result.post_id == "empty" and result.units == []
    assert (out_dir / "errors.jsonl").read_text() == ""
