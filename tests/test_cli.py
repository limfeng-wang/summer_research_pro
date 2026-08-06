import json
import subprocess
import sys

import pandas as pd


def make_record():
    return {
        "post_id": "p1",
        "country": "CHI",
        "language": "zh",
        "original_title": "牙痛好用",
        "original_text": "牙痛好用\n吃了布洛芬之后不疼了",
        "relevance_label": "R1",
        "experiencer_label": "E1",
        "content_function": "C1",
        "units": [
            {
                "unit_id": "u1",
                "domain": "Coping and Management",
                "evidence_span_original": "吃了布洛芬之后不疼了",
                "surface_text_working": "吃了布洛芬",
                "working_language": "zh",
                "normalized_concept_en": "Ibuprofen use",
                "concept_status": "new_candidate",
                "support_type": "explicit",
                "assertion": "present",
                "temporality": "past",
                "sentiment_or_outcome": "effective",
                "confidence": 0.99,
                "judge_verdict": "accept",
            }
        ],
    }


def test_summarize_gold_json_output(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([make_record()], ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "dental_ai.cli", "summarize-gold", "--json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(proc.stdout)
    assert summary["posts"] == 1
    assert summary["primary_posts"] == 1
    assert summary["span_exact_matches"] == 1


def test_export_gold_units_writes_expected_files(tmp_path):
    path = tmp_path / "gold.json"
    out_dir = tmp_path / "exports"
    path.write_text(json.dumps([make_record()], ensure_ascii=False), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "dental_ai.cli",
            "export-gold-units",
            str(path),
            "--out-dir",
            str(out_dir),
            "--jsonl",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "all_posts.jsonl").exists()
    rows = pd.read_csv(out_dir / "primary_units.csv")
    assert rows["normalized_concept_en"].tolist() == ["Ibuprofen use"]


def test_run_hierarchical_mock_writes_outputs(tmp_path):
    path = tmp_path / "posts.jsonl"
    out_dir = tmp_path / "run"
    path.write_text(
        (
            '{"record_id":"p1","country":"CHI","language":"zh",'
            '"original_text":"牙疼得睡不着","text_clean":"牙疼得睡不着"}\n'
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "dental_ai.cli",
            "run-hierarchical",
            "--input",
            str(path),
            "--out-dir",
            str(out_dir),
            "--backend",
            "mock",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_dir / "annotations.jsonl").exists()
    assert (out_dir / "errors.jsonl").exists()
    assert (out_dir / "run_manifest.json").exists()
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["rows_attempted"] == 1
    assert manifest["rows_succeeded"] == 1
    assert manifest["rows_failed"] == 0
