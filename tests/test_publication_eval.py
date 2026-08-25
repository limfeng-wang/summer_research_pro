import json

import pandas as pd

from scripts.evaluate_publication_holdout import build_evaluation, load_gold, load_predictions


def test_publication_eval_scores_classification_domains_units_and_normalization(tmp_path):
    gold_path = tmp_path / "gold.xlsx"
    pred_path = tmp_path / "annotations.jsonl"
    with pd.ExcelWriter(gold_path) as writer:
        pd.DataFrame(
            [
                {
                    "record_id": "p1",
                    "国家": "CHI",
                    "语言": "zh",
                    "原文标题": "",
                    "原文正文": "牙疼得睡不着，吃了布洛芬。",
                    "牙痛相关性": "R1",
                    "经历主体": "E1",
                    "内容功能": "C1",
                    "感知病因_标签": "",
                    "感知病因_证据": "",
                    "感知病因_关键词_EN": "",
                    "症状描述_标签": 1,
                    "症状描述_证据": "牙疼",
                    "症状描述_关键词_EN": "toothache",
                    "感知后果_标签": 1,
                    "感知后果_证据": "睡不着",
                    "感知后果_关键词_EN": "sleep disturbance",
                    "应对与管理_标签": 1,
                    "应对与管理_证据": "布洛芬",
                    "应对与管理_关键词_EN": "oral analgesic",
                    "情绪表达_标签": "",
                    "情绪表达_证据": "",
                    "情绪表达_关键词_EN": "",
                },
                {
                    "record_id": "p2",
                    "国家": "JPN",
                    "语言": "ja",
                    "原文标题": "",
                    "原文正文": "口腔ケアの一般情報",
                    "牙痛相关性": "R0",
                    "经历主体": "",
                    "内容功能": "",
                    "感知病因_标签": "",
                    "感知病因_证据": "",
                    "感知病因_关键词_EN": "",
                    "症状描述_标签": "",
                    "症状描述_证据": "",
                    "症状描述_关键词_EN": "",
                    "感知后果_标签": "",
                    "感知后果_证据": "",
                    "感知后果_关键词_EN": "",
                    "应对与管理_标签": "",
                    "应对与管理_证据": "",
                    "应对与管理_关键词_EN": "",
                    "情绪表达_标签": "",
                    "情绪表达_证据": "",
                    "情绪表达_关键词_EN": "",
                },
            ]
        ).to_excel(writer, sheet_name="1", index=False)

    predictions = [
        {
            "post_id": "p1",
            "country": "CHI",
            "language": "zh",
            "original_text": "牙疼得睡不着，吃了布洛芬。",
            "relevance_label": "R1",
            "experiencer_label": "E1",
            "content_function": "C1",
            "units": [
                {
                    "unit_id": "p1_u001",
                    "domain": "Symptom Description",
                    "evidence_span_original": "牙疼",
                    "surface_text_working": "牙疼",
                    "normalized_concept_en": "Toothache",
                    "concept_status": "new_candidate",
                    "support_type": "explicit",
                    "assertion": "present",
                    "confidence": 0.9,
                    "judge_verdict": "accept",
                },
                {
                    "unit_id": "p1_u002",
                    "domain": "Coping and Management",
                    "evidence_span_original": "布洛芬",
                    "surface_text_working": "布洛芬",
                    "normalized_concept_en": "Oral analgesic",
                    "concept_status": "new_candidate",
                    "support_type": "explicit",
                    "assertion": "present",
                    "confidence": 0.9,
                    "judge_verdict": "accept",
                },
                {
                    "unit_id": "p1_u003",
                    "domain": "Emotional Expression",
                    "evidence_span_original": "not in source",
                    "surface_text_working": "焦虑",
                    "normalized_concept_en": "Anxiety",
                    "concept_status": "new_candidate",
                    "support_type": "unsupported",
                    "assertion": "present",
                    "confidence": 0.3,
                    "judge_verdict": "reject",
                },
            ],
        },
        {
            "post_id": "p2",
            "country": "JPN",
            "language": "ja",
            "original_text": "口腔ケアの一般情報",
            "relevance_label": "R0",
            "experiencer_label": None,
            "content_function": None,
            "units": [],
        },
    ]
    pred_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )

    gold = load_gold(gold_path)
    pred = load_predictions(pred_path)
    evaluation = build_evaluation(gold, pred, ["p1", "p2"])

    assert evaluation["overall"]["eligibility"]["f1"] == 1.0
    assert evaluation["overall"]["unit_quality"]["unit_count"] == 3
    assert evaluation["overall"]["unit_quality"]["accepted_unit_count"] == 2
    assert evaluation["overall"]["unit_quality"]["hallucination_proxy_rate"] == 1 / 3

    consequence_row = next(
        row
        for row in evaluation["domain_metrics"]
        if row["group_type"] == "overall" and row["domain"] == "Perceived Consequences"
    )
    assert consequence_row["fn"] == 1
    assert consequence_row["f1"] == 0.0

    normalization = evaluation["overall"]["normalization_micro"]
    assert normalization["tp"] == 2
    assert normalization["fn"] == 1
