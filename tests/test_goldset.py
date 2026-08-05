import json

import pytest

from dental_ai.goldset import (
    assert_ready_for_rag_seed,
    load_csm_gold_json,
    primary_csm_results,
    proxy_csm_results,
    summarize_csm_gold,
)
from dental_ai.schemas import CSMDomain, ExperiencerLabel


def make_record(**overrides):
    data = {
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
    data.update(overrides)
    return data


def test_load_csm_gold_json_validates_and_accepts_current_aliases(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([make_record()], ensure_ascii=False), encoding="utf-8")

    results = load_csm_gold_json(path)

    assert len(results) == 1
    assert results[0].units[0].surface_text_working == "吃了布洛芬"
    assert results[0].domain_labels[CSMDomain.COPING_AND_MANAGEMENT] == 1


def test_load_csm_gold_json_rejects_missing_evidence_span(tmp_path):
    record = make_record()
    record["units"][0]["evidence_span_original"] = "不存在的证据"
    path = tmp_path / "gold.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence_span_original not found"):
        load_csm_gold_json(path)


def test_primary_and_proxy_filters():
    primary = make_record(post_id="p1", experiencer_label="E1")
    proxy = make_record(post_id="p2", experiencer_label="E2")
    records = [
        load_csm_gold_json_from_obj(primary),
        load_csm_gold_json_from_obj(proxy),
    ]

    assert [result.post_id for result in primary_csm_results(records)] == ["p1"]
    assert [result.post_id for result in proxy_csm_results(records)] == ["p2"]
    assert proxy_csm_results(records)[0].experiencer_label == ExperiencerLabel.E2


def test_summarize_csm_gold_counts_core_fields():
    result = load_csm_gold_json_from_obj(make_record())

    summary = summarize_csm_gold([result])

    assert summary["posts"] == 1
    assert summary["units"] == 1
    assert summary["span_exact_matches"] == 1
    assert summary["countries"] == {"CHI": 1}
    assert summary["content_functions"] == {"C1": 1}
    assert summary["primary_posts"] == 1
    assert summary["proxy_posts"] == 0


def test_assert_ready_for_rag_seed_requires_primary_records():
    proxy = load_csm_gold_json_from_obj(make_record(experiencer_label="E2"))

    with pytest.raises(ValueError, match="Not enough primary CSM records"):
        assert_ready_for_rag_seed([proxy], min_primary_posts=1)


def test_assert_ready_for_rag_seed_can_require_proxy_records():
    primary = load_csm_gold_json_from_obj(make_record(experiencer_label="E1"))

    with pytest.raises(ValueError, match="Not enough proxy CSM records"):
        assert_ready_for_rag_seed([primary], min_primary_posts=1, min_proxy_posts=1)

    proxy = load_csm_gold_json_from_obj(make_record(post_id="p2", experiencer_label="E2"))

    assert_ready_for_rag_seed([primary, proxy], min_primary_posts=1, min_proxy_posts=1)


def load_csm_gold_json_from_obj(obj):
    from dental_ai.schemas import ExtractionResult

    result = ExtractionResult.model_validate(obj)
    result.validate_evidence_spans()
    return result
