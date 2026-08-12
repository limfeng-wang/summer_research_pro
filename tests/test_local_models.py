import json

from dental_ai.local_models import (
    LocalCombinedClassifier,
    _extract_csm_payload,
    _extract_label_payload,
    _extract_json_object,
    _extract_judge_verdict_payload,
    _normalize_csm_payload_enums,
    apply_deterministic_csm_safeguards,
    apply_judge_verdict_payload,
    apply_classification_safeguards,
    apply_commercial_safeguard,
    apply_generic_procedure_demote_safeguard,
    apply_relevance_safeguard,
    apply_short_lived_pain_narrative_safeguard,
    apply_weak_commercial_demote_safeguard,
    has_lived_pain_burden_sequence,
    has_strong_commercial_evidence,
    mark_units_needing_human_review,
    repair_evidence_spans,
)
from dental_ai.schemas import (
    AssertionStatus,
    CSMDomain,
    ConceptStatus,
    ContentFunctionLabel,
    Country,
    ExperiencerLabel,
    ExtractionResult,
    JudgeVerdict,
    Language,
    NarrativeUnit,
    RelevanceLabel,
    SourcePost,
    SupportType,
)


def test_extract_label_payload_recovers_labels_from_truncated_json():
    text = (
        '{\n'
        '  "experiencer_label": "E3",\n'
        '  "content_function": "C3",\n'
        '  "experiencer_evidence": "",\n'
        '  "content_function_evidence": "long evidence that never closes'
    )

    assert _extract_label_payload(text, ["experiencer_label", "content_function"]) == {
        "experiencer_label": "E3",
        "content_function": "C3",
    }


def test_combined_classifier_returns_relevance_and_r1_labels_with_safeguards():
    class StubLM:
        def generate_json_text(self, system_prompt, user_payload, config):
            return json.dumps(
                {
                    "relevance_label": "R1",
                    "experiencer_label": "E1",
                    "content_function": "C5",
                }
            )

    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="昨晚牙疼的一晚上没合眼 但是不想吃药",
    )

    relevance, experiencer, content_function = LocalCombinedClassifier(StubLM()).classify(post)

    assert relevance == RelevanceLabel.R1
    assert experiencer == ExperiencerLabel.E1
    assert content_function == ContentFunctionLabel.C1


def test_combined_classifier_recovers_r1_labels_from_truncated_evidence_json():
    class StubLM:
        def generate_json_text(self, system_prompt, user_payload, config):
            return (
                '{\n'
                '  "relevance_label": "R1",\n'
                '  "experiencer_label": "E1",\n'
                '  "content_function": "C4",\n'
                '  "relevance_evidence": "牙痛,真的太难受了",\n'
                '  "experiencer_evidence": "我和牙说能不能别疼了",\n'
                '  "content_function_evidence": "牙发炎痛到哭,这个诀窍'
            )

    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="牙痛,真的太难受了。我和牙说能不能别疼了。我立马下单牙痛宁喷剂,这支喷剂真的好用。",
    )

    relevance, experiencer, content_function = LocalCombinedClassifier(StubLM()).classify(post)

    assert relevance == RelevanceLabel.R1
    assert experiencer == ExperiencerLabel.E1
    assert content_function == ContentFunctionLabel.C4


def test_extractor_units_are_reset_to_needs_human_review_before_judge():
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[
            NarrativeUnit(
                domain=CSMDomain.SYMPTOM_DESCRIPTION,
                evidence_span_original="牙疼",
                surface_text_working="牙疼",
                normalized_concept_en="Tooth pain",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            )
        ],
    )

    sanitized = mark_units_needing_human_review(result)

    assert sanitized.units[0].judge_verdict == JudgeVerdict.NEEDS_HUMAN_REVIEW


def test_compact_judge_payload_updates_verdicts_by_unit_id():
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[
            NarrativeUnit(
                unit_id="p1_u001",
                domain=CSMDomain.SYMPTOM_DESCRIPTION,
                evidence_span_original="牙疼",
                surface_text_working="牙疼",
                normalized_concept_en="Tooth pain",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.NEEDS_HUMAN_REVIEW,
            ),
            NarrativeUnit(
                unit_id="p1_u002",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="挂号",
                surface_text_working="挂号",
                normalized_concept_en="Appointment registration",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.NEEDS_HUMAN_REVIEW,
            ),
        ],
    )

    judged = apply_judge_verdict_payload(
        result,
        {
            "unit_verdicts": [
                {"unit_id": "p1_u001", "judge_verdict": "accept"},
                {"unit_id": "p1_u002", "judge_verdict": "reject"},
            ]
        },
    )

    assert [unit.judge_verdict for unit in judged.units] == [JudgeVerdict.ACCEPT, JudgeVerdict.REJECT]


def test_compact_judge_payload_keeps_missing_units_for_human_review():
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[
            NarrativeUnit(
                unit_id="p1_u001",
                domain=CSMDomain.SYMPTOM_DESCRIPTION,
                evidence_span_original="牙疼",
                surface_text_working="牙疼",
                normalized_concept_en="Tooth pain",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            )
        ],
    )

    judged = apply_judge_verdict_payload(result, {"unit_verdicts": []})

    assert judged.units[0].judge_verdict == JudgeVerdict.NEEDS_HUMAN_REVIEW


def test_deterministic_csm_safeguards_reject_admin_cost_and_negated_units():
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[
            NarrativeUnit(
                unit_id="p1_u001",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="检查费70r,拍了一个牙片",
                surface_text_working="拍牙片检查",
                normalized_concept_en="Dental imaging fee",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
            NarrativeUnit(
                unit_id="p1_u002",
                domain=CSMDomain.SYMPTOM_DESCRIPTION,
                evidence_span_original="有龋齿,不痛,没发炎",
                surface_text_working="无疼痛和炎症",
                normalized_concept_en="Absence of dental pain and inflammation",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
            NarrativeUnit(
                unit_id="p1_u003",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="拔完牙24h内可以冰敷一下",
                surface_text_working="术后冰敷",
                normalized_concept_en="Cold therapy after extraction",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
            NarrativeUnit(
                unit_id="p1_u004",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="我拔完牙疼到睡不着, 冰敷后缓解很多",
                surface_text_working="冰敷缓解术后疼痛",
                normalized_concept_en="Cold therapy relieved post-extraction pain",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
            NarrativeUnit(
                unit_id="p1_u005",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="希望我的牙能好着",
                surface_text_working="希望牙能好",
                normalized_concept_en="Dental health aspiration",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
            NarrativeUnit(
                unit_id="p1_u006",
                domain=CSMDomain.SYMPTOM_DESCRIPTION,
                evidence_span_original="磨牙的时候有点慌",
                surface_text_working="磨牙时紧张",
                normalized_concept_en="Anxiety during dental procedure",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
            NarrativeUnit(
                unit_id="p1_u007",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="歯医者に電話しなきゃ",
                surface_text_working="需要联系牙医",
                normalized_concept_en="Dental care seeking intention",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
            NarrativeUnit(
                unit_id="p1_u008",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="발치하고 봉(?) 심어서 (아픕니다!)",
                surface_text_working="拔牙过程疼痛",
                normalized_concept_en="Dental procedure with pain",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            ),
        ],
    )

    safeguarded = apply_deterministic_csm_safeguards(result)

    assert [unit.judge_verdict for unit in safeguarded.units] == [
        JudgeVerdict.REJECT,
        JudgeVerdict.REJECT,
        JudgeVerdict.REJECT,
        JudgeVerdict.ACCEPT,
        JudgeVerdict.REJECT,
        JudgeVerdict.REJECT,
        JudgeVerdict.REJECT,
        JudgeVerdict.ACCEPT,
    ]
    assert safeguarded.units[0].support_type == SupportType.UNSUPPORTED
    assert safeguarded.units[1].support_type == SupportType.UNSUPPORTED
    assert safeguarded.units[2].support_type == SupportType.UNSUPPORTED
    assert safeguarded.units[4].support_type == SupportType.UNSUPPORTED
    assert safeguarded.units[5].support_type == SupportType.UNSUPPORTED
    assert safeguarded.units[6].support_type == SupportType.UNSUPPORTED


def test_repair_evidence_spans_fixes_near_exact_model_typos():
    post = SourcePost(
        post_id="p1",
        country=Country.KOR,
        language=Language.KO,
        text_clean="어젠 치아 조각해서 발치하고 봉(?) 심어서 (아픕니다!) 더 정줄 놓음.",
    )
    result = ExtractionResult(
        post_id="p1",
        country=Country.KOR,
        language=Language.KO,
        units=[
            NarrativeUnit(
                unit_id="p1_u001",
                domain=CSMDomain.COPING_AND_MANAGEMENT,
                evidence_span_original="어젠 치아 조각해서 발치하고 봉(?) 심해서 (아픕니다!) 더 정줄 놓음",
                surface_text_working="拔牙过程疼痛",
                normalized_concept_en="Dental procedure with pain",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.EXPLICIT,
                assertion=AssertionStatus.PRESENT,
                confidence=0.9,
                judge_verdict=JudgeVerdict.ACCEPT,
            )
        ],
    )

    repaired = repair_evidence_spans(result, post)

    assert repaired.units[0].evidence_span_original == "어젠 치아 조각해서 발치하고 봉(?) 심어서 (아픕니다!) 더 정줄 놓음"


def test_repair_evidence_spans_fixes_short_repeated_terminal_punctuation():
    post = SourcePost(
        post_id="p1",
        country=Country.JPN,
        language=Language.JA,
        text_clean="歯が痛いぞ!虫歯?!疲れ?!?!",
    )
    result = ExtractionResult(
        post_id="p1",
        country=Country.JPN,
        language=Language.JA,
        units=[
            NarrativeUnit(
                unit_id="p1_u001",
                domain=CSMDomain.PERCEIVED_CAUSE,
                evidence_span_original="虫歯?!?",
                surface_text_working="怀疑是蛀牙",
                normalized_concept_en="Dental caries",
                concept_status=ConceptStatus.NEW_CANDIDATE,
                support_type=SupportType.UNSUPPORTED,
                assertion=AssertionStatus.UNCERTAIN,
                confidence=0.9,
                judge_verdict=JudgeVerdict.REJECT,
            )
        ],
    )

    repaired = repair_evidence_spans(result, post)

    assert repaired.units[0].evidence_span_original == "虫歯?!"


def test_normalize_csm_payload_enums_repairs_model_enum_drift():
    payload = {
        "post_id": "p1",
        "country": "CHI",
        "language": "zh",
        "units": [
            {
                "unit_id": "p1_u001",
                "domain": "Symptom Description",
                "evidence_span_original": "牙疼",
                "surface_text_working": "牙疼",
                "normalized_concept_en": "Dental pain",
                "concept_status": "new_candidate",
                "support_type": "explicit",
                "assertion": "past",
                "sentiment_or_outcome": "mixed",
                "confidence": 0.9,
                "judge_verdict": "needs_human_review",
            },
            {
                "unit_id": "p1_u002",
                "domain": "Coping and Management",
                "evidence_span_original": "现在吃药",
                "surface_text_working": "现在吃药",
                "normalized_concept_en": "Medication use",
                "concept_status": "new_candidate",
                "support_type": "explicit",
                "assertion": "current",
                "sentiment_or_outcome": "relieved",
                "confidence": 0.9,
                "judge_verdict": "needs_human_review",
            },
        ],
    }

    normalized = _normalize_csm_payload_enums(payload)
    result = ExtractionResult.model_validate(normalized)

    assert result.units[0].assertion == AssertionStatus.PRESENT
    assert result.units[0].temporality.value == "past"
    assert result.units[0].sentiment_or_outcome.value == "unknown"
    assert result.units[1].assertion == AssertionStatus.PRESENT
    assert result.units[1].temporality.value == "current"
    assert result.units[1].sentiment_or_outcome.value == "effective"


def test_json_object_parser_strips_thinking_text():
    text = '<think>reasoning that should not be parsed</think>{"unit_verdicts":[]}'

    assert _extract_json_object(text) == {"unit_verdicts": []}


def test_csm_payload_recovers_units_from_missing_comma_json():
    post = SourcePost(post_id="p1", country=Country.CHI, language=Language.ZH, text_clean="牙疼, 吃了布洛芬")
    text = """
{
  "post_id": "p1",
  "country": "CHI",
  "language": "zh",
  "units": [
    {
      "domain": "Symptom Description",
      "evidence_span_original": "牙疼",
      "surface_text_working": "牙疼",
      "working_language": "zh",
      "normalized_concept_en": "Tooth pain",
      "concept_status": "new_candidate",
      "support_type": "explicit",
      "assertion": "present",
      "temporality": "current",
      "sentiment_or_outcome": "negative",
      "confidence": 0.9,
      "judge_verdict": "needs_human_review"
    }
    {
      "domain": "Coping and Management",
      "evidence_span_original": "吃了布洛芬",
      "surface_text_working": "服用布洛芬",
      "working_language": "zh",
      "normalized_concept_en": "Ibuprofen use",
      "concept_status": "new_candidate",
      "support_type": "explicit",
      "assertion": "present",
      "temporality": "past",
      "sentiment_or_outcome": "unknown",
      "confidence": 0.8,
      "judge_verdict": "needs_human_review"
    }
  ]
}
"""

    payload = _extract_csm_payload(text, post)

    assert payload["post_id"] == "p1"
    assert [unit["domain"] for unit in payload["units"]] == [
        "Symptom Description",
        "Coping and Management",
    ]


def test_csm_payload_does_not_recover_non_unit_json_fragments():
    post = SourcePost(post_id="p1", country=Country.CHI, language=Language.ZH, text_clean="牙疼")
    text = '{"post_id":"p1","units":[{"domain":"Symptom Description"}'

    try:
        _extract_csm_payload(text, post)
    except Exception as exc:
        assert type(exc).__name__ == "JSONDecodeError"
    else:
        raise AssertionError("expected malformed extraction without full units to fail")


def test_judge_payload_recovers_verdicts_from_missing_comma_json():
    text = """
{
  "unit_verdicts": [
    {
      "unit_id": "p1_u001",
      "judge_verdict": "reject",
      "reason": "admin cost"
    }
    {
      "unit_id": "p1_u002",
      "judge_verdict": "accept",
      "reason": "supported pain management"
    }
  ]
}
"""

    payload = _extract_judge_verdict_payload(text)

    assert payload == {
        "unit_verdicts": [
            {"unit_id": "p1_u001", "judge_verdict": "reject"},
            {"unit_id": "p1_u002", "judge_verdict": "accept"},
        ]
    }


def test_judge_payload_does_not_recover_without_verdicts():
    text = '{"unit_verdicts": [{"unit_id": "p1_u001", "reason": "missing verdict"}'

    try:
        _extract_judge_verdict_payload(text)
    except Exception as exc:
        assert type(exc).__name__ == "JSONDecodeError"
    else:
        raise AssertionError("expected malformed judge output without verdicts to fail")


def test_commercial_safeguard_promotes_product_advertorial_to_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "喝了一包芬必得精氨酸布洛芬颗粒, 真的就快速缓解了我的牙痛。"
            "不愧是西药大品牌! 全家都很信任它! #芬必得 #牙痛止痛药"
        ),
    )

    assert apply_commercial_safeguard(post, ContentFunctionLabel.C3) == ContentFunctionLabel.C4


def test_commercial_safeguard_preserves_noncommercial_c3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="牙痛急救办法: 可以短时间冰敷, 但如果持续疼痛需要及时就医。",
    )

    assert apply_commercial_safeguard(post, ContentFunctionLabel.C3) == ContentFunctionLabel.C3


def test_relevance_safeguard_demotes_oral_ulcer_only_posts():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="口腔溃疡无痛治好。用双氧水消毒, 第二天早上就不那么痛了。",
    )

    assert apply_relevance_safeguard(post, RelevanceLabel.R1) == RelevanceLabel.R0


def test_commercial_safeguard_requires_promotion_not_care_logistics():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "作者本人的是阻生齿, 拔牙两次。第一次手术费975, 第二次手术费1300左右。"
            "挂号一定要趁早, 可以手机预约挂号。拔完牙24h内可以冰敷。"
            "价格不同地区会有所差异, 仅供参考。"
        ),
    )

    assert not has_strong_commercial_evidence(post)
    assert apply_commercial_safeguard(post, ContentFunctionLabel.C1) == ContentFunctionLabel.C1
    assert apply_commercial_safeguard(post, ContentFunctionLabel.C3) == ContentFunctionLabel.C3


def test_personal_cost_process_without_lived_burden_stays_c3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "拔牙×2, 第一次手术费975, 第二次手术费1300左右。"
            "作者本人的是阻生齿。首先关于挂号的小tips: 可以提前预约挂号。"
            "拔牙后的注意事项: 拔完牙24h内可以冰敷一下。"
        ),
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C3,
    )


def test_weak_commercial_demote_safeguard_turns_personal_cost_post_back_to_c3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "拔牙×2, 第一次手术费975, 第二次手术费1300左右。"
            "作者本人的是阻生齿。首先关于挂号的小tips: 可以在手机上提前预约挂号。"
            "拔牙后的注意事项: 拔完牙24h内可以冰敷一下。"
        ),
    )

    assert apply_weak_commercial_demote_safeguard(post, ContentFunctionLabel.C4) == ContentFunctionLabel.C3
    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C4) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C3,
    )


def test_generic_procedure_demote_safeguard_preserves_lived_pain_narrative():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "前段时间我智齿发炎疼到睡不着, 吃饭只能喝流食。"
            "后来去冲洗, 回家冰敷后终于缓解。"
        ),
    )

    assert has_lived_pain_burden_sequence(post)
    assert apply_generic_procedure_demote_safeguard(post, ExperiencerLabel.E1, ContentFunctionLabel.C1) == (
        ContentFunctionLabel.C1
    )


def test_generic_procedure_demote_safeguard_demotes_wisdom_tooth_logistics():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "拔牙×2, 第一次手术费975, 第二次手术费1300左右。"
            "作者本人的是阻生齿, 有龋齿, 不痛, 没发炎。"
            "首先关于挂号的小tips: 可以提前预约挂号。"
            "拔牙后的注意事项: 拔完牙24h内可以冰敷一下。"
        ),
    )

    assert not has_lived_pain_burden_sequence(post)
    assert apply_generic_procedure_demote_safeguard(post, ExperiencerLabel.E1, ContentFunctionLabel.C1) == (
        ContentFunctionLabel.C3
    )


def test_short_lived_pain_narrative_safeguard_promotes_meaningful_c5_to_c1():
    posts = [
        SourcePost(
            post_id="j1",
            country=Country.JPN,
            language=Language.JA,
            text_clean="歯が痛くて眠れないので 早く病院あいてよぉぉぉ...",
        ),
        SourcePost(
            post_id="j2",
            country=Country.JPN,
            language=Language.JA,
            text_clean="歯が痛くて咽び泣いてる。",
        ),
        SourcePost(
            post_id="k1",
            country=Country.KOR,
            language=Language.KO,
            text_clean="갑자기 개 큰 치통 생겨서 괴로움",
        ),
        SourcePost(
            post_id="c1",
            country=Country.CHI,
            language=Language.ZH,
            text_clean="昨晚疼的一晚上没合眼 但是不想吃药",
        ),
    ]

    for post in posts:
        assert apply_short_lived_pain_narrative_safeguard(
            post,
            ExperiencerLabel.E1,
            ContentFunctionLabel.C5,
        ) == ContentFunctionLabel.C1
        assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C5) == (
            ExperiencerLabel.E1,
            ContentFunctionLabel.C1,
        )


def test_classification_safeguards_promote_omitted_subject_lived_pain_to_e1_c1():
    post = SourcePost(
        post_id="c1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="根管治疗第三天\n牙疼真要命,治疗也很疼,想把那颗坏牙直接拔掉,大家好好刷牙吧",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C1,
    )


def test_classification_safeguards_keep_generic_pain_rhetoric_as_e3_c3():
    post = SourcePost(
        post_id="c1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="牙痛不是病,痛起来真要命! 弄清楚牙痛的真正原因才能对症下药。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )


def test_classification_safeguards_demote_weak_c4_lived_pain_to_c1():
    post = SourcePost(
        post_id="c1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="我已中招\n昨晚疼的一晚上没合眼\n但是不想吃药\n希望借着这个病毒瘦上十来斤#牙疼 #病毒",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C4) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C1,
    )


def test_classification_safeguards_keep_procedure_heavy_posts_c3_after_short_pain_rule():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "拔牙×2, 第一次手术费975, 第二次手术费1300左右。"
            "作者本人的是阻生齿, 有龋齿, 不痛, 没发炎。"
            "首先关于挂号的小tips: 可以提前预约挂号。"
            "拔牙后的注意事项: 拔完牙24h内可以冰敷一下, 镇静止痛。"
        ),
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C1) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C3,
    )


def test_classification_safeguards_keep_spray_advertorial_as_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="别慌!这支牙髓炎喷剂来啦。草本植萃, 使用超简单, 喷头对准疼痛牙齿覆盖喷涂。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C4) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C4,
    )


def test_short_lived_pain_narrative_safeguard_preserves_generic_c3_and_c4():
    generic = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="牙痛急救办法: 冰敷、盐水漱口、及时就医。",
    )
    advertorial = SourcePost(
        post_id="p2",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="牙疼后喝了一包芬必得, 不愧是大品牌, #芬必得 #牙痛止痛药",
    )

    assert apply_short_lived_pain_narrative_safeguard(
        generic,
        ExperiencerLabel.E1,
        ContentFunctionLabel.C3,
    ) == ContentFunctionLabel.C3
    assert apply_short_lived_pain_narrative_safeguard(
        advertorial,
        ExperiencerLabel.E1,
        ContentFunctionLabel.C4,
    ) == ContentFunctionLabel.C4


def test_classification_safeguards_preserve_first_person_advertorial_as_e1_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="喝了一包芬必得小绿盒, 真的缓解了我的牙痛。不愧是大品牌! #芬必得 #牙痛止痛药",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C4,
    )


def test_classification_safeguards_demote_broad_dental_hashtag_c4_to_c1():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean=(
            "大半夜智齿的蛀牙掉了一块!\n"
            "我真是醉了牙牙龈痛了两天我以为上火了就没管,半夜感觉牙里面有东西,"
            "一摸牙掉了一块,我早就准备去看牙来着,一直没带要去。"
            "#牙齿 #看牙 #补牙 #牙科"
        ),
    )

    assert not has_strong_commercial_evidence(post)
    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C4) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C1,
    )


def test_classification_safeguards_demote_rhetorical_quick_tips_c2_to_c3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_title="牙疼快速处理三招",
        text_clean="牙疼快速处理三招\n牙疼如何快速处理?\n自然是止疼药啦,家里常备布洛芬\n温盐水漱口\n冰敷",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C2) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )


def test_classification_safeguards_preserve_clinic_account_promotion_as_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="智齿冠周炎症状和治疗一篇说清楚。#上海看牙 #上海颂皓口腔 @上海颂皓口腔",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C4,
    )


def test_classification_safeguards_promote_medical_ad_license_to_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="补牙贵吗?首颗98 z250。专业补牙, 价格透明。(黔)医广(2024)第12-05-824 #补牙 #都匀牙美家",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C4,
    )


def test_classification_safeguards_do_not_treat_generic_oral_care_hashtags_as_c4():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="保护牙齿的关键是每天刷牙、用牙线、喝完饮料漱口。#口腔护理 #牙齿护理 #刷牙误区",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C4) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )


def test_classification_safeguards_demote_rhetorical_c2_to_c3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_title="智齿分类太复杂?一张图看懂!",
        text_clean="智齿分类太复杂?一张图看懂! 近中阻生、水平阻生、垂直阻生分别是什么。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C2) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )


def test_classification_safeguards_detect_specific_author_case():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="我自己是中位近中阻生智齿, 拔的时候医生说不算复杂, 现在恢复得很顺利。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3)[0] == ExperiencerLabel.E1


def test_classification_safeguards_demote_general_knowledge_to_e3():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        text_clean="什么是智齿冠周炎? 常见病因、症状、护理和治疗方法一篇说清楚。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )


def test_classification_safeguards_promote_genuine_question_to_c2():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_title="牙疼求助",
        text_clean="我牙疼三天了, 晚上睡不着, 要不要去急诊? 有没有人知道怎么办?",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E1, ContentFunctionLabel.C1) == (
        ExperiencerLabel.E1,
        ContentFunctionLabel.C2,
    )


def test_classification_safeguards_do_not_promote_rhetorical_education_title_to_c2():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_title="牙痛怎么办? 8个急救办法",
        text_clean="牙痛怎么办? 8个急救办法: 冰敷、盐水漱口、及时就医。",
    )

    assert apply_classification_safeguards(post, ExperiencerLabel.E3, ContentFunctionLabel.C3) == (
        ExperiencerLabel.E3,
        ContentFunctionLabel.C3,
    )


def test_relevance_safeguard_keeps_japanese_tooth_pain_variants():
    examples = [
        "親知らずが痛すぎて、早く抜きたいです。",
        "最近夜になると左半分の歯が全部痛い",
        "なんか前に治療した歯が痛いからロキソニン飲んで冷やして寝る...",
        "昨日の矯正で下前歯が終わってます...ゴム掛けしている歯も痛むし、痛い所だらけです。",
        "前回の虫歯治療・・・10数分後・・・えっ痛っ!!!!!",
        "歯の痛みがなくなって、ゆっくり休んでいます。",
    ]

    for index, text in enumerate(examples):
        post = SourcePost(
            post_id=f"jp{index}",
            country=Country.JPN,
            language=Language.JA,
            text_clean=text,
        )

        assert apply_relevance_safeguard(post, RelevanceLabel.R1) == RelevanceLabel.R1


def test_relevance_safeguard_still_demotes_non_pain_japanese_dental_noise():
    examples = [
        "3ヶ月先まで予約取れない歯医者とか。虫歯今から初期だけど、3ヶ月後は神経ぬいたりやん。",
        "久々の歯科定期検診で生まれてこの方25年間虫歯ゼロ記録更新。",
        "虫歯菌みたいなぬいかわいい。",
        "歯科衛生指導ソングの紹介です。",
    ]

    for index, text in enumerate(examples):
        post = SourcePost(
            post_id=f"jp_noise{index}",
            country=Country.JPN,
            language=Language.JA,
            text_clean=text,
        )

        assert apply_relevance_safeguard(post, RelevanceLabel.R1) == RelevanceLabel.R0
