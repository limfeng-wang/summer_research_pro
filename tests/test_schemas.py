import pytest
from pydantic import ValidationError

from dental_ai.schemas import (
    AssertionStatus,
    CSMDomain,
    ConceptStatus,
    Country,
    ExtractionResult,
    JudgeVerdict,
    Language,
    NarrativeUnit,
    SourcePost,
    SupportType,
    flatten_units,
)


def make_unit(**overrides):
    data = {
        "domain": CSMDomain.COPING_AND_MANAGEMENT,
        "evidence_span_original": "吃了布洛芬",
        "surface_text_working": "布洛芬",
        "working_language": Language.ZH,
        "normalized_concept_en": "Oral analgesic",
        "concept_status": ConceptStatus.EXISTING_DICTIONARY,
        "support_type": SupportType.EXPLICIT,
        "assertion": AssertionStatus.PRESENT,
        "confidence": 0.92,
        "judge_verdict": JudgeVerdict.ACCEPT,
    }
    data.update(overrides)
    return NarrativeUnit(**data)


def test_domain_labels_are_derived_from_accepted_units():
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[
            make_unit(),
            make_unit(
                domain=CSMDomain.PERCEIVED_CAUSE,
                evidence_span_original="智齿发炎",
                surface_text_working="智齿发炎",
                normalized_concept_en="Inflamed wisdom tooth",
                judge_verdict=JudgeVerdict.REJECT,
            ),
        ],
    )

    labels = result.domain_labels

    assert labels[CSMDomain.COPING_AND_MANAGEMENT] == 1
    assert labels[CSMDomain.PERCEIVED_CAUSE] == 0


def test_validate_against_post_checks_metadata_and_evidence_spans():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_text="智齿发炎疼得睡不着，吃了布洛芬还是没用。",
    )
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[make_unit()],
    )

    result.validate_against_post(post)


def test_validate_against_post_rejects_ungrounded_spans():
    post = SourcePost(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        original_text="智齿发炎疼得睡不着。",
    )
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[make_unit()],
    )

    with pytest.raises(ValueError, match="evidence_span_original not found"):
        result.validate_against_post(post)


def test_unmapped_concepts_can_leave_normalized_concept_empty():
    unit = make_unit(
        normalized_concept_en="",
        concept_status=ConceptStatus.UNMAPPED,
        support_type=SupportType.IMPLICIT,
        judge_verdict=JudgeVerdict.NEEDS_HUMAN_REVIEW,
    )

    assert unit.normalized_concept_en == ""


def test_mapped_concepts_require_normalized_concept():
    with pytest.raises(ValidationError, match="normalized_concept_en is required"):
        make_unit(normalized_concept_en="")


def test_legacy_unit_field_aliases_are_accepted():
    unit = NarrativeUnit(
        domain=CSMDomain.COPING_AND_MANAGEMENT,
        evidence_span="歯が痛いので薬を飲んだ",
        surface_text_original="吃了止痛药",
        normalized_concept="Ibuprofen use",
        concept_status=ConceptStatus.NEW_CANDIDATE,
        support_type=SupportType.EXPLICIT,
        assertion=AssertionStatus.PRESENT,
        confidence=0.8,
        judge_verdict=JudgeVerdict.ACCEPT,
    )

    assert unit.evidence_span_original == "歯が痛いので薬を飲んだ"
    assert unit.surface_text_working == "吃了止痛药"
    assert unit.working_language == Language.ZH
    assert unit.normalized_concept_en == "Ibuprofen use"


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        make_unit(confidence=1.5)


def test_assigns_missing_unit_ids_deterministically():
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[make_unit(), make_unit(unit_id="p1_u002")],
    ).with_assigned_unit_ids()

    assert [unit.unit_id for unit in result.units] == ["p1_u001", "p1_u002"]


def test_duplicate_unit_ids_are_rejected():
    with pytest.raises(ValidationError, match="unit_id values must be unique"):
        ExtractionResult(
            post_id="p1",
            country=Country.CHI,
            language=Language.ZH,
            units=[make_unit(unit_id="u1"), make_unit(unit_id="u1")],
        )


def test_flatten_units_exports_unit_rows():
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[make_unit(unit_id="p1_u001")],
    )

    rows = flatten_units([result])

    assert rows == [
        {
            "post_id": "p1",
            "country": "CHI",
            "language": "zh",
            "relevance_label": "",
            "experiencer_label": "",
            "content_function": "",
            "unit_id": "p1_u001",
            "domain": "Coping and Management",
            "evidence_span_original": "吃了布洛芬",
            "surface_text_working": "布洛芬",
            "working_language": "zh",
            "normalized_concept_en": "Oral analgesic",
            "concept_status": "existing_dictionary",
            "support_type": "explicit",
            "assertion": "present",
            "temporality": "unknown",
            "sentiment_or_outcome": "unknown",
            "confidence": 0.92,
            "judge_verdict": "accept",
        }
    ]
