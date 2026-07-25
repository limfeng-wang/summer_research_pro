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
        "evidence_span": "吃了布洛芬",
        "surface_text": "布洛芬",
        "normalized_concept": "Oral analgesic",
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
                evidence_span="智齿发炎",
                surface_text="智齿发炎",
                normalized_concept="Inflamed wisdom tooth",
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
        text="智齿发炎疼得睡不着，吃了布洛芬还是没用。",
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
        text="智齿发炎疼得睡不着。",
    )
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[make_unit()],
    )

    with pytest.raises(ValueError, match="evidence_span not found"):
        result.validate_against_post(post)


def test_unmapped_concepts_can_leave_normalized_concept_empty():
    unit = make_unit(
        normalized_concept="",
        concept_status=ConceptStatus.UNMAPPED,
        support_type=SupportType.IMPLICIT,
        judge_verdict=JudgeVerdict.NEEDS_HUMAN_REVIEW,
    )

    assert unit.normalized_concept == ""


def test_mapped_concepts_require_normalized_concept():
    with pytest.raises(ValidationError, match="normalized_concept is required"):
        make_unit(normalized_concept="")


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
            "unit_id": "p1_u001",
            "domain": "Coping and Management",
            "evidence_span": "吃了布洛芬",
            "surface_text": "布洛芬",
            "normalized_concept": "Oral analgesic",
            "concept_status": "existing_dictionary",
            "support_type": "explicit",
            "assertion": "present",
            "temporality": "unknown",
            "sentiment_or_outcome": "unknown",
            "confidence": 0.92,
            "judge_verdict": "accept",
        }
    ]
