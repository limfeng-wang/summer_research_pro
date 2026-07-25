import pandas as pd
import pytest

from dental_ai.data_io import (
    PostColumnMap,
    infer_post_column_map,
    load_posts,
    read_extractions_jsonl,
    read_posts_jsonl,
    write_extractions_jsonl,
    write_posts_jsonl,
    write_unit_table,
)
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
)


def make_unit(**overrides):
    data = {
        "unit_id": "p1_u001",
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


def test_load_posts_from_csv_with_explicit_constants(tmp_path):
    path = tmp_path / "posts.csv"
    pd.DataFrame(
        [
            {"Post_ID": "p1", "Text": "智齿发炎疼得睡不着"},
            {"Post_ID": "p2", "Text": "吃了布洛芬还是没用"},
        ]
    ).to_csv(path, index=False)

    posts = load_posts(path, country=Country.CHI, language=Language.ZH)

    assert posts == [
        SourcePost(post_id="p1", country=Country.CHI, language=Language.ZH, text="智齿发炎疼得睡不着"),
        SourcePost(post_id="p2", country=Country.CHI, language=Language.ZH, text="吃了布洛芬还是没用"),
    ]


def test_load_posts_requires_country_and_language_when_not_mapped(tmp_path):
    path = tmp_path / "posts.csv"
    pd.DataFrame([{"post_id": "p1", "text": "牙疼"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="country must be provided"):
        load_posts(path)


def test_load_posts_accepts_explicit_column_map(tmp_path):
    path = tmp_path / "posts.csv"
    pd.DataFrame(
        [
            {
                "custom_id": "p1",
                "body": "歯が痛い",
                "country_code": "JPN",
                "lang": "ja",
            }
        ]
    ).to_csv(path, index=False)

    posts = load_posts(
        path,
        column_map=PostColumnMap(
            post_id="custom_id",
            text="body",
            country="country_code",
            language="lang",
        ),
    )

    assert posts[0] == SourcePost(
        post_id="p1",
        country=Country.JPN,
        language=Language.JA,
        text="歯が痛い",
    )


def test_infer_post_column_map_rejects_ambiguous_required_columns():
    with pytest.raises(ValueError, match="Ambiguous post_id"):
        infer_post_column_map(["post_id", "id", "text"])


def test_posts_jsonl_round_trip(tmp_path):
    path = tmp_path / "posts.jsonl"
    posts = [SourcePost(post_id="p1", country=Country.KOR, language=Language.KO, text="치통")]

    write_posts_jsonl(posts, path)

    assert read_posts_jsonl(path) == posts


def test_extractions_jsonl_round_trip(tmp_path):
    path = tmp_path / "extractions.jsonl"
    results = [
        ExtractionResult(
            post_id="p1",
            country=Country.CHI,
            language=Language.ZH,
            units=[make_unit()],
        )
    ]

    write_extractions_jsonl(results, path)

    assert read_extractions_jsonl(path) == results


def test_write_unit_table_exports_only_accepted_by_default(tmp_path):
    path = tmp_path / "units.csv"
    result = ExtractionResult(
        post_id="p1",
        country=Country.CHI,
        language=Language.ZH,
        units=[
            make_unit(),
            make_unit(
                unit_id="p1_u002",
                judge_verdict=JudgeVerdict.REJECT,
            ),
        ],
    )

    write_unit_table([result], path)

    rows = pd.read_csv(path)
    assert rows["unit_id"].tolist() == ["p1_u001"]
